"""SmoothPPO's smoothness terms on the policy mean (CAPS / L2C2: ``caps_temporal``, ``caps_spatial``) and the adaptive
learning rate's bounds (``lr_min`` / ``lr_max``), on a toy actor, critic and rollout storage (CPU)."""
import pytest
import torch
from rsl_rl.algorithms.ppo import PPO
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict

from mp4_recovery.gpu import algorithms
from mp4_recovery.gpu.algorithms import (SmoothPPO, caps_pair_indices, caps_spatial_loss, caps_temporal_loss)

N_ENVS, T, N_OBS, N_CRIT, N_ACT = 16, 8, 45, 50, 12
GROUPS = {"actor": ["actor"], "critic": ["critic"]}


def _obs(n, g):
    return TensorDict({"actor": torch.randn(n, N_OBS, generator=g) * 2.0 + 0.5,
                       "critic": torch.randn(n, N_CRIT, generator=g)}, batch_size=[n])


def _models():
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(1)
    obs = _obs(N_ENVS, g)
    actor = MLPModel(obs, GROUPS, "actor", N_ACT, hidden_dims=(32, 32), obs_normalization=True,
                     distribution_cfg={"class_name": "GaussianDistribution", "init_std": 0.1, "std_type": "scalar"})
    critic = MLPModel(obs, GROUPS, "critic", 1, hidden_dims=(32, 32), obs_normalization=True)
    actor.update_normalization(_obs(512, g))      # non-trivial normaliser statistics
    critic.update_normalization(_obs(512, g))
    return obs, actor, critic


def _fill(storage, actor, critic, seed=2, zero_adv=False):
    """A rollout of random observations, the actor's own samples, some dones, random advantages and returns."""
    g = torch.Generator().manual_seed(seed)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        for t in range(T):
            o = _obs(N_ENVS, g)
            tr = RolloutStorage.Transition()
            tr.observations = o
            with torch.no_grad():
                tr.actions = actor(o, stochastic_output=True)
                tr.actions_log_prob = actor.get_output_log_prob(tr.actions)
                tr.distribution_params = tuple(p.clone() for p in actor.output_distribution_params)
                tr.values = critic(o)
            tr.rewards = torch.randn(N_ENVS, generator=g)
            tr.dones = (torch.rand(N_ENVS, generator=g) < 0.15).to(torch.uint8)
            storage.add_transition(tr)
    storage.advantages.copy_(torch.zeros_like(storage.advantages) if zero_adv
                             else torch.randn(storage.advantages.shape, generator=g))
    storage.returns.copy_(storage.values + (0.0 if zero_adv else 0.1 * torch.randn(storage.values.shape, generator=g)))


def _alg(cls=SmoothPPO, seed=2, zero_adv=False, **kw):
    obs, actor, critic = _models()
    storage = RolloutStorage("rl", N_ENVS, T, obs, [N_ACT], "cpu")
    base = dict(num_learning_epochs=5, num_mini_batches=4, entropy_coef=0.0, learning_rate=1e-3, desired_kl=0.01)
    base.update(kw.pop("ppo", {}))
    alg = cls(actor, critic, storage, device="cpu", **base, **kw)
    _fill(storage, actor, critic, seed=seed, zero_adv=zero_adv)
    return alg


def _update(alg, seed=7):
    torch.manual_seed(seed)
    out = alg.update()
    return out, torch.random.get_rng_state()


def _params(alg):
    return [p.detach().clone() for p in list(alg.actor.parameters()) + list(alg.critic.parameters())]


@pytest.mark.parametrize("kw", [{}, {"caps_obs_noise": 0.05},
                                {"caps_temporal": 0.0, "caps_spatial": 0.0, "lr_min": 1e-5, "lr_max": 1e-2}])
def test_lambda_zero_is_the_stock_update(kw):
    stock = _alg(PPO)
    ours = _alg(**kw)
    _, rng0 = _update(stock)
    out, rng1 = _update(ours)
    for a, b in zip(_params(stock), _params(ours)):
        assert torch.equal(a, b)
    assert stock.learning_rate == ours.learning_rate
    assert torch.equal(rng0, rng1)          # the global RNG stream is the stock one
    assert "caps_temporal" not in out


def test_lambda_positive_changes_the_update_but_not_the_rng_stream():
    stock = _alg(PPO)
    ours = _alg(caps_temporal=1.0, caps_spatial=1.0, caps_obs_noise=0.05)
    _, rng0 = _update(stock)
    out, rng1 = _update(ours)
    assert not all(torch.equal(a, b) for a, b in zip(_params(stock), _params(ours)))
    assert torch.equal(rng0, rng1)
    assert out["caps_temporal"] > 0.0 and out["caps_spatial"] > 0.0 and out["kl"] > 0.0


def test_pair_indices_exclude_done_boundaries():
    dones = torch.zeros(4, 3, 1, dtype=torch.uint8)
    dones[1, 0] = 1             # env 0's episode ends at t 1: (1, 2) of env 0 is not a pair
    dones[0, 2] = 1             # env 2: (0, 1) is not a pair
    dones[3, 1] = 1             # the last step: no partner anyway
    got = caps_pair_indices(dones).tolist()
    want = [t * 3 + e for t in range(3) for e in range(3) if (t, e) not in ((1, 0), (0, 2))]
    assert got == want


def test_update_draws_every_valid_pair_once_per_epoch(monkeypatch):
    alg = _alg(caps_temporal=0.5)
    seen = []
    real = algorithms.caps_temporal_loss

    def spy(actor, flat_obs, idx, num_envs):
        seen.append(idx.clone())
        return real(actor, flat_obs, idx, num_envs)

    monkeypatch.setattr(algorithms, "caps_temporal_loss", spy)
    valid = set(caps_pair_indices(alg.storage.dones).tolist())
    dones = alg.storage.dones.reshape(T, N_ENVS)
    assert 0 < len(valid) < (T - 1) * N_ENVS          # the toy rollout has resets inside it
    _update(alg)
    assert len(seen) == 5 * 4
    for ep in range(5):
        idx = torch.cat(seen[ep * 4:(ep + 1) * 4]).tolist()
        assert sorted(idx) == sorted(valid)             # each valid pair exactly once per epoch
        for i in idx:
            assert dones[i // N_ENVS, i % N_ENVS] == 0 and i // N_ENVS < T - 1


def test_temporal_pairs_are_consecutive_steps_of_one_env():
    """A mean that reads the observation's first float: o(t, e)[0] = t + 100 e, so every pair differs by exactly 1."""
    obs, actor, _ = _models()
    actor.mlp = torch.nn.Sequential(torch.nn.Linear(N_OBS, N_ACT, bias=False))
    with torch.no_grad():
        actor.mlp[0].weight.zero_()
        actor.mlp[0].weight[0, 0] = 1.0
    actor.obs_normalizer = torch.nn.Identity()
    x = torch.zeros(T, N_ENVS, N_OBS)
    x[..., 0] = torch.arange(T)[:, None] + 100.0 * torch.arange(N_ENVS)[None, :]
    flat = TensorDict({"actor": x.flatten(0, 1)}, batch_size=[T * N_ENVS])
    dones = torch.zeros(T, N_ENVS, 1, dtype=torch.uint8)
    dones[3, 5] = 1
    idx = caps_pair_indices(dones)
    assert caps_temporal_loss(actor, flat, idx, N_ENVS).item() == pytest.approx(1.0)


def test_spatial_noise_is_in_normalised_units():
    """sigma is the std of the perturbation of the normalised input: a linear mean W z gives E||W eps||^2 =
    sigma^2 ||W||_F^2, whatever the normaliser's statistics."""
    obs, actor, _ = _models()
    actor.mlp = torch.nn.Sequential(torch.nn.Linear(N_OBS, N_ACT))
    g = torch.Generator().manual_seed(3)
    o = _obs(20000, g)
    with torch.no_grad():
        clean = actor(o)
        got = caps_spatial_loss(actor, o, clean, 0.05, torch.Generator().manual_seed(4)).item()
    want = 0.05 ** 2 * actor.mlp[0].weight.pow(2).sum().item()
    assert got == pytest.approx(want, rel=0.05)


def test_the_loss_decreases_on_a_toy_actor():
    """Zero advantages: the actor's only gradient is the smoothness loss; one update lowers both terms."""
    alg = _alg(zero_adv=True, caps_temporal=10.0, caps_spatial=10.0, caps_obs_noise=0.5)
    flat = alg.storage.observations.flatten(0, 1)
    idx = caps_pair_indices(alg.storage.dones)

    def terms():
        with torch.no_grad():
            mu = alg.actor(flat)
            return (caps_temporal_loss(alg.actor, flat, idx, N_ENVS).item(),
                    caps_spatial_loss(alg.actor, flat, mu, 0.5, torch.Generator().manual_seed(5)).item())

    t0, s0 = terms()
    out, _ = _update(alg)
    t1, s1 = terms()
    assert t1 < 0.9 * t0 and s1 < 0.9 * s0, (t0, t1, s0, s1)
    assert out["caps_temporal"] > 0.0 and out["caps_spatial"] > 0.0


def _record_lr(alg):
    lrs = []
    step = alg.optimizer.step

    def rec(*a, **k):
        lrs.append(alg.optimizer.param_groups[0]["lr"])
        return step(*a, **k)

    alg.optimizer.step = rec
    return lrs


def test_lr_ceiling_under_a_small_kl():
    """desired_kl 1: every mini-batch's KL is below half of it, so the rate grows x1.5 each step until the ceiling."""
    alg = _alg(lr_max=5e-5, ppo={"learning_rate": 1e-5, "desired_kl": 1.0})
    lrs = _record_lr(alg)
    out, _ = _update(alg)
    assert max(lrs) == pytest.approx(5e-5) and all(lr <= 5e-5 for lr in lrs)
    assert alg.learning_rate == pytest.approx(5e-5) and lrs[-1] == pytest.approx(5e-5)
    assert out["lr_mean"] <= 5e-5 and 0.0 < out["kl"] < 0.5
    stock = _alg(PPO, ppo={"learning_rate": 1e-5, "desired_kl": 1.0})     # without the ceiling it goes past 5e-5
    _update(stock)
    assert stock.learning_rate > 5e-5


def test_lr_start_above_the_ceiling_is_clamped():
    alg = _alg(lr_max=5e-5, ppo={"learning_rate": 1e-4, "desired_kl": 1.0})
    lrs = _record_lr(alg)
    _update(alg)
    assert all(lr <= 5e-5 for lr in lrs)


def test_lr_floor_under_a_large_kl():
    alg = _alg(lr_min=2e-4, ppo={"learning_rate": 1e-3, "desired_kl": 1e-12})
    lrs = _record_lr(alg)
    _update(alg)
    assert min(lrs) == pytest.approx(2e-4) and all(lr >= 2e-4 for lr in lrs)


def test_terms_skipped_in_the_actor_warmup():
    alg = _alg(caps_temporal=1.0, caps_spatial=1.0, caps_obs_noise=0.05, actor_warmup_iters=1)
    actor0 = [p.detach().clone() for p in alg.actor.parameters()]
    out, _ = _update(alg)
    assert out["caps_temporal"] == 0.0 and out["caps_spatial"] == 0.0
    assert all(torch.equal(a, p) for a, p in zip(actor0, alg.actor.parameters()))   # the actor did not move


def test_bad_options():
    with pytest.raises(ValueError):
        _alg(caps_spatial=1.0)                  # no noise
    with pytest.raises(ValueError):
        _alg(lr_min=1e-2, lr_max=1e-3)
    with pytest.raises(ValueError):
        _alg(caps_temporal=-1.0)


