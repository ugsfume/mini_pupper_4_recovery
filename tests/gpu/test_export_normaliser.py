"""An actor with an observation normaliser exports as the plain numpy policy and equals rsl_rl's actor."""
import numpy as np

from mp4_recovery.policy_file import NumpyPolicy, export


def _rsl_actor(hidden, mean, std):
    """rsl_rl's own actor model (MLPModel, obs_normalization=True) with these normaliser statistics."""
    import torch
    from tensordict import TensorDict
    from rsl_rl.models.mlp_model import MLPModel
    from mp4_recovery import obs_norm
    obs = TensorDict({"actor": torch.zeros(2, 45)}, batch_size=[2])
    m = MLPModel(obs, {"actor": ["actor"]}, "actor", 12, hidden_dims=hidden, activation="elu", obs_normalization=True,
                 distribution_cfg={"class_name": "GaussianDistribution", "init_std": 0.1, "std_type": "scalar"})
    sd = m.state_dict()
    sd.update(obs_norm.state(mean, std, 1000))
    m.load_state_dict(sd, strict=True)
    m.eval()
    return m, sd


def test_folded_normaliser_export_equals_the_torch_actor(tmp_path):
    """A 512-256-128 actor with an observation normaliser exports as the plain 45-in / 12-out numpy policy (the
    normaliser folded into the first layer) that equals rsl_rl's actor with its normaliser to 1e-5."""
    import torch
    from tensordict import TensorDict
    from mp4_recovery import obs_norm
    g = torch.Generator().manual_seed(3)
    mean = torch.randn(45, generator=g) * 0.5
    std = 0.05 + 2.0 * torch.rand(45, generator=g)
    std[6:9] = 0.0                                          # a constant input (the zeroed command): std + eps
    m, sd = _rsl_actor((512, 256, 128), mean, std)
    ck = tmp_path / "m.pt"
    torch.save({"actor_state_dict": sd}, ck)
    path = export(ck, tmp_path / "exp", scale=[1.57] * 12, band_deg=[90.0], kind="getup")
    p = NumpyPolicy(path)
    assert p.layers[0][0].shape == (512, 45) and p.layers[-1][0].shape == (12, 128) and len(p.layers) == 4
    assert p.meta["obs_normalizer"]["folded"] and p.meta["hidden"] == [512, 256, 128]
    x = (mean + std * torch.randn(256, 45, generator=g)).numpy().astype(np.float32)
    with torch.no_grad():
        want = m(TensorDict({"actor": torch.from_numpy(x)}, batch_size=[256])).numpy()
    assert np.abs(p(x) - want).max() < 1e-5, np.abs(p(x) - want).max()
    d = np.load(path)
    assert np.abs(p(d["fixture_obs"]) - d["fixture_act"]).max() < 1e-5
    # the fold itself: W' x + b' = W (x - mean) / (std + eps) + b
    w, b = np.random.default_rng(0).normal(size=(4, 45)), np.random.default_rng(1).normal(size=4)
    w2, b2 = obs_norm.fold(w, b, mean.numpy(), std.numpy())
    want = (x - mean.numpy()) / (std.numpy() + obs_norm.EPS) @ w.T + b
    assert np.abs(x @ w2.T + b2 - want).max() < 1e-4


def test_load_actor_any_size_with_normaliser(tmp_path):
    import torch
    from tensordict import TensorDict
    from mp4_recovery.gpu.actor import load_actor
    mean, std = torch.linspace(-1, 1, 45), torch.linspace(0.1, 2.0, 45)
    m, sd = _rsl_actor((64, 32), mean, std)
    torch.save({"actor_state_dict": sd}, tmp_path / "m.pt")
    net = load_actor(tmp_path / "m.pt", device="cpu")
    x = torch.randn(16, 45)
    with torch.no_grad():
        assert torch.allclose(net(x), m(TensorDict({"actor": x}, batch_size=[16])), atol=1e-6)
