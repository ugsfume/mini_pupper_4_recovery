"""The target low-pass filter: the runner's decoder (contract.ActionDecoder) with alpha 0 skips it, bit for
bit; the filter's form and its state (the previous applied target, seeded by reset); the training ramp; the
export meta and the get-up task's options; then on the GPU: the simulator's decoder (mdp.ServoAction) against the
runner's over random actions through the band, the step limit and the get-up bounds, alpha 0 bit-identical to the
decoder without the filter (scalar and per-env), and the reset seeding."""
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mp4_recovery import limits as L
from mp4_recovery.contract import STEP_LIMIT_DEG, ActionDecoder, Contract, policy_filter_alpha

C = Contract.load()
TICK = 3100 / 10240                                     # one firmware tick, deg
GETUP_SCALE = [1.57] * 12
GETUP_BAND = [90.0] * 12


def decode_unfiltered(d: ActionDecoder, raw) -> dict:
    """ActionDecoder.__call__ without the filter, on ``d``'s state."""
    a = np.clip(np.asarray(raw, dtype=float), -1.0, 1.0)
    rel = np.clip(d.scale * a, -d.band, d.band)
    rel = np.clip(rel, d.prev_rel - d.limit, d.prev_rel + d.limit)
    phys = d.physical(rel)
    dd, phys = d.c.quantise_rel(phys) if d.quantise else (None, phys)
    rel = phys - d.offset
    d.prev_rel = rel
    d.prev_action = rel / d.scale
    return {"dd": dd, "rel": phys, "rel_policy": rel, "action": d.prev_action.copy()}


# ------------------------------------------------------------------------------------------------------------- CPU
@pytest.mark.parametrize("kw", [dict(scale=0.25), dict(scale=GETUP_SCALE, band_deg=GETUP_BAND,
                                                      bounds=L.decoder_bounds("getup")),
                                dict(scale=0.5, bounds=L.decoder_bounds("walk"), stand_offset_abd_deg=10.0)])
def test_numpy_alpha_zero_is_the_decoder_without_the_filter(kw):
    filtered, plain = ActionDecoder(C, filter_alpha=0.0, **kw), ActionDecoder(C, **kw)
    rng = np.random.default_rng(1)
    rel0 = np.radians(rng.uniform(-60, 60, 12))
    filtered.reset(rel0)
    plain.reset(rel0)
    for _ in range(200):
        raw = 2.0 * rng.normal(size=12)
        x, y = filtered(raw), decode_unfiltered(plain, raw)
        assert all(np.array_equal(x[k], y[k]) for k in ("dd", "rel", "rel_policy", "action"))
        assert np.array_equal(filtered.prev_rel, plain.prev_rel) and np.array_equal(filtered.prev_action, plain.prev_action)


@pytest.mark.parametrize("alpha", [0.5, 0.75])
def test_numpy_filter_form_and_reset_seeding(alpha):
    """Inside the clamps a step moves (1 - alpha) of the way from the previous applied target to the commanded one;
    reset(rel0) seeds the state (the measured pose, clipped to the band) and the previous action follows it."""
    d = ActionDecoder(C, GETUP_SCALE, band_deg=GETUP_BAND, quantise=False, filter_alpha=alpha)
    rel0 = np.radians(np.linspace(-30, 30, 12))
    d.reset(rel0)
    assert np.allclose(d.prev_rel, rel0) and np.allclose(d.prev_action, rel0 / 1.57)
    tgt = rel0 + np.radians(8.0)                                # within the band and 20 deg / step
    out = d(tgt / 1.57)
    assert np.allclose(out["rel"], rel0 + (1 - alpha) * np.radians(8.0))
    assert np.allclose(d.prev_rel, out["rel"]) and np.allclose(out["action"], out["rel"] / 1.57)
    for _ in range(60):                                         # held: converges on the target
        out = d(tgt / 1.57)
    assert np.allclose(out["rel"], tgt, atol=1e-6)
    far = ActionDecoder(C, GETUP_SCALE, band_deg=GETUP_BAND, quantise=False, filter_alpha=alpha)
    far.reset(np.zeros(12))
    out = far(np.ones(12))                                      # 90 deg away: (1 - alpha) x 90 > 20: the step limit
    assert np.allclose(np.degrees(out["rel"]), min((1 - alpha) * 90.0, STEP_LIMIT_DEG))
    clipped = ActionDecoder(C, GETUP_SCALE, band_deg=GETUP_BAND, filter_alpha=alpha)
    clipped.reset(np.radians([120.0] * 12))                     # a pose beyond the band: the state is clipped to it
    assert np.allclose(np.degrees(clipped.prev_rel), 90.0)
    with pytest.raises(ValueError):
        ActionDecoder(C, 0.25, filter_alpha=1.0)


def test_policy_filter_alpha_reader():
    assert policy_filter_alpha({}) == 0.0 and policy_filter_alpha({"action_filter_alpha": None}) == 0.0
    assert policy_filter_alpha({"action_filter_alpha": 0.75}) == 0.75
    with pytest.raises(ValueError):
        policy_filter_alpha({"action_filter_alpha": -0.1})


def test_ramp_schedule():
    from mp4_recovery.gpu.getup_task import filter_alpha_at, policy_filter_alpha_of
    assert [filter_alpha_at(k, 0.75, 0.5, 200) for k in (0, 100, 150, 200, 600)] == pytest.approx(
        [0.5, 0.625, 0.6875, 0.75, 0.75])
    assert filter_alpha_at(0, 0.75) == filter_alpha_at(999, 0.75) == 0.75          # no ramp: the final alpha
    assert filter_alpha_at(50, 0.75, None, 200) == 0.75                            # start None: ALPHA throughout
    assert filter_alpha_at(-5, 0.75, 0.5, 200) == 0.5
    assert policy_filter_alpha_of({"filter_alpha": 0.6}) == 0.6
    assert policy_filter_alpha_of({"filter_alpha": 0.6}, {"action_filter_alpha": 0.7}) == 0.7
    assert policy_filter_alpha_of({}, {}) == 0.0


def test_ramp_hook_counts_this_runs_iterations():
    """install_filter_ramp: alpha 0 before the first iteration's rollout, then after each iteration's log the next
    iteration's; a resumed run (start_it 2498) starts at k = 0; each iteration's alpha written and kept."""
    from mp4_recovery.gpu.getup_task import install_filter_ramp
    seen, written = [], []

    class Servo:
        def set_filter_alpha(self, a):
            seen.append(a)

    class Writer:
        def add_scalar(self, tag, v, it):
            written.append((tag, v, it))

    logger = SimpleNamespace(writer=Writer(), log=lambda **kw: None)
    runner = SimpleNamespace(logger=logger)
    st = install_filter_ramp(runner, Servo(), 0.75, 0.5, 4)
    assert seen == [0.5]
    for it in range(2498, 2504):
        runner.logger.log(it=it, start_it=2498, total_it=2504)
    assert [a for _, a in st["log"]] == pytest.approx([0.5, 0.5625, 0.625, 0.6875, 0.75, 0.75])
    assert [it for it, _ in st["log"]] == list(range(2498, 2504))
    assert written[0] == ("Getup/filter_alpha", 0.5, 2498) and written[-1][1:] == (0.75, 2503)


def test_export_meta_round_trip(tmp_path):
    from mp4_recovery.gpu.actor import mlp
    from mp4_recovery.policy_file import NumpyPolicy, export, with_meta
    from mp4_recovery.deploy.runner import band_refusal
    sd = {f"mlp.{k}": v for k, v in mlp(45, 12, (64, 32)).state_dict().items()}
    ck = tmp_path / "m.pt"
    torch.save({"actor_state_dict": sd}, ck)
    b = L.decoder_bounds("getup").to_meta()
    plain = NumpyPolicy(export(ck, tmp_path / "plain", scale=GETUP_SCALE, band_deg=[90.0], kind="getup", bounds=b))
    assert plain.meta["action_filter_alpha"] == 0.0 and policy_filter_alpha(plain.meta) == 0.0
    filt = NumpyPolicy(export(ck, tmp_path / "f", scale=GETUP_SCALE, band_deg=[90.0], kind="getup", bounds=b,
                              action_filter_alpha=0.75))
    assert policy_filter_alpha(filt.meta) == 0.75 and band_refusal(filt.meta) is None
    re = NumpyPolicy(with_meta(tmp_path / "f", tmp_path / "re", action_filter_alpha=0.5))
    assert policy_filter_alpha(re.meta) == 0.5
    assert all(np.array_equal(x, y) for (x, _), (y, _) in zip(re.layers, filt.layers))
    unfiltered = {k: v for k, v in filt.meta.items() if k != "action_filter_alpha"}
    assert policy_filter_alpha(unfiltered) == 0.0               # no such key: no filter
    with pytest.raises(ValueError):
        export(ck, tmp_path / "bad", scale=GETUP_SCALE, kind="getup", band_deg=[90.0], action_filter_alpha=1.2)


def test_export_script_takes_the_runs_option(tmp_path):
    """export_policy: the run's option filter_alpha into the meta; --from-policy keeps the source's;
    --filter-alpha overrides."""
    import json
    import subprocess
    import sys
    from pathlib import Path
    from mp4_recovery.gpu.actor import mlp
    from mp4_recovery.gpu.env_cfg import describe
    from mp4_recovery.gpu.getup_task import GetupOptions
    repo = Path(__file__).resolve().parents[2]
    run = tmp_path / "run"
    run.mkdir()
    torch.save({"actor_state_dict": {f"mlp.{k}": v for k, v in mlp(45, 12, (64, 32)).state_dict().items()}}, run / "model_1.pt")
    opt = GetupOptions(filter_alpha=0.75, filter_alpha_start=0.5, filter_ramp_iters=200)
    (run / "config.json").write_text(json.dumps({"task": describe(opt), "getup_task": True}, default=str))

    def meta(d):
        return json.loads(str(np.load(d / "policy.npz")["meta"]))

    def go(*argv):
        r = subprocess.run([sys.executable, "-B", "-m", "mp4_recovery.export_policy", *map(str, argv)],
                           capture_output=True, text=True, cwd=repo)
        assert r.returncode == 0, r.stderr

    go(run / "model_1.pt", tmp_path / "p")
    assert meta(tmp_path / "p")["action_filter_alpha"] == 0.75 and meta(tmp_path / "p")["kind"] == "getup"
    go(tmp_path / "p", tmp_path / "q", "--from-policy")
    assert meta(tmp_path / "q")["action_filter_alpha"] == 0.75
    go(tmp_path / "p", tmp_path / "r", "--from-policy", "--filter-alpha", "0.6")
    assert meta(tmp_path / "r")["action_filter_alpha"] == 0.6
    go(run / "model_1.pt", tmp_path / "s", "--filter-alpha", "0")
    assert meta(tmp_path / "s")["action_filter_alpha"] == 0.0


def test_getup_options_wire_the_filter():
    from mp4_recovery.gpu.env_cfg import describe
    from mp4_recovery.gpu.getup_task import GetupOptions, make_getup_env_cfg
    off = make_getup_env_cfg(GetupOptions())
    assert off.actions["servo"].filter_alpha == 0.0 and off.rewards["action_rate"].params["applied"] is False
    on = make_getup_env_cfg(GetupOptions(filter_alpha=0.75, filter_alpha_start=0.5,
                                         filter_ramp_iters=200))
    assert on.actions["servo"].filter_alpha == 0.75 and on.rewards["action_rate"].params["applied"] is True
    d = describe(GetupOptions(filter_alpha=0.75, filter_alpha_start=0.5, filter_ramp_iters=200))
    assert (d["options"]["filter_alpha"], d["options"]["filter_alpha_start"], d["options"]["filter_ramp_iters"]) == \
        (0.75, 0.5, 200)


# ------------------------------------------------------------------------------------------------------------- GPU
def _getup_env(n, **kw):
    from mjlab.envs import ManagerBasedRlEnv
    from mp4_recovery.gpu.env_cfg import noise_profile
    from mp4_recovery.gpu.getup_task import GetupOptions, make_getup_env_cfg
    cfg = make_getup_env_cfg(GetupOptions(num_envs=n, noise_end=noise_profile(), **kw))
    cfg.seed = 7
    return ManagerBasedRlEnv(cfg=cfg, device="cuda:0")


def process_unfiltered(servo, actions):
    """ServoAction.process_actions without the filter, returning (prev_rel, applied, target)
    without touching the term's state."""
    a = actions.clamp(-1.0, 1.0)
    rel = torch.maximum(torch.minimum(servo.scale * a, servo.band), -servo.band)
    rel = torch.maximum(torch.minimum(rel, servo.prev_rel + servo.limit), servo.prev_rel - servo.limit)
    phys = servo.physical(rel)
    if servo.cfg.quantise:
        _, phys = servo.tc.quantise_rel(phys)
    rel = phys - servo.off if servo._shifted else phys
    target = servo.tc.stand_phys + phys if servo._shifted else servo.tc.stand + rel
    return rel, rel / servo.scale, target


def _compare(servo, n, alphas, g, b, quantise, steps=60):
    """Both decoders from the sim's state, ``steps`` random actions; the runner's synced to the sim's applied target
    after each step. Returns (worst target difference, deg; the share of values beyond a tick; the largest step)."""
    servo.cfg.quantise = quantise
    decs = [ActionDecoder(C, servo.scale.cpu().numpy(), band_deg=GETUP_BAND, bounds=b, filter_alpha=alphas[i],
                          quantise=quantise) for i in range(n)]
    for i, d in enumerate(decs):
        d.prev_rel = servo.prev_rel[i].cpu().numpy().astype(float)
    worst, beyond, moved = 0.0, 0, 0.0
    for k in range(steps):
        raw = 2.0 * torch.randn(n, 12, device="cuda:0", generator=g)
        prev = servo.prev_rel.clone()
        servo.process_actions(raw)
        tgt = (servo.target_model - servo.tc.stand).cpu().numpy()          # no offset: the physical rel
        act = servo.applied_action.cpu().numpy()
        moved = max(moved, float(torch.rad2deg((servo.prev_rel - prev).abs()).max()))
        for i in range(n):
            out = decs[i](raw[i].cpu().numpy())
            diff = np.abs(np.degrees(out["rel"] - tgt[i]))
            worst, beyond = max(worst, float(diff.max())), beyond + int((diff > TICK + 0.01).sum())
            assert np.allclose(out["action"] * 1.57, act[i] * 1.57, atol=math.radians(diff.max() + 1e-4))
            decs[i].prev_rel = servo.prev_rel[i].cpu().numpy().astype(float)    # stay on the sim's tick
        model = servo.target_model.cpu().numpy()
        assert np.all(model >= b.lo - math.radians(TICK + 0.02)) and np.all(model <= b.hi + math.radians(TICK + 0.02))
    servo.cfg.quantise = True
    return worst, beyond / (steps * n * 12), moved


def test_sim_and_runner_filter_the_same_targets():
    """ServoAction and ActionDecoder on the same raw actions (2 x N(0, 1): the band, the step limit and the get-up
    bounds all bind), alpha 0.5 and 0.75 as a scalar, then a per-env (n, 1) alpha (0 / 0.5 / 0.75 by robot), from
    the seeded reset. Unquantised: the same targets to 1e-4 deg. Quantised: to a tick where float32 and float64 round
    differently at a tick edge, except rarely where such a tick on an abduction moves the get-up's hip envelope
    (the alpha-0 decoder does the same: the per-env alpha-0 robots). And alpha 0 (scalar and per-env
    zeros) bit-identical to the decoder without the filter."""
    from mp4_recovery.gpu import mdp
    n = 6
    env = _getup_env(n, limits=True)
    env.reset()
    servo = mdp.servo_action(env)
    b = L.decoder_bounds("getup")
    g = torch.Generator(device="cuda:0").manual_seed(0)
    per_env = torch.tensor([0.0, 0.5, 0.75] * 2, device="cuda:0")[:, None]
    for alpha in (0.5, 0.75, per_env):
        servo.set_filter_alpha(alpha)
        al = per_env[:, 0].tolist() if isinstance(alpha, torch.Tensor) else [alpha] * n
        worst, _, moved = _compare(servo, n, al, g, b, quantise=False)
        assert worst < 1e-4, (alpha, worst)
        assert moved >= STEP_LIMIT_DEG - 1e-3, moved             # the step limit was reached
        worst, share, _ = _compare(servo, n, al, g, b, quantise=True)
        assert worst < 1.0 and share < 0.01, (alpha, worst, share)
    # alpha 0: no filter, bit for bit (the scalar skips the filter; per-env zeros run it)
    for alpha in (0.0, torch.zeros(n, 1, device="cuda:0")):
        servo.set_filter_alpha(alpha)
        for k in range(30):
            raw = 2.0 * torch.randn(n, 12, device="cuda:0", generator=g)
            rel, applied, target = process_unfiltered(servo, raw)
            servo.process_actions(raw)
            assert torch.equal(servo.prev_rel, rel) and torch.equal(servo.applied_action, applied)
            assert torch.equal(servo.target_model, target)
    env.close()


def test_sim_reset_seeds_the_filter_state():
    """A reset (seed_from_measured) puts the filter's state at the measured pose (clipped to the band, into the
    bounds), the previous action observed with it; the first step moves (1 - alpha) of the way from there."""
    from mp4_recovery.gpu import mdp
    n = 8
    env = _getup_env(n, filter_alpha=0.75)
    env.reset()
    servo = mdp.servo_action(env)
    assert servo.alpha == 0.75
    meas = servo.measured_rel()
    seeded = torch.maximum(torch.minimum(meas, servo.band), -servo.band)
    assert torch.allclose(servo.prev_rel, seeded) and torch.allclose(servo.applied_action, seeded / servo.scale)
    prev = servo.prev_rel.clone()
    step = math.radians(8.0)
    servo.process_actions((prev + step) / servo.scale)          # 8 deg from the seeded state everywhere
    inside = (prev.abs() + step < servo.band - 1e-3)
    got = torch.rad2deg(servo.prev_rel - prev)[inside]
    assert torch.allclose(got, torch.full_like(got, 2.0), atol=TICK + 1e-3)     # 0.25 x 8 deg, to a tick
    env.close()
