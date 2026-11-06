"""eval/motion_meter.py: the reversal count, the band power shares, the windows and the summaries on synthetic
series (CPU), then the meter on the GPU: a scripted sinusoidal target on robots left on their backs gives the expected
reversal rate, the impact hook reads forces and detaches, and getup_eval / getup_search carry the motion block."""
import math

import numpy as np
import pytest
import torch



from mp4_recovery.eval import motion_meter as M  # noqa: E402
DT = M.DT


def _sine(f, amp, steps=200, phase=0.3):
    t = np.arange(steps) * DT
    x = amp * np.sin(2 * np.pi * f * t + phase)
    v = amp * 2 * np.pi * f * np.cos(2 * np.pi * f * t + phase)
    return x, v


# ------------------------------------------------------------------------------------------------------------- CPU
@pytest.mark.parametrize("f,amp,r5,r10", [(4.0, 15.0, 8.0, 8.0), (4.0, 4.0, 8.0, 0.0), (2.0, 15.0, 4.0, 4.0),
                                          (8.0, 15.0, 16.0, 16.0), (0.5, 3.0, 0.0, 0.0)])
def test_sinusoid_reversal_rate(f, amp, r5, r10):
    """A sinusoidal target of frequency f reverses 2 f times a second where its swing (2 x amp) clears the prominence
    within +-15 steps (0.3 s); edge extrema may miss a side: +-0.6 per s over 4 s."""
    x, _ = _sine(f, amp)
    tg = np.repeat(x[:, None], 12, 1)
    assert M.reversal_rate(tg, 5.0) == pytest.approx(r5, abs=0.6)
    assert M.reversal_rate(tg, 10.0) == pytest.approx(r10, abs=0.6)


def test_reversal_details():
    """A plateau (a clamp) is one extremum; a step that never returns is none; the +-15-step window: a slow 0.5 Hz
    swing of 30 deg p-p reaches 10 deg within 0.3 s of the peak, 1 deg does not."""
    t = np.arange(300) * DT
    clipped = np.clip(20 * np.sin(2 * np.pi * 3 * t), -10, 10)
    assert M.reversals(clipped, 5.0) == pytest.approx(2 * 3 * 6, abs=2)
    assert M.reversals(np.concatenate([np.zeros(50), np.full(50, 30.0)]), 5.0) == 0
    assert M.reversals(np.zeros(100), 1.0) == 0
    x, _ = _sine(0.5, 15.0, 300)
    assert M.reversals(x, 10.0) == 0 and M.reversals(x, 1.0) >= 5


@pytest.mark.parametrize("f,s3,s6", [(4.0, 1.0, 0.0), (1.0, 0.0, 0.0), (8.0, 1.0, 1.0), (2.0, 0.0, 0.0)])
def test_band_shares_of_a_sinusoid(f, s3, s6):
    _, v = _sine(f, 10.0)
    p3, p6 = M.band_shares(np.repeat(v[:, None], 12, 1))
    assert p3 == pytest.approx(s3, abs=0.02) and p6 == pytest.approx(s6, abs=0.02)


def test_band_shares_of_a_mixture():
    """Equal-amplitude velocity sinusoids at 1, 4 and 8 Hz: a third of the power each."""
    t = np.arange(400) * DT
    v = sum(np.sin(2 * np.pi * f * t + f) for f in (1.0, 4.0, 8.0))
    p3, p6 = M.band_shares(v[:, None])
    assert p3 == pytest.approx(2 / 3, abs=0.02) and p6 == pytest.approx(1 / 3, abs=0.02)


def _series(T=300, n=3, f=4.0, amp=15.0, up_at=(100, 10, None), succ=(4.0, -1.0, -1.0)):
    x, v = _sine(f, amp, T)
    tg = np.repeat(np.repeat(x[:, None, None], n, 1), 12, 2)
    jv = np.repeat(np.repeat(v[:, None, None], n, 1), 12, 2)
    tilt = np.full((T, n), 90.0)
    for r, k in enumerate(up_at):
        if k is not None:
            tilt[k:, r] = 5.0
    return {"tg": tg, "tg0": tg[0], "act": tg / 180.0, "act0": tg[0] / 180.0, "jv": jv, "tilt": tilt}, np.array(succ)


def test_windows_and_summary():
    """Robot 0 comes up at step 100 and succeeds at 4.0 s (step 200): windows [0, 100) and [100, 200); robot 1 is
    under 15 deg from step 10: its window is the minimum 25 steps; robot 2 never: the whole trial, no after window."""
    s, ts = _series()
    res = M.compute(s, ts)
    tr = res["trials"]
    assert tr["window_end_step"] == [100, 25, 300] and tr["after_end_step"] == [200, 300, 300]
    assert tr["getup.rev10"][0] == pytest.approx(8.0, abs=0.6) and tr["getup.rev10"][2] == pytest.approx(8.0, abs=0.3)
    assert tr["after.rev5"][2] is None                                        # no after window
    assert tr["after.rev5"][0] == pytest.approx(8.0, abs=0.6)
    g = res["summary"]["getup"]
    assert g["rev10"]["n"] == 3 and res["summary"]["after"]["rev5"]["n"] == 2
    assert tr["getup.pow3"][0] == pytest.approx(1.0, abs=0.02) and tr["getup.pow6"][2] < 0.02
    # da2: per step sum over 12 joints of the squared change in action units (tg / 180 here)
    da = np.diff(s["act"][:100, 0, 0]) ** 2
    assert tr["getup.da2"][0] == pytest.approx(12 * da.sum() / 100, rel=1e-3)
    sp = np.abs(s["jv"][:100, 0, 0])
    assert tr["getup.speed_rms"][0] == pytest.approx(math.sqrt((sp ** 2).mean()), rel=1e-4)
    assert tr["getup.sat"][0] == 0.0
    assert set(g) == set(M.MOTION_KEYS) and "trial" not in res["summary"]


def test_step_limit_saturation_share():
    """A target that climbs the full 20 deg every other step saturates half its steps."""
    T, n = 100, 1
    x = 20.0 * ((np.arange(T) + 1) // 2)
    tg = np.repeat(x[:, None, None], 12, 2)
    s = {"tg": tg, "tg0": np.zeros((n, 12)), "act": tg / 180, "act0": np.zeros((n, 12)), "jv": np.zeros((T, n, 12)),
         "tilt": np.full((T, n), 90.0)}
    tr = M.compute(s, np.array([-1.0]))["trials"]
    assert tr["getup.sat"][0] == pytest.approx(0.5)
    assert tr["getup.pow3"][0] is None                         # no motion: no power share


def test_impact_windows():
    """Per-step peaks and touchdowns are split by window; the trial covers both."""
    s, ts = _series(n=1, up_at=(100,), succ=(5.0,))
    T = s["tilt"].shape[0]
    bf = np.zeros((T, 1))
    bf[50, 0], bf[150, 0] = 30.0, 80.0
    af = np.zeros((T, 1))
    af[120, 0] = 0.5
    s.update(base_f=bf, abd_f=af, td=[[(60, 0.4, 0.4, M.PART_LEG, 1), (140, 0.9, 1.1, M.PART_FOOT, 2)]])
    res = M.compute(s, ts)
    tr = res["trials"]
    assert tr["getup.base_force_peak_N"] == [30.0] and tr["after.base_force_peak_N"] == [80.0]
    assert tr["trial.base_force_peak_N"] == [80.0] and tr["trial.abd_limit_peak_Nm"] == [0.5]
    assert tr["getup.touchdown_vz_mps"] == [0.4] and tr["after.touchdown_vz_max_mps"] == [1.1]
    assert tr["trial.airborne"] == [3] and tr["trial.touchdown_part"] == [M.PART_LEG]
    assert res["summary"]["trial"]["base_force_peak_N"]["p99"] == 80.0


# ------------------------------------------------------------------------------------------------------------- GPU
def _env(n, seed=3):
    from mjlab.envs import ManagerBasedRlEnv
    from mp4_recovery.gpu.env_cfg import noise_profile
    from mp4_recovery.gpu.getup_task import GetupOptions, make_getup_env_cfg
    opt = GetupOptions(num_envs=n, noise_end=noise_profile(), band_deg=(90.0, 180.0, 180.0),
                       limits=True, fixed_starts=("random",), episode_s=1.0e3)
    cfg = make_getup_env_cfg(opt)
    cfg.seed = seed
    return ManagerBasedRlEnv(cfg=cfg, device="cuda:0")


def test_meter_on_robots_on_their_backs():
    """Robots on their backs with a 3 Hz, +-12 deg sinusoidal target on every servo never come up: the whole trial is
    the get-up window, the reversal rate 2 x 3 per s at 5 and 10 deg, the target-change cost is the decoder's applied
    actions'; the base rests on the floor (its weight-scale force), and detach restores the physics step."""
    from mp4_recovery.gpu import mdp
    from mp4_recovery.eval import getup_eval as ev
    n, f, steps = 8, 3.0, 150
    env = _env(n)
    act = mdp.servo_action(env)
    step0 = env.sim.step
    meter = M.MotionMeter(env)
    assert env.sim.step != step0
    ev.set_start(env, "back")
    torch.manual_seed(3)
    env.reset()
    meter.begin(env)
    amp = math.radians(12.0) / act.scale
    applied = [act.applied_action.cpu().numpy().copy()]
    for k in range(steps):
        a = (amp * math.sin(2 * math.pi * f * (k + 1) * DT + 0.3)).expand(n, 12)
        env.step(a)
        meter.update(env)
        applied.append(act.applied_action.cpu().numpy().copy())
    res = meter.finish(np.full(n, -1.0))
    meter.detach()
    assert env.sim.step == step0
    env.close()
    tr, g = res["trials"], res["summary"]["getup"]
    assert tr["window_end_step"] == [steps] * n
    assert g["rev5"]["mean"] == pytest.approx(2 * f, abs=0.8) and g["rev10"]["mean"] == pytest.approx(2 * f, abs=0.8)
    assert g["pow3"]["mean"] > 0.5 and g["sat"]["mean"] < 0.02         # the first step may jump from the reset pose
    applied = np.stack(applied)                                                    # (steps + 1, n, 12)
    expect = (np.diff(applied, axis=0) ** 2).sum(-1).mean(0)                       # per robot
    assert np.allclose(tr["getup.da2"], expect, rtol=1e-4, atol=1e-5)          # trials rounded to 5 places
    t = res["summary"]["trial"]
    assert 2.0 < t["base_force_peak_N"]["mean"] < 500.0
    assert t["abd_limit_peak_Nm"]["n"] == n and t["abd_limit_peak_Nm"]["max"] >= 0.0


def test_getup_eval_and_search_carry_the_motion_block():
    """run_start(meter=...) puts the motion summary into the start's summary and per-trial arrays of length n;
    getup_search.evaluate feeds a meter the same way."""
    from mp4_recovery.eval import getup_eval as ev
    from mp4_recovery.gpu import getup_search as gs
    n = 4
    env = _env(n)
    meter = M.MotionMeter(env)
    pol = lambda obs: torch.zeros(obs.shape[0], 12, device=obs.device)       # noqa: E731
    r = ev.run_start(env, pol, "left", 1.0, 3, meter=meter)
    assert set(r["summary"]["motion"]) >= {"getup", "after", "trial", "window_steps"}
    assert len(r["trials"]["motion"]["getup.rev5"]) == n and len(r["trials"]["success"]) == n
    x = torch.zeros(1, 5 * 13)
    x[0, 60:] = 0.3
    out = gs.evaluate(env, x.cuda(), 5, "left", n, 3, meter=meter, trial_s=1.0)
    m = meter.finish(out["t_success"].reshape(-1))
    meter.detach()
    env.close()
    assert m["summary"]["getup"]["rev5"]["n"] == n
