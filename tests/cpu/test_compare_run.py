"""compare_run: parsing and arithmetic (no simulation). The fixtures are simulated logs from
``tests/data/compare_run``, in the real formats."""
import json
import math
from pathlib import Path

import numpy as np
import pytest

import compare_run as tc  # noqa: E402
from mp4_recovery.contract import Contract  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data" / "compare_run"


def test_dd_rel_about_the_logs_stand_matches_the_contract():
    c = Contract.load()
    dd = np.asarray(c.stand_dd, float) + np.arange(12) * 7.0
    a = tc.rel_deg_from_dd(dd, c.stand_dd, c.signs)
    b = np.degrees(c.rel_from_dd(dd) - c.rel_from_dd(c.stand_dd))
    assert a == pytest.approx(b, abs=1e-9)
    assert tc.dd_from_rel_deg(a, c.stand_dd, c.signs) == pytest.approx(dd)


def test_tracking_is_the_one_step_error():
    t = np.arange(0, 2, 0.005)
    pos = np.tile(10.0 * t[:, None], (1, 12))        # 10 deg/s
    ts = np.arange(0, 1.9, 0.02)
    tgt = np.tile(10.0 * ts[:, None], (1, 12))
    assert tc.tracking_errors(t, pos, ts, tgt) == pytest.approx(0.2, abs=1e-9)   # 20 ms at 10 deg/s
    assert len(tc.tracking_errors(t, pos, ts + 1.0, tgt)) < len(ts)              # past the positions: dropped


def test_delay_by_cross_correlation():
    t = np.arange(0, 6, 0.005)
    ts = np.arange(0, 6, 0.02)
    f = lambda x: 20 * np.sin(2 * np.pi * 0.7 * x) + 8 * np.sin(2 * np.pi * 2.3 * x)  # noqa: E731
    assert tc.delay_ms(ts, f(ts), t, f(t - 0.05)) == pytest.approx(50 - 10, abs=5)   # the held target itself lags half a frame
    assert math.isnan(tc.delay_ms(ts, np.zeros_like(ts), t, f(t)))


def test_pass_arithmetic_and_slow():
    assert tc.stream_pass(25.0, 10.0, False) == (True, 25.0)
    assert tc.stream_pass(25.1, 10.0, False)[0] is False
    assert tc.stream_pass(15.0, 0.1, True) == (True, 15.0)
    assert tc.stream_pass(15.1, 50.0, True)[0] is False
    assert tc.is_slow("x/run_back_s14_x0p25.npz") and not tc.is_slow("x/run_back_s14_x1p00.npz")
    assert tc.is_slow("a.npz", {"speed": 0.25}) and not tc.is_slow("a.npz", {"speed": 1.0})
    assert tc.ratio(1.0, 0.5) == 2.0 and math.isnan(tc.ratio(1.0, 0.0))
    temp = np.full((100, 12), 300.0)
    temp[-20:, 3] = 360.0
    assert tc.housing_rise(temp)[3] == pytest.approx(6.0) and tc.housing_rise(temp)[0] == 0.0


def test_zoh_keeps_a_gap():
    t = np.array([0.0, 0.02, 0.1])
    g, r = tc.zoh_grid(t, np.array([[1.0], [2.0], [3.0]]))
    assert len(g) == 6 and r[:, 0].tolist() == [1, 2, 2, 2, 2, 3]


def test_planned_sends_ramp_hold_clip():
    stand = np.full(12, 1500.0)
    delta = np.zeros((3, 12))
    delta[1, 0] = 2000.0                     # past a 90 deg band and a 20 deg step
    rows, i0, n, clipped = tc.planned_sends(stand, stand, delta, np.array([90.0, 180.0, 180.0] * 4), 20.0)
    assert (i0, n) == (150, 3) and len(rows) == 150 + 3 + 100
    assert rows[i0 + 1, 0] == 1500.0 + 200.0 and clipped >= 1
    assert rows[-1] == pytest.approx(stand)


@pytest.mark.parametrize("g", [(0, 0, -1), (0, -1, 0), (0, 1, 0), (1, 0, 0), (0.3, -0.5, 0.8), (0, 0, 1)])
def test_quat_from_gravity(g):
    g = np.array(g, float) / np.linalg.norm(g)
    for yaw in (0.0, 1.1):
        w, x, y, z = tc.quat_from_gravity(g, yaw)
        R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                      [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                      [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
        assert R.T @ np.array([0, 0, -1.0]) == pytest.approx(g, abs=1e-9)


def test_handback_and_band_exit():
    tilt = np.r_[np.full(10, 90.0), np.full(40, 5.0)]
    assert tc.handback_steps(tilt, np.zeros(50), np.zeros(50)) == 34
    assert tc.handback_steps(tilt, np.ones(50), np.zeros(50)) == -1
    x = np.zeros(50)
    x[20:25] = 5.0                           # too short
    assert tc.first_exit(x, np.zeros(50) - 1, np.ones(50)) is None
    x[30:] = -5.0
    assert tc.first_exit(x, np.zeros(50) - 1, np.ones(50)) == (30, "below")
    assert tc.first_exit(x, np.zeros(50) - 1, np.ones(50), margin=5.0) is None
    d = tc.first_divergence({"hip tracking": (x, np.zeros(50) - 1, np.ones(50), 0.0),
                             "gyro": (x * 0, np.zeros(50) - 1, np.ones(50), 0.0)})
    assert d["channel"] == "hip tracking" and d["t_s"] == pytest.approx(0.6)


def test_readings():
    div = {"channel": "knee tracking", "t_s": 0.4, "direction": "above"}
    assert tc.floor_reading(True, 2.0, 0.9, [1.8, 2.2], None)[0] == "inside"
    assert tc.floor_reading(False, 10.0, 0.4, [2.0], div)[0] == "policy"
    tag, line = tc.floor_reading(False, 10.0, 0.98, [2.0], div)
    assert tag == "plant-gap" and "knee tracking" in line
    assert tc.floor_reading(True, 5.0, 0.9, [1.8, 2.2], div)[0] == "plant-gap"
    assert tc.floor_reading(True, 2.0, 0.01, [2.0], div)[0] == "sim-pessimistic"
    assert tc.floor_reading(None, 1.0, 0.9, [2.0], None)[0] == "stopped"


def _roll(T=300):
    """A clean roll from the side to upright, gravity and gyro consistent (dg/dt = -w x g)."""
    phi = np.radians(90.0) * np.clip(1 - np.arange(T) * 0.02 / 3.0, 0, 1)
    g = np.stack([np.zeros(T), -np.sin(phi), -np.cos(phi)], 1)     # roll phi about body x
    w = np.stack([np.gradient(phi, 0.02), np.zeros(T), np.zeros(T)], 1)
    return g, w, np.degrees(phi)


def test_gravity_checks():
    g, w, tilt = _roll()
    assert tc.gravity_checks(g, w, tilt, g)["mismatch"] == []
    assert tc.gravity_checks(-g, w, tilt, g)["mismatch"]            # negated: accel, filter and upright all say so
    assert tc.gravity_checks(-g, w, tilt, -g)["mismatch"]           # the accel too: the filter and upright still do
    flip = g * np.array([1, -1, 1])
    m = tc.gravity_checks(flip, w, tilt, flip)["mismatch"]          # one axis, everywhere: dg/dt = -w x g breaks
    assert any("dg_" in s for s in m)


def test_obs_outside_lists_a_channel():
    real = np.zeros((100, 45))
    real[:50, 4] = 2.0
    out = tc.obs_outside(real, -np.ones(45), np.ones(45))
    assert [r["channel"] for r in out] == ["gravity_y"] and out[0]["share"] == 0.5


def test_parse_stream_fixture():
    log = tc.load_stream_log(DATA / "fake_stream_back_x1.npz")
    assert log["cap"] == 2000 and log["pos_dd"].shape[1] == 12 and log["t1"] > log["t0"]
    r = tc.robot_report(log, Contract.load().signs)
    assert r["sends"] > 100 and r["err"].shape[1] == 12 and np.isfinite(r["delay"]).any()


def test_parse_run_fixture_and_real_series():
    run = tc.load_run(DATA / "fake_left")
    w = tc.getup_window(run)
    assert w["up"] is True and w["t_s"] > 0.5 and w["k1"] - w["k0"] == len(run["steps"]["obs"])
    real = tc.real_series(run, w)
    assert real["obs"].shape[1] == 45 and np.isnan(real["err"][-1]).all() and np.isfinite(real["err"][:-1]).all()
    s = tc.run_start(real)
    assert np.linalg.norm(s["g"]) == pytest.approx(1.0) and s["rel"].shape == (12,)
    assert json.loads(json.dumps(run["meta"]))["getups"][0]["end"] == "hand-back"


def test_fixture_actions_reproduce_and_gravity_clean():
    from mp4_recovery.deploy.runner import load_mount
    run = tc.load_run(DATA / "fake_left")
    w = tc.getup_window(run)
    real = tc.real_series(run, w)
    assert real["raw"].shape == (w["k1"] - w["k0"], 12)
    _, R = load_mount(run["meta"]["mount"])
    idx = np.searchsorted(run["main"]["tick"].astype(np.int64), run["steps"]["tick"].astype(np.int64))
    acc = (R @ run["main"]["acc_mg"][idx].astype(float).T).T
    g_acc = -acc / np.linalg.norm(acc, axis=1, keepdims=True)
    assert tc.gravity_checks(real["obs"][:, 3:6], real["obs"][:, :3], real["tilt"], g_acc)["mismatch"] == []


def test_getup_window_refusals():
    with pytest.raises(SystemExit):
        tc.getup_window({"prefix": "x", "meta": {"getups": []}, "steps": {"mode": np.zeros(5)}})
    run = {"prefix": "x", "meta": {"getups": [{"step": 0}], "stop_reason": "enter"}, "steps": {"mode": np.ones(5)}}
    w = tc.getup_window(run)
    assert w["up"] is None and w["t_s"] == pytest.approx(0.1)
