"""The runner's modes and stops against the fake robot: walk, fall, get-up, retry, hand-back, limp."""
import json
import math
from pathlib import Path

import numpy as np

from fake_esp32 import fall_and_rise, roll
from mp4_recovery.deploy import runner as rn
from robot_helpers import RAMP_TICKS, C, go

POLICY_START = RAMP_TICKS + 4


def sent_after_last_step(tmp_path, z) -> int:
    """SET_ALL frames sent after the last logged step's: 1 is the DISARM frame alone (limp); 50 or more is a ramp to the stand."""
    raw = np.load(tmp_path / "run.npz")
    return int((raw["sent_seq"][-1] - z["seq"][-1]) & 0xFFFF)


def test_run_writes_the_gains_holds_the_stand_and_logs(robot, tmp_path):
    ids = {name: i for i, name in enumerate(rn.p.PARAMS)}
    for s in range(1, 13):
        robot.params[s][ids["kp_position"]] = 0.0
    rc, z, meta = go(robot, tmp_path, seconds=0.4)
    assert rc == 0 and meta["stop_reason"] == "completed" and not meta["warnings"]
    assert all(robot.params[s][ids["kp_position"]] == 80.0 for s in range(1, 13))
    assert robot.caps_armed == {rn.CAP_MA}
    assert np.all(np.diff(z["tick"].astype(int)) == 4) and np.all(z["dd"] == np.asarray(C.stand_dd))
    assert json.loads((tmp_path / "run.stand.json").read_text())["stand_dd"] == list(C.stand_dd)
    raw = np.load(tmp_path / "run.npz")
    assert len(raw["sent_seq"]) > 0 and len(raw["pos_dd"]) > 0 and not robot.armed


def test_step_limit_acts(robot, tmp_path):
    rc, z, _ = go(robot, tmp_path, walker_bias=1.0, seconds=0.2)
    step = np.abs(np.diff(z["rel"], axis=0))
    assert rc == 0 and step.max() <= math.radians(20.0) + 0.006


def test_refuses_without_imu(robot, tmp_path):
    robot.imu_ok = False
    try:
        go(robot, tmp_path)
    except SystemExit as exc:
        assert "IMU" in str(exc)
    else:
        raise AssertionError("no refusal")


def test_refuses_a_hot_start(robot, tmp_path):
    robot.temps_dc = [460] * 12
    try:
        go(robot, tmp_path)
    except SystemExit as exc:
        assert "degC" in str(exc)
    else:
        raise AssertionError("no refusal")
    assert not robot.armed_seen


def test_offsets_warn_only(robot, tmp_path):
    robot.offsets[1] += 3.0
    rc, _, meta = go(robot, tmp_path, seconds=0.2)
    assert rc == 0 and any("offsets" in w for w in meta["warnings"])


def test_a_latched_fault_needs_clear(robot, tmp_path):
    robot.latched = rn.p.ST_FAULT_CUR
    robot.fault_servo = 2
    try:
        go(robot, tmp_path)
    except Exception as exc:
        assert "--clear" in str(exc)
    else:
        raise AssertionError("no refusal")
    assert not robot.armed_seen
    (tmp_path / "again").mkdir()
    rc, _, meta = go(robot, tmp_path / "again", "--clear", seconds=0.2)
    assert rc == 0 and meta["stop_reason"] == "completed"


def test_imu_dropout_stops(robot, tmp_path):
    orig = robot.before_tick

    def tick():
        orig()
        if len(robot.armed_seen) > POLICY_START + 20:
            robot.imu_ok = False
    robot.before_tick = tick
    _, _, meta = go(robot, tmp_path, seconds=2.0)
    assert meta["stop_reason"].startswith("stop: IMU frame invalid")


def hot_after(robot, ticks, seconds=None, dc=672):
    """A housing reads ``dc`` from ``ticks`` armed ticks on, for ``seconds`` (None: from then on)."""
    orig = robot.before_tick

    def tick():
        orig()
        n = len(robot.armed_seen) - ticks
        robot.temps_dc[1] = dc if n >= 0 and (seconds is None or n * 0.005 < seconds) else 336
    robot.before_tick = tick


def test_a_brief_doubled_temperature_reading_does_not_stop(robot, tmp_path):
    hot_after(robot, POLICY_START + 20, seconds=0.1)     # long enough for several steps to read it
    _, _, meta = go(robot, tmp_path, seconds=1.0)
    assert meta["stop_reason"] == "completed"


def test_a_sustained_rise_stops_after_a_second(robot, tmp_path):
    hot_after(robot, POLICY_START + 20, dc=560)
    _, z, meta = go(robot, tmp_path, seconds=4.0)
    assert meta["stop_reason"] == "stop: servo at 56.0 degC"
    since_hot = (len(z["tick"]) + 1) * 0.02 - (POLICY_START + 20 - RAMP_TICKS) * 0.005
    assert 0.95 <= since_hot <= 1.2, since_hot
    assert sent_after_last_step(tmp_path, z) >= 50      # walking: a ramp to the stand


def test_a_sustained_rise_in_a_getup_goes_limp(robot, tmp_path):
    robot.script, robot.script_at = roll([(0.3, 90.0)]), POLICY_START + 10
    hot_after(robot, POLICY_START + 100, dc=560)
    _, z, meta = go(robot, tmp_path, seconds=4.0)
    assert meta["stop_reason"] == "stop: servo at 56.0 degC (in GETUP: limp)"
    assert sent_after_last_step(tmp_path, z) == 2       # the step that saw it, and the DISARM frame


def test_fall_enters_getup_and_hands_back(robot, tmp_path):
    """Walk, a scripted roll to 90 deg, a get-up within 0.4 s (the bit sent and echoed), a scripted rise, a hand-back when the
    criterion has held 25 steps, the walking policy's first target stepped from the measured pose."""
    robot.script, robot.script_at = roll(fall_and_rise(lie_s=0.6, up_s=3.0)), POLICY_START + 100
    gb = np.zeros(12)
    gb[1] = math.radians(30.0) / 1.57           # the get-up holds the FR hip 30 deg off: rms 8.7 deg
    rc, z, meta = go(robot, tmp_path, seconds=4.5, walker_bias=-1.0, getup_bias=gb)
    assert rc == 0 and meta["stop_reason"] == "completed", meta["stop_reason"]
    mode = z["mode"]
    (ev,) = meta["getups"]
    entry = int(np.argmax(mode == 1))
    assert ev["step"] == entry and ev["end"] == "hand-back" and "gyro" in ev["trigger"]
    assert (z["tick"][entry] - robot.script_tick) * 0.005 <= 0.4
    assert robot.getup_frames > 0
    assert z["getup_echo"][entry + 2:ev["end_step"]].all() and not z["getup_echo"][ev["end_step"] + 2:].any()
    h = ev["end_step"]
    assert mode[h] == 0 and mode[h - 1] == 1 and not mode[h:].any()
    pos_deg = np.degrees(z["obs"][:, 9:21])
    crit = (z["tilt_cf_deg"] < 12.0) & (z["gyro_norm"] < 0.8) & (np.sqrt(np.mean(pos_deg ** 2, axis=1)) < 10.0)
    assert crit[h - 24:h + 1].all() and not crit[h - 25]
    assert np.all(np.abs(np.degrees(z["rel"][h]) - pos_deg[h]) <= 20.0 + 0.32)
    assert np.all(np.abs(np.degrees(z["rel"][entry]) - pos_deg[entry]) <= 20.0 + 0.32)
    assert not robot.armed


def test_getup_gives_up_at_the_timeout(robot, tmp_path):
    robot.script, robot.script_at = roll([(0.3, 90.0)]), POLICY_START + 50
    _, z, meta = go(robot, tmp_path, "--getup-timeout", "1.5", seconds=5.0)
    assert meta["stop_reason"] == "disarm: get-up: not up after 1.5 s (limp)"
    (ev,) = meta["getups"]
    assert ev["end"] == "timeout" and 1.5 < ev["seconds"] <= 1.56
    assert sent_after_last_step(tmp_path, z) == 1 and not robot.armed


def test_one_fall_past_the_maximum_goes_limp(robot, tmp_path):
    robot.script, robot.script_at = roll(fall_and_rise(lie_s=0.4, up_s=1.0) * 3), POLICY_START + 25
    _, _, meta = go(robot, tmp_path, "--getup-max", "2", seconds=10.0)
    assert meta["stop_reason"].startswith("disarm: fall 3 (") and "--getup-max 2 reached" in meta["stop_reason"]
    assert [e["end"] for e in meta["getups"]] == ["hand-back"] * 2


def test_missing_echo_disarms(robot, tmp_path):
    robot.echo = False
    robot.script, robot.script_at = roll([(0.3, 90.0)]), POLICY_START + 20
    _, _, meta = go(robot, tmp_path, seconds=3.0)
    assert meta["stop_reason"] == "disarm: get-up: no ST_GETUP echo"
    assert any("echo" in w for w in meta["warnings"])


def test_fall_in_the_stand_ramp_starts_in_getup(robot, tmp_path):
    robot.script, robot.script_at = roll([(0.3, 90.0)]), 5
    _, z, meta = go(robot, tmp_path, "--getup-timeout", "0.5", seconds=2.0)
    (ev,) = meta["getups"]
    assert ev["step"] == 0 and ev["trigger"].startswith("stand ramp/hold") and z["mode"][0] == 1


def test_slow_topple_goes_to_getup(robot, tmp_path):
    robot.script, robot.script_at = roll([(3.0, 70.0)]), POLICY_START + 10
    _, _, meta = go(robot, tmp_path, "--getup-timeout", "0.5", seconds=5.0)
    assert "for 0.3 s" in meta["getups"][0]["trigger"]


def test_end_of_time_in_a_getup_goes_limp(robot, tmp_path):
    robot.script, robot.script_at = roll([(0.3, 90.0)]), POLICY_START + 10
    _, z, meta = go(robot, tmp_path, seconds=1.0)
    assert meta["stop_reason"] == "completed in GETUP (limp)" and sent_after_last_step(tmp_path, z) == 1


def test_end_of_time_walking_ramps_to_the_stand(robot, tmp_path):
    _, z, meta = go(robot, tmp_path, seconds=0.2)
    assert meta["stop_reason"] == "completed" and sent_after_last_step(tmp_path, z) >= 50


def test_get_up_observes_its_phase_from_each_entry(robot, tmp_path):
    robot.script, robot.script_at = roll(fall_and_rise(lie_s=0.6, up_s=1.0) * 2), POLICY_START + 10
    _, z, meta = go(robot, tmp_path, seconds=7.0)
    for ev in meta["getups"]:
        k0, k1 = ev["step"], ev["end_step"]
        phi = (np.arange(k1 - k0)) * 0.02 / 4.0
        assert np.allclose(z["obs"][k0:k1, 6], phi, atol=1e-5)
        assert np.allclose(z["obs"][k0:k1, 7], np.sin(2 * np.pi * phi), atol=1e-5)
    walk = z["mode"] == 0
    assert np.all(z["obs"][walk, 6] == 0.0)


def test_phase_restart_retries_a_getup_stuck_on_its_side(robot, tmp_path):
    robot.script, robot.script_at = roll(fall_and_rise(lie_s=5.6, up_s=1.0)), POLICY_START + 10
    _, z, meta = go(robot, tmp_path, seconds=8.0)
    (ev,) = meta["getups"]
    assert ev["end"] == "hand-back" and ev["phase_restarts"] == [ev["step"] + 250]
    k = ev["phase_restarts"][0]
    assert z["obs"][k - 1, 6] == 1.0 and z["obs"][k, 6] == 0.0


def test_phase_restart_needs_the_tilt(robot, tmp_path):
    """Upright, the clock stays at its end: a get-up about to hand back is not restarted."""
    gb = np.full(12, math.radians(25.0) / 1.57)  # the pose too far off for a hand-back
    robot.script, robot.script_at = roll([(0.3, 90.0), (0.5, 90.0), (0.3, 0.0)]), POLICY_START + 10
    _, _, meta = go(robot, tmp_path, "--getup-timeout", "6.5", seconds=9.0, getup_bias=gb)
    (ev,) = meta["getups"]
    assert ev["end"] == "timeout" and "phase_restarts" not in ev


def test_placed_start_goes_to_getup_and_hands_back(robot, tmp_path):
    robot.set_body_imu((0.0, 0.0, -1000.0), (0.0, 0.0, 0.0))     # on its back, then a rise once armed
    robot.script_at = 1
    robot.script = roll([(0.001, 180.0), (0.5, 180.0), (0.5, 0.0), (3.0, 0.0)])
    _, z, meta = go(robot, tmp_path, "--start-mode", "getup", "--getup-max", "1", seconds=3.0)
    (ev,) = meta["getups"]
    assert ev["step"] == 0 and ev["trigger"] == "start mode getup" and meta["start_tilt_deg"] > 170.0
    assert z["mode"][0] == 1 and ev["end"] == "hand-back"
    raw = np.load(tmp_path / "run.npz")
    assert raw["sent_pos_dd"][0].tolist() == z["dd"][0].tolist()    # no ramp before the first step


def test_frames_keep_flowing_up_to_the_first_step(robot, tmp_path):
    """Between the ramp's last frame and the first step only the same target, on every STATE frame, without a flag."""
    _, z, _ = go(robot, tmp_path, seconds=0.2)
    raw = np.load(tmp_path / "run.npz")
    first = int(np.searchsorted(raw["sent_t"], z["t_send"][0]))
    ramp_n = 10                                 # the 0.2 s ramp
    repeats = raw["sent_pos_dd"][ramp_n:first]
    assert len(repeats) >= 3 and np.all(repeats == raw["sent_pos_dd"][ramp_n - 1])
    assert np.diff(raw["sent_t"][ramp_n - 1:first + 1]).max() < 0.1
    assert all(fr[1] == 0 for fr in robot.frames[ramp_n:first])


def test_every_policy_runs_a_step_before_arming(robot, tmp_path, monkeypatch):
    calls = []
    orig = rn.NumpyPolicy.__call__

    def counted(self, obs):
        calls.append(len(robot.frames))
        return orig(self, obs)
    monkeypatch.setattr(rn.NumpyPolicy, "__call__", counted)
    go(robot, tmp_path, seconds=0.1)
    assert calls[:2] == [0, 0] and len(calls) >= 7


def test_the_warm_up_leaves_nothing_behind(robot, tmp_path):
    """The first observed target and the first history are the run's own, as in a run without the warm-up."""
    rc, z, _ = go(robot, tmp_path, walker_bias=0.5, seconds=0.2)
    assert rc == 0
    assert np.allclose(z["obs"][0, 33:45], 0.0) and np.allclose(z["heading_psi"][0], 0.0, atol=1e-6)


def test_placed_start_upright_warns(robot, tmp_path):
    _, _, meta = go(robot, tmp_path, "--start-mode", "getup", "--getup-timeout", "0.2", seconds=1.0)
    assert any("not lying down" in w for w in meta["warnings"])


def test_heading_loop_steers_against_a_turn(robot, tmp_path):
    robot.script, robot.script_at = (lambda t: ((0.0, 0.0, 1000.0), (0.0, 0.0, 0.5 if t > 0.2 else 0.0))), POLICY_START
    _, z, _ = go(robot, tmp_path, seconds=1.0)
    assert z["wz_cmd"][-1] < -0.2 and z["obs"][-1, 8] == z["wz_cmd"][-1].astype(np.float32)
    assert np.all(z["heading_psi"][-1] > 0.0)


def test_commanded_turn_is_sent_without_the_loop(robot, tmp_path):
    robot.script, robot.script_at = (lambda t: ((0.0, 0.0, 1000.0), (0.0, 0.0, 0.5))), POLICY_START
    _, z, _ = go(robot, tmp_path, "--wz", "0.3", seconds=0.6)
    assert np.allclose(z["wz_cmd"], 0.3) and np.all(z["heading_err"] == 0.0)


def test_no_heading_loop(robot, tmp_path):
    robot.script, robot.script_at = (lambda t: ((0.0, 0.0, 1000.0), (0.0, 0.0, 0.5))), POLICY_START
    _, z, meta = go(robot, tmp_path, "--no-heading-loop", seconds=0.4)
    assert meta["heading_loop"] is None and len(z["wz_cmd"]) == 0 and np.all(z["obs"][:, 8] == 0.0)


def test_history_policy_clears_its_history_at_a_hand_back(robot, tmp_path, monkeypatch):
    seen = []
    orig = rn.HistoryStacker.reset

    def reset(self):
        seen.append(1)
        orig(self)
    monkeypatch.setattr(rn.HistoryStacker, "reset", reset)
    walker = tmp_path / "w"
    d = walker
    d.mkdir()
    meta = {"omni": True, "history": 2, "contract_sha256": rn.paths.sha256(rn.CONTRACT_JSON)}
    np.savez(d / "policy.npz", n_layers=1, scale=np.full(12, 0.3, dtype=np.float32), meta=json.dumps(meta),
             fixture_obs=np.zeros((2, 90), dtype=np.float32), fixture_act=np.zeros((2, 12), dtype=np.float32),
             W0=np.zeros((12, 90), dtype=np.float32), b0=np.zeros(12, dtype=np.float32))
    from robot_helpers import getup_policy
    g = getup_policy(tmp_path / "g")
    robot.script, robot.script_at = roll(fall_and_rise(lie_s=0.6, up_s=1.0)), POLICY_START + 10
    rc = rn.main(["--port", robot.port, "run", "--policy", str(d), "--getup-policy", str(g), "--stand-ramp", "0.2",
                  "--seconds", "3", "--log", str(tmp_path / "run")])
    z = np.load(tmp_path / "run.policy.npz")
    assert rc == 0 and json.loads(str(z["meta"]))["getups"][0]["end"] == "hand-back"
    assert len(seen) >= 2                       # at the start, and at the hand-back


def test_the_simulation_only_get_up_runs_from_its_flag(robot, tmp_path):
    root = Path(__file__).resolve().parents[2]
    robot.script, robot.script_at = roll(fall_and_rise(lie_s=0.6, up_s=1.0)), POLICY_START + 10
    rc = rn.main(["--port", robot.port, "run", "--policy", str(root / "policies/walker"),
                  "--getup-policy", str(root / "policies/getup_sim"), "--stand-ramp", "0.2", "--seconds", "3",
                  "--log", str(tmp_path / "run")])
    meta = json.loads(str(np.load(tmp_path / "run.policy.npz")["meta"]))
    assert rc == 0 and meta["getups"][0]["end"] == "hand-back", meta["stop_reason"]


def test_a_bad_target_in_the_loop_disarms(robot, tmp_path, monkeypatch):
    calls = []

    def refusal(dec, d):
        calls.append(1)
        return "scripted" if len(calls) > 2 + 5 else None     # the two of the warm-up, then five steps
    monkeypatch.setattr(rn, "target_refusal", refusal)
    _, z, meta = go(robot, tmp_path, seconds=1.0)
    assert meta["stop_reason"] == "disarm: target check: scripted" and len(z["tick"]) == 5
    assert sent_after_last_step(tmp_path, z) == 1


# ---- the watches ---------------------------------------------------------------------------------------------------
def stall(amps, servo=2):
    def f(i, pos, tgt):
        return amps * 1000.0 if i == servo else 0.0
    return f


def test_current_watch_flags_a_long_stall_and_never_stops(robot, tmp_path, capsys):
    robot.current_fn = stall(1.6)
    robot.guard_cur = (100000, 10.0)
    robot.script, robot.script_at = roll([(0.3, 90.0)]), POLICY_START + 10
    _, z, meta = go(robot, tmp_path, "--getup-timeout", "2.5", seconds=4.0)
    assert meta["stop_reason"].startswith("disarm: get-up: not up")
    (flag,) = meta["current_flags"]
    assert flag["servo"] == 3 and flag["seconds_over"] > 1.0 and flag["peak_A"] == 1.6
    assert "ADVISORY" in capsys.readouterr().out and z["current_flagged"].any()


def test_current_watch_is_silent_while_walking(robot, tmp_path):
    robot.current_fn = stall(1.3)
    _, _, meta = go(robot, tmp_path, seconds=1.5)
    assert meta["stop_reason"] == "completed" and not meta["current_flags"]


def test_reading_near_end_rule():
    r = rn.reading_near_end(np.array([0, 50, 51, 1550, 3049, 3050, 3112, 2000, 2000, 2000, 2000, 2000]))
    assert r.tolist() == [True, True, False, False, False, True, True, False, False, False, False, False]


def test_reading_watch_flags_once_per_get_up_and_never_stops(robot, tmp_path, capsys):
    robot.script, robot.script_at = roll([(0.3, 90.0)]), POLICY_START + 10
    orig = robot.before_tick

    def tick():
        orig()
        if len(robot.armed_seen) > POLICY_START + 80:
            robot.pos[2] = 3112.0
    robot.before_tick = tick
    _, z, meta = go(robot, tmp_path, "--getup-timeout", "1.5", seconds=3.0)
    assert meta["stop_reason"].startswith("disarm: get-up: not up")
    (flag,) = meta["reading_flags"]
    assert flag["servo"] == 3 and flag["track_end"] == 3100.0 and flag["getup"] == 1
    assert capsys.readouterr().out.count("reads 3112 dd") == 1 and z["reading_flagged"].any()


def test_a_start_reading_in_the_zone_prints_then_the_start_refuses(robot, tmp_path, capsys):
    robot.pos[2] = 3060.0
    try:
        go(robot, tmp_path)
    except Exception as exc:
        assert "implausible" in str(exc)
    else:
        raise AssertionError("no refusal")
    assert "at the start" in capsys.readouterr().out and not robot.armed_seen
