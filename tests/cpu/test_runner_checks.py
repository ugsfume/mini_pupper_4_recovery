"""The refusals: policies the robot must not run and targets it must not send; the shipped policies pass."""
import json

import numpy as np
import pytest

from mp4_recovery import paths
from mp4_recovery.contract import ActionDecoder, Bounds, contract_tendon_coefs, policy_bounds
from mp4_recovery.deploy import checks
from mp4_recovery.deploy import runner as rn
from robot_helpers import C, getup_policy, go, make_policy

WALKER, GETUP = paths.POLICIES / "walker", paths.POLICIES / "getup"


def real_meta(path):
    return json.loads(str(np.load(path / "policy.npz")["meta"]))


@pytest.mark.parametrize("policy", [WALKER, GETUP])
def test_shipped_policies_pass_check(policy, capsys):
    assert rn.main(["check", "--policy", str(policy)]) == 0
    out = capsys.readouterr().out
    assert "FAIL" not in out and "DIFFERS" not in out


def test_shipped_policies_observe_the_standard_layout():
    w, g = rn.NumpyPolicy(WALKER / "policy.npz"), rn.NumpyPolicy(GETUP / "policy.npz")
    assert rn.layout_refusal(w) is None and rn.layout_refusal(g) is None
    assert rn.policy_kind(w.meta) == "walk" and rn.policy_kind(g.meta) == "getup" and g.meta["getup_phase_s"] > 0


def test_check_fails_on_wrong_fixtures(tmp_path, capsys):
    d = make_policy(tmp_path / "p")
    z = dict(np.load(d / "policy.npz"))
    z["fixture_act"] = z["fixture_act"] + 1.0
    np.savez(d / "policy.npz", **z)
    assert rn.main(["check", "--policy", str(d)]) == 1
    assert "FAIL" in capsys.readouterr().out


def test_check_fails_on_another_contract(tmp_path, capsys):
    d = make_policy(tmp_path / "p", contract_sha256="0" * 64)
    assert rn.main(["check", "--policy", str(d)]) == 1
    assert "DIFFERS" in capsys.readouterr().out


def test_bands_the_robot_refuses():
    assert "wider than the robot's 90" in checks.band_refusal({"band_deg": [91.0] * 12})
    assert "walking band" in checks.band_refusal({"band_deg": [30.0] * 12})
    assert "only with decoder bounds" in checks.band_refusal({"kind": "getup", "band_deg": [120.0] * 12})
    assert "unknown policy kind" in checks.band_refusal({"kind": "dance"})
    assert checks.band_refusal({"kind": "getup", "band_deg": [90.0] * 12}) is None
    assert checks.band_refusal({}) is None


def test_bounds_must_match_the_contract_calibration():
    meta = real_meta(GETUP)
    assert checks.band_refusal(meta) is None
    meta["bounds"]["info"]["contract_stand_dd"] = [v + 10 for v in C.stand_dd]
    assert "another calibration" in checks.band_refusal(meta)


def test_a_stand_offset_needs_bounds():
    assert "without decoder bounds" in checks.band_refusal({"stand_offset_abd_deg": 12.0, "band_deg": [20, 60, 60] * 4})


def test_run_refuses_policies_of_the_wrong_kind(robot, tmp_path, capsys):
    g = getup_policy(tmp_path / "g")
    rc = rn.main(["--port", robot.port, "run", "--policy", str(g), "--getup-policy", str(g)])
    assert rc == 2 and "not a walking policy" in capsys.readouterr().out
    w = make_policy(tmp_path / "w")
    rc = rn.main(["--port", robot.port, "run", "--policy", str(w), "--getup-policy", str(w)])
    assert rc == 2 and "not a get-up policy" in capsys.readouterr().out
    assert not robot.armed_seen


def test_run_refuses_a_get_up_without_its_phase_clock(robot, tmp_path, capsys):
    rc = go(robot, tmp_path, getup_meta={"getup_phase_s": 0.0})[0]
    assert rc == 2 and "phase clock" in capsys.readouterr().out and not robot.armed_seen


def test_run_refuses_a_walker_without_the_whole_command(robot, tmp_path, capsys):
    rc = go(robot, tmp_path, walker_meta={"omni": False})[0]
    assert rc == 2 and "whole command" in capsys.readouterr().out


def test_target_refusal_near_a_track_end():
    dec = ActionDecoder(C, np.full(12, 0.3))
    assert checks.target_refusal(dec, {"rel_exact": np.zeros(12), "dd": np.asarray(C.stand_dd)}) is None
    dd = np.array(C.stand_dd)
    dd[4] = 3000
    assert "servo 5" in checks.target_refusal(dec, {"rel_exact": np.zeros(12), "dd": dd})


def test_target_refusal_outside_the_bounds():
    b = policy_bounds(real_meta(GETUP))
    dec = ActionDecoder(C, np.full(12, 1.57), band_deg=[90.0] * 12, bounds=b)
    far = np.full(12, 4.0)
    assert "outside the policy's bounds" in checks.target_refusal(dec, {"rel_exact": far, "dd": None})


def test_selftest_passes_the_shipped_policies_and_catches_a_dead_zone():
    for path in (WALKER, GETUP):
        pol = rn.NumpyPolicy(path / "policy.npz")
        assert checks.decoder_selftest(rn.policy_decoder(pol, C), C) is None
    wide = Bounds(lo=C.stand_rel_rad - 4.0, hi=C.stand_rel_rad + 4.0, kc=contract_tendon_coefs(C),
                  calf_lo=np.full(4, -10.0), calf_hi=np.full(4, 10.0))
    dec = ActionDecoder(C, np.full(12, 3.0), band_deg=[170.0] * 12, bounds=wide)
    assert "track end" in checks.decoder_selftest(dec, C)


def test_selftest_failure_refuses_before_arming(robot, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(rn, "decoder_selftest", lambda dec, c: "scripted")
    assert go(robot, tmp_path)[0] == 2
    assert "decoder self-test: scripted" in capsys.readouterr().out and not robot.armed_seen


def test_the_contract_stand_is_its_calibration():
    assert C.dd_from_ticks(C.ticks_from_servo_deg(np.zeros(12))).tolist() == list(C.stand_dd)


def test_shipped_policies_run_the_sequence_on_the_fake_robot(robot, tmp_path):
    """The real networks through a fall, a get-up and a hand-back."""
    from fake_esp32 import fall_and_rise, roll
    robot.script, robot.script_at = roll(fall_and_rise(lie_s=1.0, up_s=2.0)), 100
    orig = robot.before_tick

    def tick():                                 # upright again: the joints at the stand whatever the policy asks
        orig()
        if robot.getup and robot.script_tick is not None and (robot.tick - robot.script_tick) * 0.005 > 1.3:
            robot.pos = [float(v) for v in C.stand_dd]
    robot.before_tick = tick
    log = tmp_path / "run"
    rc = rn.main(["--port", robot.port, "run", "--seconds", "4", "--log", str(log)])
    z = np.load(log.with_suffix(".policy.npz"))
    meta = json.loads(str(z["meta"]))
    assert rc == 0 and meta["stop_reason"] == "completed" and meta["getups"][0]["end"] == "hand-back"
    assert meta["policy_sha256"] == paths.sha256(WALKER / "policy.npz")

