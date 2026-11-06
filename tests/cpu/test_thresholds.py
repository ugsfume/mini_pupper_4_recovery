"""eval/thresholds.py on synthetic getup_eval outputs: the limits pass and fail where they should, and a missing start is
missing, never a pass."""
import json

from mp4_recovery.eval import thresholds as t

LIMITS = {"rev5": 2.0, "pow3": 0.4, "speed_rms": 5.0, "base_force_peak_N": 150.0, "abd_limit_peak_Nm": 0.8}


def motion(rev5=1.0, pow3=0.3, speed=3.0, force=100.0, abd=0.5):
    window = {k: {"mean": v} for k, v in (("rev5", rev5), ("pow3", pow3), ("speed_rms", speed))}
    trial = {k: {"p99": v} for k, v in (("base_force_peak_N", force), ("abd_limit_peak_Nm", abd))}
    return {"getup": window, "trial": trial}


def result(success=None, t50=2.0, t90=3.0, mot=None, starts=t.STARTS):
    success = success or {}
    return {"seed": 12, "robots": 256, "seconds": 8.0,
            "starts": {s: {"summary": {"success": success.get(s, 0.97), "t_success_p50": t50, "t_success_p90": t90,
                                       "motion": (mot or {}).get(s, motion())}} for s in starts}}


THRESHOLDS = {"starts": {s: LIMITS for s in t.STARTS}}


def test_a_clean_result_passes():
    assert t.evaluate(result(), THRESHOLDS)["verdict"] == t.PASS


def test_success_time_motion_and_impact_fail():
    bad = t.evaluate(result(success={"nose": 0.89, "back": 0.86}), THRESHOLDS)
    assert bad["failed"] == ["nose"] and bad["starts"]["back"]["status"] == t.PASS
    assert t.evaluate(result(t90=6.6), THRESHOLDS)["starts"]["left"]["checks"]["t90"]["status"] == t.FAIL
    rough = t.evaluate(result(mot={"left": motion(rev5=2.1)}), THRESHOLDS)
    assert rough["failed"] == ["left"] and rough["starts"]["left"]["checks"]["rev5"]["status"] == t.FAIL
    hard = t.evaluate(result(mot={"back": motion(force=151.0)}), THRESHOLDS)
    assert hard["failed"] == ["back"]


def test_crouch_has_no_success_minimum():
    res = t.evaluate(result(success={"crouch": 0.1}), THRESHOLDS)
    assert res["verdict"] == t.PASS and "success" not in res["starts"]["crouch"]["checks"]


def test_a_missing_start_is_incomplete():
    res = t.evaluate(result(starts=("left", "right")), THRESHOLDS)
    assert res["verdict"] == "incomplete" and "nose" in res["missing"]


def test_a_missing_time_is_never_a_pass():
    res = t.evaluate(result(t50=None), THRESHOLDS)
    assert res["verdict"] == "incomplete"


def test_packaged_thresholds_cover_every_start_and_limit():
    d = json.loads(t.THRESHOLDS_JSON.read_text())
    for start in t.STARTS:
        assert set(d["starts"][start]) == set(t.MOTION + t.IMPACT)


def test_cli_exit_codes(tmp_path):
    good = tmp_path / "r.json"
    good.write_text(json.dumps(result()))
    limits = tmp_path / "t.json"
    limits.write_text(json.dumps(THRESHOLDS))
    assert t.main([str(good), "--thresholds", str(limits), "--json", str(tmp_path / "o.json")]) == 0
    assert json.loads((tmp_path / "o.json").read_text())["verdict"] == t.PASS
    good.write_text(json.dumps(result(success={"left": 0.1})))
    assert t.main([str(good), "--thresholds", str(limits)]) == 1
