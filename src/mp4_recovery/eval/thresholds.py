"""Check a get-up evaluation against fixed success, time and motion limits, per start.

    python -m mp4_recovery.eval.thresholds RESULT.json [--json OUT.json]

``RESULT.json`` is the output of ``getup_eval --motion``, normally 256 robots for 8 s on each of left, right, nose,
back, midfall, crouch and random. The limits on motion and impact are in ``data/motion_thresholds.json``. A start
missing from the result is reported as missing and the verdict is then "incomplete", never a pass. Crouch has no
success minimum.

Limits per start:

- success at least 90 % (left, right, nose) or 85 % (back, midfall, random);
- time to a held stand: median at most 5.0 s and 90th percentile at most 6.5 s;
- in the get-up window (``motion_meter``), the robots' mean of ``rev5`` (target reversals per joint per second),
  ``pow3`` (share of joint velocity power at 3 Hz or above) and ``speed_rms`` (rms leg joint speed);
- over the trial, the 99th percentile over robots of the peak base-floor force and the peak abduction limit force.

The exit status is 0 on a pass, 1 otherwise.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mp4_recovery import paths

THRESHOLDS_JSON = paths.DATA / "motion_thresholds.json"
SUCCESS_MIN = {"left": 0.90, "right": 0.90, "nose": 0.90, "back": 0.85, "midfall": 0.85, "random": 0.85}
STARTS = ("left", "right", "nose", "back", "midfall", "crouch", "random")
T50_MAX, T90_MAX = 5.0, 6.5
MOTION = ("rev5", "pow3", "speed_rms")
IMPACT = ("base_force_peak_N", "abd_limit_peak_Nm")
PASS, FAIL, MISSING = "pass", "fail", "missing"


def _at_most(value, limit) -> str:
    return MISSING if value is None or limit is None else PASS if value <= limit else FAIL


def _motion_value(summary: dict, metric: str):
    window, stat = ("trial", "p99") if metric in IMPACT else ("getup", "mean")
    try:
        return summary["motion"][window][metric][stat]
    except (KeyError, TypeError):
        return None


def check_start(summary: dict, start: str, limits: dict) -> dict:
    """Each limit's value, limit and status for one start's summary."""
    checks = {}
    if start in SUCCESS_MIN:
        ok = summary["success"] >= SUCCESS_MIN[start] - 1e-9
        checks["success"] = {"value": summary["success"], "limit": SUCCESS_MIN[start], "status": PASS if ok else FAIL}
    for name, key, limit in (("t50", "t_success_p50", T50_MAX), ("t90", "t_success_p90", T90_MAX)):
        checks[name] = {"value": summary.get(key), "limit": limit, "status": _at_most(summary.get(key), limit)}
    for name in MOTION + IMPACT:
        value = _motion_value(summary, name)
        checks[name] = {"value": value, "limit": limits.get(name), "status": _at_most(value, limits.get(name))}
    return checks


def evaluate(result: dict, thresholds: dict | None = None) -> dict:
    """``result``: ``getup_eval --motion``'s output. Returns per-start checks and the verdict."""
    thresholds = thresholds or json.loads(THRESHOLDS_JSON.read_text())
    starts, failed, missing = {}, [], []
    for start in STARTS:
        r = result["starts"].get(start)
        if r is None:
            starts[start] = {"status": MISSING}
            missing.append(start)
            continue
        checks = check_start(r["summary"], start, thresholds["starts"][start])
        states = {c["status"] for c in checks.values()}
        status = FAIL if FAIL in states else MISSING if MISSING in states else PASS
        starts[start] = {"status": status, "checks": checks}
        (failed if status == FAIL else missing if status == MISSING else []).append(start)
    verdict = FAIL if failed else "incomplete" if missing else PASS
    return {"verdict": verdict, "failed": failed, "missing": missing, "starts": starts,
            "settings": {k: result.get(k) for k in ("seed", "robots", "seconds")}}


def _cell(c: dict) -> str:
    v, lim = c["value"], c["limit"]
    return f"{'-' if v is None else format(v, '.3f')}/{'-' if lim is None else format(lim, '.3f')}"


def print_table(res: dict) -> None:
    print("settings", json.dumps(res["settings"]))
    cols = ("success", "t50", "t90") + MOTION + IMPACT
    print(f"{'start':8s} " + " ".join(f"{c:>18s}" for c in cols) + "  status (value/limit)")
    for start, row in res["starts"].items():
        cells = [_cell(row["checks"][c]) if c in row.get("checks", {}) else "-" for c in cols]
        print(f"{start:8s} " + " ".join(f"{c:>18s}" for c in cells) + f"  {row['status']}")
    print("verdict", res["verdict"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("result", help="getup_eval --motion's JSON")
    ap.add_argument("--thresholds", default=str(THRESHOLDS_JSON))
    ap.add_argument("--json", help="write the checks to this file")
    a = ap.parse_args(argv)
    res = evaluate(json.loads(Path(a.result).read_text()), json.loads(Path(a.thresholds).read_text()))
    print_table(res)
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1))
    return 0 if res["verdict"] == PASS else 1


if __name__ == "__main__":
    sys.exit(main())
