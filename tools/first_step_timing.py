#!/usr/bin/env python3
"""Measure the gap in SET_ALL frames across the walking policy's first step, without motors or a robot: the runner against a fake
board on a pseudo-terminal, in a fresh process per start, as a real launch is. The firmware disarms 100 ms after the last frame.

    env PYTHONPATH=src python3 -B tools/first_step_timing.py --starts 10

``plain`` runs without the two safeguards around the first step (``Runner.warm_up``, ``Runner.keep_fed``); ``safeguarded`` is
the runner as it ships. Per start it prints the gap, how long the first step took from its STATE frame to its SET_ALL, and whether the
fake board's watchdog tripped. Run it on the Pi to see the Pi's numbers: page cache and bytecode matter, so also try it right
after a reboot. Numpy and pyserial only.
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def child(mode: str, port: str, log: str) -> int:
    from mp4_recovery.deploy import runner as rn
    rn.HOLD_AFTER_S = 0.1
    if mode == "plain":
        rn.Runner.warm_up = lambda *a, **k: None
        rn.Runner.keep_fed = lambda self: None
    return rn.main(["--port", port, "run", "--seconds", "0.2", "--log", log])


def first_call_report() -> str:
    """What a fresh process pays for np.median, the slowest first call of the policy step on numpy 2."""
    code = ("import sys, time; import numpy as np; x = np.zeros((3, 12)); before = 'numpy.ma' in sys.modules;"
            "t = time.perf_counter(); np.median(x, axis=0); a = time.perf_counter() - t;"
            "t = time.perf_counter(); np.median(x, axis=0); b = time.perf_counter() - t;"
            "print(f\"numpy {np.__version__}: first np.median {a * 1e3:.2f} ms, second {b * 1e3:.3f} ms; numpy.ma loaded before: {before}, after: {'numpy.ma' in sys.modules}\")")
    return subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True, check=True).stdout.strip()


def one_start(mode: str, tmp: Path, n: int) -> dict:
    import numpy as np

    from fake_esp32 import FakeRobot
    from mp4_recovery.contract import Contract
    from mp4_recovery.deploy import runner as rn
    board = FakeRobot(Contract.load(), rn.load_mount()[1], rn.IDENTIFIED)
    log = tmp / f"{mode}_{n}"
    p = subprocess.run([sys.executable, "-B", __file__, "--child", mode, board.port, str(log)], capture_output=True, text=True)
    tripped = bool(board.latched & rn.p.ST_WATCHDOG)
    board.close()
    if p.returncode:
        raise SystemExit(p.stdout + p.stderr)
    z = np.load(f"{log}.policy.npz")
    r = np.load(f"{log}.npz")
    st, steps = r["sent_t"], z["t_send"]
    if len(steps):
        i = int(np.searchsorted(st, steps[0]))
        # sends from the ramp's end: the walker's first step and any frame kept flowing before it
        ramp_n = int(round(1.0 / rn.CONTROL_DT))
        gap = float(np.max(np.diff(st[ramp_n - 1:i + 1])) * 1e3)
        compute = float((steps[0] - z["t_rx"][0]) * 1e3)
    else:                                            # the first step never logged: the watchdog tripped at it
        gap = float(np.max(np.diff(st[:-1])) * 1e3)
        compute = float("nan")
    return {"gap_ms": gap, "compute_ms": compute, "tripped": tripped,
            "stop": json.loads(str(z["meta"]))["stop_reason"]}


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        return child(*sys.argv[2:5])
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--starts", type=int, default=10)
    a = ap.parse_args()
    sys.path.insert(0, str(ROOT / "tests" / "cpu"))
    print(first_call_report())
    import numpy as np
    with tempfile.TemporaryDirectory() as t:
        for mode in ("plain", "safeguarded"):
            rows = [one_start(mode, Path(t), n) for n in range(a.starts)]
            print(f"{mode}: largest gap ms {[round(r['gap_ms'], 1) for r in rows]}")
            print(f"{' ' * len(mode)}  first step, its STATE frame to its SET_ALL, ms {[round(r['compute_ms'], 1) for r in rows]}")
            print(f"{' ' * len(mode)}  worst {max(r['gap_ms'] for r in rows):.1f} ms, median "
                  f"{np.median([r['gap_ms'] for r in rows]):.1f} ms; watchdog trips {sum(r['tripped'] for r in rows)} of {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
