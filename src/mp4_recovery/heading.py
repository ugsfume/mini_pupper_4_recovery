"""The outer heading loop's control law, shared by the Pi runner (numpy floats) and the evaluation tools (torch
tensors, one value per env): PI on the heading error into a yaw-rate command, clipped, with the integrator clamped and,
by default, conditional integration (the integrator holds while the output is saturated in the error's direction).
numpy only at import: the runner on the Pi has no torch."""
from __future__ import annotations

import numpy as np

KP, KI, WZ_MAX, I_MAX = 2.0, 0.5, 0.4, 0.5
ANTIWINDUP = ("cond", "none")


def _ops(x):
    if hasattr(x, "clamp"):                                   # a torch tensor
        import torch
        return (lambda v, lo, hi: v.clamp(lo, hi)), torch.where
    return np.clip, np.where


def pi_step(err, integ, dt: float, kp: float = KP, ki: float = KI, wz_max: float = WZ_MAX, i_max: float = I_MAX,
            antiwindup: str = "cond"):
    """One step: ``err`` the heading error (rad, + = turn left to correct), ``integ`` the integrator (rad s).
    Returns (wz, new integ). ``antiwindup``: "none" integrates always (only the clamp at ±i_max); "cond" skips the
    update when the output before it is at the limit and the error has the output's sign (standard clamping: while
    saturated, an integrator of the opposite sign is held too, so it unwinds once |kp·err + ki·integ| < wz_max)."""
    if antiwindup not in ANTIWINDUP:
        raise ValueError(f"antiwindup must be one of {ANTIWINDUP}")
    clip, where = _ops(err)
    nxt = clip(integ + err * dt, -i_max, i_max)
    if antiwindup == "cond":
        u0 = kp * err + ki * integ
        nxt = where((abs(u0) >= wz_max) & (err * u0 > 0), integ, nxt)
    return clip(kp * err + ki * nxt, -wz_max, wz_max), nxt
