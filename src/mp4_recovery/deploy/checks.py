"""What the runner refuses: policies the robot must not run, and targets it must not send."""
from __future__ import annotations

import copy
import math

import numpy as np

from mp4_recovery.contract import (BAND_DEG, POLICY_KINDS, ROBOT_BAND_MAX_DEG, ROBOT_GETUP_BAND_MAX_DEG, TRACK_DD, ActionDecoder,
                                   Contract, calibration_refusal, contract_tendon_coefs, policy_band_deg, policy_bounds,
                                   policy_kind, policy_stand_offset_deg)

TARGET_END_DEG = 15.0          # every target's dd at least this far from a potentiometer's track end
BOUNDS_TOL_RAD = 1e-9
SELFTEST_STEPS = 400


def band_refusal(meta: dict) -> str | None:
    """Why the robot must not run a policy, or None: its band is wider than 90 deg (180 for a get-up policy with decoder
    bounds, which keep its targets out of the dead zones), a walking policy's is wider than the walking band, or its bounds
    were made for another calibration."""
    try:
        band = policy_band_deg(meta)
    except ValueError as exc:
        return str(exc)
    kind = policy_kind(meta)
    if kind not in POLICY_KINDS:
        return f"unknown policy kind {kind!r}"
    try:
        bounds = policy_bounds(meta)
        offset = policy_stand_offset_deg(meta)
    except (ValueError, KeyError, TypeError) as exc:
        return f"bounds: {exc}"
    widest = ROBOT_GETUP_BAND_MAX_DEG if kind == "getup" and bounds is not None else ROBOT_BAND_MAX_DEG
    if band.max() > widest:
        why = f"band {band.max():g} deg is wider than the robot's {widest:g}"
        if kind == "getup" and bounds is None and band.max() <= ROBOT_GETUP_BAND_MAX_DEG:
            why += f" (a get-up policy runs up to {ROBOT_GETUP_BAND_MAX_DEG:g} only with decoder bounds)"
        return why
    if kind == "walk" and np.any(band > BAND_DEG):
        return "a walk policy's band is wider than the walking band 20/60 deg"
    if kind == "getup" and "bounds" in meta and bounds is None:
        return "a get-up policy exported without decoder bounds (export it with the limits)"
    if offset and bounds is None:
        return f"a stand offset ({offset:g} deg) without decoder bounds"
    c = Contract.load()
    if bounds is not None:
        if not np.array_equal(contract_tendon_coefs(c), bounds.kc):
            return "bounds: tendon coefficients differ from the contract's"
    return calibration_refusal(meta, c)


def target_refusal(dec: ActionDecoder, d: dict) -> str | None:
    """Why a decoded target must not be sent, or None: outside the decoder's bounds (before the tick quantisation; by
    construction never), or its dd within ``TARGET_END_DEG`` of a track end."""
    if dec.bounds is not None:
        v = dec.bounds.violation_rel(d["rel_exact"], dec.c.stand_rel_rad)
        if v > BOUNDS_TOL_RAD:
            return f"a target {math.degrees(v):.3g} deg outside the policy's bounds"
    dd = d.get("dd")
    if dd is not None:
        m = 10.0 * TARGET_END_DEG
        bad = np.flatnonzero((np.asarray(dd) < TRACK_DD[0] + m) | (np.asarray(dd) > TRACK_DD[1] - m))
        if bad.size:
            i = int(bad[0])
            return f"servo {i + 1} target {int(dd[i])} dd within {TARGET_END_DEG:g} deg of its track end"
    return None


def decoder_selftest(dec: ActionDecoder, contract: Contract, seed: int = 0, steps: int = SELFTEST_STEPS) -> str | None:
    """Before arming: drive a copy of the decoder from extreme and random measured poses with saturated and random actions;
    every target must pass ``target_refusal``. Returns the first failure or None."""
    rng = np.random.default_rng(seed)
    d0 = copy.deepcopy(dec)
    ends = np.stack([contract.rel_from_dd(np.full(12, TRACK_DD[0])), contract.rel_from_dd(np.full(12, TRACK_DD[1]))])
    starts = [None, np.full(12, math.pi), np.full(12, -math.pi), ends[0], ends[1],
              rng.uniform(-math.pi, math.pi, 12), rng.uniform(ends.min(0), ends.max(0))]
    per = max(1, steps // len(starts))
    for rel0 in starts:
        d0.reset(rel0)
        for k in range(per):
            r = k % 4
            a = (np.ones(12) if r == 0 else -np.ones(12) if r == 1 else rng.choice([-1.0, 1.0], 12) if r == 2
                 else rng.uniform(-1.0, 1.0, 12))
            why = target_refusal(d0, d0(a))
            if why:
                return why
    return None
