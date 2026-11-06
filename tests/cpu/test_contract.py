"""The contract: firmware units, coordinates, the action decoder and the velocity filter."""

import numpy as np

from mp4_recovery.contract import (DD0, ActionDecoder, Contract, VelocityFilter, STEP_LIMIT_DEG)

C = Contract.load()


def test_stand_ticks_and_dd_match_the_stand():
    """servo_write(0 + offset) gives the contract's stand ticks and dd (its own formula, main.c:2526)."""
    ticks = C.ticks_from_servo_deg(np.zeros(12))
    assert ticks.tolist() == C.stand_ticks.tolist()
    assert C.dd_from_ticks(ticks).tolist() == C.stand_dd.tolist()




def test_dd_inverse_is_exact_on_the_tick_grid():
    """A dd produced from a tick reads back within half a tick (0.15 deg) of the angle that produced it."""
    rng = np.random.default_rng(1)
    s = rng.uniform(-40, 40, (200, 12))
    dd = C.dd_from_ticks(C.ticks_from_servo_deg(s))
    back = C.servo_deg_from_dd(dd)
    assert np.abs(back - s).max() < 3100 / 10240 + 0.11   # one tick (truncation) plus the dd truncation
    assert abs(DD0 - 1553.02734375) < 1e-12


def test_decoder_step_limit_band_and_quantisation():
    d = ActionDecoder(C, scale=[1.0] * 12)
    out = d(np.full(12, 1.0))                       # 57 deg asked; 20 deg allowed in one step
    assert np.all(np.degrees(np.abs(out["rel"])) <= STEP_LIMIT_DEG + 0.31)
    out2 = d(np.full(12, 1.0))
    assert np.all(np.degrees(np.abs(out2["rel"] - out["rel"])) <= STEP_LIMIT_DEG + 0.31)
    for _ in range(5):
        out = d(np.full(12, 1.0))
    ab = [0, 3, 6, 9]
    assert np.all(np.degrees(np.abs(out["rel"][ab])) <= 20.0 + 0.31)      # the abduction's band
    assert np.allclose(out["action"], out["rel"])                          # scale 1


def test_velocity_filter_ramp_and_spike():
    vf = VelocityFilter()
    t = np.arange(40) * 0.005
    rng = np.random.default_rng(3)
    x = 0.8 * t[:, None] * np.ones(12) + rng.normal(0, 1e-4, (40, 12))
    x[20, 3] += 0.05                                # a one-sample chatter spike
    out = []
    for k in range(40):
        vf.push(x[k])
        out.append(vf.velocity())
    out = np.array(out)
    assert np.abs(out[10:, 0] - 0.8).max() < 0.05          # a ramp reads its slope
    # The median replaces the spike by a neighbouring sample: on a ramp the error is one sample's increment (here
    # 0.8 rad/s x 5 ms = 4 mrad, weighted up to 1.5 / 25 ms), against 0.05 rad x 60 = 3 rad/s without the median.
    assert np.abs(out[18:26, 3] - 0.8).max() < 0.35
