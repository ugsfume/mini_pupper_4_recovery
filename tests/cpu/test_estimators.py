"""The complementary filter, the fall detector and the heading loop, on synthetic and real IMU traces."""
import math
from pathlib import Path

import numpy as np
import pytest

from mp4_recovery.deploy import estimators as es
from mp4_recovery.deploy import runner as rn
from mp4_recovery.deploy.mp4link import protocol as p

ROLL_DPS = 300.0                # the scripted falls: 90 deg in 0.3 s
DATA = Path(__file__).resolve().parent / "data"


def roll_frames(phi_deg, dt=0.005):
    """Body-frame (gyro rad/s, acc mg) of a roll about the body x axis following ``phi_deg`` per frame."""
    phi = np.radians(np.asarray(phi_deg, dtype=float))
    rate = np.gradient(phi, dt)
    acc = np.stack([np.zeros_like(phi), 1000 * np.sin(phi), 1000 * np.cos(phi)], -1)
    gyro = np.stack([rate, np.zeros_like(phi), np.zeros_like(phi)], -1)
    return gyro, acc


def test_filter_follows_a_real_roll():
    t = np.arange(0, 1.0, 0.005)
    gyro, acc = roll_frames(np.clip((t - 0.2) * ROLL_DPS, 0, 90))
    gf = es.GravityFilter()
    tilt = np.array([gf.update(w, a, 0.005) for w, a in zip(gyro, acc)])
    assert tilt[int(round(0.5 / 0.005))] > 80.0 and abs(tilt[-1] - 90.0) < 2.0


def test_filter_follows_a_roll_the_accelerometer_does_not_show():
    t = np.arange(0, 0.5, 0.005)
    gyro, _ = roll_frames(np.clip((t - 0.2) * ROLL_DPS, 0, 90))
    gf = es.GravityFilter()
    tilt = [gf.update(w, [0, 0, 1000], 0.005) for w in gyro]
    assert tilt[-1] > 75.0


@pytest.mark.parametrize("seed", range(5))
def test_edge_strike_does_not_move_it(seed):
    """2 g frames pointing anywhere for 50 ms and a 0.2-3.4 g burst of 100 ms, with a gyro jolt: below 10 deg."""
    rng = np.random.default_rng(seed)
    gf = es.GravityFilter()
    peak = 0.0
    for i in range(400):
        acc, w = np.array([0.0, 0.0, 1000.0]), np.zeros(3)
        if 100 <= i < 110:
            th, az = math.radians(rng.uniform(0, 165)), rng.uniform(0, 2 * math.pi)
            acc = 2000 * np.array([math.sin(th) * math.cos(az), math.sin(th) * math.sin(az), math.cos(th)])
            w = rng.normal(0, 1.0, 3)
        if 200 <= i < 220:
            th, az = math.radians(rng.uniform(0, 165)), rng.uniform(0, 2 * math.pi)
            acc = rng.uniform(200, 3400) * np.array([math.sin(th) * math.cos(az), math.sin(th) * math.sin(az), math.cos(th)])
        peak = max(peak, gf.update(w, acc, 0.005))
    assert peak < 10.0, peak


def test_filter_starts_only_near_1g():
    gf = es.GravityFilter()
    assert math.isnan(gf.update([0, 0, 0], [0, 300, 40], 0.005))
    assert math.isnan(gf.update([0, 0, 0], [0, 1700, 0], 0.005))
    assert gf.update([0, 0, 0], [0, 0, 990], 0.005) < 1e-6


def test_filter_settles_on_a_slope_and_holds_against_bias():
    phi = math.radians(20.0)
    acc = [0, 1000 * math.sin(phi), 1000 * math.cos(phi)]
    gf = es.GravityFilter()
    gf.update([0, 0, 0], [0, 0, 1000], 0.005)
    for _ in range(1000):
        t = gf.update([0.02, 0.0, 0.0], acc, 0.005)
    assert abs(t - 20.0) < 2.0


def test_filter_stays_upright_on_recorded_trots():
    """Accelerometer captures of the real robot, no gyro recorded: far from the fall detector's 45 deg."""
    d = np.load(DATA / "sku2_imu_captures.npz")
    R = rn.IMU_MOUNTS["flip_x"]
    peak = {}
    for name in sorted({k.split("__")[0] for k in d.files}):
        acc, flags = d[name + "__acc_mg"], d[name + "__flags"]
        ok = [bool(fl & p.ST_IMU_OK) and np.linalg.norm(a) > es.ACC_DROPOUT_MG for a, fl in zip(acc, flags)]
        if sum(ok) < 0.9 * len(ok):
            continue
        gf = es.GravityFilter()
        peak[name] = max(gf.update([0, 0, 0], R @ a, 0.005) for a, o in zip(acc, ok) if o)
    motion = {k: v for k, v in peak.items() if "trot" in k or "bounce" in k or "keyframes" in k}
    assert motion and max(motion.values()) < 15.0, motion


def test_fall_detector_rules():
    fd = es.FallDetector()
    for i in range(59):
        assert not fd.update(61.0, 0.0, i * 0.005)
    assert fd.update(61.0, 0.0, 60 * 0.005) and "60" in fd.trigger
    fd.reset()
    assert not fd.update(50.0, 1.9, 0.0) and fd.update(46.0, 2.1, 0.005) and "gyro" in fd.trigger
    fd.reset()
    for i in range(100):                        # interrupted holds never add up
        fd.update(61.0 if i % 40 < 30 else 50.0, 0.0, i * 0.005)
    assert not fd.fallen


def test_getup_phase_values():
    assert es.getup_phase(0.0, 4.0) == pytest.approx((0.0, 0.0, 1.0))
    assert es.getup_phase(1.0, 4.0) == pytest.approx((0.25, 1.0, 0.0), abs=1e-9)
    assert es.getup_phase(9.0, 4.0) == pytest.approx((1.0, 0.0, 1.0), abs=1e-9)


def test_tilt_and_gravity_frames():
    assert es.tilt_deg([0, 0, -1000]) == pytest.approx(0.0)
    assert es.tilt_deg([0, 1000, 0]) == pytest.approx(90.0)
    assert math.isnan(es.tilt_deg([0, 0, 0]))


# ---- the heading loop -----------------------------------------------------------
def walk(loop, seconds=6.0, drift=-0.1, tilt_acc=(0.0, 0.0, -1.0), noise=0.0, seed=0):
    """The loop closed on a body that yaws by its command plus a constant drift."""
    rng = np.random.default_rng(seed)
    g = np.asarray(tilt_acc, dtype=float)
    wz = 0.0
    for _ in range(int(seconds / 0.02)):
        for _ in range(4):
            loop.push(g * -(drift + wz), g + rng.normal(0, noise, 3))
        wz = loop.wz()
    return loop.psi, wz


def test_heading_loop_holds_the_line_against_a_drift():
    psi, wz = walk(es.HeadingLoop())
    assert abs(psi) < math.radians(2.0) and wz == pytest.approx(0.1, abs=0.02)


def test_heading_about_gravity_on_a_slope():
    phi = math.radians(20.0)
    psi, _ = walk(es.HeadingLoop(), tilt_acc=(0.0, -math.sin(phi), -math.cos(phi)))
    assert abs(psi) < math.radians(2.0)


def test_heading_survives_a_vibrating_accelerometer():
    psi, _ = walk(es.HeadingLoop(), noise=0.5)
    assert abs(psi) < math.radians(3.0)


def test_heading_command_is_clipped_and_wraps():
    loop = es.HeadingLoop()
    loop.psi = 3.0
    assert loop.wz() == pytest.approx(-0.4)
    loop.psi = 3 * math.pi
    assert abs(loop.error()) == pytest.approx(math.pi)


def test_relatch_holds_the_new_heading():
    loop = es.HeadingLoop()
    loop.psi, loop.integ = 1.0, 0.3
    loop.relatch()
    assert loop.error() == 0.0 and loop.integ == 0.0 and loop.wz() == 0.0
