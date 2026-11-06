"""What the runner estimates from the STATE stream: the policy's observation, the gravity direction and tilt, whether the
robot has fallen, and its heading."""
from __future__ import annotations

import math

import numpy as np

from mp4_recovery.contract import CONTROL_DT, Contract, VelocityFilter, assemble_observation
from mp4_recovery.heading import I_MAX, KI, KP, WZ_MAX, pi_step

DEG = math.pi / 180.0
ACC_DROPOUT_MG = 1.0           # the firmware sends exact zeros without a fresh sample; real frames reach 25 mg under vibration
CF_TAU_S = 1.0                 # s: the accelerometer's pull on the gyro-integrated gravity
CF_GATE_G = (0.15, 0.5)        # g from 1 g: full pull within the first, none beyond the second
FALL_TILT_DEG, FALL_HOLD_S = 60.0, 0.3
FALL_FAST_TILT_DEG, FALL_FAST_GYRO = 45.0, 2.0


def tilt_deg(acc_mg) -> float:
    """The body's tilt from upright. Body z = -sensor z whatever the IMU's mounting about z."""
    a = np.asarray(acc_mg, dtype=float)
    n = np.linalg.norm(a)
    if n <= 0.0:
        return float("nan")
    return math.degrees(math.acos(max(-1.0, min(1.0, -a[2] / n))))


def imu_valid(st, imu_ok_flag: int) -> bool:
    """The IMU flag set and not the firmware's dropout zeros. The magnitude itself is no test: walking, single frames
    read 0.35-2.1 g."""
    n = float(np.linalg.norm(np.asarray(st.acc_mg, dtype=float)))
    return bool(st.flags & imu_ok_flag) and n > ACC_DROPOUT_MG


class GravityFilter:
    """The gravity direction in the body frame from a complementary filter: the gyro integrated (gravity is fixed in the
    world, so in the body frame it turns by -omega dt) and pulled towards the accelerometer's direction with a slow gain, the
    pull weighted by how close the frame's magnitude is to 1 g. An edge strike (0.2-3.4 g, single frames pointing anywhere)
    barely moves it, while a real roll is followed by the gyro at once. Fed every STATE frame with the body-frame gyro (rad/s,
    bias removed) and accelerometer (mg); starts at the first frame within ``gate_g[1]`` of 1 g (NaN before)."""

    def __init__(self, tau_s: float = CF_TAU_S, gate_g: tuple = CF_GATE_G):
        self.tau_s = tau_s
        self.gate_g = gate_g
        self.g = None

    def update(self, gyro_body, acc_body_mg, dt: float) -> float:
        acc = np.asarray(acc_body_mg, dtype=float)
        n = float(np.linalg.norm(acc))
        if self.g is None:
            if n <= ACC_DROPOUT_MG or abs(n / 1000.0 - 1.0) >= self.gate_g[1]:
                return float("nan")
            self.g = -acc / n
            return self.tilt()
        g = self.g
        w = np.asarray(gyro_body, dtype=float)
        wn = float(np.linalg.norm(w))
        th = wn * dt
        if th > 0.0:                                   # g turned by -th about w/|w| (Rodrigues)
            k = w / wn
            c, s = math.cos(th), math.sin(th)
            g = g * c - np.cross(k, g) * s + k * float(np.dot(k, g)) * (1.0 - c)
        if n > ACC_DROPOUT_MG:
            dev = abs(n / 1000.0 - 1.0)
            lo, hi = self.gate_g
            wt = min(1.0, max(0.0, (hi - dev) / (hi - lo)))
            g = g + min(1.0, dt / self.tau_s) * wt * (-acc / n - g)
        self.g = g / max(float(np.linalg.norm(g)), 1e-12)
        return self.tilt()

    def tilt(self) -> float:
        """Degrees from upright."""
        if self.g is None:
            return float("nan")
        return math.degrees(math.acos(max(-1.0, min(1.0, -float(self.g[2])))))


class FallDetector:
    """A fall: the filtered tilt above ``FALL_TILT_DEG`` for ``FALL_HOLD_S``, or above ``FALL_FAST_TILT_DEG`` while |gyro|
    exceeds ``FALL_FAST_GYRO`` (rad/s). ``fallen`` latches until ``reset()``; ``trigger`` says which rule fired."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.since = None
        self.fallen = False
        self.trigger = None

    def update(self, tilt: float, gyro_norm: float, t: float) -> bool:
        if self.fallen:
            return True
        if tilt > FALL_TILT_DEG:
            self.since = t if self.since is None else self.since
            if t - self.since >= FALL_HOLD_S - 1e-9:
                self.fallen, self.trigger = True, f"tilt > {FALL_TILT_DEG:g} deg for {FALL_HOLD_S:g} s"
        else:
            self.since = None
        if not self.fallen and tilt > FALL_FAST_TILT_DEG and gyro_norm > FALL_FAST_GYRO:
            self.fallen, self.trigger = True, f"tilt > {FALL_FAST_TILT_DEG:g} deg and |gyro| > {FALL_FAST_GYRO:g} rad/s"
        return self.fallen


def getup_phase(t_s: float, phase_s: float) -> tuple:
    """The get-up policy's command inputs: (phi, sin 2 pi phi, cos 2 pi phi), phi = ``t_s`` / ``phase_s``, held at 1."""
    phi = min(1.0, max(0.0, t_s / phase_s))
    return (phi, math.sin(2 * math.pi * phi), math.cos(2 * math.pi * phi))


class ObservationBuilder:
    """STATE frames -> the policy's observation, as ``contract.py`` defines it."""

    def __init__(self, contract: Contract, mount: np.ndarray, gyro_bias_rad_s=(0.0, 0.0, 0.0)):
        self.c = contract
        self.R = np.asarray(mount, dtype=float)
        self.bias = np.asarray(gyro_bias_rad_s, dtype=float)       # sensor frame
        self.vf = VelocityFilter()
        self.last_tick: int | None = None
        self.missing = 0
        self.latest = None

    def reset(self) -> None:
        self.vf.reset()
        self.last_tick = None
        self.latest = None

    def push(self, st) -> None:
        """Every STATE frame, in order. A missing tick repeats the previous sample."""
        rel = self.c.rel_from_dd(np.asarray(st.pos_dd, dtype=float))
        if self.last_tick is not None:
            gap = (int(st.tick) - self.last_tick - 1) & 0xFFFFFFFF
            if 0 < gap < 1000:
                self.missing += gap
                for _ in range(min(gap, 6)):
                    self.vf.push(self.vf.hist[-1])
        self.vf.push(rel)
        self.last_tick = int(st.tick)
        self.latest = st

    def gyro_body(self, st) -> np.ndarray:
        g = np.asarray(st.gyr_ddps, dtype=float) * 0.1 * DEG - self.bias
        return self.R @ g

    def gravity_body(self, st) -> np.ndarray:
        a = self.R @ np.asarray(st.acc_mg, dtype=float)
        return -a / np.linalg.norm(a)

    def observe(self, command, prev_action, pos_offset=None) -> np.ndarray:
        """``command``: three floats. ``pos_offset``: the policy's stand offset, subtracted from the measured positions."""
        st = self.latest
        pos = self.c.rel_from_dd(np.asarray(st.pos_dd, dtype=float))
        if pos_offset is not None and np.any(pos_offset):
            pos = pos - pos_offset
        return assemble_observation(self.gyro_body(st), self.gravity_body(st), command, pos, self.vf.velocity(), prev_action)


class HeadingLoop:
    """Holds a heading while no turn is commanded. The heading is the gyro's rate about gravity integrated over every STATE
    frame (gravity low-passed, the accelerometer being vibration-dominated while walking); each policy step the error to the
    target heading becomes a yaw-rate command (PI, clipped; ``heading.pi_step``)."""

    def __init__(self, tau_g_s: float = 1.0, frame_dt: float = 0.005):
        self.tau_g_s = tau_g_s
        self.alpha = frame_dt / tau_g_s
        self.dt = frame_dt
        self.reset()

    def reset(self) -> None:
        self.psi, self.integ, self.g_lp = 0.0, 0.0, None
        self.target = 0.0
        self.err = 0.0

    def relatch(self) -> None:
        """Hold the current heading from now on, the integrator cleared."""
        self.target = self.psi
        self.integ = 0.0
        self.err = 0.0

    def push(self, gyro_body, gravity_body) -> None:
        g = np.asarray(gravity_body, dtype=float)
        self.g_lp = g.copy() if self.g_lp is None else self.g_lp + self.alpha * (g - self.g_lp)
        up = -self.g_lp / max(np.linalg.norm(self.g_lp), 1e-9)
        self.psi += float(np.dot(np.asarray(gyro_body, dtype=float), up)) * self.dt

    def error(self) -> float:
        """The heading error (rad, + = turn left to correct), wrapped to +-pi."""
        return float(-(((self.psi - self.target) + np.pi) % (2 * np.pi) - np.pi))

    def wz(self, dt: float = CONTROL_DT) -> float:
        self.err = self.error()
        wz, integ = pi_step(self.err, self.integ, dt, KP, KI, WZ_MAX, I_MAX, "cond")
        self.integ = float(integ)
        return float(wz)

    def describe(self) -> dict:
        return {"kp": KP, "ki": KI, "wz_max": WZ_MAX, "i_max": I_MAX, "tau_g_s": self.tau_g_s, "antiwindup": "cond"}
