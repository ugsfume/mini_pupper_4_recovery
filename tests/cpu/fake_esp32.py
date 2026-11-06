"""A fake ESP32 on a pty speaking the link's frames, for tests that run without a robot.

``FakeEsp32`` keeps the firmware's rules: STATE at 200 Hz, arming, the 100 ms watchdog, the per-servo current guard (walking
limits, and get-up limits while the latest SET_ALL carries SA_GETUP, echoed as ST_GETUP; a get-up limit of 0 mA turns the guard
off in that mode), offsets, parameters. Joints follow their targets with a first-order lag.

``FakeRobot`` adds the Mini Pupper v4's calibration (from the contract) and IMU mounting (data/imu_mount.json), and an IMU played
from a script: ``script(t)`` -> (body-frame acc mg, body-frame gyro rad/s), t in s from the moment ``script_at`` armed ticks
have run.
"""
from __future__ import annotations

import math
import os
import select
import struct
import threading
import time
import tty

import numpy as np

from mp4_recovery.deploy.mp4link import protocol as p


class FakeEsp32:
    def __init__(self, tick_s: float = 0.005, watchdog_s: float = 0.1):
        self.master, slave = os.openpty()
        tty.setraw(slave)
        self.port = os.ttyname(slave)
        self._slave = slave
        self.tick_s, self.watchdog_s = tick_s, watchdog_s
        self.parser = p.Parser(accept=(p.MAGIC_REQ, p.MAGIC_SET_ALL))
        self.lock = threading.Lock()
        self.armed = False
        self.latched = 0
        self.fault_servo = 0
        self.seq = 0
        self.enable = 0
        self.cap = 0
        self.target = [1550.0] * 12
        self.pos = [1500.0 + 10 * i for i in range(12)]
        self.cur_ma = [0] * 12
        self.current_fn = None                          # (i, pos_dd, target_dd) -> mA
        self.walls: dict[int, tuple[float, float]] = {}
        self.temps_dc = [315] * 12
        self.acc = [0, 0, 1000]
        self.gyr = [0, 0, 0]
        self.imu_ok = True
        self.echo = True                                # False: a firmware without the get-up mode
        self.last_set_all = 0.0
        self.params = {i: {pid: 0.0 for pid in range(len(p.PARAMS))} for i in range(1, 13)}
        self.nvs: dict[int, float] = {}                 # stored offsets
        self.offsets: dict[int, float] = {}             # RAM table
        self.armed_seen: list[tuple[int, list[float]]] = []   # (enable, targets) per armed tick
        self.caps_armed: set[int] = set()
        self.frames: list[tuple[list[int], int, int, int]] = []   # every SET_ALL: (targets, flags, enable, cap)
        self.tick = 0
        self.guard_cur = (1500, 0.5)                    # (mA, s) walking
        self.guard_getup_cur = (0, 1.0)                 # (mA, s) get-up mode
        self.getup = False                              # SA_GETUP of the latest SET_ALL
        self.over_cur_s = [0.0] * 12
        self._guard_getup = False
        self._guard_t: float | None = None
        self._getup_since = 0.0
        self._run = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._run = False
        self._thread.join(timeout=1.0)
        os.close(self.master)
        os.close(self._slave)

    # ---- frames ----------------------------------------------------------
    def _respond(self, cmd, servo, status, val=0.0) -> None:
        pos = int(self.pos[servo - 1]) if 1 <= servo <= 12 else 0
        body = p.RSP.pack(p.MAGIC_RSP, cmd, servo, status, pos, 0, val)
        os.write(self.master, body + struct.pack("<H", p.crc16(body)))

    def _request(self, frame: bytes) -> None:
        _, cmd, servo, pad, a1, a2 = p.REQ.unpack(frame[:8])
        ok_servo = 1 <= servo <= 12
        if cmd == p.CMD_IDLE:
            self.armed = False
            return self._respond(cmd, servo, 0)
        if cmd == p.CMD_PING:
            return self._respond(cmd, servo, 0)
        if cmd in (p.CMD_SET_POS, p.CMD_TORQUE) or self.armed:
            return self._respond(cmd, servo, 6)
        if cmd == p.CMD_SAVE_OFFSETS:
            self.nvs.update(self.offsets)
            return self._respond(cmd, servo, 0, float(len(self.offsets)))
        if not ok_servo:
            return self._respond(cmd, servo, 1)
        value = struct.unpack("<f", struct.pack("<I", a1 | a2 << 16))[0]
        if cmd == p.CMD_GET_PARAM:
            return self._respond(cmd, servo, 0, self.params[servo][a1])
        if cmd == p.CMD_SET_PARAM:
            self.params[servo][pad] = value
            return self._respond(cmd, servo, 0, self.params[servo][pad])
        if cmd == p.CMD_GET_OFFSET:
            if servo not in self.offsets:
                return self._respond(cmd, servo, p.STATUS_UNSET, 0.0)
            return self._respond(cmd, servo, 0, self.offsets[servo])
        if cmd == p.CMD_SET_OFFSET:
            if not -90.0 <= value <= 90.0:
                return self._respond(cmd, servo, 2)
            self.offsets[servo] = value
            return self._respond(cmd, servo, 0, value)
        if cmd == p.CMD_GET_LIVE:
            return self._respond(cmd, servo, 0 if a1 == p.LIVE_TEMPERATURE_C else 3, 31.5)
        return self._respond(cmd, servo, 5)

    def _set_all(self, frame: bytes) -> None:
        v = p.SET_ALL.unpack(frame[:34])
        flags = v[3]
        self.seq, self.enable, self.cap = v[2], v[4], v[5]
        self.target = [float(x) for x in v[6:18]]
        self.frames.append(([int(x) for x in v[6:18]], flags, v[4], v[5]))
        self.last_set_all = time.monotonic()
        self.getup = bool(flags & p.SA_GETUP)
        if flags & p.SA_CLEAR:
            self.latched, self.fault_servo = 0, 0
        if flags & p.SA_DISARM:
            self.armed = False
        elif flags & p.SA_ARM:
            if self.latched & p.ST_FAULTS:
                self.latched |= p.ST_REJECTED
            else:
                self.armed = True
                self.latched &= ~(p.ST_WATCHDOG | p.ST_REJECTED)

    def currents(self) -> list[int]:
        if self.current_fn is None:
            return [int(c) for c in self.cur_ma]
        return [int(max(-32768, min(32767, self.current_fn(i, self.pos[i], self.target[i])))) for i in range(12)]

    def before_tick(self) -> None:
        """Hook: the IMU, positions or temperatures of the next frame."""

    def _state(self) -> bytes:
        self.before_tick()
        flags = (self.latched | (p.ST_ARMED if self.armed else 0) | (p.ST_IMU_OK if self.imu_ok else 0)
                 | (p.ST_GETUP if self.getup and self.echo else 0))
        body = p.STATE.pack(p.MAGIC_STATE, 1, self.seq, self.tick, int(time.monotonic() * 1e6) & 0xFFFFFFFF,
                            flags, self.fault_servo, 0x0F,
                            *[int(x) for x in self.pos], *self.currents(), *[int(t) for t in self.temps_dc],
                            *[int(v) for v in self.acc], *[int(v) for v in self.gyr], 0)
        return body + struct.pack("<H", p.crc16(body))

    def _guard(self, now: float) -> None:
        dt = 0.0 if self._guard_t is None else now - self._guard_t
        self._guard_t = now
        limit_ma, limit_s = self.guard_getup_cur if self.getup else self.guard_cur
        if self.getup != self._guard_getup:
            if self.getup:
                self._getup_since = now
            elif now - self._getup_since >= self.guard_getup_cur[1]:
                self.over_cur_s = [0.0] * 12
            self._guard_getup = self.getup
        cur = self.currents()
        trip = 0
        for i in range(12):
            if not (self.armed and self.enable >> i & 1) or (self.getup and limit_ma == 0):
                self.over_cur_s[i] = 0.0
                continue
            self.over_cur_s[i] = self.over_cur_s[i] + dt if abs(cur[i]) > limit_ma else 0.0
            if not trip and self.over_cur_s[i] > limit_s:
                trip = i + 1
        if trip and self.armed:
            self.armed = False
            self.latched |= p.ST_FAULT_CUR
            if not self.fault_servo:
                self.fault_servo = trip

    # ---- loop ------------------------------------------------------------
    def _loop(self) -> None:
        next_t = time.monotonic()
        while self._run:
            r, _, _ = select.select([self.master], [], [], max(0.0, next_t - time.monotonic()))
            if r:
                try:
                    data = os.read(self.master, 4096)
                except OSError:
                    return
                for magic, frame in self.parser.feed(data):
                    with self.lock:
                        (self._request if magic == p.MAGIC_REQ else self._set_all)(frame)
            if time.monotonic() >= next_t:
                next_t += self.tick_s
                with self.lock:
                    now = time.monotonic()
                    if self.armed and now - self.last_set_all > self.watchdog_s:
                        self.armed = False
                        self.latched |= p.ST_WATCHDOG
                    self._guard(now)
                    if self.armed:
                        self.armed_seen.append((self.enable, list(self.target)))
                        self.caps_armed.add(self.cap)
                        for i in range(12):
                            if self.enable >> i & 1:
                                self.pos[i] += 0.5 * (self.target[i] - self.pos[i])
                                if i in self.walls:
                                    lo, hi = self.walls[i]
                                    self.pos[i] = min(max(self.pos[i], lo), hi)
                    try:
                        os.write(self.master, self._state())
                    except OSError:
                        return
                    self.tick += 1


class FakeRobot(FakeEsp32):
    def __init__(self, contract, mount, gains, tick_s: float = 0.005, watchdog_s: float = 0.1):
        self.R = np.asarray(mount, dtype=float)
        self.script = None
        self.script_at = 0
        self.script_tick = None
        self.getup_frames = 0
        super().__init__(tick_s, watchdog_s)
        self.offsets = {i + 1: float(v) for i, v in enumerate(contract.offsets_deg)}
        self.pos = [float(v) for v in contract.stand_dd]
        ids = {name: i for i, name in enumerate(p.PARAMS)}
        for s in range(1, 13):
            for name, val in gains.items():
                self.params[s][ids[name]] = val
        self.set_body_imu((0.0, 0.0, 1000.0), (0.0, 0.0, 0.0))

    def set_body_imu(self, acc_mg, gyro_rad_s) -> None:
        self.acc = [int(round(v)) for v in self.R @ np.asarray(acc_mg, dtype=float)]
        self.gyr = [int(round(v)) for v in self.R @ np.degrees(np.asarray(gyro_rad_s, dtype=float)) * 10.0]

    def _set_all(self, frame: bytes) -> None:
        super()._set_all(frame)
        if p.SET_ALL.unpack(frame[:34])[3] & p.SA_GETUP:
            self.getup_frames += 1

    def before_tick(self) -> None:
        if self.script is None:
            return
        if self.script_tick is None and len(self.armed_seen) >= self.script_at:
            self.script_tick = self.tick
        if self.script_tick is not None:
            self.set_body_imu(*self.script((self.tick - self.script_tick) * 0.005))


def roll(segments, axis="roll"):
    """A script from (duration s, angle at its end deg) segments, linear in between, from 0 deg; held after. A segment
    shorter than a tick is a placement, not a motion: no gyro. ``roll`` turns about the body x axis, ``pitch`` about body y
    (+ = nose down)."""
    times = np.cumsum([0.0] + [s[0] for s in segments])
    angles = np.array([0.0] + [s[1] for s in segments])

    def f(t):
        phi = float(np.interp(t, times, angles))
        i = np.searchsorted(times, t, side="right") - 1
        rate = 0.0 if i >= len(segments) or segments[i][0] < 0.005 else math.radians((angles[i + 1] - angles[i]) / segments[i][0])
        r = math.radians(phi)
        if axis == "pitch":
            return (-1000 * math.sin(r), 0.0, 1000 * math.cos(r)), (0.0, rate, 0.0)
        return (0.0, 1000 * math.sin(r), 1000 * math.cos(r)), (rate, 0.0, 0.0)
    return f


def fall_and_rise(lie_s=0.5, up_s=1.2):
    """A fall (90 deg in 0.3 s), lying ``lie_s``, a rise back in 0.3 s, upright ``up_s``."""
    return [(0.3, 90.0), (lie_s, 90.0), (0.3, 0.0), (up_s, 0.0)]
