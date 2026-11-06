"""Run the walking and get-up policies on a Mini Pupper v4 (SKU2) from its Raspberry Pi 5.

    python3 -m mp4_recovery.deploy.runner check [--policy DIR]
    python3 -m mp4_recovery.deploy.runner imu --seconds 25
    python3 -m mp4_recovery.deploy.runner run --vx 0 --seconds 40 --log logs/run1

The walking policy stands and walks. When the robot falls, the get-up policy takes over; once the robot is standing the
walking policy takes over again.

**The loop.** One policy step per 4 STATE frames (20 ms, locked to the ESP32's clock). Every frame goes into the velocity
filter, the fall detector and the heading loop, in tick order. Each step builds the observation from the newest frame, runs
the policy, decodes its action into targets (``contract.ActionDecoder``: clip, scale, band, 20 deg step limit, joint limits,
the firmware's ticks) and sends one SET_ALL.

**The sequence.** One second with the servos idle measures the gyro's bias: keep the robot still. A ramp to the walking
policy's stand follows, then the policy loop. With ``--start-mode getup`` (the robot placed lying down) there is no ramp: the
loop starts in the get-up.

**A fall** (``estimators.FallDetector`` on the complementary-filter tilt) during the ramp or walking switches to the get-up
policy at the next step: its decoder seeded from the measured pose, the observation rebuilt, the command inputs carrying its
phase clock, and SA_GETUP on every SET_ALL (the firmware's get-up mode: current guard off; its ST_GETUP echo must follow within 0.2 s).
**A retry:** when the phase has been at its end for ``--getup-phase-restart`` s and the tilt is still above 45 deg, the clock
restarts. **The hand-back:** tilt < 12 deg, |gyro| < 0.8 rad/s and the pose within 10 deg rms of the stand, held for 25 steps,
returns to the walking policy, its decoder seeded from the measured pose and the heading re-latched. Not up after
``--getup-timeout`` s, or one fall more than ``--getup-max``: limp, and the run ends.

**Stops.** The end of ``--seconds``, Enter, an invalid IMU frame, a non-finite action, a STATE gap, two late steps in a row or
a servo at 55 degC for 1 s: ramp to the stand, hold, disarm; during a get-up, limp at once (no ramp from a fallen pose). Ctrl-C, a
link fault or a target that fails its check: limp at once. A latched fault refuses the next launch until ``--clear``.

**The first step.** The firmware disarms 100 ms after the last frame, and between the ramp's last send and the first step nothing
else is sent. So one step of each policy runs before arming (``Runner.warm_up``: first calls are slow, and hold the interpreter),
and until the first step the last target is repeated on every STATE frame (``Runner.keep_fed``).

**Refusals before arming.** A latched fault, a servo gain that does not read back, no valid IMU, a servo at 45 degC, a joint
reading within 10 deg of a track end, a policy whose bounds were made for another calibration (``checks.band_refusal``), a
decoder that can reach a target too near a track end (``checks.decoder_selftest``). A difference between the robot's offsets
and the contract's is printed and logged.

**Watches (they print and log; nothing stops).** During a get-up, a servo above 1.2 A for more than 1 s. At the start and
during a get-up, a position reading within 5 deg of a track end.

**Logs.** ``<log>.npz``: every STATE and SET_ALL. ``<log>.policy.npz``: per step the STATE tick used, times, observation, raw
action, target, command, ``mode`` (0 walk, 1 get-up), filtered tilt and |gyro|; its meta lists every get-up (``getups``: entry,
trigger, restarts, end). ``<log>.stand.json``: the stand and the robot's offsets. Logs are written at the end of the run.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import select
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mp4_recovery import paths
from mp4_recovery.contract import (CONTRACT_JSON, CONTROL_DT, TRACK_DD, ActionDecoder, Contract, HistoryStacker,
                                   has_heading_obs, obs_size, policy_band_deg, policy_bounds, policy_filter_alpha, policy_kind,
                                   policy_stand_offset_deg)
from mp4_recovery.deploy.checks import TARGET_END_DEG, band_refusal, decoder_selftest, target_refusal
from mp4_recovery.deploy.estimators import (ACC_DROPOUT_MG, CF_GATE_G, CF_TAU_S, DEG, FALL_FAST_GYRO, FALL_FAST_TILT_DEG,  # noqa: F401
                                            FALL_HOLD_S, FALL_TILT_DEG, FallDetector, GravityFilter, HeadingLoop,
                                            ObservationBuilder, getup_phase, imu_valid, tilt_deg)
from mp4_recovery.deploy.mp4link import protocol as p
from mp4_recovery.deploy.mp4link.cli import GAINS as IDENTIFIED
from mp4_recovery.deploy.mp4link.cli import (Pacer, TempWatch, check_flags, preflight, present, read_offsets, save_log,
                                             smoothstep)
from mp4_recovery.policy_file import NumpyPolicy, fixture_tolerance

TICKS_PER_STEP = int(round(CONTROL_DT / 0.005))
TEMP_STOP_C = 55.0
TEMP_START_C = 45.0
CAP_MA = 2000                  # the frame's cap: in the servos' control mode 1, duty <= min(max_pwm, kff x cap)
HOLD_AFTER_S = 5.0             # at the stand before disarming at the end of a run
HANDBACK_TILT_DEG, HANDBACK_GYRO, HANDBACK_RMS_DEG, HANDBACK_STEPS = 12.0, 0.8, 10.0, 25
ECHO_TIMEOUT_S = 0.2
PHASE_RESTART_TILT_DEG = 45.0
WATCH_AMPS, WATCH_S = 1.2, 1.0
READING_WATCH_DEG = 5.0
READING_WATCH_DD = READING_WATCH_DEG * 10.0
MODE_WALK, MODE_GETUP = 0, 1

# The IMU's mounting, sensor -> body. data/imu_mount.json names the one measured; `imu` prints gravity in each candidate.
IMU_MOUNTS = {
    "flip_x": np.diag([1.0, -1.0, -1.0]),
    "flip_y": np.diag([-1.0, 1.0, -1.0]),
    "swap_xy": np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]]),
    "swap_xy_neg": np.array([[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]]),
}
IMU_MOUNT_JSON = paths.DATA / "imu_mount.json"


class StopRun(Exception):
    """A graceful stop: ramp to the stand, hold, disarm."""


class Disarm(Exception):
    """An immediate stop: limp now."""


class FallDetected(Exception):
    """A fall during the stand ramp or hold: the policy loop starts in the get-up."""


def load_mount(name: str | None = None) -> tuple[str, np.ndarray]:
    """A mounting by name; by default the measured one."""
    name = name or json.loads(IMU_MOUNT_JSON.read_text())["mount"]
    return name, IMU_MOUNTS[name]


def reading_near_end(pos_dd) -> np.ndarray:
    """Per servo: the measured ``pos_dd`` within ``READING_WATCH_DD`` of a track end, or past it."""
    d = np.asarray(pos_dd, dtype=float)
    return (d <= TRACK_DD[0] + READING_WATCH_DD) | (d >= TRACK_DD[1] - READING_WATCH_DD)


def policy_act(pol: NumpyPolicy):
    """``act(obs, k)``: the policy on its observation, stacked with its meta's history (cleared at k 0)."""
    hist = int(pol.meta.get("history", 0) or 0)
    if not hist:
        return lambda obs, k: pol(obs)
    stack = HistoryStacker(hist)

    def act(obs, k):
        if k == 0:
            stack.reset()
        return pol(stack(obs))
    return act


def policy_decoder(pol: NumpyPolicy, contract: Contract) -> ActionDecoder:
    return ActionDecoder(contract, pol.scale, band_deg=policy_band_deg(pol.meta), bounds=policy_bounds(pol.meta),
                         stand_offset_abd_deg=policy_stand_offset_deg(pol.meta), filter_alpha=policy_filter_alpha(pol.meta))


@dataclass
class GetupSpec:
    act: object                # act(obs, k), k = steps since this get-up began
    decoder: ActionDecoder
    max_getups: int
    timeout_s: float
    phase_s: float             # the time the phase clock takes to reach its end
    phase_restart_s: float     # the time it stays there before a retry; 0: never


class Runner:
    def __init__(self, link, contract: Contract, mount: np.ndarray, *, stdin=None, out=print):
        self.link, self.c, self.mount = link, contract, mount
        self.stdin = stdin
        self.out = out
        self.stand_dd = np.asarray(contract.stand_dd, dtype=int)
        self.armed = False
        self.clear = False                                # the first arming frame also clears a latched fault
        self.temp_watch = TempWatch(TEMP_STOP_C)
        self.last_dd = None
        self.steps: dict[str, list] = {k: [] for k in (
            "tick", "t_rx", "t_send", "seq", "obs", "raw", "rel", "dd", "tilt_deg", "late", "heading_psi", "heading_integ",
            "heading_err", "wz_cmd", "cmd", "mode", "tilt_cf_deg", "gyro_norm", "getup_echo", "cur_max_a", "current_flagged",
            "reading_flagged")}
        self.stop_reason = ""
        self.warnings: list[str] = []
        self.gyro_bias = np.zeros(3)
        self.robot_offsets = None
        self.command_initial = None
        self.mode = MODE_WALK
        self.fall = FallDetector()
        self.fall_pending = False                         # a fall seen, the get-up not yet entered
        self.stand_phase = False                          # the stand ramp or hold: a fall there raises FallDetected
        self.gf: GravityFilter | None = None
        self.tilt_cf = float("nan")
        self.gyro_norm = 0.0
        self._gf_tick = None
        self._rec_i = 0
        self.getups: list[dict] = []
        self.current_flags: list[dict] = []
        self._cw_tick = None
        self._cw_over = np.zeros(12)
        self._cw_active: list[dict | None] = [None] * 12
        self.reading_flags: list[dict] = []
        self._rw_seen: dict[int, dict] = {}
        self._rw_tick0 = None

    # ---- checks ---------------------------------------------------------------------------------------------------
    def preflight(self, clear: bool = False) -> None:
        st = self.link.wait_state(1.0)
        preflight(self.link, clear)
        self.clear = clear
        present(self.link)
        if not st.flags & p.ST_IMU_OK:
            raise SystemExit("IMU absent: power-cycle the board and check ping before a policy run")
        for i in range(1, 13):
            for name, value in {**IDENTIFIED, "ki_position": 0.0, "kp_current": 0.0}.items():
                self.link.set_param(i, name, value)
        for i in range(1, 13):
            for name, want in IDENTIFIED.items():
                got = self.link.get_param(i, name)
                if abs(got - want) > 1e-6 + 1e-3 * abs(want):
                    raise SystemExit(f"servo {i} {name} = {got:g}, not {want:g} after writing it")
        offs, stored = read_offsets(self.link)
        self.robot_offsets = offs
        if not all(stored) or np.any(np.abs(offs - self.c.offsets_deg) > 1e-3):
            msg = (f"WARNING: the robot's calibration offsets {np.round(offs, 2).tolist()} differ from the contract's "
                   f"{np.round(self.c.offsets_deg, 2).tolist()}; the policy runs on the contract's stand")
            self.warnings.append(msg)
            self.out(msg)
        temps = np.asarray(st.temp_dc, dtype=float) / 10.0
        if temps.max() >= TEMP_START_C:
            raise SystemExit(f"a servo is at {temps.max():.1f} degC: wait below {TEMP_START_C:.0f} degC")

    def measure_gyro_bias(self, seconds: float = 1.0) -> np.ndarray:
        """The gyro's mean over ``seconds`` with the servos idle (sensor frame, rad/s)."""
        end = time.monotonic() + seconds
        st, g = None, []
        while time.monotonic() < end:
            st = self.link.wait_state(0.1, after=st)
            if imu_valid(st, p.ST_IMU_OK):
                g.append(np.asarray(st.gyr_ddps, dtype=float) * 0.1 * DEG)
        if len(g) < 0.5 * seconds / 0.005:
            raise SystemExit("too few valid IMU frames to measure the gyro bias")
        self.gyro_bias = np.mean(g, axis=0)
        return self.gyro_bias

    # ---- sending --------------------------------------------------------------------------------------------------
    def send(self, dd) -> int:
        dd = np.asarray(dd)
        if not np.all(np.isfinite(dd)):
            raise StopRun("non-finite target")
        dd = np.clip(np.rint(dd), 0, 3100).astype(int)
        mode = p.SA_GETUP if self.mode == MODE_GETUP else 0
        flags = 0 if self.armed else p.SA_ARM | (p.SA_CLEAR if self.clear else 0)
        seq = self.link.set_all(dd, enable=0x0FFF, flags=flags | mode, cur_cap_ma=CAP_MA)
        self.armed = True
        self.last_dd = dd
        return seq

    def watch(self) -> None:
        """After each send: link faults, IMU, temperatures, Enter."""
        try:
            self.link.check()
            st = self.link.latest
            if st.flags & (p.ST_FAULTS | p.ST_WATCHDOG | p.ST_REJECTED):
                check_flags(self.link)
        except Exception as exc:
            raise Disarm(f"link: {exc}") from exc
        st = self.link.latest
        if not imu_valid(st, p.ST_IMU_OK):
            raise StopRun("IMU frame invalid")
        self.feed_imu()
        hot = self.temp_watch.update(st)
        if hot is not None:
            raise StopRun(f"servo at {hot:.1f} degC")
        if self.enter_pressed():
            raise StopRun("Enter")

    def enter_pressed(self) -> bool:
        if self.stdin is None:
            return False
        r, _, _ = select.select([self.stdin], [], [], 0)
        if r:
            self.stdin.readline()
            return True
        return False

    # ---- the complementary filter and the fall detector ------------------------------------------------------------
    def start_imu(self) -> None:
        """Start the complementary filter from the next frame (the gyro's bias known)."""
        self.gf = GravityFilter()
        self._gf_tick = None
        self._rec_i = len(self.link.record)

    def imu_update(self, s) -> None:
        """One STATE frame (each tick once, in order; invalid IMU frames skipped)."""
        if self.gf is None or not imu_valid(s, p.ST_IMU_OK):
            return
        tick = int(s.tick)
        if self._gf_tick is not None:
            gap = (tick - self._gf_tick) & 0xFFFFFFFF
            if gap == 0 or gap >= 0x80000000:
                return                                    # already fed, or older
            dt = min(gap, 20) * 0.005
        else:
            dt = 0.005
        self._gf_tick = tick
        gyro = self.mount @ (np.asarray(s.gyr_ddps, dtype=float) * 0.1 * DEG - self.gyro_bias)
        self.tilt_cf = self.gf.update(gyro, self.mount @ np.asarray(s.acc_mg, dtype=float), dt)
        self.gyro_norm = float(np.linalg.norm(gyro))
        if self.mode == MODE_WALK:
            self.fall.update(self.tilt_cf, self.gyro_norm, tick * 0.005)
            if self.fall.fallen:
                self.fall_pending = True

    def feed_imu(self) -> None:
        """Every recorded frame not yet fed."""
        if self.gf is None:
            return
        rec = self.link.record
        for s in rec[self._rec_i:]:
            self.imu_update(s)
        self._rec_i = len(rec)
        if self.fall_pending and self.stand_phase:
            raise FallDetected(self.fall.trigger)

    # ---- the current watch (get-up only; it prints and logs, nothing stops) -----------------------------------------
    def current_watch_reset(self, why: str = "mode change") -> None:
        for ev in self._cw_active:
            if ev is not None:
                ev.setdefault("end", why)
        self._cw_tick = None
        self._cw_over = np.zeros(12)
        self._cw_active = [None] * 12

    def current_watch(self, s, getup_n: int) -> None:
        """A servo above ``WATCH_AMPS`` for more than ``WATCH_S``: printed and flagged once per episode."""
        tick = int(s.tick)
        if self._cw_tick is not None:
            gap = (tick - self._cw_tick) & 0xFFFFFFFF
            if gap == 0 or gap >= 0x80000000:
                return
            dt = min(gap, 20) * 0.005
        else:
            dt = 0.005
        self._cw_tick = tick
        amps = np.abs(np.asarray(s.cur_ma, dtype=float)) / 1000.0
        over = amps > WATCH_AMPS
        self._cw_over = np.where(over, self._cw_over + dt, 0.0)
        names = self.c.meta.get("servo_order") or [str(i + 1) for i in range(12)]
        for i in range(12):
            ev = self._cw_active[i]
            if over[i] and ev is None and self._cw_over[i] > WATCH_S + 1e-9:
                ev = {"servo": i + 1, "name": names[i], "getup": getup_n, "tick": tick, "seconds_over": 0.0, "peak_A": 0.0}
                self._cw_active[i] = ev
                self.current_flags.append(ev)
                self.out(f"ADVISORY (log only): servo {i + 1} ({names[i]}) above {WATCH_AMPS:g} A for "
                         f"{WATCH_S:g} s in GETUP {getup_n} ({amps[i]:.2f} A)")
            if ev is not None:
                if over[i]:
                    ev["seconds_over"] = float(self._cw_over[i])
                    ev["peak_A"] = max(ev["peak_A"], float(amps[i]))
                else:
                    ev["end"], ev["end_tick"] = "below", tick
                    self._cw_active[i] = None

    # ---- the reading watch (the run's start and get-ups; it prints and logs, nothing stops) --------------------------
    def reading_watch_reset(self) -> None:
        self._rw_seen = {}

    def reading_watch(self, s, getup_n: int) -> None:
        """A servo reading near a track end: printed and flagged once per episode. Frames without fresh feedback from every
        board are skipped: their positions can be a boot's zeros."""
        if s.boards != 0x0F:
            return
        tick = int(s.tick)
        if self._rw_tick0 is None:
            self._rw_tick0 = tick
        d = np.asarray(s.pos_dd, dtype=float)
        near = reading_near_end(d)
        if not near.any():
            return
        names = self.c.meta.get("servo_order") or [str(i + 1) for i in range(12)]
        for i in np.flatnonzero(near).tolist():
            low = d[i] < 0.5 * (TRACK_DD[0] + TRACK_DD[1])
            ev = self._rw_seen.get(i)
            if ev is not None:
                ev["extreme_dd"] = min(ev["extreme_dd"], float(d[i])) if low else max(ev["extreme_dd"], float(d[i]))
                continue
            t_s = ((tick - self._rw_tick0) & 0xFFFFFFFF) * 0.005
            ev = {"servo": i + 1, "name": names[i], "dd": float(d[i]), "track_end": TRACK_DD[0] if low else TRACK_DD[1],
                  "getup": getup_n, "tick": tick, "t_s": t_s, "extreme_dd": float(d[i])}
            self._rw_seen[i] = ev
            self.reading_flags.append(ev)
            where = f"in GETUP {getup_n}" if getup_n else "at the start"
            self.out(f"ADVISORY (log only): servo {i + 1} ({names[i]}) reads {d[i]:.0f} dd, within "
                     f"{READING_WATCH_DEG:g} deg of the track end {ev['track_end']:.0f}, {where}, t {t_s:.2f} s")

    def reading_watch_start(self) -> None:
        """The first frame with fresh feedback from every board; the preflight, next, reports a link that has none."""
        try:
            for _ in range(20):
                st = self.link.wait_state(1.0)
                if st.boards == 0x0F:
                    self.reading_watch(st, 0)
                    return
        except Exception:
            return

    # ---- the first step -----------------------------------------------------------------------------------------------
    def warm_up(self, ob: ObservationBuilder, heading: HeadingLoop | None, walker_act, walker_decoder: ActionDecoder,
                getup: GetupSpec) -> None:
        """One step of each policy, nothing sent. A first call can hold the interpreter for tens of ms with every thread
        stalled, the serial reader's too: on numpy 2, the velocity filter's first ``np.median`` imports ``numpy.ma``."""
        st = self.link.wait_state(0.5)
        ob.push(st)
        if heading is not None:
            heading.push(ob.gyro_body(st), ob.gravity_body(st))
            heading.wz()
        for act, dec, command in ((walker_act, walker_decoder, (0.0, 0.0, 0.0)),
                                  (getup.act, getup.decoder, getup_phase(0.0, getup.phase_s))):
            obs = ob.observe(command, dec.prev_action, dec.obs_offset)
            target_refusal(dec, dec(np.asarray(act(obs, 0), dtype=float)))
            dec.reset()
        tilt_deg(st.acc_mg)
        ob.reset()
        if heading is not None:
            heading.reset()

    def keep_fed(self) -> None:
        """The last target again, so that the firmware's watchdog does not run down while the first step waits for its frames."""
        if self.armed:
            self.send(self.last_dd)

    # ---- moving to the stand --------------------------------------------------------------------------------------
    def ramp(self, to_dd, seconds: float, frm=None, stoppable: bool = True) -> None:
        frm = np.asarray(self.last_dd if frm is None else frm, dtype=float)
        to = np.asarray(to_dd, dtype=float)
        pacer = Pacer(CONTROL_DT)
        n = max(1, int(round(seconds / CONTROL_DT)))
        for k in range(n):
            self.send(frm + (to - frm) * smoothstep((k + 1) / n))
            pacer.wait()
            try:
                self.watch()
            except StopRun:
                if stoppable:
                    raise

    def hold(self, dd, seconds: float, stoppable: bool = True) -> None:
        pacer = Pacer(CONTROL_DT)
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            self.send(dd)
            pacer.wait()
            try:
                self.watch()
            except StopRun:
                if stoppable:
                    raise

    def safe_end(self) -> None:
        """Ramp to the stand, hold, disarm."""
        try:
            if self.armed and self.last_dd is not None:
                self.ramp(self.stand_dd, 1.0, stoppable=False)
                self.hold(self.stand_dd, HOLD_AFTER_S, stoppable=False)
        except Disarm as exc:
            self.stop_reason += f"; {exc}"
        finally:
            self.link.disarm()
            self.armed = False

    # ---- the loop -------------------------------------------------------------------------------------------------
    def loop(self, walker_act, command, seconds: float, ob: ObservationBuilder, walker_decoder: ActionDecoder,
             heading: HeadingLoop | None, getup: GetupSpec, start_getup: bool = False, start_trigger: str | None = None) -> None:
        """Policy steps locked to the STATE ticks. ``command``: (vx, vy, wz). ``start_getup``: a fall already happened, the
        first step is in the get-up (``start_trigger`` names it when the robot was placed lying down)."""
        link = self.link
        rec = link.record
        idx = len(rec)
        self.keep_fed()
        st = link.wait_state(0.05)
        ob.reset()
        walker_decoder.reset()
        ob.push(st)
        if heading is not None:
            heading.reset()
        command = tuple(float(v) for v in command)
        self.command_initial = list(command)
        next_tick = int(st.tick) + TICKS_PER_STEP
        late_run = 0
        n_steps = int(round(seconds / CONTROL_DT))
        k_mode = 0                            # steps since the current mode began
        if start_getup:
            self.fall_pending = True
        g_entry = g_echo = g_phase = 0        # the get-up's entry tick, last echo tick, phase clock's origin tick
        hb_run = 0                            # consecutive steps meeting the hand-back criterion
        for k in range(n_steps):
            while True:                       # every frame up to the step's tick, in order
                try:
                    st = link.wait_state(0.025, after=st)
                except Exception as exc:
                    raise StopRun(f"STATE gap: {exc}") from exc
                new = rec[idx:]
                idx = len(rec)
                for s in new:
                    if ob.last_tick is None or ((int(s.tick) - ob.last_tick) & 0xFFFFFFFF) < 1000:
                        if int(s.tick) != ob.last_tick:
                            ob.push(s)
                            if heading is not None and imu_valid(s, p.ST_IMU_OK):
                                heading.push(ob.gyro_body(s), ob.gravity_body(s))
                    if not imu_valid(s, p.ST_IMU_OK):
                        raise StopRun("IMU frame invalid")
                    self.imu_update(s)
                    if self.mode == MODE_GETUP:
                        if s.flags & p.ST_GETUP:
                            g_echo = int(s.tick)
                        self.current_watch(s, len(self.getups))
                        self.reading_watch(s, len(self.getups))
                if ((int(st.tick) - next_tick) & 0xFFFFFFFF) < 0x80000000:
                    break
                if k == 0:
                    self.keep_fed()
            late = (int(st.tick) - next_tick) & 0xFFFFFFFF
            late_run = late_run + 1 if late > 0 else 0
            if late_run >= 2:
                raise StopRun("two late steps in a row")
            next_tick = int(st.tick) + TICKS_PER_STEP
            tcf, gn = self.tilt_cf, self.gyro_norm
            pos = self.c.rel_from_dd(np.asarray(st.pos_dd, dtype=float))
            if self.mode == MODE_WALK and self.fall_pending:                  # walk -> get-up
                g_now = self.gf.g if self.gf is not None else None
                ev = {"n": len(self.getups) + 1, "step": k, "tick": int(st.tick), "t": time.monotonic(),
                      "trigger": (start_trigger or "stand ramp/hold: " + str(self.fall.trigger))
                      if start_getup and k == 0 else str(self.fall.trigger), "tilt_cf_deg": self.tilt_cf, "which": "getup",
                      "roll_deg": None if g_now is None else math.degrees(math.atan2(-g_now[1], -g_now[2])),
                      "gravity": None if g_now is None else [float(v) for v in g_now]}
                if len(self.getups) >= getup.max_getups:
                    raise Disarm(f"fall {ev['n']} ({ev['trigger']}): --getup-max {getup.max_getups} reached")
                self.getups.append(ev)
                self.out(f"fall ({ev['trigger']}, tilt {self.tilt_cf:.0f} deg): GETUP {ev['n']}")
                self.mode, self.fall_pending = MODE_GETUP, False
                ob.reset()
                ob.push(st)
                getup.decoder.reset(pos)
                g_entry = g_echo = g_phase = int(st.tick)
                self.current_watch_reset()
                if not (start_trigger and k == 0):         # a placed start's reading watch goes on from the run's start
                    self.reading_watch_reset()
                self.reading_watch(st, ev["n"])
                hb_run, k_mode = 0, 0
            elif self.mode == MODE_GETUP:
                ev = self.getups[-1]
                if ((int(st.tick) - g_echo) & 0xFFFFFFFF) * 0.005 > ECHO_TIMEOUT_S:
                    msg = "GETUP: no ST_GETUP echo for 0.2 s: this firmware has no get-up mode"
                    self.warnings.append(msg)
                    self.out("WARNING: " + msg)
                    ev.update(end="no echo", end_step=k)
                    raise Disarm("get-up: no ST_GETUP echo")
                ok = (tcf < HANDBACK_TILT_DEG and gn < HANDBACK_GYRO
                      and math.degrees(float(np.sqrt(np.mean(pos ** 2)))) < HANDBACK_RMS_DEG)
                hb_run = hb_run + 1 if ok else 0
                t_in = ((int(st.tick) - g_entry) & 0xFFFFFFFF) * 0.005
                if hb_run >= HANDBACK_STEPS:                                  # get-up -> walk
                    ev.update(end="hand-back", end_step=k, end_tick=int(st.tick), seconds=t_in)
                    self.out(f"hand-back after {t_in:.2f} s in GETUP {ev['n']}")
                    self.mode = MODE_WALK
                    self.current_watch_reset()
                    self.reading_watch_reset()
                    self.fall.reset()
                    ob.reset()
                    ob.push(st)
                    walker_decoder.reset(pos)
                    if heading is not None:
                        heading.relatch()
                    k_mode = 0
                elif t_in > getup.timeout_s:
                    ev.update(end="timeout", end_step=k, seconds=t_in)
                    raise Disarm(f"get-up: not up after {getup.timeout_s:g} s (limp)")
                elif getup.phase_restart_s > 0:
                    t_ph = ((int(st.tick) - g_phase) & 0xFFFFFFFF) * 0.005
                    if t_ph >= getup.phase_s + getup.phase_restart_s - 1e-9 and tcf > PHASE_RESTART_TILT_DEG:
                        g_phase = int(st.tick)
                        ev.setdefault("phase_restarts", []).append(k)
                        self.out(f"GETUP {ev['n']}: phase restarted after {t_in:.2f} s (tilt {tcf:.0f} deg)")
            cmd = command
            walking = self.mode == MODE_WALK
            if heading is not None:
                if not walking:               # during a get-up the heading is tracked, the loop idle
                    heading.err = 0.0
                    cmd = (command[0], command[1], 0.0)
                elif abs(command[2]) < 1e-6:
                    cmd = (command[0], command[1], command[2] + heading.wz())
                else:                         # a commanded turn is sent alone
                    heading.err = 0.0
                self.steps["heading_psi"].append(heading.psi)
                self.steps["heading_integ"].append(heading.integ)
                self.steps["heading_err"].append(heading.err)
                self.steps["wz_cmd"].append(cmd[2])
            self.steps["cmd"].append((*command, cmd[2]))
            if walking:
                obs = ob.observe(cmd, walker_decoder.prev_action, walker_decoder.obs_offset)
                raw = np.asarray(walker_act(obs, k_mode), dtype=float)
                dec = walker_decoder
            else:
                phi_t = ((int(st.tick) - g_phase) & 0xFFFFFFFF) * 0.005
                obs = ob.observe(getup_phase(phi_t, getup.phase_s), getup.decoder.prev_action, getup.decoder.obs_offset)
                raw = np.asarray(getup.act(obs, k_mode), dtype=float)
                dec = getup.decoder
            k_mode += 1
            if not np.all(np.isfinite(raw)):
                raise StopRun("non-finite action")
            d = dec(raw)
            why = target_refusal(dec, d)
            if why:
                raise Disarm(f"target check: {why}")
            t_send = time.monotonic()
            seq = self.send(d["dd"])
            if k == 0:
                self.out("policy start")
            self.watch()
            for key, val in (("tick", int(st.tick)), ("t_rx", st.t_rx), ("t_send", t_send), ("seq", seq),
                             ("obs", obs), ("raw", raw), ("rel", d["rel"]), ("dd", d["dd"]),
                             ("tilt_deg", tilt_deg(st.acc_mg)), ("late", late), ("mode", int(not walking)),
                             ("tilt_cf_deg", tcf), ("gyro_norm", gn), ("getup_echo", bool(st.flags & p.ST_GETUP)),
                             ("cur_max_a", float(np.max(np.abs(np.asarray(st.cur_ma, dtype=float)))) / 1000.0),
                             ("current_flagged", any(e is not None for e in self._cw_active)),
                             ("reading_flagged", not walking and bool(self._rw_seen))):
                self.steps[key].append(val)

    def save(self, log: str, meta: dict) -> None:
        base = Path(log)
        base.parent.mkdir(parents=True, exist_ok=True)
        meta = dict(meta, stop_reason=self.stop_reason, warnings=self.warnings, gyro_bias_rad_s=self.gyro_bias.tolist(),
                    stand=self.stand_dd.tolist(), cap=CAP_MA)
        save_log(self.link, str(base.with_suffix(".npz")), None, meta)
        arrays = {k: np.asarray(v) for k, v in self.steps.items()}
        np.savez_compressed(base.with_suffix(".policy.npz"), meta=json.dumps(meta), **arrays)
        offs = self.robot_offsets if self.robot_offsets is not None else self.c.offsets_deg
        base.with_suffix(".stand.json").write_text(json.dumps(
            {"stand_dd": self.stand_dd.tolist(), "stand_ticks": np.asarray(self.c.stand_ticks).tolist(),
             "offsets_deg": np.asarray(offs, dtype=float).tolist(), "source": "the contract; offsets read from the robot"},
            indent=1))


# ---------------------------------------------------------------------------------------------------- commands
def layout_refusal(pol: NumpyPolicy) -> str | None:
    """Why the runner cannot feed this policy, or None: it observes more than the standard terms."""
    if has_heading_obs(pol.meta):
        return "this policy observes the heading error: the runner does not feed that input"
    want = obs_size() * max(int(pol.meta.get("history", 0) or 0), 1)
    have = int(pol.layers[0][0].shape[1])
    return None if have == want else f"the policy has {have} inputs, its meta says {want}"


def cmd_check(a) -> int:
    """No port: the policy under this numpy against its export's fixtures, its contract, its layout, its band and bounds,
    the decoder self-test."""
    path = Path(a.policy) / "policy.npz"
    pol = NumpyPolicy(path)
    d = np.load(path)
    err = float(np.abs(pol(d["fixture_obs"]) - d["fixture_act"]).max())
    tol = fixture_tolerance(d["fixture_act"])
    want = pol.meta.get("contract_sha256")
    have = paths.sha256(CONTRACT_JSON)
    # a contract lists the calibrations it replaced; they differ only in the offsets and the stand
    earlier = {c["sha256"] for c in json.loads(CONTRACT_JSON.read_text()).get("calibrations_superseded", [])}
    same = want == have or want in earlier
    layout_why = layout_refusal(pol)
    x = d["fixture_obs"][0]
    t0 = time.perf_counter()
    for _ in range(1000):
        pol(x)
    us = (time.perf_counter() - t0) * 1e3
    print(f"fixtures: max |numpy - torch| {err:.2e} ({'ok' if err < tol else 'FAIL'}, limit {tol:.2e})")
    print(f"contract: {'ok' if want == have else 'ok (exported for an earlier calibration)' if same else 'DIFFERS from the policy'}"
          f" ({have[:12]})")
    print(f"inputs: {int(pol.layers[0][0].shape[1])} ({'ok' if not layout_why else 'FAIL: ' + layout_why})")
    print(f"inference: {us:.1f} us per step; numpy {np.__version__}")
    band_why = band_refusal(pol.meta)
    print(f"kind {policy_kind(pol.meta)}, band deg {policy_band_deg(pol.meta).tolist()} "
          f"({'FAIL: ' + band_why if band_why else 'ok'})")
    test_why = "not run (the band check failed)"
    if not band_why:
        b = policy_bounds(pol.meta)
        print(f"bounds: {'none' if b is None else b.mode}, stand offset {policy_stand_offset_deg(pol.meta):g} deg outward, "
              f"target filter alpha {policy_filter_alpha(pol.meta):g}")
        c = Contract.load()
        test_why = decoder_selftest(ActionDecoder(c, d["scale"], band_deg=policy_band_deg(pol.meta), bounds=b,
                                                  stand_offset_abd_deg=policy_stand_offset_deg(pol.meta),
                                                  filter_alpha=policy_filter_alpha(pol.meta)), c)
    print(f"decoder self-test (targets inside the bounds, {TARGET_END_DEG:g} deg from the track ends): "
          f"{'FAIL: ' + test_why if test_why else 'ok'}")
    return 0 if err < tol and same and not layout_why and not band_why and not test_why else 1


def cmd_imu(link, a) -> int:
    """The servos stay idle. Prints the tilt and the gravity direction in each mounting candidate, to tell which one fits:
    nose down, gravity's x is positive; right side down, its y is negative."""
    link.record = []
    end = time.monotonic() + a.seconds
    st, last_print = None, 0.0
    while time.monotonic() < end:
        st = link.wait_state(0.5, after=st)
        if time.monotonic() - last_print > 0.2:
            last_print = time.monotonic()
            acc = np.asarray(st.acc_mg, dtype=float)
            cands = "  ".join(f"{k}: g=({', '.join(f'{v:+.2f}' for v in -(R @ acc) / np.linalg.norm(acc))})"
                              for k, R in IMU_MOUNTS.items()) if np.linalg.norm(acc) > 0 else "acc 0"
            print(f"tilt {tilt_deg(acc):5.1f}  imu {'ok' if st.flags & p.ST_IMU_OK else 'ABSENT'}  "
                  f"gyro dps {np.round(np.asarray(st.gyr_ddps) * 0.1, 1).tolist()}  {cands}", flush=True)
    if a.log:
        save_log(link, a.log, None, {"cmd": "imu", "seconds": a.seconds})
    return 0


def cmd_run(link, a) -> int:
    contract = Contract.load()
    mount_name, mount = load_mount()
    walker = NumpyPolicy(Path(a.policy) / "policy.npz")
    gpol = NumpyPolicy(Path(a.getup_policy) / "policy.npz")
    why = None
    if policy_kind(walker.meta) != "walk":
        why = f"--policy is a {policy_kind(walker.meta)!r} policy, not a walking policy"
    elif policy_kind(gpol.meta) != "getup":
        why = f"--getup-policy is a {policy_kind(gpol.meta)!r} policy, not a get-up policy"
    elif not walker.meta.get("omni"):
        why = "--policy does not observe the whole command (vx, vy, wz)"
    elif not gpol.meta.get("getup_phase_s"):
        why = "--getup-policy has no phase clock (meta getup_phase_s)"
    else:
        for flag, pol in (("--policy", walker), ("--getup-policy", gpol)):
            why = band_refusal(pol.meta) or layout_refusal(pol)
            if why:
                why = f"{flag}: {why}"
                break
    if why:
        print(f"refused: {why}", flush=True)
        return 2
    decoder = policy_decoder(walker, contract)
    getup = GetupSpec(act=policy_act(gpol), decoder=policy_decoder(gpol, contract), max_getups=int(a.getup_max),
                      timeout_s=float(a.getup_timeout), phase_s=float(gpol.meta["getup_phase_s"]),
                      phase_restart_s=float(a.getup_phase_restart))
    for flag, dec in (("--policy", decoder), ("--getup-policy", getup.decoder)):
        why = decoder_selftest(dec, contract)
        if why:
            print(f"refused: {flag}: decoder self-test: {why}", flush=True)
            return 2
    heading = HeadingLoop() if a.heading_loop else None
    runner = Runner(link, contract, mount, stdin=sys.stdin if sys.stdin.isatty() else None)
    runner.reading_watch_start()              # before the preflight, whose position check refuses these readings
    runner.preflight(a.clear)
    link.record = []
    runner.measure_gyro_bias(1.0)
    ob = ObservationBuilder(contract, mount, runner.gyro_bias)
    runner.start_imu()
    walk_stand = decoder.stand_dd()
    walker_act = policy_act(walker)
    meta = {"cmd": "run", "start_mode": a.start_mode, "seconds": a.seconds, "vx": a.vx, "vy": a.vy, "wz": a.wz,
            "policy": str(Path(a.policy).resolve()), "policy_sha256": paths.sha256(Path(a.policy) / "policy.npz"),
            "getup_policy": str(Path(a.getup_policy).resolve()),
            "getup_policy_sha256": paths.sha256(Path(a.getup_policy) / "policy.npz"),
            "contract_sha256": paths.sha256(CONTRACT_JSON), "mount": mount_name, "stand_dd_policy": [int(v) for v in walk_stand],
            "heading_loop": None if heading is None else heading.describe(), "getup_max": getup.max_getups,
            "getup_timeout_s": getup.timeout_s, "getup_phase_s": getup.phase_s, "getup_phase_restart_s": getup.phase_restart_s,
            "target_end_deg": TARGET_END_DEG,
            "getup_rules": {"cf_tau_s": CF_TAU_S, "cf_gate_g": list(CF_GATE_G),
                            "fall": [FALL_TILT_DEG, FALL_HOLD_S, FALL_FAST_TILT_DEG, FALL_FAST_GYRO],
                            "handback": [HANDBACK_TILT_DEG, HANDBACK_GYRO, HANDBACK_RMS_DEG, HANDBACK_STEPS],
                            "restart_tilt_deg": PHASE_RESTART_TILT_DEG, "echo_timeout_s": ECHO_TIMEOUT_S},
            "current_watch": {"amps": WATCH_AMPS, "seconds": WATCH_S}, "reading_watch": {"deg": READING_WATCH_DEG}}
    runner.warm_up(ob, heading, walker_act, decoder, getup)
    # A full collection over the log's objects can stall every thread for tens of ms: none while armed.
    gc.collect()
    gc.freeze()
    gc.disable()
    try:
        start = present(link)
        fell = False
        if a.start_mode == "getup":
            fell = True
            tilt0 = tilt_deg(link.latest.acc_mg)
            meta["start_tilt_deg"] = tilt0
            if tilt0 < FALL_FAST_TILT_DEG:
                msg = f"--start-mode getup with the robot at {tilt0:.0f} deg tilt (not lying down): the get-up starts anyway"
                runner.warnings.append(msg)
                print("WARNING: " + msg, flush=True)
        else:
            runner.stand_phase = True
            try:
                runner.ramp(walk_stand, a.stand_ramp, frm=start)
                runner.hold(walk_stand, a.hold_before)
            except FallDetected:
                fell = True
            finally:
                runner.stand_phase = False
        runner.loop(walker_act, (a.vx, a.vy, a.wz), a.seconds, ob, decoder, heading, getup, start_getup=fell,
                    start_trigger="start mode getup" if a.start_mode == "getup" else None)
        runner.stop_reason = "completed"
        if runner.mode == MODE_GETUP:
            runner.stop_reason = "completed in GETUP (limp)"
            link.disarm()
        else:
            runner.safe_end()
    except StopRun as exc:
        runner.stop_reason = f"stop: {exc}"
        print(runner.stop_reason, flush=True)
        if runner.mode == MODE_GETUP:
            runner.stop_reason += " (in GETUP: limp)"
            link.disarm()
        else:
            runner.safe_end()
    except (Disarm, KeyboardInterrupt) as exc:
        runner.stop_reason = f"disarm: {str(exc) or 'Ctrl-C'}"
        link.disarm()
        print(runner.stop_reason, flush=True)
    except BaseException:
        link.disarm()
        raise
    finally:
        gc.enable()
        if a.log:
            meta.update(missing_ticks=ob.missing, command_initial=runner.command_initial,
                        reading_flags=runner.reading_flags, getups=runner.getups)
            runner.current_watch_reset("run end")
            meta["current_flags"] = runner.current_flags
            runner.save(a.log, meta)
    lat = np.asarray(link.latency, dtype=float) * 1000
    if len(lat):
        print(f"SET_ALL -> STATE echo ms: p50 {np.percentile(lat, 50):.2f} p95 {np.percentile(lat, 95):.2f}; "
              f"missing ticks {ob.missing}; steps {len(runner.steps['tick'])}; {runner.stop_reason}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python3 -m mp4_recovery.deploy.runner", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default="/dev/ttyAMA0")
    ap.add_argument("--baud", type=int, default=921_600)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("check", help="no port: check a policy against this contract and this numpy")
    s.add_argument("--policy", default=str(paths.POLICIES / "walker"))
    s = sub.add_parser("imu", help="servos idle: print the tilt and gravity in each IMU mounting candidate")
    s.add_argument("--seconds", type=float, default=60.0)
    s.add_argument("--log")
    s = sub.add_parser("run", help="the two policies, closed loop")
    s.add_argument("--policy", default=str(paths.POLICIES / "walker"), help="the walking policy's directory")
    s.add_argument("--getup-policy", default=str(paths.POLICIES / "getup"), help="the get-up policy's directory")
    s.add_argument("--vx", type=float, default=0.0, help="m/s, + = forward")
    s.add_argument("--vy", type=float, default=0.0, help="m/s, + = left")
    s.add_argument("--wz", type=float, default=0.0, help="rad/s, + = turn left")
    s.add_argument("--seconds", type=float, default=10.0, help="the policy loop's length, get-ups included")
    s.add_argument("--heading-loop", action=argparse.BooleanOptionalAction, default=True,
                   help="hold the heading with the gyro while no turn is commanded (default on)")
    s.add_argument("--stand-ramp", type=float, default=1.0, help="s of the ramp to the stand (default %(default)s)")
    s.add_argument("--hold-before", type=float, default=0.0, help="s at the stand before the policy (default %(default)s)")
    s.add_argument("--start-mode", choices=("walk", "getup"), default="walk",
                   help="getup: the robot is placed lying down; no ramp, the loop starts in the get-up")
    s.add_argument("--getup-max", type=int, default=10, help="get-ups per run; the next fall goes limp (default %(default)s)")
    s.add_argument("--getup-timeout", type=float, default=20.0,
                   help="s in a get-up before giving up and going limp (default %(default)s)")
    s.add_argument("--getup-phase-restart", type=float, default=1.0, metavar="S",
                   help="the retry: restart the get-up's phase clock S s after it reached its end, if the robot is still "
                        "tilted over 45 deg; 0: never (default %(default)s)")
    s.add_argument("--clear", action="store_true", help="clear a latched fault before arming")
    s.add_argument("--log", help="log base path, without suffix")
    a = ap.parse_args(argv)
    if a.cmd == "check":
        return cmd_check(a)
    from mp4_recovery.deploy.mp4link.link import Link
    link = Link(a.port, a.baud)
    try:
        return cmd_imu(link, a) if a.cmd == "imu" else cmd_run(link, a)
    finally:
        link.close()


if __name__ == "__main__":
    sys.exit(main())
