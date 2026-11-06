"""Command line for the servo link: ``python3 -m mp4_recovery.deploy.mp4link <command>``.

Servo ids 1..12: 1-3 FR, 4-6 FL, 7-9 RR, 10-12 RL (abduction, hip, knee). Positions are the servo's own deci-degrees
(dd, 0.1 deg, 0..3100). Every motion command disarms on exit, and the ESP32 disarms by itself 100 ms after the last frame.

Commands: ``ping`` checks the link; ``config`` writes the servo gains, which every power-off of the servos loses;
``offsets`` reads the calibration; ``dump`` reads every servo parameter; ``watch`` prints the joints while the limp legs are
moved by hand; ``centre`` holds servos at their centre to mount the legs; ``calib`` holds the stand pose and adjusts the
offsets, which the ESP32 stores in flash; ``stream`` plays a recorded get-up on a lifted robot; ``idle`` disarms.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import select
import sys
import time

import numpy as np
import serial

from . import protocol as p
from .link import DEFAULT_BAUD, DEFAULT_PORT, Link, LinkError

NAMES = ["FR abd", "FR hip", "FR knee", "FL abd", "FL hip", "FL knee",
         "RR abd", "RR hip", "RR knee", "RL abd", "RL hip", "RL knee"]
LEGS = ("FR", "FL", "RR", "RL")
ABDUCTION = (1, 4, 7, 10)
# The gains the policies were trained for; the servos' flash holds softer ones.
GAINS = {"control_mode": 1.0, "kp_position": 80.0, "kd_position": 400.0, "kff_current": 0.001,
         "max_pwm_duty_cycle": 0.99}
MAX_STEP_DEG = 15.0            # the most a target moves per send when holding poses
# The firmware's calibration formula, in float32: stand code = 511 + (int)(offset * 10240 / 3100), and the servo target
# is (uint16)(3100 - code * 3100 / 1024) dd.
CENTRE_TICK = 511
OFFSET_MAX_DEG = 90.0          # stand codes 511 +- 297 (214..808), far from 0 and 1023
ONE_TICK_DEG = 3100.0 / 10240.0
# A joint's potentiometer has a dead zone beyond 0 and 3100 dd.
TRACK_END_DEG = 15.0           # a joint measured this close to an end stops a stream
STREAM_END_MARGIN_DEG = 20.0   # a stream with a target this close to an end is refused
WATCH_NEAR_END_DEG = 5.0       # `watch` flags a reading this close to an end
# A stream plays a get-up: every frame in the firmware's get-up current mode, in the get-up policy's band and step limit.
STREAM_ABD_BAND_DEG, STREAM_HIP_KNEE_BAND_DEG = 90.0, 180.0
STREAM_STEP_DEG = 20.0
STREAM_START_C = 45.0          # a stream starts only below this
STREAM_ECHO_S = 0.2            # the firmware must echo the get-up mode within this of the first frame
STREAM_GAP_S = 0.1             # no STATE this long stops a stream
TEMP_HOLD_S = 1.0              # a housing must stay over its limit this long, as the firmware's own cut-out asks


# ---- helpers ------------------------------------------------------------------
class Pacer:
    """Wall-clock pacing: sleep, then spin the last millisecond."""

    def __init__(self, period: float):
        self.period = period
        self.next = time.monotonic()
        self.stamps: list[float] = []

    def wait(self) -> None:
        self.next += self.period
        delay = self.next - time.monotonic()
        if delay > 0.002:
            time.sleep(delay - 0.001)
        while time.monotonic() < self.next:
            pass
        if time.monotonic() - self.next > self.period:   # overrun: restart the grid
            self.next = time.monotonic()
        self.stamps.append(time.monotonic())


class TempWatch:
    """Whether a housing has stayed at ``limit_c`` or above for ``hold_s`` of ticks. A single frame can read double the real
    temperature, so one hot reading must not count."""

    def __init__(self, limit_c: float, hold_s: float = TEMP_HOLD_S):
        self.limit_dc = limit_c * 10.0
        self.hold_ticks = hold_s / 0.005
        self.since = [None] * 12

    def update(self, st) -> float | None:
        """A frame; returns the temperature [degC] of a servo that has stayed hot, or None."""
        tick = int(st.tick)
        for i, t in enumerate(st.temp_dc):
            if t < self.limit_dc:
                self.since[i] = None
                continue
            if self.since[i] is None:
                self.since[i] = tick
            if ((tick - self.since[i]) & 0xFFFFFFFF) >= self.hold_ticks:
                return float(t) / 10.0
        return None


def smoothstep(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


def parse_ids(text: str) -> list[int]:
    ids: list[int] = []
    for part in text.split(","):
        if "-" in part:
            a, b = part.split("-")
            ids += list(range(int(a), int(b) + 1))
        elif part:
            ids.append(int(part))
    if any(i < 1 or i > 12 for i in ids):
        raise SystemExit("servo ids are 1..12")
    return ids


def offset_ticks(offsets_deg) -> np.ndarray:
    off = np.asarray(offsets_deg, dtype=np.float32)
    return CENTRE_TICK + np.trunc(off * np.float32(10240.0) / np.float32(3100.0)).astype(int)


def ticks_to_dd(ticks) -> np.ndarray:
    t = np.asarray(ticks, dtype=np.float32)
    return np.trunc(np.float32(3100.0) - t * np.float32(3100.0) / np.float32(1024.0)).astype(int)


def read_offsets(link: Link) -> tuple[np.ndarray, list[bool]]:
    """The twelve stored offsets [deg] (unset reads 0) and which are stored."""
    vals = [link.get_offset(i) for i in range(1, 13)]
    return np.array([0.0 if v is None else v for v in vals]), [v is not None for v in vals]


def stand_from_robot(link: Link) -> np.ndarray:
    """The stand pose [dd] from the offsets stored on the robot."""
    return ticks_to_dd(offset_ticks(read_offsets(link)[0])).astype(float)


def require_config(link: Link, ids) -> None:
    """Holding a pose on the servos' flash gains is several times softer."""
    for i in ids:
        kff = link.get_param(i, "kff_current")
        if kff < 0.0009:
            raise LinkError("servo %d runs kff %g, not 0.001: run `config` first" % (i, kff))


def mask(ids) -> int:
    m = 0
    for i in ids:
        m |= 1 << (i - 1)
    return m


def present(link: Link) -> np.ndarray:
    """Positions from two consecutive STATE frames with fresh feedback that agree."""
    states = []
    for _ in range(20):
        st = link.wait_state(1.0)
        if st.boards == 0x0F:
            states.append(st)
            if len(states) == 2:
                break
    if len(states) < 2:
        raise LinkError("no fresh feedback from every board")
    a, b = (s.pos_dd.astype(float) for s in states)
    if np.any(a < 100) or np.any(a > 3000) or np.any(np.abs(a - b) > 20):
        raise LinkError("implausible positions (dd): %s / %s" % (a.astype(int).tolist(), b.astype(int).tolist()))
    return b


def clip_band(stand: np.ndarray, target: np.ndarray, band_deg) -> np.ndarray:
    return np.clip(target, stand - np.asarray(band_deg, dtype=float) * 10, stand + np.asarray(band_deg, dtype=float) * 10)


def preflight(link: Link, clear: bool) -> None:
    """Refuse to arm over a latched fault unless ``clear``."""
    st = link.wait_state(1.0)
    latched = st.flags & (p.ST_FAULTS | p.ST_REJECTED)
    if latched:
        what = "%s (servo %d)" % (", ".join(n for b, n in p.FLAG_NAMES.items() if latched & b), st.fault_servo)
        if not clear:
            raise LinkError("fault latched: %s; look at it, then rerun with --clear" % what)
        print("clearing latched fault:", what)


class Mover:
    """Sends targets, arms on the first frame, keeps the step limit, and (once ``stand`` is set) the band about the stand."""

    def __init__(self, link: Link, enable: int, cap: int, clear: bool = False, band_deg=None, getup: bool = False,
                 max_step_deg: float = MAX_STEP_DEG):
        self.link, self.enable, self.cap, self.clear = link, enable, cap, clear
        self.stand: np.ndarray | None = None
        self.band_deg = band_deg
        self.getup = getup
        self.max_step_deg = max_step_deg
        self.last: np.ndarray | None = None
        self.armed = False
        self.first_seq: int | None = None
        self.clipped = 0

    def send(self, target_dd: np.ndarray) -> None:
        t = np.asarray(target_dd, dtype=float).copy()
        if self.stand is not None:
            c = clip_band(self.stand, t, self.band_deg)
            self.clipped += int(np.any(c != t))
            t = c
        if self.last is not None:
            step = self.max_step_deg * 10
            c = np.clip(t, self.last - step, self.last + step)
            self.clipped += int(np.any(c != t))
            t = c
        t = np.clip(np.rint(t), 0, 3100)
        flags = 0 if self.armed else (p.SA_ARM | (p.SA_CLEAR if self.clear else 0))
        seq = self.link.set_all(t.astype(int), self.enable, flags, self.cap, getup=self.getup)
        if not self.armed:
            self.first_seq = seq
        self.armed = True
        self.last = t


def check_flags(link: Link, mover: Mover | None = None) -> None:
    """Raise if the reader stopped, a fault or the watchdog latched, or the ESP32 disarmed after our ARM."""
    link.check()
    st = link.latest
    if st is None:
        return
    if st.flags & (p.ST_FAULTS | p.ST_WATCHDOG | p.ST_REJECTED):
        raise LinkError("ESP32 disarmed: %s (servo %d)" % (", ".join(st.flag_names()), st.fault_servo))
    if mover is not None and mover.first_seq is not None and not st.armed:
        if (st.seq - mover.first_seq) & 0xFFFF < 0x8000:     # a STATE frame after our ARM
            raise LinkError("ESP32 disarmed after our ARM (an IDLE or DISARM from another port?)")


def save_log(link: Link, path: str, pacer: Pacer | None, meta: dict, states: list | None = None) -> None:
    """Every STATE frame recorded (or ``states``) and every SET_ALL sent, with ``meta`` as JSON."""
    states = list(link.record or []) if states is None else states     # the reader thread keeps appending
    sent = list(link.sent)
    arr = lambda f, dt: np.array([f(s) for s in states], dtype=dt)
    np.savez_compressed(
        path,
        t_rx=arr(lambda s: s.t_rx, float), tick=arr(lambda s: s.tick, np.uint32),
        t_us=arr(lambda s: s.t_us, np.uint32), seq=arr(lambda s: s.seq, np.uint16),
        flags=arr(lambda s: s.flags, np.uint16), boards=arr(lambda s: s.boards, np.uint8),
        pos_dd=np.array([s.pos_dd for s in states], dtype=np.uint16).reshape(-1, 12),
        cur_ma=np.array([s.cur_ma for s in states], dtype=np.int16).reshape(-1, 12),
        temp_dc=np.array([s.temp_dc for s in states], dtype=np.int16).reshape(-1, 12),
        acc_mg=np.array([s.acc_mg for s in states], dtype=np.int16).reshape(-1, 3),
        gyr_ddps=np.array([s.gyr_ddps for s in states], dtype=np.int16).reshape(-1, 3),
        imu_t_us=arr(lambda s: s.imu_t_us, np.uint32),
        sent_t=np.array([s[0] for s in sent], dtype=float),
        sent_seq=np.array([s[1] for s in sent], dtype=np.uint16),
        sent_pos_dd=np.array([np.frombuffer(s[2][10:34], dtype="<u2") for s in sent],
                             dtype=np.uint16).reshape(-1, 12),
        sent_enable=np.array([int.from_bytes(s[2][6:8], "little") for s in sent], dtype=np.uint16),
        loop_t=np.array(pacer.stamps if pacer else [], dtype=float),
        latency=np.array(link.latency, dtype=float),
        meta=json.dumps(meta),
    )
    print("log: %s (%d STATE, %d SET_ALL)" % (path, len(states), len(sent)))


def summary(link: Link, pacer: Pacer | None) -> None:
    states = link.record or list(link.states)
    print("STATE frames %d, tick gaps %d, CRC errors %d, bytes skipped %d"
          % (link.n_states, link.tick_gaps, link.parser.crc_errors, link.parser.skipped))
    if len(states) > 1:
        dt = states[-1].t_rx - states[0].t_rx
        print("STATE rate %.1f Hz" % ((len(states) - 1) / dt if dt > 0 else 0))
    if pacer and len(pacer.stamps) > 2:
        per = np.diff(pacer.stamps) * 1000
        print("loop period ms: p50 %.2f  p99 %.2f  max %.2f  (n %d)"
              % (np.percentile(per, 50), np.percentile(per, 99), per.max(), len(per)))
    if link.latency:
        lat = np.array(link.latency) * 1000
        print("SET_ALL -> STATE echo ms: p50 %.2f  p95 %.2f  max %.2f  (n %d)"
              % (np.percentile(lat, 50), np.percentile(lat, 95), lat.max(), len(lat)))
    if states:
        cur = np.abs(np.array([s.cur_ma for s in states], dtype=float))
        tmp = np.array([s.temp_dc for s in states], dtype=float) / 10
        flags = 0
        for s in states:
            flags |= s.flags
        print("max |I| mA:", " ".join("%d" % v for v in cur.max(0)))
        print("max temp C:", " ".join("%.1f" % v for v in tmp.max(0)))
        print("flags seen:", ", ".join(n for b, n in p.FLAG_NAMES.items() if flags & b) or "none")


def ramp_to(mover: Mover, pacer: Pacer, link: Link, frm: np.ndarray, to: np.ndarray, seconds: float) -> None:
    n = max(1, int(seconds / pacer.period))
    for k in range(n):
        mover.send(frm + (to - frm) * smoothstep((k + 1) / n))
        pacer.wait()
        check_flags(link, mover)


def hold(mover: Mover, pacer: Pacer, link: Link, target: np.ndarray, seconds: float) -> None:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        mover.send(target)
        pacer.wait()
        check_flags(link, mover)


def at_track_end(pos_dd):
    """Whether a measured position (or each of an array) lies within TRACK_END_DEG of 0 or 3100 dd."""
    pos = np.asarray(pos_dd, dtype=float)
    return (pos < TRACK_END_DEG * 10) | (pos > 3100 - TRACK_END_DEG * 10)


def _stdin_fd():
    """stdin's descriptor when it is a terminal (Enter stops), else None."""
    try:
        if sys.stdin is not None and sys.stdin.isatty():
            return sys.stdin.fileno()
    except (AttributeError, ValueError, OSError):
        pass
    return None


def _enter_pressed(fd) -> bool:
    if fd is None:
        return False
    if select.select([fd], [], [], 0)[0]:
        os.read(fd, 1024)
        return True
    return False


# ---- link and servo checks ------------------------------------------------------
def cmd_ping(link: Link, a) -> None:
    link.record = []
    pacer = Pacer(0.02)
    end = time.monotonic() + a.seconds
    pos = [1550] * 12
    while time.monotonic() < end:
        link.set_all(pos, enable=0, flags=0)       # disarmed: only measures the echo
        pacer.wait()
    summary(link, pacer)
    st = link.latest
    if st is None:
        raise LinkError("no STATE received")
    acc = np.array([s.acc_mg for s in link.record], dtype=float) / 1000
    gyr = np.array([s.gyr_ddps for s in link.record], dtype=float) / 10
    print("IMU %s: acc mean %s g (|a| %.3f), gyro mean %s deg/s, std %s"
          % ("ok" if st.flags & p.ST_IMU_OK else "ABSENT", np.round(acc.mean(0), 3),
             np.linalg.norm(acc.mean(0)), np.round(gyr.mean(0), 2), np.round(gyr.std(0), 2)))
    boards = np.array([[s.boards >> b & 1 for b in range(4)] for s in link.record])
    print("fresh feedback per board (CN4, CN5, CN3, CN6): %s" % np.round(boards.mean(0), 3))
    for i in range(12):
        print("  %2d %-9s pos %4d dd (%6.1f deg)  I %5d mA  T %5.1f C"
              % (i + 1, NAMES[i], st.pos_dd[i], st.pos_dd[i] / 10, st.cur_ma[i], st.temp_dc[i] / 10))
    if a.log:
        save_log(link, a.log, pacer, {"cmd": "ping", "seconds": a.seconds})


def cmd_dump(link: Link, a) -> None:
    out = {}
    print("%-4s %-9s " % ("id", "") + " ".join("%10s" % n[:10] for n in p.PARAMS) + "  temp_C  supply_mA")
    for i in range(1, 13):
        row = {}
        for name in p.PARAMS:
            try:
                row[name] = link.get_param(i, name)
            except LinkError:
                row[name] = None
        for key, lid in (("temperature_c", p.LIVE_TEMPERATURE_C), ("supply_current_ma", p.LIVE_SUPPLY_CURRENT_MA)):
            try:
                row[key] = link.get_live(i, lid)
            except LinkError:
                row[key] = None
        out[i] = row
        fmt = lambda v: "%10s" % ("-" if v is None else "%.6g" % v)
        print("%-4d %-9s " % (i, NAMES[i - 1]) + " ".join(fmt(row[n]) for n in p.PARAMS)
              + "  %6s  %9s" % (fmt(row["temperature_c"]).strip(), fmt(row["supply_current_ma"]).strip()))
    if a.out:
        with open(a.out, "w") as f:
            json.dump({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "servos": out}, f, indent=1)
        print("saved", a.out)


def cmd_config(link: Link, a) -> None:
    for i in range(1, 13):
        got = []
        for name, value in {**GAINS, "ki_position": 0.0, "kp_current": 0.0}.items():
            rb = link.set_param(i, name, value)
            if abs(rb - value) > 1e-6 + 1e-4 * abs(value):
                raise LinkError("servo %d %s: wrote %g, read back %g" % (i, name, value, rb))
            got.append("%s=%g" % (name, rb))
        print("%2d %-9s %s" % (i, NAMES[i - 1], " ".join(got)))
    print("written to RAM (not saved; a power cycle restores each servo's flash)")


def cmd_idle(link: Link, a) -> None:
    link.disarm()
    st = link.wait_state(0.5)
    print("flags:", ", ".join(st.flag_names()) or "none")


# ---- playing a get-up on a lifted robot --------------------------------------------
def stream_band() -> np.ndarray:
    return np.array([STREAM_ABD_BAND_DEG if i + 1 in ABDUCTION else STREAM_HIP_KNEE_BAND_DEG for i in range(12)])


def check_track_ends(stand: np.ndarray, delta: np.ndarray, band) -> None:
    """Refuse a stream whose targets or stand come within STREAM_END_MARGIN_DEG of a track end."""
    tgt = clip_band(stand, stand + delta, band)
    lo_dd, hi_dd = STREAM_END_MARGIN_DEG * 10, 3100 - STREAM_END_MARGIN_DEG * 10
    for i in range(12):
        dmin, dmax = min(float(tgt[:, i].min()), stand[i]), max(float(tgt[:, i].max()), stand[i])
        if lo_dd <= dmin and dmax <= hi_dd:
            continue
        bad = dmin if dmin < lo_dd else dmax
        raise SystemExit("stream: servo %d (%s) reaches %d dd (%+.1f deg from the stand %d dd), within %g deg of "
                         "the track end (0..3100 dd)"
                         % (i + 1, NAMES[i], round(bad), (bad - stand[i]) / 10, stand[i], STREAM_END_MARGIN_DEG))


def cmd_stream(link: Link, a) -> None:
    """Ramp to the stream's first pose, hold 1 s, play the file, ramp back to the stand, hold 1 s, disarm. Any stop
    returns to the stand the same way."""
    band = stream_band()
    stand = stand_from_robot(link)
    data = np.load(a.file)
    delta = np.asarray(data["delta_dd"], dtype=float)
    period = float(data["period_s"])
    if delta.ndim != 2 or delta.shape[1] != 12 or len(delta) == 0 or not np.all(np.isfinite(delta)):
        raise SystemExit("%s: delta_dd must be N x 12 finite values" % a.file)
    check_track_ends(stand, delta, band)
    link.record = []
    start = present(link)
    preflight(link, a.clear)
    t0 = link.latest.temp_dc.astype(float) / 10
    if t0.max() >= min(a.stop_c, STREAM_START_C):
        raise LinkError("a housing is at %.1f C (servo %d): let it cool before a stream"
                        % (t0.max(), int(np.argmax(t0)) + 1))
    fd = _stdin_fd()
    mover = Mover(link, 0x0FFF, a.cap, a.clear, band_deg=band, getup=True, max_step_deg=STREAM_STEP_DEG)
    pacer = Pacer(0.02)
    first = clip_band(stand, stand + delta[0], band)
    reason, t_play = None, [None, None]
    t_first = time.monotonic()
    idx, end_hit = 0, None
    hot = TempWatch(a.stop_c)

    def stop_reason() -> str | None:
        st = link.latest
        if time.monotonic() - st.t_rx > STREAM_GAP_S:
            return "state_gap"
        if not st.getup and time.monotonic() - t_first > STREAM_ECHO_S:
            return "no_getup_echo"
        if hot.update(st) is not None:
            return "temperature"
        if _enter_pressed(fd):
            return "enter"
        return None

    def track_end() -> str | None:
        """Every STATE frame since the last call, during the playback only."""
        nonlocal idx, end_hit
        new = link.record[idx:]
        idx += len(new)
        for s in new:
            if s.boards != 0x0F:              # a board without fresh feedback: its positions may be a boot's zeros
                continue
            hit = np.flatnonzero(at_track_end(s.pos_dd))
            if len(hit):
                end_hit = (int(hit[0]) + 1, int(s.pos_dd[hit[0]]))
                return "track_end"
        return None

    try:
        ramp_to(mover, pacer, link, start, first, 2.0)     # from wherever the legs are: only its end is clipped
        mover.stand = stand
        end = time.monotonic() + 1.0
        while reason is None and time.monotonic() < end:
            mover.send(first)
            pacer.wait()
            check_flags(link, mover)
            reason = stop_reason()
        pacer.period = period
        t_play[0] = time.monotonic()
        idx = len(link.record)
        for row in delta:
            if reason is not None:
                break
            mover.send(stand + row)
            pacer.wait()
            check_flags(link, mover)
            reason = stop_reason() or track_end()
        t_play[1] = time.monotonic()
        pacer.period = 0.02
        ramp_to(mover, pacer, link, mover.last, stand, 1.0)
        hold(mover, pacer, link, stand, 1.0)
    finally:
        link.disarm()
        summary(link, pacer)
        states = list(link.record or [])
        play = [s for s in states if t_play[0] is not None and t_play[0] <= s.t_rx <= (t_play[1] or math.inf)]
        if play:
            cur = np.abs(np.array([s.cur_ma for s in play], dtype=float))
            print("playback p99 |I| mA:", " ".join("%d" % v for v in np.percentile(cur, 99, axis=0)))
        if reason == "no_getup_echo":
            print("stream: no get-up mode echo (a firmware without it): stopped and returned", flush=True)
        if end_hit is not None:
            print("stream: servo %d (%s) measured at %d dd, within %g deg of a track end: stopped and returned"
                  % (end_hit[0], NAMES[end_hit[0] - 1], end_hit[1], TRACK_END_DEG), flush=True)
        echo = sum(1 for s in play if s.getup) / len(play) if play else None
        print("stream %s; targets clipped on %d sends; get-up mode echoed on %s of playback frames"
              % (reason or "completed", mover.clipped, "-" if echo is None else "%.0f %%" % (100 * echo)), flush=True)
        if a.log:
            save_log(link, a.log, pacer,
                     {"cmd": "stream", "file": a.file, "period_s": period, "stand": stand.tolist(), "cap": a.cap,
                      "band_deg": band.tolist(), "stop_reason": reason or "completed", "clipped_sends": mover.clipped,
                      "t_play": t_play, "stop_c": a.stop_c, "getup_echo_share": echo,
                      "max_step_deg": mover.max_step_deg, "track_end_deg": TRACK_END_DEG, "track_end_servo": end_hit},
                     states=states)


# ---- watching the limp legs ----------------------------------------------------
def _fmt(v, f: str) -> str:
    return "-" if v is None else f % v


def cmd_watch(link: Link, a) -> None:
    """Prints the chosen joints once a second as degrees from the stand, with their running min/max."""
    ids = list(range(1, 13)) if a.leg is None else [LEGS.index(a.leg) * 3 + n for n in (1, 2, 3)]
    ref = stand_from_robot(link)
    log_path = a.log or "watch_%s.npz" % time.strftime("%Y%m%d_%H%M%S")
    fd = _stdin_fd()
    link.record = []
    pacer = Pacer(0.02)
    pos = [1550] * 12
    lo = np.full(12, np.inf)
    hi = np.full(12, -np.inf)
    near = WATCH_NEAR_END_DEG * 10
    print("watch %s (servos idle) for %g s, degrees from the stand; Enter ends, Ctrl-C ends and saves"
          % (a.leg or "all legs", a.seconds), flush=True)
    reason, idx = "time", 0
    t0 = time.monotonic()
    last_print = t0
    try:
        while time.monotonic() - t0 < a.seconds:
            link.set_all(pos, enable=0, flags=0)
            pacer.wait()
            link.check()
            new = link.record[idx:]
            idx += len(new)
            if new:
                arr = np.array([s.pos_dd for s in new], dtype=float).reshape(-1, 12)
                lo, hi = np.minimum(lo, arr.min(0)), np.maximum(hi, arr.max(0))
            if time.monotonic() - last_print >= 1.0 and link.latest is not None:
                last_print = time.monotonic()
                st = link.latest
                print("t %.0f s" % (last_print - t0), flush=True)
                for i in ids:
                    q = float(st.pos_dd[i - 1])
                    print("  %2d %-9s %+7.1f deg  min %+7.1f  max %+7.1f%s"
                          % (i, NAMES[i - 1], (q - ref[i - 1]) / 10, (lo[i - 1] - ref[i - 1]) / 10,
                             (hi[i - 1] - ref[i - 1]) / 10,
                             "  near track end (%d dd)" % q if q < near or q > 3100 - near else ""), flush=True)
            if _enter_pressed(fd):
                reason = "enter"
                break
    except KeyboardInterrupt:
        reason = "ctrl-c"
    finally:
        seen = np.isfinite(lo)
        mn = [float((lo[i] - ref[i]) / 10) if seen[i] else None for i in range(12)]
        mx = [float((hi[i] - ref[i]) / 10) if seen[i] else None for i in range(12)]
        print("watch ended (%s); min / max deg from the stand:" % reason)
        for i in ids:
            print("  %2d %-9s min %s  max %s" % (i, NAMES[i - 1], _fmt(mn[i - 1], "%+7.1f"), _fmt(mx[i - 1], "%+7.1f")))
        save_log(link, log_path, pacer, {"cmd": "watch", "seconds": a.seconds, "leg": a.leg, "stand": ref.tolist(),
                                         "stop_reason": reason, "min_deg": mn, "max_deg": mx,
                                         "min_dd": [float(v) if seen[i] else None for i, v in enumerate(lo)],
                                         "max_dd": [float(v) if seen[i] else None for i, v in enumerate(hi)],
                                         "near_end_deg": WATCH_NEAR_END_DEG,
                                         "time": time.strftime("%Y-%m-%d %H:%M:%S")})


# ---- calibration --------------------------------------------------------------
def cmd_offsets(link: Link, a) -> None:
    offs, stored = read_offsets(link)
    ticks = offset_ticks(offs)
    dd = ticks_to_dd(ticks)
    for i in range(12):
        off = ("%+6.1f deg" % offs[i]) if stored[i] else "     unset"
        print("%2d %-9s offset %s   stand code %4d   %4d dd" % (i + 1, NAMES[i], off, ticks[i], dd[i]))
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"stand_dd": dd.tolist(), "stand_ticks": ticks.tolist(), "offsets_deg": offs.tolist(),
                       "offsets_stored": stored,
                       "source": "the robot's stored offsets, read %s; code = 511 + (int)(offset*10240/3100), "
                                 "dd = (uint16)(3100 - code*3100/1024)" % time.strftime("%Y-%m-%d %H:%M:%S")},
                      f, indent=1)
        print("written:", a.json)


def _status(link: Link, ids) -> str:
    st = link.latest
    k = np.array(ids) - 1
    return "max |I| %d mA, max housing %.1f C" % (np.abs(st.cur_ma[k]).max(), st.temp_dc[k].max() / 10.0)


def cmd_centre(link: Link, a) -> None:
    ids = parse_ids(a.ids)
    require_config(link, ids)
    start = present(link)
    preflight(link, a.clear)
    target = start.copy()
    target[np.array(ids) - 1] = ticks_to_dd([CENTRE_TICK])[0]
    mover = Mover(link, mask(ids), a.cap, a.clear)
    pacer = Pacer(0.02)
    print("servos %s to the centre: code %d = %d dd, offsets ignored. Ctrl-C ends (all idle)."
          % (a.ids, CENTRE_TICK, ticks_to_dd([CENTRE_TICK])[0]))
    try:
        ramp_to(mover, pacer, link, start, target, 2.0)
        end, last = time.monotonic() + a.seconds, 0.0
        while time.monotonic() < end:
            mover.send(target)
            pacer.wait()
            check_flags(link, mover)
            if time.monotonic() - last > 5.0:
                last = time.monotonic()
                print("  holding, %s, %.0f s left" % (_status(link, ids), end - last))
    except KeyboardInterrupt:
        pass
    finally:
        link.disarm()
    print("idle")


def _keys(fd):
    """Characters typed since the last call (non-blocking)."""
    out = ""
    while select.select([fd], [], [], 0)[0]:
        out += sys.stdin.read(1)
    return out


def cmd_calib(link: Link, a) -> None:
    offs, stored = read_offsets(link)
    if a.zero:
        offs[np.array(parse_ids(a.zero)) - 1] = 0.0
    start_offs = offs.copy()
    for item in a.set or []:
        sid, deg = item.split("=")
        offs[int(sid) - 1] = float(deg)
    if np.any(np.abs(offs) > OFFSET_MAX_DEG):
        raise SystemExit("offsets are limited to +-%g deg" % OFFSET_MAX_DEG)
    interactive = a.seconds is None
    if interactive and not sys.stdin.isatty():
        raise SystemExit("calib needs a terminal (ssh -t), or --seconds for a timed hold")
    if interactive:
        import termios
        import tty
    require_config(link, range(1, 13))
    pose = lambda: ticks_to_dd(offset_ticks(offs)).astype(float)
    start = present(link)
    preflight(link, a.clear)
    mover = Mover(link, 0x0FFF, a.cap, a.clear)
    pacer = Pacer(0.02)
    save = a.save
    fd = sys.stdin.fileno() if interactive else None
    old = termios.tcgetattr(fd) if interactive else None
    try:
        ramp_to(mover, pacer, link, start, pose(), 2.0)
        if not interactive:
            hold(mover, pacer, link, pose(), a.seconds)
        else:
            tty.setcbreak(fd)
            sel, step = 0, a.step
            print("Stand pose held (code 511 + offset). Keys: ] / [ next / previous servo, + / - step,\n"
                  "f fine (one code step, %.2f deg) / coarse, r restore this servo, s save and quit,\n"
                  "q quit without saving (Ctrl-C too)." % ONE_TICK_DEG)

            def show():
                print("  servo %2d %-9s offset %+6.2f deg   stand code %4d   step %.2f deg   [%s]"
                      % (sel + 1, NAMES[sel], offs[sel], offset_ticks(offs)[sel], step, _status(link, range(1, 13))))
            show()
            done = False
            while not done:
                for c in _keys(fd):
                    if c in "]n":
                        sel = (sel + 1) % 12
                    elif c in "[p":
                        sel = (sel - 1) % 12
                    elif c in "+=":
                        offs[sel] = min(offs[sel] + step, OFFSET_MAX_DEG)
                    elif c in "-_":
                        offs[sel] = max(offs[sel] - step, -OFFSET_MAX_DEG)
                    elif c == "f":
                        step = ONE_TICK_DEG if step != ONE_TICK_DEG else a.step
                    elif c == "r":
                        offs[sel] = start_offs[sel]
                    elif c == "s":
                        save, done = True, True
                    elif c == "q":
                        save, done = False, True
                    else:
                        continue
                    show()
                    if done:
                        break
                mover.send(pose())
                pacer.wait()
                check_flags(link, mover)
    except KeyboardInterrupt:
        save = False if interactive else save
    finally:
        if interactive:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        link.disarm()
    for i in range(12):
        flag = "" if offs[i] == start_offs[i] else "   (was %+.2f)" % start_offs[i]
        print("%2d %-9s offset %+6.2f deg%s" % (i + 1, NAMES[i], offs[i], flag))
    if not save:
        print("not saved: the robot keeps the previous offsets")
        return
    t_end = time.monotonic() + 1.0                       # offsets are refused while armed
    while link.wait_state(0.5).armed and time.monotonic() < t_end:
        pass
    for i in range(12):
        link.set_offset(i + 1, float(offs[i]))
    n = link.save_offsets()
    back, _ = read_offsets(link)
    if not np.allclose(back, offs.astype(np.float32), atol=1e-4):
        raise LinkError("read-back differs: %s" % back.tolist())
    print("saved %d offsets and read them back" % n)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python3 -m mp4_recovery.deploy.mp4link", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default=DEFAULT_PORT, help="serial port (default %(default)s)")
    ap.add_argument("--baud", type=int, default=DEFAULT_BAUD, help="default %(default)s")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("ping", help="link check: STATE rate, CRC, IMU, latency (servos stay idle)")
    s.add_argument("--seconds", type=float, default=10.0, help="default %(default)s")
    s.add_argument("--log", help="save the STATE stream and the frames sent to this .npz")
    s = sub.add_parser("watch", help="servos idle: print where the joints are while the legs are moved by hand")
    s.add_argument("--seconds", type=float, default=60.0, help="default %(default)s")
    s.add_argument("--log", help="the .npz log (default watch_<time>.npz here)")
    s.add_argument("--leg", choices=LEGS, help="print only this leg's joints (default all)")
    s = sub.add_parser("dump", help="read every servo parameter (read-only)")
    s.add_argument("--out", help="also write them to this JSON file")
    sub.add_parser("config", help="write the gains to the servos' RAM and read them back")
    s = sub.add_parser("stream", help="play a get-up on a lifted robot, in the firmware's get-up current mode")
    s.add_argument("file", help="npz with delta_dd (N x 12, targets minus the stand) and period_s")
    s.add_argument("--cap", type=int, default=600,
                   help="frame cap mA; in control mode 1 a duty cap of kff x cap (default %(default)s: duty 0.6)")
    s.add_argument("--stop-c", type=float, default=55.0, help="any housing at this degC for 1 s stops (default %(default)s)")
    s.add_argument("--clear", action="store_true", help="clear a latched fault before arming")
    s.add_argument("--log", help="the .npz log")
    s = sub.add_parser("offsets", help="read the calibration offsets stored on the robot")
    s.add_argument("--json", help="also write them, with the stand they give, to this file")
    s = sub.add_parser("centre", help="hold servos at the centre (code 511, offsets ignored) to mount the legs")
    s.add_argument("--ids", default="1-12")
    s.add_argument("--cap", type=int, default=300, help="frame cap mA (default %(default)s)")
    s.add_argument("--seconds", type=float, default=600.0)
    s.add_argument("--clear", action="store_true", help="clear a latched fault before arming")
    s = sub.add_parser("calib", help="hold the stand pose, adjust the offsets, save them on the robot")
    s.add_argument("--step", type=float, default=1.0, help="deg per key press (f: one code step)")
    s.add_argument("--cap", type=int, default=300, help="frame cap mA (default %(default)s)")
    s.add_argument("--zero", metavar="IDS", help="start these servos from offset 0 (after `centre` and mounting the leg)")
    s.add_argument("--set", action="append", metavar="ID=DEG", help="start from this offset (repeatable)")
    s.add_argument("--seconds", type=float, help="no keys: hold this long, then stop")
    s.add_argument("--save", action="store_true", help="with --seconds: save the offsets after")
    s.add_argument("--clear", action="store_true", help="clear a latched fault before arming")
    sub.add_parser("idle", help="disarm: every servo idle")

    a = ap.parse_args(argv)
    try:
        link = Link(a.port, a.baud)
    except serial.SerialException as exc:
        print("error: cannot open %s (%s)" % (a.port, exc), file=sys.stderr)
        return 1
    try:
        {"ping": cmd_ping, "dump": cmd_dump, "config": cmd_config, "stream": cmd_stream, "idle": cmd_idle,
         "offsets": cmd_offsets, "centre": cmd_centre, "calib": cmd_calib, "watch": cmd_watch}[a.cmd](link, a)
    except LinkError as exc:
        print("error:", exc, file=sys.stderr)
        return 1
    finally:
        link.close()
    return 0
