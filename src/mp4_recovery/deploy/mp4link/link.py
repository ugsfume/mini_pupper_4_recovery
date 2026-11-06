"""The serial link to the ESP32: a reader thread, requests, SET_ALL."""

from __future__ import annotations

import collections
import queue
import threading
import time

from . import protocol as p

DEFAULT_PORT = "/dev/ttyAMA0"
DEFAULT_BAUD = 921_600


class LinkError(IOError):
    pass


class Link:
    """One serial port to the ESP32. A reader thread parses every frame: ``STATE`` frames update ``latest`` and the
    ``states`` ring (and ``record`` when it is a list); responses answer :meth:`request`. ``ser`` may be any object with
    ``read``, ``write`` and ``in_waiting``."""

    def __init__(self, port: str = DEFAULT_PORT, baud: int = DEFAULT_BAUD,
                 ser=None, keep: int = 2000):
        if ser is None:
            import serial
            # DTR and RTS released: on a USB serial adapter toggling them resets the ESP32.
            ser = serial.Serial()
            ser.port, ser.baudrate, ser.timeout, ser.exclusive = port, baud, 0.01, True
            ser.dtr = False
            ser.rts = False
            ser.open()
            ser.reset_input_buffer()
        self.ser = ser
        self.parser = p.Parser()
        self.latest: p.State | None = None
        self.states: collections.deque = collections.deque(maxlen=keep)
        self.record: list | None = None
        self.n_states = 0
        self.tick_gaps = 0          # STATE ticks missed between consecutive frames
        self.latency: collections.deque = collections.deque(maxlen=20000)  # SET_ALL -> STATE echo [s]
        self.error: BaseException | None = None   # what stopped the reader thread
        self.sent: list[tuple[float, int, bytes]] = []   # (time, seq, frame) when recording
        self._seq = 0
        self._sent_at: dict[int, float] = {}
        self._responses: queue.Queue = queue.Queue()
        self._cond = threading.Condition()
        self._write_lock = threading.Lock()
        self._req_lock = threading.Lock()     # one request in flight
        self._run = True
        self._thread = threading.Thread(target=self._reader, name="mp4link-rx", daemon=True)
        self._thread.start()

    # ---- receive ---------------------------------------------------------
    def _reader(self) -> None:
        try:
            while self._run:
                data = self.ser.read(max(1, getattr(self.ser, "in_waiting", 0) or 1))
                if not data:
                    continue
                t = time.monotonic()
                for magic, frame in self.parser.feed(data):
                    if magic == p.MAGIC_STATE:
                        self._on_state(p.parse_state(frame, t))
                    elif magic == p.MAGIC_RSP:
                        self._responses.put(p.parse_response(frame))
        except Exception as exc:     # surfaced by wait_state / check
            if self._run:
                self.error = exc
                with self._cond:
                    self._cond.notify_all()

    def check(self) -> None:
        """Raise if the reader thread has stopped."""
        if self.error is not None:
            raise LinkError("serial reader stopped: %r" % (self.error,))

    def _on_state(self, st: p.State) -> None:
        with self._cond:
            if self.latest is not None:
                gap = (st.tick - self.latest.tick - 1) & 0xFFFFFFFF
                if gap < 1000:
                    self.tick_gaps += gap
            self.latest = st
            self.n_states += 1
            self.states.append(st)
            if self.record is not None:
                self.record.append(st)
            sent = self._sent_at.pop(st.seq, None)
            if sent is not None:
                self.latency.append(st.t_rx - sent)
            self._cond.notify_all()

    def wait_state(self, timeout: float = 0.5, after: p.State | None = None) -> p.State:
        """The next STATE newer than ``after`` (default: the current one)."""
        deadline = time.monotonic() + timeout
        with self._cond:
            ref = after if after is not None else self.latest
            while self.latest is None or (ref is not None and self.latest.tick == ref.tick):
                self.check()
                left = deadline - time.monotonic()
                if left <= 0:
                    raise LinkError("no STATE from the ESP32 (wiring, baud, firmware?)")
                self._cond.wait(left)
            return self.latest

    # ---- send ------------------------------------------------------------
    def _write(self, frame: bytes) -> None:
        with self._write_lock:
            self.ser.write(frame)

    def _transact(self, cmd: int, servo: int, frame: bytes, timeouts) -> p.Response | None:
        """Send ``frame`` once per timeout until a response to (cmd, servo)
        arrives. Caller holds ``_req_lock``, so every response in the queue
        belongs to this exchange or to an earlier one that timed out; the
        latter are dropped by the cmd/servo match or the drain before each send."""
        for timeout in timeouts:
            while not self._responses.empty():
                self._responses.get_nowait()
            self._write(frame)
            deadline = time.monotonic() + timeout
            while True:
                try:
                    rsp = self._responses.get(timeout=max(0.0, deadline - time.monotonic()))
                except queue.Empty:
                    break
                if rsp.cmd == cmd and rsp.servo == servo:
                    return rsp
        return None

    def request(self, cmd: int, servo: int = 0, arg1: int = 0, arg2: int = 0,
                pad: int = 0, timeout: float = 0.3, check: bool = True,
                retries: int = 2) -> p.Response:
        """One per-servo request, sent up to ``retries`` times on a timeout."""
        frame = p.pack_request(cmd, servo, arg1, arg2, pad)
        timeouts = [timeout] * max(1, retries)
        with self._req_lock:
            rsp = self._transact(cmd, servo, frame, timeouts)
        if rsp is None:
            raise LinkError("no response to cmd %d servo %d" % (cmd, servo))
        if check and not rsp.ok:
            raise LinkError("servo %d cmd %d: %s" % (servo, cmd, p.STATUS.get(rsp.status, rsp.status)))
        return rsp

    def get_param(self, servo: int, name_or_id) -> float:
        pid = p.PARAM_ID[name_or_id] if isinstance(name_or_id, str) else int(name_or_id)
        return self.request(p.CMD_GET_PARAM, servo, pid).val

    def set_param(self, servo: int, name_or_id, value: float) -> float:
        """Write a parameter to the servo's RAM; returns the read-back."""
        pid = p.PARAM_ID[name_or_id] if isinstance(name_or_id, str) else int(name_or_id)
        a1, a2 = p.float_args(value)
        return self.request(p.CMD_SET_PARAM, servo, a1, a2, pad=pid).val

    def get_live(self, servo: int, live_id: int) -> float:
        return self.request(p.CMD_GET_LIVE, servo, live_id).val

    # ---- calibration offsets (stored on the ESP32, degrees) ---------------
    def get_offset(self, servo: int) -> float | None:
        """The stored offset, or None when the servo has none."""
        rsp = self.request(p.CMD_GET_OFFSET, servo, check=False)
        if rsp.status == p.STATUS_UNSET:
            return None
        if not rsp.ok:
            raise LinkError("servo %d offset: %s" % (servo, p.STATUS.get(rsp.status, rsp.status)))
        return rsp.val

    def set_offset(self, servo: int, value: float) -> float:
        """Into the ESP32's RAM; save_offsets() stores them."""
        a1, a2 = p.float_args(value)
        return self.request(p.CMD_SET_OFFSET, servo, a1, a2).val

    def save_offsets(self) -> int:
        """Store every set offset; returns how many."""
        return int(round(self.request(p.CMD_SAVE_OFFSETS, 0).val))

    def set_all(self, pos_dd, enable: int = 0x0FFF, flags: int = 0, cur_cap_ma: int = 1500,
                getup: bool = False) -> int:
        """Send twelve targets; returns the frame's seq. ``getup`` adds SA_GETUP (the firmware's get-up mode: current guard off,
        for this frame and until a frame without it); ``State.getup`` reports whether the firmware applied it."""
        self._seq = (self._seq + 1) & 0xFFFF
        if getup:
            flags |= p.SA_GETUP
        frame = p.pack_set_all(self._seq, pos_dd, enable, flags, cur_cap_ma)
        t = time.monotonic()
        with self._cond:
            self._sent_at[self._seq] = t
            if len(self._sent_at) > 256:
                self._sent_at.pop(next(iter(self._sent_at)))
        self._write(frame)
        if self.record is not None:
            self.sent.append((t, self._seq, frame))
        return self._seq

    def disarm(self) -> None:
        """Every servo IDLE: a DISARM frame, then an IDLE request as a second path."""
        pos = self.latest.pos_dd if self.latest is not None else [1550] * 12
        self.set_all(pos, enable=0, flags=p.SA_DISARM, cur_cap_ma=0)
        try:
            self.request(p.CMD_IDLE, 1, check=False, timeout=0.2)
        except LinkError:
            pass

    def close(self) -> None:
        self._run = False
        self._thread.join(timeout=1.0)
        try:
            self.ser.close()
        except Exception:
            pass
