"""The frames of the Raspberry Pi <-> ESP32 link: requests and responses, SET_ALL (targets) and STATE (200 Hz feedback)."""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

MAGIC_REQ = 0x42
MAGIC_RSP = 0x62
MAGIC_SET_ALL = 0x43
MAGIC_STATE = 0x63
VERSION = 1

# per-servo commands
CMD_PING = 0
CMD_SET_POS = 1
CMD_IDLE = 2
CMD_TORQUE = 3
CMD_SET_KP = 4
CMD_SET_KD = 5
CMD_GET_PARAM = 6
CMD_GET_LIVE = 7
CMD_SET_SCALE = 8
CMD_RAW_REPLY = 9
CMD_SET_PARAM = 10
CMD_GET_STATUS = 11
CMD_GET_OFFSET = 12      # calibration offsets, degrees
CMD_SET_OFFSET = 13      # RAM until CMD_SAVE_OFFSETS
CMD_SAVE_OFFSETS = 14
STATUS_UNSET = 7

STATUS = {0: "ok", 1: "bad servo", 2: "bad argument", 3: "board did not answer",
          5: "unknown command", 6: "refused (armed, or a motion command)",
          7: "no offset stored", 8: "offset store unavailable or a write failed"}

# SET_ALL flags
SA_ARM = 1 << 0
SA_DISARM = 1 << 1
SA_CLEAR = 1 << 2
SA_GETUP = 1 << 3         # get-up mode: the current guard uses its get-up limits

# STATE flags
ST_ARMED = 1 << 0
ST_WATCHDOG = 1 << 1
ST_FAULT_CUR = 1 << 2
ST_FAULT_TEMP = 1 << 3
ST_FAULT_BOARD = 1 << 4
ST_IMU_OK = 1 << 5
ST_BOARD_MISSING = 1 << 6
ST_REJECTED = 1 << 7
ST_HOST_USB = 1 << 8
ST_GETUP = 1 << 9         # get-up mode: current guard off
ST_FAULTS = ST_FAULT_CUR | ST_FAULT_TEMP | ST_FAULT_BOARD

FLAG_NAMES = {ST_ARMED: "armed", ST_WATCHDOG: "watchdog", ST_FAULT_CUR: "fault-current",
              ST_FAULT_TEMP: "fault-temperature", ST_FAULT_BOARD: "fault-board",
              ST_IMU_OK: "imu", ST_BOARD_MISSING: "board-missing", ST_REJECTED: "rejected",
              ST_HOST_USB: "usb", ST_GETUP: "getup"}

# servo parameter ids
PARAMS = ("reverse_position_sensor", "min_position_adc", "max_position_adc",
          "range_position_deg", "reverse_motor", "kp_position", "kd_position", "kp_current",
          "kff_current", "max_pwm_duty_cycle", "ki_position", "control_mode")
PARAM_ID = {name: i for i, name in enumerate(PARAMS)}
LIVE_TEMPERATURE_C = 12
LIVE_SUPPLY_CURRENT_MA = 13

REQ = struct.Struct("<BBBBHH")             # + crc = 10
RSP = struct.Struct("<BBBBHhf")            # + crc = 14
SET_ALL = struct.Struct("<BBHHHH12H")      # + crc = 36
STATE = struct.Struct("<BBHIIHBB12H12h12h3h3hI")  # + crc = 106
REQ_LEN, RSP_LEN = REQ.size + 2, RSP.size + 2
SET_ALL_LEN, STATE_LEN = SET_ALL.size + 2, STATE.size + 2
assert (REQ_LEN, RSP_LEN, SET_ALL_LEN, STATE_LEN) == (10, 14, 36, 106)


def crc16(data: bytes) -> int:
    """CRC16-CCITT-FALSE (init 0xFFFF, poly 0x1021, MSB-first)."""
    crc = 0xFFFF
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


def _seal(body: bytes) -> bytes:
    return body + struct.pack("<H", crc16(body))


def pack_request(cmd: int, servo: int = 0, arg1: int = 0, arg2: int = 0, pad: int = 0) -> bytes:
    return _seal(REQ.pack(MAGIC_REQ, cmd, servo, pad & 0xFF, arg1 & 0xFFFF, arg2 & 0xFFFF))


def float_args(value: float) -> tuple[int, int]:
    """Split a float32 into the (arg1, arg2) halves of a request."""
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    return bits & 0xFFFF, bits >> 16


def pack_set_all(seq: int, pos_dd, enable: int = 0, flags: int = 0, cur_cap_ma: int = 1500) -> bytes:
    pos = [int(p) for p in pos_dd]
    if len(pos) != 12:
        raise ValueError("12 positions expected")
    if any(p < 0 or p > 0xFFFF for p in pos):
        raise ValueError("position out of range")
    return _seal(SET_ALL.pack(MAGIC_SET_ALL, VERSION, seq & 0xFFFF, flags & 0xFFFF,
                              enable & 0x0FFF, cur_cap_ma & 0xFFFF, *pos))


@dataclass
class Response:
    cmd: int
    servo: int
    status: int
    pos_dd: int
    cur_ma: int
    val: float
    raw_val: int

    @property
    def ok(self) -> bool:
        return self.status == 0


@dataclass
class State:
    seq: int
    tick: int
    t_us: int
    flags: int
    fault_servo: int
    boards: int
    pos_dd: np.ndarray      # (12,) uint16
    cur_ma: np.ndarray      # (12,) int16
    temp_dc: np.ndarray     # (12,) int16, 0.1 degC
    acc_mg: np.ndarray      # (3,) int16
    gyr_ddps: np.ndarray    # (3,) int16, 0.1 deg/s
    imu_t_us: int
    t_rx: float = 0.0       # host receive time (time.monotonic)

    @property
    def armed(self) -> bool:
        return bool(self.flags & ST_ARMED)

    @property
    def getup(self) -> bool:
        """The firmware's current guard is in get-up mode."""
        return bool(self.flags & ST_GETUP)

    def flag_names(self) -> list[str]:
        return [name for bit, name in FLAG_NAMES.items() if self.flags & bit]


def parse_response(frame: bytes) -> Response:
    magic, cmd, servo, status, pos_dd, cur_ma, val = RSP.unpack(frame[:RSP.size])
    raw_val = struct.unpack("<I", frame[8:12])[0]
    return Response(cmd, servo, status, pos_dd, cur_ma, val, raw_val)


def parse_state(frame: bytes, t_rx: float = 0.0) -> State:
    v = STATE.unpack(frame[:STATE.size])
    return State(seq=v[2], tick=v[3], t_us=v[4], flags=v[5], fault_servo=v[6], boards=v[7],
                 pos_dd=np.array(v[8:20], dtype=np.uint16),
                 cur_ma=np.array(v[20:32], dtype=np.int16),
                 temp_dc=np.array(v[32:44], dtype=np.int16),
                 acc_mg=np.array(v[44:47], dtype=np.int16),
                 gyr_ddps=np.array(v[47:50], dtype=np.int16),
                 imu_t_us=v[50], t_rx=t_rx)


_LENGTHS = {MAGIC_RSP: RSP_LEN, MAGIC_STATE: STATE_LEN,
            MAGIC_REQ: REQ_LEN, MAGIC_SET_ALL: SET_ALL_LEN}


class Parser:
    """Split a byte stream into frames; a bad CRC drops one byte and resyncs.

    ``feed`` returns ``(magic, frame_bytes)`` pairs. ``crc_errors`` counts frames
    whose magic and length looked right but whose CRC did not, and
    ``skipped`` the bytes dropped while looking for a magic byte.
    """

    def __init__(self, accept=(MAGIC_RSP, MAGIC_STATE)):
        self.buf = bytearray()
        self.accept = set(accept)
        self.crc_errors = 0
        self.skipped = 0

    def feed(self, data: bytes) -> list[tuple[int, bytes]]:
        self.buf += data
        out = []
        while self.buf:
            magic = self.buf[0]
            need = _LENGTHS.get(magic) if magic in self.accept else None
            if need is None:
                del self.buf[0]
                self.skipped += 1
                continue
            if len(self.buf) < need:
                break
            frame = bytes(self.buf[:need])
            if crc16(frame[:-2]) != struct.unpack("<H", frame[-2:])[0]:
                del self.buf[0]
                self.crc_errors += 1
                continue
            out.append((magic, frame))
            del self.buf[:need]
        return out
