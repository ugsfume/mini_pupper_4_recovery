"""The link's frames, and the serial link against the fake ESP32."""
import struct
import time

import numpy as np
import pytest

from fake_esp32 import FakeEsp32
from mp4_recovery.deploy.mp4link import protocol as p
from mp4_recovery.deploy.mp4link.link import Link, LinkError


def test_crc16_ccitt_false_vector():
    assert p.crc16(b"123456789") == 0x29B1


def test_set_all_layout():
    f = p.pack_set_all(7, list(range(100, 112)), enable=0x0FFF, flags=p.SA_ARM, cur_cap_ma=1500)
    assert len(f) == p.SET_ALL_LEN == 36 and f[0] == p.MAGIC_SET_ALL and f[1] == p.VERSION
    v = p.SET_ALL.unpack(f[:34])
    assert v[2:6] == (7, p.SA_ARM, 0x0FFF, 1500) and list(v[6:18]) == list(range(100, 112))
    assert struct.unpack("<H", f[-2:])[0] == p.crc16(f[:-2])
    with pytest.raises(ValueError):
        p.pack_set_all(1, [0] * 11)


def test_float_args_roundtrip():
    a1, a2 = p.float_args(0.001)
    assert struct.unpack("<f", struct.pack("<I", a1 | a2 << 16))[0] == pytest.approx(0.001)


def test_parser_resyncs_after_junk_and_bad_crc():
    body = p.RSP.pack(p.MAGIC_RSP, 0, 1, 0, 100, 0, 1.5)
    good = body + struct.pack("<H", p.crc16(body))
    bad = good[:-1] + bytes([good[-1] ^ 0xFF])
    parser = p.Parser()
    out = parser.feed(b"\x00\x01" + bad + good)
    assert [m for m, _ in out] == [p.MAGIC_RSP] and p.parse_response(out[0][1]).val == 1.5
    assert parser.crc_errors >= 1 and parser.skipped >= 2


@pytest.fixture
def fake():
    f = FakeEsp32()
    yield f
    f.close()


@pytest.fixture
def link(fake):
    lk = Link(fake.port, 921600)
    yield lk
    lk.close()


def test_state_stream_and_echo_latency(link):
    link.record = []
    st = None
    for _ in range(40):
        link.set_all([1550] * 12, enable=0, flags=0)
        st = link.wait_state(0.2, after=st)
    assert len(link.record) >= 30 and link.tick_gaps == 0
    assert len(link.latency) > 0 and np.median(link.latency) < 0.02


def test_params_and_refusal_while_armed(link, fake):
    assert link.set_param(3, "kp_position", 80.0) == 80.0
    assert link.get_param(3, "kp_position") == 80.0
    link.set_all([1550] * 12, enable=0x0FFF, flags=p.SA_ARM)
    link.wait_state(0.2)
    with pytest.raises(LinkError, match="refused"):
        link.set_param(3, "kp_position", 1.0)
    link.disarm()


def test_watchdog_latches_and_rearm_clears(link):
    link.set_all([1550] * 12, enable=0x0FFF, flags=p.SA_ARM)
    time.sleep(0.2)
    st = link.wait_state(0.2)
    assert st.flags & p.ST_WATCHDOG and not st.armed
    link.set_all([1550] * 12, enable=0x0FFF, flags=p.SA_ARM)
    time.sleep(0.02)
    st = link.wait_state(0.2)
    assert st.armed and not st.flags & p.ST_WATCHDOG
    link.disarm()


def test_offsets_unset_set_save(link, fake):
    assert link.get_offset(2) is None
    assert link.set_offset(2, -12.5) == -12.5
    assert link.get_offset(2) == -12.5
    assert link.save_offsets() == 1 and fake.nvs == {2: -12.5}


def test_the_get_up_echo_follows_the_bit(link):
    link.set_all([1550] * 12, enable=0, flags=0, getup=True)
    time.sleep(0.02)
    assert link.wait_state(0.2).getup
    link.set_all([1550] * 12, enable=0, flags=0)
    time.sleep(0.02)
    assert not link.wait_state(0.2).getup


def test_walking_guard_trips_and_get_up_mode_does_not(link, fake):
    fake.cur_ma = [2000] * 12
    t0 = time.monotonic()
    while time.monotonic() - t0 < 1.0:          # get-up mode: no current guard
        link.set_all([1550] * 12, enable=0x0FFF, flags=p.SA_ARM if time.monotonic() - t0 < 0.03 else 0, getup=True)
        time.sleep(0.02)
    assert link.wait_state(0.2).armed
    t0 = time.monotonic()
    while time.monotonic() - t0 < 0.8:          # walking mode: 1.5 A for 0.5 s
        link.set_all([1550] * 12, enable=0x0FFF, flags=0)
        time.sleep(0.02)
    st = link.wait_state(0.2)
    assert st.flags & p.ST_FAULT_CUR and not st.armed


def test_a_closed_port_surfaces(fake):
    lk = Link(fake.port, 921600)
    lk.wait_state(0.5)
    lk.close()
    with pytest.raises(Exception):
        lk.ser.write(b"x")
