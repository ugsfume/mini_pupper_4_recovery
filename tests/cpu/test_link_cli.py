"""The link's command line against the fake ESP32."""
import json
import math
import types

import numpy as np
import pytest

from fake_esp32 import FakeEsp32
from mp4_recovery.contract import Contract
from mp4_recovery.deploy.mp4link import cli
from mp4_recovery.deploy.mp4link import protocol as p

C = Contract.load()


@pytest.fixture
def fake():
    f = FakeEsp32()
    f.offsets = {i + 1: float(v) for i, v in enumerate(C.offsets_deg)}
    f.pos = [float(v) for v in C.stand_dd]
    ids = {name: i for i, name in enumerate(p.PARAMS)}
    for s in range(1, 13):
        f.params[s][ids["kff_current"]] = 0.001
    yield f
    f.close()


def test_the_stand_formula_matches_the_contract():
    assert cli.ticks_to_dd(cli.offset_ticks(C.offsets_deg)).tolist() == list(C.stand_dd)
    assert cli.offset_ticks(C.offsets_deg).tolist() == list(C.stand_ticks)


def test_config_writes_and_verifies(fake, capsys):
    assert cli.main(["--port", fake.port, "config"]) == 0
    ids = {name: i for i, name in enumerate(p.PARAMS)}
    assert fake.params[7][ids["kp_position"]] == 80.0 and fake.params[7][ids["kff_current"]] == pytest.approx(0.001)
    assert "written to RAM" in capsys.readouterr().out


def test_ping_and_dump_are_read_only(fake, capsys, tmp_path):
    assert cli.main(["--port", fake.port, "ping", "--seconds", "0.2"]) == 0
    out = capsys.readouterr().out
    assert "IMU ok" in out and "fresh feedback" in out
    assert cli.main(["--port", fake.port, "dump", "--out", str(tmp_path / "d.json")]) == 0
    assert len(json.loads((tmp_path / "d.json").read_text())["servos"]) == 12
    assert not fake.armed_seen


def test_offsets_json(fake, tmp_path):
    f = tmp_path / "o.json"
    assert cli.main(["--port", fake.port, "offsets", "--json", str(f)]) == 0
    d = json.loads(f.read_text())
    assert d["stand_dd"] == list(C.stand_dd) and all(d["offsets_stored"])
    assert np.allclose(d["offsets_deg"], C.offsets_deg, atol=1e-5)


def test_latched_fault_refused_until_clear(fake, capsys):
    fake.latched = p.ST_FAULT_TEMP
    fake.fault_servo = 4
    assert cli.main(["--port", fake.port, "calib", "--seconds", "0.1"]) == 1
    assert "fault latched" in capsys.readouterr().err
    assert not fake.armed_seen
    assert cli.main(["--port", fake.port, "calib", "--seconds", "0.1", "--clear"]) == 0
    assert fake.armed_seen


def test_implausible_position_blocks_motion(fake, capsys):
    fake.pos[3] = 50.0
    assert cli.main(["--port", fake.port, "calib", "--seconds", "0.1"]) == 1
    assert "implausible" in capsys.readouterr().err and not fake.armed_seen


def test_centre_holds_the_centre_on_the_chosen_servos(fake):
    assert cli.main(["--port", fake.port, "centre", "--ids", "2,3", "--seconds", "0.1"]) == 0
    enable, targets = fake.armed_seen[-1]
    centre = int(cli.ticks_to_dd([cli.CENTRE_TICK])[0])
    assert enable == 0b110 and targets[1] == targets[2] == centre
    assert not fake.armed


def test_calib_refuses_flash_gains_then_holds_and_saves(fake, capsys):
    ids = {name: i for i, name in enumerate(p.PARAMS)}
    fake.params[5][ids["kff_current"]] = 0.00022
    assert cli.main(["--port", fake.port, "calib", "--seconds", "0.1"]) == 1
    assert "run `config` first" in capsys.readouterr().err
    fake.params[5][ids["kff_current"]] = 0.001
    assert cli.main(["--port", fake.port, "calib", "--seconds", "0.1", "--set", "2=11", "--save"]) == 0
    assert fake.nvs[2] == 11.0
    assert fake.armed_seen[-1][1][1] == cli.ticks_to_dd(cli.offset_ticks(fake.nvs[2]))


@pytest.mark.parametrize("deg,rc", [(90, 0), (-90, 0)])
def test_calib_accepts_90(fake, deg, rc):
    fake.pos = [1550.0] * 12
    assert cli.main(["--port", fake.port, "calib", "--seconds", "0.05", f"--set=1={deg}"]) == rc


def test_calib_refuses_91(fake):
    with pytest.raises(SystemExit, match="limited"):
        cli.main(["--port", fake.port, "calib", "--seconds", "0.05", "--set=1=91"])


def test_set_offset_is_refused_past_90(fake, capsys):
    from mp4_recovery.deploy.mp4link.link import Link, LinkError
    lk = Link(fake.port, 921600)
    try:
        with pytest.raises(LinkError):
            lk.set_offset(1, 91.0)
    finally:
        lk.close()


# ---- stream --------------------------------------------------------------------
def stream_file(tmp_path, delta):
    f = tmp_path / "s.npz"
    np.savez(f, delta_dd=np.asarray(delta, dtype=float), period_s=0.02)
    return f


def stream(fake, tmp_path, delta, *extra):
    log = tmp_path / "run.npz"
    rc = cli.main(["--port", fake.port, "stream", str(stream_file(tmp_path, delta)), "--log", str(log), *extra])
    return rc, log


def test_stream_sets_get_up_mode_on_every_frame_and_logs_it(fake, tmp_path, capsys):
    rc, log = stream(fake, tmp_path, np.zeros((20, 12)))
    assert rc == 0
    armed = [fr for fr in fake.frames if not fr[1] & p.SA_DISARM]
    assert armed and all(fr[1] & p.SA_GETUP for fr in armed)
    meta = json.loads(str(np.load(log)["meta"]))
    assert meta["stop_reason"] == "completed" and meta["getup_echo_share"] == 1.0
    assert not fake.armed


def test_stream_stops_without_the_echo(fake, tmp_path, capsys):
    fake.echo = False
    rc, log = stream(fake, tmp_path, np.zeros((400, 12)))
    assert rc == 0
    assert json.loads(str(np.load(log)["meta"]))["stop_reason"] == "no_getup_echo"
    assert "no get-up mode echo" in capsys.readouterr().out


def test_stream_plays_a_180_deg_hip_at_20_deg_per_send(fake, tmp_path, capsys):
    stand = np.asarray(C.stand_dd, dtype=float)
    room = int(min(3100 - 200 - stand[1], 1800))
    rows = np.zeros((math.ceil(room / 200) + 1, 12))
    rows[:, 1] = np.minimum(np.arange(rows.shape[0]) * 200.0, room)
    rc, log = stream(fake, tmp_path, rows)
    meta = json.loads(str(np.load(log)["meta"]))
    assert rc == 0 and meta["clipped_sends"] == 0 and meta["stop_reason"] == "completed"


def test_stream_clips_abductions_to_90_deg(fake, tmp_path, capsys):
    rows = np.zeros((60, 12))
    rows[:, 0] = np.minimum(np.arange(60) * 100.0, 950.0)
    stream(fake, tmp_path, rows)
    sent = max(abs(fr[0][0] - int(C.stand_dd[0])) for fr in fake.frames if fr[2])
    assert sent <= 900


def test_stream_refuses_a_target_near_a_track_end_before_arming(fake, tmp_path):
    rows = np.zeros((3, 12))
    rows[:, 1] = 3100 - 100 - float(C.stand_dd[1])
    with pytest.raises(SystemExit, match="track end"):
        stream(fake, tmp_path, rows)
    assert not fake.frames


def test_stream_stops_and_returns_at_a_track_end_reading(fake, tmp_path, capsys):
    def later():
        fake.pos[1] = 3050.0
    rows = np.zeros((300, 12))
    orig = fake.before_tick
    count = {"n": 0}

    def tick():
        orig()
        count["n"] += 1 if fake.armed else 0
        if count["n"] > 700:                    # into the playback: the ramp and hold take 3 s
            later()
    fake.before_tick = tick
    rc, log = stream(fake, tmp_path, rows)
    meta = json.loads(str(np.load(log)["meta"]))
    assert rc == 0 and meta["stop_reason"] == "track_end" and meta["track_end_servo"][0] == 2
    assert "within 15 deg of a track end" in capsys.readouterr().out


def test_stream_refuses_a_hot_start(fake, tmp_path, capsys):
    fake.temps_dc = [460] * 12
    rc, _ = stream(fake, tmp_path, np.zeros((5, 12)))
    assert rc == 1 and "let it cool" in capsys.readouterr().err and not fake.armed_seen


def test_stream_a_hot_housing_stops_and_returns(fake, tmp_path):
    orig = fake.before_tick

    def tick():
        orig()
        if len(fake.armed_seen) > 500:
            fake.temps_dc = [560] * 12
    fake.before_tick = tick
    rc, log = stream(fake, tmp_path, np.zeros((300, 12)))
    assert rc == 0 and json.loads(str(np.load(log)["meta"]))["stop_reason"] == "temperature"
    assert not fake.armed


def frame(tick, dc):
    return types.SimpleNamespace(tick=tick, temp_dc=[dc] * 12)


def test_temp_watch_ignores_a_one_frame_spike():
    w = cli.TempWatch(55.0)
    seq = [336, 672, 336] + [336] * 400
    assert all(w.update(frame(t, dc)) is None for t, dc in enumerate(seq))


def test_temp_watch_needs_a_second_above_the_limit():
    w = cli.TempWatch(55.0)
    got = [w.update(frame(t, 560)) for t in range(201)]
    assert got[199] is None and got[200] == 56.0
    w = cli.TempWatch(55.0)
    for t in range(150):
        assert w.update(frame(t, 560)) is None
    assert w.update(frame(150, 500)) is None    # a dip starts the second again
    for t in range(151, 351):
        assert w.update(frame(t, 560)) is None


def test_missing_port_is_a_one_line_error(capsys):
    assert cli.main(["--port", "/dev/does-not-exist", "ping"]) == 1
    assert capsys.readouterr().err.startswith("error: cannot open")
