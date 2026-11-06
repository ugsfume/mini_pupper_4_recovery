"""The tools: a new contract, the deployed subset, the summary of logs."""
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from mp4_recovery import paths
from mp4_recovery.contract import CONTRACT_JSON, Contract
from mp4_recovery.deploy import checks
from mp4_recovery.deploy.mp4link import cli
from robot_helpers import go

ROOT = Path(__file__).resolve().parents[2]


def load_tool(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def offsets_file(tmp_path, offsets, stored=True):
    ticks = cli.offset_ticks(offsets)
    f = tmp_path / "offsets.json"
    f.write_text(json.dumps({"offsets_deg": [float(v) for v in offsets], "offsets_stored": [stored] * 12,
                             "stand_ticks": ticks.tolist(), "stand_dd": cli.ticks_to_dd(ticks).tolist()}))
    return f


def test_make_contract_for_another_unit(tmp_path, capsys):
    tool = load_tool("make_contract")
    c0 = Contract.load()
    new = c0.offsets_deg.astype(np.float32) + 6.0
    out = tmp_path / "contract.json"
    assert tool.main([str(offsets_file(tmp_path, new)), "--note", "unit 2", "--out", str(out)]) == 0
    c = Contract.load(out)
    d = json.loads(out.read_text())
    assert np.allclose(c.offsets_deg, new) and d["calibration"] == "unit 2"
    assert c.stand_dd.tolist() == cli.ticks_to_dd(cli.offset_ticks(new)).tolist()
    assert d["calibrations_superseded"][-1]["sha256"] == paths.sha256(CONTRACT_JSON)
    for key in ("signs", "tendon_coefs", "stand_policy_rad", "joint_order"):
        assert d[key] == c0.meta[key]
    assert "re-export" in capsys.readouterr().out


def test_make_contract_refuses_unset_offsets_and_a_wrong_stand(tmp_path):
    tool = load_tool("make_contract")
    c0 = Contract.load()
    with pytest.raises(SystemExit, match="not every offset"):
        tool.main([str(offsets_file(tmp_path, c0.offsets_deg, stored=False)), "--note", "x", "--out", str(tmp_path / "c.json")])
    f = offsets_file(tmp_path, c0.offsets_deg)
    d = json.loads(f.read_text())
    d["stand_dd"][0] += 10
    f.write_text(json.dumps(d))
    with pytest.raises(SystemExit, match="not the one derived"):
        tool.main([str(f), "--note", "x", "--out", str(tmp_path / "c.json")])


def test_a_shipped_policy_is_refused_on_a_new_calibration(tmp_path, monkeypatch):
    tool = load_tool("make_contract")
    c0 = Contract.load()
    out = tmp_path / "contract.json"
    tool.main([str(offsets_file(tmp_path, c0.offsets_deg.astype(np.float32) + 6.0)), "--note", "unit 2", "--out", str(out)])
    meta = json.loads(str(np.load(paths.POLICIES / "getup" / "policy.npz")["meta"]))
    assert checks.band_refusal(meta) is None
    new = Contract.load(out)
    monkeypatch.setattr(Contract, "load", classmethod(lambda cls, path=out: new))
    assert "another calibration" in checks.band_refusal(meta)


def test_deployed_subset_needs_only_numpy_and_pyserial(tmp_path):
    dest = tmp_path / "walk"
    subprocess.run(["sh", str(ROOT / "tools" / "deploy.sh"), "-", str(dest)], check=True, capture_output=True)
    assert not (dest / "src" / "mp4_recovery" / "gpu").exists() and not (dest / "src" / "mp4_recovery" / "limits.py").exists()
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"} | {"PYTHONPATH": str(dest / "src")}
    code = ("import sys; import mp4_recovery.deploy.runner, mp4_recovery.deploy.mp4link.cli;"
            "bad = [m for m in sys.modules if m.split('.')[0] in ('torch', 'mjlab', 'mujoco', 'warp')];"
            "assert not bad, bad; assert mp4_recovery.deploy.runner.__file__.startswith(sys.argv[1])")
    subprocess.run([sys.executable, "-B", "-c", code, str(dest)], check=True, env=env, cwd=dest)
    for name in ("walker", "getup", "getup_sim"):
        subprocess.run([sys.executable, "-B", "-m", "mp4_recovery.deploy.runner", "check", "--policy", f"policies/{name}"],
                       check=True, capture_output=True, env=env, cwd=dest)
    for pol in ("walker", "getup"):
        p = subprocess.run([sys.executable, "-B", "-m", "mp4_recovery.deploy.runner", "check", "--policy", f"policies/{pol}"],
                           env=env, cwd=dest, capture_output=True, text=True)
        assert p.returncode == 0, p.stdout + p.stderr


def test_summarise_runs_on_a_fresh_log(robot, tmp_path, capsys):
    from fake_esp32 import fall_and_rise, roll
    robot.script, robot.script_at = roll(fall_and_rise(lie_s=0.5, up_s=0.8)), 60
    rc, _, _ = go(robot, tmp_path, seconds=3.0)
    assert rc == 0
    tool = load_tool("summarise_runs")
    capsys.readouterr()
    assert tool.main([str(tmp_path), "--json", str(tmp_path / "s.json")]) == 0
    assert "fall: 1 get-ups, 1 up, 1 on the first attempt, 0 never up" in capsys.readouterr().out
    row = json.loads((tmp_path / "s.json").read_text())["runs"][0]
    assert row["events"][0]["up"] and row["steps"] == 150


def test_summarise_filters_a_one_frame_temperature_spike():
    tool = load_tool("summarise_runs")
    t = np.full((10, 12), 33.6)
    t[5, 1] = 67.2
    assert tool.median3(t).max() == pytest.approx(33.6)
