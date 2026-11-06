"""Shared by the robot-side tests: synthetic policies and a run against the fake robot."""
import gc
import json

import numpy as np
import pytest

from fake_esp32 import FakeRobot
from mp4_recovery import paths
from mp4_recovery.contract import CONTRACT_JSON, Contract
from mp4_recovery.deploy import runner as rn

C = Contract.load()
MOUNT = rn.load_mount()[1]
RAMP_TICKS = 40                 # armed ticks at the policy's first step: the 0.2 s ramp these tests use


def make_policy(d, bias=0.0, scale=(0.10, 0.40, 0.42), **meta_extra):
    """A one-layer policy whose action is ``bias`` (a scalar or 12 values) whatever it observes."""
    d.mkdir(parents=True)
    meta = {"contract_sha256": paths.sha256(CONTRACT_JSON), "omni": True, **meta_extra}
    b = np.broadcast_to(np.asarray(bias, dtype=np.float32), (12,)).copy()
    s = np.broadcast_to(np.asarray(scale * 4 if len(scale) == 3 else scale, dtype=np.float32), (12,)).copy()
    np.savez(d / "policy.npz", n_layers=1, scale=s, meta=json.dumps(meta), fixture_obs=np.zeros((2, 45), dtype=np.float32),
             fixture_act=np.tile(b, (2, 1)), W0=np.zeros((12, 45), dtype=np.float32), b0=b)
    return d


def getup_policy(d, bias=0.0, **meta):
    """A get-up policy: kind getup, the 90 deg band, 1.57 rad per unit, a 4 s phase clock."""
    m = {"kind": "getup", "band_deg": [90.0] * 12, "getup_phase_s": 4.0, **meta}
    return make_policy(d, bias, scale=(1.57,) * 12, **m)


@pytest.fixture
def robot(monkeypatch):
    gc.collect()   # the fake shares this process: a collection of earlier tests' garbage would stall its STATE
    monkeypatch.setattr(rn, "HOLD_AFTER_S", 0.2)
    f = FakeRobot(C, MOUNT, rn.IDENTIFIED)
    yield f
    f.close()


def go(robot, tmp_path, *extra, seconds=3.0, walker_bias=0.0, getup_bias=0.0, walker_meta=None, getup_meta=None):
    """One ``run`` with synthetic policies; returns (exit code, the step log, its meta)."""
    walker = make_policy(tmp_path / "walker", walker_bias, **(walker_meta or {}))
    getup = getup_policy(tmp_path / "getup", getup_bias, **(getup_meta or {}))
    log = tmp_path / "run"
    rc = rn.main(["--port", robot.port, "run", "--policy", str(walker), "--getup-policy", str(getup), "--stand-ramp", "0.2",
                  "--seconds", str(seconds), "--log", str(log), *extra])
    if rc:
        return rc, None, None
    z = np.load(log.with_suffix(".policy.npz"))
    return rc, z, json.loads(str(z["meta"]))
