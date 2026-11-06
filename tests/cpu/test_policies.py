"""The shipped policies: arrays pinned, the numpy forward pass against their fixtures, metadata free of local paths, the
walker's meta what ``export_policy --from-policy`` writes."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from mp4_recovery import paths
from mp4_recovery.contract import CONTRACT_JSON, Bounds
from mp4_recovery.deploy import runner as rn
from mp4_recovery.policy_file import NumpyPolicy, fixture_tolerance

REPO = Path(__file__).resolve().parents[2]
ARRAYS = {
    "walker": "9a2568ff39eda4f8cfacb3822d2ebece1c39b009e0380efeb58a80498ba2aef1",
    "getup": "1f1e04bd8f99fb99e6de25fd0711f21081def083af7f56f68ea33773cdcdbcf8",
    "getup_sim": "0c15fd538e56de60d4b23c8f1e4379dad986ca014001ef4d1fce18a690146fae",
}
LOCAL = ("/home", "/tmp", "wandb", "runs/")


def arrays_sha256(path) -> str:
    h = hashlib.sha256()
    with np.load(path, allow_pickle=False) as d:
        for k in sorted(d.files):
            if k != "meta":
                h.update(k.encode())
                h.update(d[k].tobytes())
    return h.hexdigest()


@pytest.mark.parametrize("name", ARRAYS)
def test_arrays_and_fixtures(name):
    path = REPO / "policies" / name / "policy.npz"
    assert arrays_sha256(path) == ARRAYS[name]
    p = NumpyPolicy(path)
    d = np.load(path, allow_pickle=False)
    assert np.abs(p(d["fixture_obs"]) - d["fixture_act"]).max() < fixture_tolerance(d["fixture_act"])


@pytest.mark.parametrize("name", ARRAYS)
def test_meta_and_config(name):
    meta = NumpyPolicy(REPO / "policies" / name / "policy.npz").meta
    assert meta["contract_sha256"] == paths.sha256(CONTRACT_JSON)
    Bounds.from_meta(meta["bounds"])
    for text in (json.dumps(meta), (REPO / "policies" / name / "config.json").read_text()):
        assert not any(s in text for s in LOCAL), text[:200]


@pytest.mark.parametrize("name", ARRAYS)
def test_runner_check_passes(name):
    assert rn.main(["check", "--policy", str(REPO / "policies" / name)]) == 0


def test_sim_only_policy_says_so():
    assert NumpyPolicy(REPO / "policies/getup_sim/policy.npz").meta["tested"] == "simulation only"
    assert "tested" not in NumpyPolicy(REPO / "policies/getup/policy.npz").meta


def test_export_records_no_local_path_and_walker_from_policy(tmp_path):
    out = tmp_path / "walker"
    subprocess.run([sys.executable, "-m", "mp4_recovery.export_policy", "--from-policy", str(REPO / "policies/walker"),
                    str(out), "--limits", "--stand-offset-abd", "12"], check=True, cwd=REPO,
                   env={"PYTHONPATH": str(REPO / "src"), "PATH": "/usr/bin:/bin"})
    assert arrays_sha256(out / "policy.npz") == ARRAYS["walker"]
    made, shipped = (NumpyPolicy(p / "policy.npz").meta for p in (out, REPO / "policies/walker"))
    assert str(tmp_path) not in json.dumps(made) and str(REPO) not in json.dumps(made)
    for k in ("kind", "limits", "stand_offset_abd_deg", "contract_sha256", "bounds", "history", "omni"):
        assert made[k] == shipped[k], k
    assert (out / "config.json").read_text() == (REPO / "policies/walker/config.json").read_text()


def test_walker_recipe_writes_the_same_arrays(tmp_path):
    # the recipe unsets PYTHONPATH: the installed package, or this checkout's src through the script below
    runner = tmp_path / "python"
    runner.write_text(f'#!/bin/sh\nPYTHONPATH={REPO / "src"} exec {sys.executable} "$@"\n')
    runner.chmod(0o755)
    out = tmp_path / "w"
    subprocess.run(["sh", "scripts/recipes/walker.sh", str(out)], check=True, cwd=REPO, env={"PYTHON": str(runner),
                                                                                             "PATH": "/usr/bin:/bin"})
    assert arrays_sha256(out / "policy.npz") == ARRAYS["walker"]


def recipe_stub(tmp_path):
    """A ``python`` that logs its arguments and leaves what training leaves: a run folder with its checkpoints."""
    stub = tmp_path / "python"
    stub.write_text("""#!/bin/sh
echo "$@" >> "$OUT/calls"
while [ $# -gt 0 ]; do
  case $1 in --tag) tag=$2;; --out) out=$2;; --iters) iters=$2;; esac
  shift
done
case $out in
  */clone) mkdir -p "$out"; : > "$out/model_bc.pt";;
  *) mkdir -p "$out/run_$tag"; : > "$out/run_$tag/model_950.pt"; : > "$out/run_$tag/model_$iters.pt";;
esac
""")
    stub.chmod(0o755)
    return stub


def recipe(tmp_path, out, **env):
    r = subprocess.run(["sh", "scripts/recipes/getup.sh"], check=True, cwd=REPO, capture_output=True, text=True,
                       env={"PYTHON": str(recipe_stub(tmp_path)), "OUT": str(out), "PATH": "/usr/bin:/bin", **env})
    calls = (out / "calls").read_text().splitlines() if (out / "calls").exists() else []
    return r.stdout, calls


STEPS = {"wide": "1500", "limits": "1000", "filter": "253", "smooth1": "400", "smooth2": "200", "smooth3": "600", "smooth4": "600"}


def steps_of(calls):
    out = {}
    for c in calls:
        a = c.split()
        if "--tag" in a and a[a.index("--tag") + 1] != "cfg":
            out[a[a.index("--tag") + 1]] = a[a.index("--iters") + 1]
    return out


def test_getup_recipe_runs_its_steps_once_each(tmp_path):
    out = tmp_path / "s"
    stdout, calls = recipe(tmp_path, out)
    assert steps_of(calls) == STEPS and list(steps_of(calls)) == list(STEPS)
    assert sum("getup_bc" in c for c in calls) == 1 and "--seed 41" in calls[0]
    assert sum("--widen-limits-deg" in c for c in calls) == 2      # only the config for the clone, and `wide`
    assert stdout.strip().endswith("run_smooth4/model_950.pt")
    before = len(calls)
    assert len(recipe(tmp_path, out)[1]) == before                 # all done: nothing runs again


def test_getup_recipe_starts_from_a_checkpoint(tmp_path):
    out = tmp_path / "s"
    _, calls = recipe(tmp_path, out, START="checkpoints/wide_1499.pt")
    want = {k: v for k, v in STEPS.items() if k != "wide"}
    assert steps_of(calls) == want and "--resume checkpoints/wide_1499.pt" in calls[0]
    assert not any("getup_bc" in c for c in calls)


def test_getup_recipe_starts_from_every_shipped_checkpoint(tmp_path):
    _, calls = recipe(tmp_path, tmp_path / "a", START="checkpoints/clone.pt")
    assert steps_of(calls) == STEPS and "--init-actor checkpoints/clone.pt" in calls[0]
    assert not any("getup_bc" in c for c in calls)
    _, calls = recipe(tmp_path, tmp_path / "b", START="checkpoints/smooth3_3947.pt")
    assert steps_of(calls) == {"smooth4": "600"} and "--resume checkpoints/smooth3_3947.pt" in calls[0]


def test_checkpoints_load_as_plain_tensors():
    import torch
    for name, it in (("clone.pt", None), ("wide_1499.pt", 1499), ("smooth3_3947.pt", 3947)):
        ck = torch.load(REPO / "checkpoints" / name, map_location="cpu", weights_only=True)
        assert "obs_normalizer._mean" in ck["actor_state_dict"]
        assert ck.get("iter") == it
        if it is not None:
            assert "obs_normalizer._mean" in ck["critic_state_dict"] and ck["optimizer_state_dict"]
