"""Multi-start behaviour cloning (``gpu/getup_bc.py``): sources in their own band, mirrored in radians, balanced per
start; the clone (any size, with an observation normaliser) taken by ``train --init-actor --obs-norm`` and exported as the
plain 45-in / 12-out numpy policy."""
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
LEFT = REPO / "data/searched_motions/search_left.json"


def _bc():
    from mp4_recovery.gpu import getup_bc
    return getup_bc


def _saved(band, scale=None, goal=None):
    k = 2
    post = np.zeros((k, 12))
    post[0, 1], post[0, 0], post[1, 4] = 1.0, -1.0, 0.5           # FR hip full, FR abduction full, FL hip half
    d = {"band_deg": list(band), "best": {"postures": post.tolist(), "holds_s": [0.5, 0.6]}}
    if scale:
        d["scale"] = list(scale)
    if goal:
        d["goal"] = goal
    return d


def test_sources_in_the_policy_units_mirrored_in_radians():
    bc = _bc()
    from mp4_recovery.gpu.getup_task import getup_scale
    pband = (30.0, 127.0, 119.0) * 4
    pscale = getup_scale(pband)                                    # 1.57 abduction, the hips' band in rad
    post, hold = bc.source_actions(_saved((30, 90, 90)), False, pscale, pband)
    assert torch.allclose(hold, torch.tensor([0.5, 0.6]))
    # FR hip: 1 x 1.57 rad (inside the source's 90 deg); abduction -1 x 1.57 clipped to -30 deg
    assert abs(float(post[0, 1]) * pscale[1] - 1.57) < 1e-5
    assert abs(float(post[0, 0]) * pscale[0] + math.radians(30)) < 1e-5
    assert abs(float(post[1, 4]) * pscale[4] - 0.5 * 1.57) < 1e-5
    mpost, _ = bc.source_actions(_saved((30, 90, 90)), True, pscale, pband)   # FR <-> FL, abduction and hip negated
    assert abs(float(mpost[0, 4]) * pscale[4] + 1.57) < 1e-5
    assert abs(float(mpost[0, 3]) * pscale[3] - math.radians(30)) < 1e-5
    with pytest.raises(AssertionError):                             # a source wider than the policy's band
        bc.source_actions(_saved((30, 150, 90)), False, getup_scale((30, 90, 90)), (30.0, 90.0, 90.0) * 4)


def test_balance_per_start():
    bc = _bc()
    g = torch.Generator().manual_seed(0)
    up = bc.balance({"left": 100, "nose": 30, "back": 0}, g)
    assert len(up["left"]) == len(up["nose"]) == 100 and len(up["back"]) == 0
    assert set(up["nose"].tolist()) == set(range(30))              # every sample used
    srcs = bc.parse_sources([f"left={LEFT}"])
    assert [(s["start"], s["mirror"]) for s in srcs] == [("left", False), ("right", True)]


def test_bc_obs_norm_clone_into_ppo_and_export(tmp_path):
    """BC (left + its mirror, 64-32 net, normaliser) -> train --init-actor --obs-norm --obs-norm-freeze 1
    (1 iteration: the normaliser still the BC's) -> export: the numpy policy equals the torch actor with its
    normaliser."""
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"} | {"WANDB_MODE": "disabled", "PYTHONPATH": str(REPO / "src")}
    bc_dir = tmp_path / "bc"
    r = subprocess.run([sys.executable, "-B", "-m", "mp4_recovery.gpu.getup_bc", "--seq", f"left={LEFT}", "--band", "90", "90", "90",
                        "--phase-s", "4", "--robots", "16", "--rounds", "1", "--epochs", "2", "--hidden", "64", "32",
                        "--obs-norm", "--out", str(bc_dir)], cwd=REPO, env=env, capture_output=True, text=True,
                       timeout=900)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    info = json.loads((bc_dir / "bc.json").read_text())
    assert set(info["closed_loop"]) == {"left", "right"} and info["samples_used"]["left"] == info["samples_used"]["right"]
    ck = torch.load(bc_dir / "model_bc.pt", weights_only=False)["actor_state_dict"]
    assert ck["mlp.0.weight"].shape == (64, 45) and "obs_normalizer._mean" in ck
    runs = tmp_path / "runs"
    r = subprocess.run([sys.executable, "-B", "-m", "mp4_recovery.gpu.train", "--getup-band", "90", "90",
                        "90", "--getup-phase", "4", "--init-actor", str(bc_dir / "model_bc.pt"), "--obs-norm",
                        "--obs-norm-freeze", "1", "--actor-hidden", "64", "32", "--critic-hidden", "64", "32",
                        "--iters", "1", "--envs", "16", "--seed", "1", "--out", str(runs), "--tag", "bcn"],
                       cwd=REPO, env=env, capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    (run,) = runs.glob("*_bcn")
    cfg = json.loads((run / "config.json").read_text())
    assert cfg["runner"]["actor"]["hidden_dims"] == [64, 32] and cfg["runner"]["actor"]["obs_normalization"]
    model = max(run.glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    sd = torch.load(model, weights_only=False)["actor_state_dict"]
    for k in ("obs_normalizer._mean", "obs_normalizer._std"):        # frozen: the BC's statistics
        assert torch.allclose(sd[k].cpu(), ck[k]), k
    assert int(sd["obs_normalizer.count"]) == int(ck["obs_normalizer.count"])
    assert "OBS_NORM_FREEZE" in r.stdout
    out = tmp_path / "exp"
    r = subprocess.run([sys.executable, "-B", "-m", "mp4_recovery.export_policy", str(model), str(out)], cwd=REPO, env=env,
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    from mp4_recovery.gpu.actor import load_actor
    from mp4_recovery.policy_file import NumpyPolicy
    p = NumpyPolicy(out / "policy.npz")
    assert p.meta["kind"] == "getup" and p.meta["getup_phase_s"] == 4.0 and p.meta["obs_normalizer"]["folded"]
    assert p.layers[0][0].shape == (64, 45) and p.layers[-1][0].shape == (12, 32)
    x = np.random.default_rng(2).normal(0.0, 1.0, (64, 45)).astype(np.float32)
    with torch.no_grad():
        want = load_actor(model, device="cpu")(torch.from_numpy(x)).numpy()
    assert np.abs(p(x) - want).max() < 1e-5


@pytest.mark.parametrize("name", ["wide_1499.pt", "smooth3_3947.pt"])
def test_resume_from_the_shipped_checkpoints(tmp_path, name):
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"} | {"WANDB_MODE": "disabled",
                                                                        "PYTHONPATH": str(REPO / "src")}
    r = subprocess.run([sys.executable, "-B", "-m", "mp4_recovery.gpu.train", "--limits", "--getup-phase", "4", "--getup-band",
                        "90", "180", "180", "--getup-episode-s", "8", "--obs-norm", "--critic-warmup", "1", "--iters", "1",
                        "--envs", "64", "--out", str(tmp_path), "--tag", "t", "--resume", str(REPO / "checkpoints" / name)],
                       cwd=REPO, env=env, capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]


def test_init_actor_from_the_shipped_clone(tmp_path):
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"} | {"WANDB_MODE": "disabled",
                                                                        "PYTHONPATH": str(REPO / "src")}
    r = subprocess.run([sys.executable, "-B", "-m", "mp4_recovery.gpu.train", "--limits", "--getup-phase", "4", "--getup-band",
                        "90", "180", "180", "--getup-episode-s", "8", "--obs-norm", "--actor-hidden", "512", "256", "128",
                        "--critic-hidden", "512", "256", "128", "--critic-warmup", "1", "--iters", "1", "--envs", "64",
                        "--out", str(tmp_path), "--tag", "t", "--widen-limits-deg", "60",
                        "--init-actor", str(REPO / "checkpoints/clone.pt")],
                       cwd=REPO, env=env, capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
