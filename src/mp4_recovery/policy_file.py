"""Export a policy for the robot: the actor's mean net as numpy arrays, with what the runner needs to use it.

``export(checkpoint, out)`` writes ``out/policy.npz``: the layers (``W0, b0, ...``, ELU between them), the action
scale, the contract's hash, the observation layout (``meta``: ``history``, ``omni``, ``obs_terms_extra``), the policy's
kind and action band (``kind``: "walk" or "getup"; ``band_deg``: 12 floats about the stand, the runner's decoder band), the
joint limits (``limits``: trained / evaluated with the measured limits; ``bounds``: the decoder's clamps,
``contract.Bounds.to_meta()``, or None; ``stand_offset_abd_deg``: the abductions' stand offset, deg outward), the
target filter (``action_filter_alpha``: the decoder's low-pass alpha, 0 without one, ``contract.policy_filter_alpha``),
and fixtures (observations and the torch actions for them), so that the robot side can check its numpy forward pass
(``NumpyPolicy``, numpy only) against training (``fixture_tolerance``). An actor with an observation normaliser (``obs_norm``) has it folded
into ``W0, b0`` (meta ``obs_normalizer``), so the runner needs nothing extra; the fixtures come from the torch actor with its
normaliser. Metas name checkpoints and source policies by hash only, never by path.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from mp4_recovery.contract import CONTRACT_JSON, POLICY_KINDS, Bounds, band_deg_array, obs_size, policy_filter_alpha


def fixture_tolerance(act) -> float:
    """How far the numpy forward pass may be from the torch outputs ``act``: float32 rounding grows with the outputs' size."""
    return 1e-5 * max(1.0, float(np.abs(act).max()))


class NumpyPolicy:
    def __init__(self, path):
        d = np.load(path, allow_pickle=False)
        self.layers = [(d[f"W{i}"], d[f"b{i}"]) for i in range(int(d["n_layers"]))]
        self.scale = d["scale"]
        self.meta = json.loads(str(d["meta"]))

    def __call__(self, obs) -> np.ndarray:
        x = np.asarray(obs, dtype=np.float32)
        for i, (w, b) in enumerate(self.layers):
            x = x @ w.T + b
            if i < len(self.layers) - 1:
                x = np.where(x > 0, x, np.expm1(np.minimum(x, 0.0))).astype(np.float32)     # ELU
        return x


def export(checkpoint, out_dir, scale, fixtures: int = 256, seed: int = 0, omni: bool = True,
           history: int = 0, band_deg=None, kind: str = "walk", getup_phase_s: float = 0.0, bounds: dict | None = None,
           stand_offset_abd_deg: float = 0.0, action_filter_alpha: float = 0.0) -> Path:
    """``omni``: the policy observes the whole command (vx, vy, wz); the runner refuses vy / wz for one that does
    not. ``band_deg``: the band the policy was trained with (``contract.band_deg_array``; default the walking band);
    ``kind``: "walk" or "getup". ``getup_phase_s`` > 0 (a phased get-up): meta ``getup_phase_s``; the command's three
    floats then carry (phi, sin 2 pi phi, cos 2 pi phi), phi = the time since the get-up began / getup_phase_s, held at 1
    (the runner feeds it from its GETUP entry). ``bounds``: the decoder's clamps (``limits.decoder_bounds(kind).to_meta()``; meta ``limits``
    true); ``stand_offset_abd_deg``: the abduction stand offset (deg outward); ``action_filter_alpha``: the target
    filter the policy was trained with (its final alpha; 0: none)."""
    if kind not in POLICY_KINDS:
        raise ValueError(f"kind {kind!r}: one of {POLICY_KINDS}")
    band = band_deg_array(band_deg)
    Bounds.from_meta(bounds)                             # validated (None passes)
    alpha = policy_filter_alpha({"action_filter_alpha": action_filter_alpha})     # validated
    import torch
    from mp4_recovery.gpu.actor import load_actor
    from mp4_recovery.paths import sha256
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    sd = ck["actor_state_dict"]
    lin = sorted({int(k.split(".")[1]) for k in sd if k.startswith("mlp.") and k.endswith(".weight")})
    arrays = {}
    for i, idx in enumerate(lin):
        arrays[f"W{i}"] = sd[f"mlp.{idx}.weight"].numpy().astype(np.float32)
        arrays[f"b{i}"] = sd[f"mlp.{idx}.bias"].numpy().astype(np.float32)
    from mp4_recovery import obs_norm
    norm = obs_norm.read(sd)          # an observation normaliser: folded into the first layer (the format unchanged)
    if norm is not None:
        arrays["W0"], arrays["b0"] = obs_norm.fold(arrays["W0"], arrays["b0"], norm[0].numpy(), norm[1].numpy())
    actor = load_actor(checkpoint, device="cpu")          # the torch reference, the normaliser included
    rng = np.random.default_rng(seed)
    obs = rng.normal(0.0, 0.5, (fixtures, arrays["W0"].shape[1])).astype(np.float32)
    assert arrays["W0"].shape[1] == obs_size() * max(history, 1), arrays["W0"].shape
    with torch.no_grad():
        act = actor(torch.from_numpy(obs)).numpy()
    meta = {"checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
            "contract_sha256": sha256(CONTRACT_JSON),
            "obs": "ang_vel 3, gravity 3, command 3, pos 12, vel 12, prev_action 12 (contract.py)",
            "activation": "elu", "omni": bool(omni), "history": int(history),
            "kind": kind, "band_deg": [float(v) for v in band], "limits": bounds is not None, "bounds": bounds,
            "stand_offset_abd_deg": float(stand_offset_abd_deg), "action_filter_alpha": alpha,
            "hidden": [int(arrays[f"W{i}"].shape[0]) for i in range(len(lin) - 1)]}
    if norm is not None:
        meta["obs_normalizer"] = {"folded": True, "eps": obs_norm.EPS}
    if getup_phase_s:
        if kind != "getup":
            raise ValueError("getup_phase_s needs kind 'getup'")
        meta["getup_phase_s"] = float(getup_phase_s)
    path = out / "policy.npz"
    np.savez(path, n_layers=len(lin), scale=np.asarray(scale, dtype=np.float32), meta=json.dumps(meta),
             fixture_obs=obs, fixture_act=act, **arrays)
    p = NumpyPolicy(path)
    err = float(np.abs(p(obs) - act).max())
    assert err < fixture_tolerance(act), err
    return path


def with_meta(src_dir, out_dir, **updates) -> Path:
    """A copy of an exported policy (``src_dir/policy.npz``, the same arrays) with meta keys replaced (e.g. ``bounds``,
    ``limits``, ``stand_offset_abd_deg``: a stand change without retraining); the source's hash is recorded as
    ``source_policy_sha256``."""
    from mp4_recovery.paths import sha256
    src = Path(src_dir) / "policy.npz"
    d = dict(np.load(src, allow_pickle=False))
    meta = json.loads(str(d.pop("meta")))
    if "bounds" in updates:
        Bounds.from_meta(updates["bounds"])
    if "action_filter_alpha" in updates:
        updates = {**updates, "action_filter_alpha": policy_filter_alpha(updates)}
    meta.update(updates)
    meta["source_policy_sha256"] = sha256(src)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "policy.npz"
    np.savez(path, meta=json.dumps(meta), **d)
    p = NumpyPolicy(path)
    assert float(np.abs(p(d["fixture_obs"]) - d["fixture_act"]).max()) < fixture_tolerance(d["fixture_act"])
    return path
