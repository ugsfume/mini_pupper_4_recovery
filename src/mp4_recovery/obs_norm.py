"""Empirical observation normalisation for a policy network: statistics from the clone's data, frozen early in PPO,
folded into the first layer at export.

rsl_rl's ``EmpiricalNormalization`` (rsl-rl-lib 5.0.1) maps x -> (x - mean) / (std + eps) with eps 1e-2 (std = sqrt of the
running, biased variance); an ``MLPModel`` built with ``obs_normalization=True`` keeps it as ``obs_normalizer`` (state-dict
keys ``obs_normalizer._mean`` (1, n), ``_var``, ``_std``, ``count``) and updates it on every env step. This module writes and
reads that state, folds it into the first linear layer (W' = W / (std + eps), b' = b - W' mean: the exported numpy policy
keeps its plain format), builds the torch reference of a checkpoint's actor, and freezes the statistics for a run's first
iterations.
"""
from __future__ import annotations

import torch
from torch import nn

EPS = 1.0e-2                 # rsl_rl EmpiricalNormalization's default eps
PREFIX = "obs_normalizer."
STEPS_PER_ITER = 24          # update_normalization calls per PPO iteration


def stats(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(mean, std) over the batch axis as rsl_rl accumulates them (biased variance)."""
    x = x.reshape(-1, x.shape[-1]).double()
    return x.mean(0).float(), x.var(0, unbiased=False).sqrt().float()


def state(mean: torch.Tensor, std: torch.Tensor, count: int) -> dict:
    """The ``obs_normalizer.*`` entries of an rsl_rl actor state dict with these statistics."""
    mean, std = mean.detach().float().cpu().reshape(1, -1), std.detach().float().cpu().reshape(1, -1)
    return {PREFIX + "_mean": mean, PREFIX + "_var": std ** 2, PREFIX + "_std": std,
            PREFIX + "count": torch.tensor(int(count), dtype=torch.long)}


def read(sd: dict) -> tuple[torch.Tensor, torch.Tensor] | None:
    """(mean, std) from an actor state dict, None without a normaliser."""
    if PREFIX + "_mean" not in sd:
        return None
    return sd[PREFIX + "_mean"].reshape(-1).float(), sd[PREFIX + "_std"].reshape(-1).float()


def fold(w0, b0, mean, std, eps: float = EPS):
    """The first layer acting on raw observations: W' = W / (std + eps), b' = b - W' mean (numpy or torch, float64
    inside)."""
    import numpy as np
    w, b = np.asarray(w0, dtype=np.float64), np.asarray(b0, dtype=np.float64)
    m, s = np.asarray(mean, dtype=np.float64).reshape(-1), np.asarray(std, dtype=np.float64).reshape(-1)
    w2 = w / (s + eps)[None, :]
    return w2.astype(np.float32), (b - w2 @ m).astype(np.float32)


class Normalize(nn.Module):
    """x -> (x - mean) / (std + eps), as rsl_rl's normaliser in eval mode."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor, eps: float = EPS):
        super().__init__()
        self.register_buffer("mean", mean.detach().float().reshape(-1).clone())
        self.register_buffer("std", std.detach().float().reshape(-1).clone())
        self.eps = float(eps)

    def forward(self, x):
        return (x - self.mean) / (self.std + self.eps)


def hidden_dims(sd: dict) -> tuple[int, ...]:
    """The hidden sizes of an rsl_rl MLP actor (``mlp.<k>.weight``, Linear layers at even k)."""
    lin = sorted(int(k.split(".")[1]) for k in sd if k.startswith("mlp.") and k.endswith(".weight"))
    return tuple(int(sd[f"mlp.{i}.weight"].shape[0]) for i in lin[:-1])


def freeze_updates(model, iters: int, steps_per_iter: int = STEPS_PER_ITER) -> dict:
    """The model's normaliser statistics are not updated for its first ``iters`` iterations of this process
    (``iters`` < 0: never). Wraps ``model.update_normalization``; returns the counter (``calls``, ``skipped``)."""
    orig = model.update_normalization
    limit = None if iters < 0 else int(iters) * steps_per_iter
    st = {"calls": 0, "skipped": 0, "limit": limit}

    def update_normalization(obs):
        st["calls"] += 1
        if limit is None or st["calls"] <= limit:
            st["skipped"] += 1
            return
        orig(obs)

    model.update_normalization = update_normalization
    return st
