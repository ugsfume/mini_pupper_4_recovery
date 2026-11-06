"""The actor's mean net: an ELU MLP as rsl_rl lays it out (``mlp.{0,2,...}``), and the reference to a checkpoint's."""
from __future__ import annotations

import torch
import torch.nn as nn


def mlp(inp: int, out: int, hidden: tuple) -> nn.Sequential:
    layers, prev = [], inp
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.ELU()]
        prev = h
    layers.append(nn.Linear(prev, out))
    return nn.Sequential(*layers)


def load_actor(path, device: str = "cuda:0") -> nn.Sequential:
    """The checkpoint's actor mean net (any hidden sizes); with an observation normaliser (``obs_norm``) the net
    starts with it, so it acts on raw observations as the rsl_rl actor does."""
    from mp4_recovery import obs_norm
    ck = torch.load(path, map_location=device, weights_only=False)
    sd = ck["actor_state_dict"]
    w = {k.removeprefix("mlp."): v for k, v in sd.items() if k.startswith("mlp.")}
    hid = obs_norm.hidden_dims(sd)
    net = mlp(w["0.weight"].shape[1], w[f"{2 * len(hid)}.weight"].shape[0], hid).to(device)
    net.load_state_dict(w, strict=True)
    norm = obs_norm.read(sd)
    if norm is not None:
        net = nn.Sequential(obs_norm.Normalize(*norm), *net).to(device)
    net.eval()
    return net
