"""Clone searched open-loop get-ups into the get-up network (45 in, 12 out, the task's action scale) by behaviour
cloning, from several starts into one policy. ``scripts/recipes/getup.sh`` shows the command to use.

    python -m mp4_recovery.gpu.getup_bc --seq left=SEARCH_LEFT.json --seq right=SEARCH_RIGHT.json --seq nose=SEARCH_NOSE.json \
        --seq back=SEARCH_BACK.json --band 30 90 90 --phase-s 4 --hidden 512 256 128 --obs-norm --out runs/getup_bc/clone

Sources: ``--seq START=PATH[:mirror]`` (repeatable; START left, right, nose, back; several JSONs per start allowed;
``:mirror`` plays the sequence left <-> right; without any right source the left ones are mirrored for it). Each JSON (``getup_search``) is played in its own band and scale: target = clip(a x its scale, +-its
band), re-expressed in the policy's action units (target / the policy's scale); the policy's band (``--config``'s, else
``--band``, else every source's band joint by joint at its widest) must hold every source's. ``trial_s`` is the JSON's
trial length (default 6 s).

Data: the get-up env (the task's observation noise at its end amplitudes, randomised plants, the stiff contacts; the
limits as the run's options say), ``--robots`` robots per source and ``--rounds`` rounds: the start with the orientation
perturbed by U(+-``--perturb``) deg in roll and pitch (left: roll +90; right: roll -90; nose: pitch +75; back: roll 180),
playing the source's best candidate; and the training slices (crouch, partial) playing action 0 (the
stand). Only the trials that meet the success criterion (without the gyro: tilt, joint rms and the base held 0.5 s) are
kept: (actor observation, the action sent). Each start's samples are repeated to the largest start's count, so every
searched start weighs the same; the slices'
samples are added unweighted.

The network (an ELU MLP, ``--hidden``; with ``--obs-norm`` behind an observation normaliser whose mean and std are the
training set's, ``mp4_recovery.obs_norm``) is fitted by mean squared error; the checkpoint holds ``actor_state_dict`` as
rsl_rl names it, with a ``config.json`` beside it (``--config`` copied, else written from the options) so ``export_policy``
and ``train --init-actor`` take it. The clone is then replayed closed loop from every start (and ``--eval-extra``, e.g.
midfall) on fresh draws: the per-start replay success, with and without the gyro. The state's ambiguity is reported: over
the posture steps (action not 0), the actions of each sample's nearest neighbour in the standardised observation, for
the closest 10 % of pairs.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv

from mp4_recovery import obs_norm
from mp4_recovery.gpu.actor import mlp
from mp4_recovery.gpu.env_cfg import describe, noise_profile
from mp4_recovery.gpu.getup_task import (EXACT, SLICES, GetupOptions, band12, getup_scale, goal_now,
                                        make_getup_env_cfg, mirror_joints, policy_filter_alpha_of)
from mp4_recovery.eval.getup_eval import start_term
from mp4_recovery.gpu.getup_search import DT, TRIAL_S, max_seq_s, unpack

STARTS = ("left", "right", "nose", "back")
PREV = slice(33, 45)                    # the previous action in the 45-float observation (contract.py)
HOLD_STEPS = int(round(0.5 / DT))


def build(robots: int, seed: int, band, phase_s: float = 0.0, limits: bool = False,
          action_scale=None, filter_alpha: float = 0.0, widen_limits_deg: float | None = None) -> ManagerBasedRlEnv:
    """``filter_alpha``: the decoder's target filter (``--config``'s option ``filter_alpha``; 0: none), so the
    sources are replayed and the clone runs through the decoder the policy will have."""
    opt = GetupOptions(num_envs=robots, noisy=True, noise_ramp=False, randomise=True,
                       noise_end=noise_profile(), band_deg=tuple(band), fixed_starts=("random",), episode_s=1.0e3,
                       phase_s=phase_s, **({"action_scale": tuple(action_scale)} if action_scale else {}),
                       limits=limits, widen_limits_deg=widen_limits_deg, filter_alpha=float(filter_alpha))
    cfg = make_getup_env_cfg(opt)
    cfg.seed = seed
    return ManagerBasedRlEnv(cfg=cfg, device="cuda:0")


def set_exact(env, roll_pitch, perturb_deg: float):
    term = start_term(env)
    n, dev = env.num_envs, env.device
    ex = torch.tensor(roll_pitch, device=dev).expand(n, 2).clone()
    ex += torch.deg2rad((2 * torch.rand(n, 2, device=dev) - 1) * perturb_deg)
    term.fixed_slice, term._exact = None, ex
    term._fixed_random = torch.zeros(n, dtype=torch.bool, device=dev)


def set_slice(env, name: str):
    term = start_term(env)
    term._exact, term._fixed_random = None, None
    term.fixed_slice = torch.full((env.num_envs,), SLICES.index(name), dtype=torch.long, device=env.device)


def source_actions(saved: dict, mirror: bool, scale, band) -> tuple[torch.Tensor, torch.Tensor]:
    """A search JSON's best sequence in the policy's action units: (postures (k, 12), holds (k,) s). The targets are
    the source's (clip(a x its scale, +-its band), mirrored in radians with ``mirror``), divided by the policy's
    ``scale``; each must lie within the policy's ``band`` (deg)."""
    seq = saved["best"]
    k = len(seq["holds_s"])
    x = torch.tensor(np.concatenate([np.asarray(seq["postures"]).reshape(-1), seq["holds_s"]]),
                     dtype=torch.float32)[None]
    post, hold = unpack(x, k, max_seq_s(float(saved.get("trial_s", TRIAL_S))))
    sband = torch.deg2rad(torch.tensor(band12(saved["band_deg"]), dtype=torch.float32))
    sscale = torch.tensor(saved.get("scale") or getup_scale(saved["band_deg"]), dtype=torch.float32)
    rad = torch.maximum(torch.minimum(post[0] * sscale, sband), -sband)
    if mirror:
        perm, sign = mirror_joints()
        rad = rad[:, perm] * torch.tensor(sign, dtype=torch.float32)
    pband = torch.deg2rad(torch.tensor(band12(band), dtype=torch.float32))
    excess = float((rad.abs() - pband).max())
    assert excess <= 1e-4, f"a source target lies {np.degrees(excess):.1f} deg outside the policy's band: widen --band"
    return rad / torch.tensor(scale, dtype=torch.float32), hold[0]


@torch.no_grad()
def trial(env, act_fn, seconds: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """One trial from the reset: (obs (T, n, 45), actions (T, n, 12), keep (T, n): every step of the trials that
    succeeded, info: per env success (no gyro / gyro) and its time). ``act_fn(step, obs) -> (n, 12)``."""
    n, dev = env.num_envs, env.device
    obs, _ = env.reset()
    T = int(round(seconds / DT))
    run = torch.zeros(n, dtype=torch.long, device=dev)
    run_g = torch.zeros(n, dtype=torch.long, device=dev)
    t_ok = torch.full((n,), -1, dtype=torch.long, device=dev)
    t_okg = torch.full((n,), -1, dtype=torch.long, device=dev)
    O, A = [], []
    for s in range(T):
        a = act_fn(s, obs["actor"])
        O.append(obs["actor"].clone())
        A.append(a.clone())
        obs, *_ = env.step(a)
        run = torch.where(goal_now(env, None), run + 1, torch.zeros_like(run))
        run_g = torch.where(goal_now(env), run_g + 1, torch.zeros_like(run_g))
        t_ok = torch.where((run >= HOLD_STEPS) & (t_ok < 0), s, t_ok)
        t_okg = torch.where((run_g >= HOLD_STEPS) & (t_okg < 0), s, t_okg)
    ok = t_ok >= 0
    keep = ok[None, :].expand(T, -1)
    return torch.stack(O), torch.stack(A), keep, {"success": ok, "success_gyro": t_okg >= 0,
                                                  "t_success": (t_ok.float() + 1) * DT}


def sequence_fn(post: torch.Tensor, hold: torch.Tensor, n: int, dev):
    post, ends = post.to(dev), hold.to(dev).cumsum(-1)
    k = len(ends)

    def act(s, _obs):
        i = int((ends <= s * DT).sum())
        return (post[i] if i < k else torch.zeros(12, device=dev)).expand(n, 12)
    return act


PLAYED_SLICES = ("crouch", "partial")           # the training slices played with action 0


def parse_sources(items: list[str]) -> list[dict]:
    out = []
    for item in items:
        name, path = item.split("=", 1)
        mirror = path.endswith(":mirror")
        path = path[: -len(":mirror")] if mirror else path
        assert name in STARTS, (name, STARTS)
        saved = json.loads(Path(path).read_text())
        out.append({"start": name, "path": path, "mirror": mirror, "saved": saved,
                    "trial_s": float(saved.get("trial_s", TRIAL_S))})
    if not any(s["start"] == "right" for s in out):
        out += [dict(s, start="right", mirror=not s["mirror"], auto=True) for s in out if s["start"] == "left"]
    return out


def balance(groups: dict[str, int], gen: torch.Generator) -> dict[str, torch.Tensor]:
    """Per group (its sample count), the indices into it, repeated to the largest count (with a random remainder)."""
    nz = {g: c for g, c in groups.items() if c > 0}
    if not nz:
        return {g: torch.arange(c) for g, c in groups.items()}
    target = max(nz.values())
    out = {}
    for g, c in groups.items():
        if c == 0:
            out[g] = torch.zeros(0, dtype=torch.long)
            continue
        reps = [torch.arange(c)] * (target // c)
        rest = torch.randperm(c, generator=gen)[: target % c]
        out[g] = torch.cat(reps + [rest])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seq", action="append", default=[], help="START=search.json[:mirror] (left, right, nose, back; "
                                                               "repeatable); without right: the left ones mirrored")
    ap.add_argument("--config", help="a get-up run's config.json (band, scale, phase, options); else --band etc.")
    ap.add_argument("--band", type=float, nargs="+", help="without --config: the policy's band (3 or 12 deg; default "
                                                          "the sources' widest per joint)")
    ap.add_argument("--phase-s", type=float, default=0.0, help="without --config: the phase input's period (s)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--robots", type=int, default=512)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--perturb", type=float, default=10.0)
    ap.add_argument("--eval-extra", default="", help="slices also replayed closed loop (e.g. midfall,crouch)")
    ap.add_argument("--eval-robots", type=int, help="robots per closed-loop start (default --robots)")
    ap.add_argument("--epochs", type=int, default=30,
                    help="passes over the training set (batch 4096; a small set needs more, about 16 k gradient steps)")
    ap.add_argument("--prev-action-noise", type=float, default=0.0, metavar="S",
                    help="training: N(0, S) added to the observed previous action (action units) of every sample: a "
                         "held posture's action equals its previous action, and a clone that copies it holds whatever "
                         "pose it starts in (the copycat problem)")
    ap.add_argument("--std", type=float, default=0.1)
    ap.add_argument("--widen-limits-deg", type=float, metavar="DEG",
                    help="hip and knee targets may go DEG past their track ends (train's option; --config's by default)")
    ap.add_argument("--hidden", type=int, nargs="+", default=[128, 128, 64],
                    help="the actor's hidden sizes (train --actor-hidden)")
    ap.add_argument("--obs-norm", action="store_true",
                    help="an observation normaliser (rsl_rl's EmpiricalNormalization) in front of the net, its mean and "
                         "std from the BC dataset, saved in the checkpoint (train --obs-norm --init-actor loads it; "
                         "export folds it into the first layer)")
    ap.add_argument("--seed", type=int, default=21)
    a = ap.parse_args()
    sources = parse_sources(a.seq)
    assert sources, "no --seq"
    if a.config:
        cfg_json = json.loads(Path(a.config).read_text())
        copt = cfg_json["task"]["options"]
        band, phase_s = band12(copt["band_deg"]), float(copt.get("phase_s", 0.0) or 0.0)
        scale, limits = tuple(copt["action_scale"]), bool(copt.get("limits", False))
        filter_alpha = policy_filter_alpha_of(copt)
        widen = copt.get("widen_limits_deg")
        if a.widen_limits_deg is not None and a.widen_limits_deg != widen:
            raise SystemExit(f"--widen-limits-deg {a.widen_limits_deg}, --config says {widen}")
    else:
        band = band12(a.band) if a.band else tuple(float(max(v)) for v in zip(*[band12(s["saved"]["band_deg"])
                                                                                  for s in sources]))
        phase_s, scale, limits = a.phase_s, getup_scale(band), a.widen_limits_deg is not None
        filter_alpha, widen = 0.0, a.widen_limits_deg
    torch.manual_seed(a.seed)
    env = build(a.robots, a.seed, band, phase_s, limits, scale, filter_alpha, widen)
    n, dev = env.num_envs, env.device
    data = {}                                  # group -> [(obs, act)]
    stats = {}
    for src in sources:
        post, hold = source_actions(src["saved"], src["mirror"], scale, band)
        key = f"{src['start']}:{Path(src['path']).stem}" + (":mirror" if src["mirror"] else "")
        for r in range(a.rounds):
            set_exact(env, EXACT[src["start"]], a.perturb)
            O, A, keep, info = trial(env, sequence_fn(post, hold, n, dev), src["trial_s"])
            data.setdefault(src["start"], []).append((O[keep].cpu(), A[keep].cpu()))
            stats.setdefault(key, []).append(float(info["success"].float().mean()))
        print("replay", key, "success per round", stats[key], flush=True)
    for sl in PLAYED_SLICES:
        for r in range(a.rounds):
            set_slice(env, sl)
            O, A, keep, info = trial(env, lambda s, o: torch.zeros(n, 12, device=dev), TRIAL_S)
            data.setdefault(sl, []).append((O[keep].cpu(), A[keep].cpu()))
            stats.setdefault(sl, []).append(float(info["success"].float().mean()))
        print("slice", sl, "success per round", stats[sl], flush=True)
    groups = {g: (torch.cat([o for o, _ in v]), torch.cat([x for _, x in v])) for g, v in data.items()}
    counts = {g: len(v[0]) for g, v in groups.items()}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    gen = torch.Generator().manual_seed(a.seed)
    # validation: 10 % of each group's raw samples, held out before the balance repeats any of them
    split = {g: torch.randperm(c, generator=gen) for g, c in counts.items()}
    val_idx = {g: split[g][: c // 10] for g, c in counts.items()}
    tr_raw = {g: split[g][c // 10:] for g, c in counts.items()}
    searched = [g for g in groups if g in STARTS]
    bal = balance({g: len(tr_raw[g]) for g in searched}, gen)
    idx = {g: tr_raw[g][bal[g]] for g in searched}
    idx.update({g: tr_raw[g] for g in groups if g not in STARTS})
    X = torch.cat([groups[g][0][idx[g]] for g in groups]).to(dev)
    Y = torch.cat([groups[g][1][idx[g]] for g in groups]).to(dev)
    Xv = torch.cat([groups[g][0][val_idx[g]] for g in groups]).to(dev)
    Yv = torch.cat([groups[g][1][val_idx[g]] for g in groups]).to(dev)
    used = {g: int(len(idx[g])) for g in groups}
    print("samples per group (raw)", counts, "(training, balanced)", used, flush=True)
    assert len(X), "no successful trial: nothing to clone"
    # ambiguity: actions of near-duplicate observations (nearest neighbour in the standardised observation)
    moving = torch.nonzero(Y.abs().amax(-1) > 1e-6).flatten()          # the sequences' postures, not the stand
    sub = moving[torch.randperm(len(moving), device=X.device)[:4000]]
    xs = (X - X.mean(0)) / X.std(0).clamp_min(1e-6)
    ambiguity = {}
    if len(sub) > 10:
        d = torch.cdist(xs[sub], xs[sub])
        d.fill_diagonal_(float("inf"))
        nn_d, nn_i = d.min(-1)
        close = nn_d <= torch.quantile(nn_d, 0.1)
        amb = (Y[sub][close] - Y[sub][nn_i[close]]).abs().amax(-1)
        ambiguity = {"nn_close_pairs": int(close.sum()), "action_diff_p50": float(amb.median()),
                     "action_diff_p90": float(amb.quantile(0.9)), "share_over_0p5": float((amb > 0.5).float().mean())}
    print("ambiguity", ambiguity, flush=True)
    core = mlp(45, 12, tuple(a.hidden)).cuda()
    norm_sd = {}
    if a.obs_norm:
        mean, std = obs_norm.stats(X)
        net = torch.nn.Sequential(obs_norm.Normalize(mean, std), core).cuda()
        norm_sd = obs_norm.state(mean, std, len(X))
    else:
        net = core
    opt = torch.optim.Adam(core.parameters(), lr=1e-3)
    m = len(X)
    tr = torch.arange(m, device=X.device)
    hist = []
    for ep in range(a.epochs):
        ix = tr[torch.randperm(len(tr), device=X.device)]
        for b in range(0, len(ix), 4096):
            j = ix[b:b + 4096]
            xb = X[j]
            if a.prev_action_noise > 0.0:      # the observed previous action perturbed (against copying it)
                xb = xb.clone()
                xb[:, PREV] += a.prev_action_noise * torch.randn_like(xb[:, PREV])
            loss = ((net(xb) - Y[j]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            vl = float(((net(Xv) - Yv) ** 2).mean())
        hist.append(vl)
        if ep % 5 == 0 or ep == a.epochs - 1:
            print("epoch", ep, "val mse", round(vl, 5), flush=True)
    sd = {f"mlp.{k}": v.cpu() for k, v in core.state_dict().items()}
    sd.update(norm_sd)
    sd["distribution.std_param"] = torch.full((12,), a.std)
    torch.save({"actor_state_dict": sd, "bc": {"args": vars(a), "hidden": list(a.hidden), "obs_norm": a.obs_norm}},
               out / "model_bc.pt")
    if a.config:
        shutil.copy(a.config, out / "config.json")
    else:
        opt_ = GetupOptions(band_deg=tuple(band), phase_s=phase_s)
        (out / "config.json").write_text(json.dumps({
            "task": describe(opt_), "noise": "measured",
            "runner": {"actor": {"hidden_dims": list(a.hidden), "obs_normalization": a.obs_norm}}}, indent=1,
            default=str))
    summary = {"args": vars(a), "band_deg": list(band), "action_scale": list(scale), "phase_s": phase_s,
            "filter_alpha": filter_alpha,
            "limits": limits, "sources": [{k: v for k, v in s.items() if k != "saved"} for s in sources],
            "rollout_success": stats, "samples_raw": counts, "samples_used": used, "samples": m,
            "samples_val": int(len(Xv)), "ambiguity": ambiguity, "val_mse": hist, "closed_loop": {}}
    (out / "bc.json").write_text(json.dumps(summary, indent=1))
    print("BC", out / "model_bc.pt", flush=True)
    # closed loop: the clone from every start, fresh draws (bc.json rewritten after each start)
    net.eval()
    if a.eval_robots and a.eval_robots != a.robots:
        env.close()
        env = build(a.eval_robots, a.seed + 1, band, phase_s, limits, scale, filter_alpha, widen)
        n = env.num_envs
    torch.manual_seed(a.seed + 1)
    closed = {}
    pol = lambda s, o: net(o)                                   # noqa: E731
    for start in dict.fromkeys(s["start"] for s in sources):
        tr_s = max(s["trial_s"] for s in sources if s["start"] == start)
        set_exact(env, EXACT[start], a.perturb)
        _, _, _, info = trial(env, pol, tr_s)
        t = info["t_success"][info["success"]]
        res = {"success": float(info["success"].float().mean()),
               "success_gyro": float(info["success_gyro"].float().mean()),
               "t_success_p50": float(t.median()) if len(t) else None}
        closed[start] = res
        print("BC closed loop", start, json.dumps(res), flush=True)
        (out / "bc.json").write_text(json.dumps(dict(summary, closed_loop=closed), indent=1))
    for sl in [s for s in a.eval_extra.split(",") if s]:
        set_slice(env, sl)
        _, _, _, info = trial(env, pol, TRIAL_S)
        closed[sl] = {"success": float(info["success"].float().mean()),
                      "success_gyro": float(info["success_gyro"].float().mean())}
        print("BC closed loop", sl, json.dumps(closed[sl]), flush=True)
        (out / "bc.json").write_text(json.dumps(dict(summary, closed_loop=closed), indent=1))
    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
