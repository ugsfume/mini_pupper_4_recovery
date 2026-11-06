"""Search an open-loop get-up: a sequence of key postures found by the cross-entropy method (CEM).

    python -m mp4_recovery.gpu.getup_search --limits --start left --json OUT.json
    python -m mp4_recovery.gpu.getup_search --limits --start back --band 90 150 150 --json OUT.json
    python -m mp4_recovery.gpu.getup_search --limits --replay OUT.json --start right --mirror --robots 256 --seed 12 --json VAL.json
    MUJOCO_GL=egl python -m mp4_recovery.gpu.getup_search --replay OUT.json --start left --robots 1 --seed 12 --video X.mp4 --json VAL.json

A candidate = K postures (12 joint targets each, in action units: target = stand + clip(a, +-1) x scale, inside the
band) with a hold time each (0.3-1.5 s; at most the trial less 1.5 s in all), then the stand (action 0) to the end of the
trial (6 s; ``--trial-s``, 8 s for the back). It is played through the get-up env's action path (the band, the 20
deg/step limit, the firmware's quantisation; the decoder seeded from the measured pose) on randomised robots (friction,
center of mass, mass, delay; the stiff contacts and leg collisions) from a fixed start (yaw random, joints the stand +
U(+-0.5 band), base 0.06-0.10 m: ``getup_eval``'s starts).

Score = the share of robots meeting the success criterion (tilt < 12 deg, joint rms < 10 deg from the stand, base above
0.05 m, held 0.5 s; the gyro condition reported, not scored) minus ``--current-cost`` x the mean time any servo is above
0.8 A (BAM, |tau| / kt) as a fraction of the trial, plus ``--shaping`` x the best uprightness reached. CEM: ``--pop``
candidates x ``--robots`` robots per generation (pop x robots <= 1024 envs), the best ``--elite`` fraction refits a
diagonal Gaussian (smoothed), the best candidate ever kept. Generation 0 holds random postures known to reach a stand
from this start (``data/getup_postures_ids.json``), scaled to this band.

``--replay`` evaluates a saved sequence (``--mirror``: left <-> right); with ``--motion`` it measures the motion quality and
impact of every robot as ``getup_eval --motion`` does (``motion_meter``), into each sequence's result.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv

from mp4_recovery.gpu.env_cfg import noise_profile
from mp4_recovery.gpu.getup_task import (GetupOptions, getup_scale, goal_now, make_getup_env_cfg, mirror_joints,
                                        tilt_cos, tilt_deg)
from mp4_recovery.eval.getup_eval import set_start
from mp4_recovery.eval.motion_meter import MotionMeter
from mp4_recovery.eval.terrain_eval import bam_currents

DT = 0.02
TRIAL_S = 6.0
HOLD_S = (0.3, 1.5)
MAX_SEQ_S = 4.5             # the stand gets >= 1.5 s at the end: the 0.5 s hold and some settling
END_S = TRIAL_S - MAX_SEQ_S  # a longer trial (--trial-s) keeps these 1.5 s at its end
HIGH_A = 0.8
POSTURE_IDS = Path(__file__).resolve().parents[1] / "data/getup_postures_ids.json"
SEED_POSES = ("left_side", "right_side", "back", "nose_down")
SEED_BANDS = {"b20_60": (20, 60, 60), "b45_90": (45, 90, 90), "b90_150": (90, 150, 120), "b150": (150, 150, 150)}
SEED_START = {"left": "left_side", "right": "right_side", "back": "back", "nose": "nose_down"}


LIMITS = {"limits": False,     # --limits: the measured joint ranges and the get-up decoder's clamps (limits.py)
          "filter_alpha": 0.0}  # --filter-alpha: the decoder's target filter (0: none)


def build(robots: int, seed: int, band, render: bool = False) -> ManagerBasedRlEnv:
    opt = GetupOptions(num_envs=robots, noisy=False, noise_ramp=False, randomise=True,
                       noise_end=noise_profile(), band_deg=tuple(band), fixed_starts=("random",),
                       visuals=render, episode_s=1.0e3, limits=LIMITS["limits"],
                       filter_alpha=float(LIMITS["filter_alpha"]))
    cfg = make_getup_env_cfg(opt)
    cfg.seed = seed
    if render:
        cfg.viewer.width, cfg.viewer.height = 960, 544
        cfg.viewer.distance, cfg.viewer.elevation, cfg.viewer.azimuth = 0.5, -20.0, 135.0
        cfg.viewer.env_idx, cfg.viewer.max_extra_envs = 0, 0
    return ManagerBasedRlEnv(cfg=cfg, device="cuda:0", render_mode="rgb_array" if render else None)


def posture_seeds(start: str, band, k: int) -> list[np.ndarray]:
    """Random posture sequences that reached a stand from this start (every band but 20/60), in this band's action
    units (target rad / this scale, clipped to +-1), the first ``k`` postures, each held 0.75 s."""
    if start not in SEED_START or not POSTURE_IDS.exists():
        return []
    ids = json.loads(POSTURE_IDS.read_text())["success_ids"]
    conds = [(p, b) for p in SEED_POSES for b in SEED_BANDS]
    nc, per = len(conds), 256
    n = nc * per
    g = torch.Generator(device="cuda:0").manual_seed(11 + 1)       # the ids index this generator's draws
    torch.rand(n, 12, 2, device="cuda:0", generator=g)
    torch.rand(n, 12, 2, device="cuda:0", generator=g)
    torch.rand(n, 12, 2, device="cuda:0", generator=g)
    post = (2 * torch.rand(n, 8, 12, device="cuda:0", generator=g) - 1).cpu().numpy()
    scale = np.asarray(getup_scale(band))
    out = []
    for b, bdeg in SEED_BANDS.items():
        if b == "b20_60":
            continue
        for i in ids.get(f"{SEED_START[start]}|{b}", []):
            assert conds[i % nc] == (SEED_START[start], b)
            rad = post[i, :k] * np.radians(np.asarray(bdeg * 4, dtype=float))[None]
            a = np.clip(rad / scale[None], -1.0, 1.0)
            out.append(np.concatenate([a.reshape(-1), np.full(k, 0.75)]))
    return out


def max_seq_s(trial_s: float = TRIAL_S) -> float:
    return trial_s - END_S


def unpack(x: torch.Tensor, k: int, max_seq: float = MAX_SEQ_S):
    """(n, k*12 + k) -> postures (n, k, 12) clipped to +-1, holds (n, k) clipped to HOLD_S, scaled to ``max_seq``
    (``MAX_SEQ_S`` for the 6 s trial; ``max_seq_s(trial_s)``)."""
    post = x[:, :k * 12].reshape(-1, k, 12).clamp(-1.0, 1.0)
    hold = x[:, k * 12:].clamp(*HOLD_S)
    total = hold.sum(-1, keepdim=True)
    hold = torch.where(total > max_seq, hold * max_seq / total, hold)
    return post, hold


@torch.no_grad()
def evaluate(env, x: torch.Tensor, k: int, start: str, robots: int, seed: int, frames: list | None = None,
             mirror: bool = False, trial_s: float = TRIAL_S,
             meter: MotionMeter | None = None) -> dict:
    """Play each of the len(x) candidates on ``robots`` consecutive envs for ``trial_s``; per candidate: success
    share (the criterion with the height, no gyro), with the gyro too, the median time of success, the mean time above
    0.8 A, and the mean best uprightness reached (``upright_max``: (1 + max cos tilt) / 2). ``meter``: a
    ``MotionMeter`` on ``env`` fed every step (the caller finishes it with ``t_success``)."""
    n, dev = env.num_envs, env.device
    pop = x.shape[0]
    assert pop * robots == n, (pop, robots, n)
    post, hold = unpack(x.to(dev), k, max_seq_s(trial_s))
    if mirror:
        perm, sign = mirror_joints()
        post = post[..., perm] * torch.tensor(sign, device=dev)
    cand = torch.arange(n, device=dev) // robots
    ends = hold.cumsum(-1)[cand]                                         # (n, k)
    set_start(env, start)
    torch.manual_seed(seed)
    env.reset()
    if meter is not None:
        meter.begin(env)
    hold_steps = int(round(0.5 / DT))
    run = torch.zeros(n, dtype=torch.long, device=dev)
    run_g = torch.zeros(n, dtype=torch.long, device=dev)
    t_succ = torch.full((n,), -1.0, device=dev)
    t_succ_g = torch.full((n,), -1.0, device=dev)
    high = torch.zeros(n, device=dev)
    prog_max = torch.zeros(n, device=dev)
    for s in range(int(round(trial_s / DT))):
        t = s * DT
        idx = (ends <= t).sum(-1)                                        # the posture being held (k: the stand)
        a = torch.where((idx < k)[:, None], post[cand, idx.clamp(max=k - 1)], torch.zeros(n, 12, device=dev))
        env.step(a)
        if meter is not None:
            meter.update(env)
        if frames is not None:
            frames.append(env.render())
        ok = goal_now(env, None)
        okg = goal_now(env)
        run = torch.where(ok, run + 1, torch.zeros_like(run))
        run_g = torch.where(okg, run_g + 1, torch.zeros_like(run_g))
        t_succ = torch.where((run >= hold_steps) & (t_succ < 0), torch.full_like(t_succ, t + DT), t_succ)
        t_succ_g = torch.where((run_g >= hold_steps) & (t_succ_g < 0), torch.full_like(t_succ_g, t + DT), t_succ_g)
        c = bam_currents(env)
        if c is not None:
            high += (c > HIGH_A).any(-1).float() * DT
        prog_max = torch.maximum(prog_max, (tilt_cos(env) + 1.0) / 2.0)
    succ = (t_succ >= 0).float().view(pop, robots)
    return {"success": succ.mean(-1).cpu().numpy(), "success_gyro": (t_succ_g >= 0).float().view(pop, robots)
            .mean(-1).cpu().numpy(), "t_success": t_succ.view(pop, robots).cpu().numpy(),
            "time_above_0p8A": high.view(pop, robots).mean(-1).cpu().numpy(),
            "final_tilt_p50": tilt_deg(env).view(pop, robots).median(-1).values.cpu().numpy(),
            "upright_max": prog_max.view(pop, robots).mean(-1).cpu().numpy()}


def search(a) -> dict:
    k, d = a.k, a.k * 13
    trial_s = a.trial_s
    env = build(a.pop * a.robots, a.seed, a.band)
    dev = env.device
    rng = torch.Generator(device=dev).manual_seed(a.seed)
    seeds = posture_seeds(a.start, a.band, k)
    if a.init:                                       # a saved search's best and mean (mirrored with --init-mirror)
        saved = json.loads(Path(a.init).read_text())
        perm, sign = mirror_joints()
        init = []
        for which in ("best", "mean"):
            post = np.asarray(saved[which]["postures"], dtype=float)
            assert post.shape[0] == k, (post.shape, k)
            if a.init_mirror:
                post = post[:, perm] * np.asarray(sign)[None]
            init.append(np.concatenate([post.reshape(-1), saved[which]["holds_s"]]))
        seeds = init + seeds
    mu = torch.zeros(d, device=dev)
    mu[k * 12:] = 0.75
    sigma = torch.full((d,), 0.6, device=dev)
    sigma[k * 12:] = 0.4
    if a.init:
        mu = torch.tensor(seeds[0], dtype=torch.float32, device=dev)
        sigma = sigma * 0.5
    elif seeds:
        s = torch.tensor(np.stack(seeds), dtype=torch.float32, device=dev)
        mu[:k * 12] = s[:, :k * 12].mean(0)
    best, best_score, hist = None, -1e9, []
    n_elite = max(2, int(round(a.elite * a.pop)))
    for gen in range(a.gens):
        t0 = time.time()
        x = mu + sigma * torch.randn(a.pop, d, device=dev, generator=rng)
        if gen == 0 and seeds:
            m = min(len(seeds), a.pop // 2)
            x[:m] = torch.tensor(np.stack(seeds[:m]), dtype=torch.float32, device=dev)
        if best is not None:
            x[0] = best                                                   # re-evaluated on new spawns
        x[:, :k * 12] = x[:, :k * 12].clamp(-1.0, 1.0)
        x[:, k * 12:] = x[:, k * 12:].clamp(*HOLD_S)
        r = evaluate(env, x, k, a.start, a.robots, a.seed * 1000 + gen, trial_s=trial_s)
        score = torch.tensor(r["success"] - a.current_cost * r["time_above_0p8A"] / trial_s
                             + a.shaping * r["upright_max"], device=dev)
        order = torch.argsort(score, descending=True)
        if best is not None:
            best_score = float(score[0])                                  # the kept best, scored afresh
        if float(score[order[0]]) > best_score or best is None:
            best_score, best = float(score[order[0]]), x[order[0]].clone()
        el = x[order[:n_elite]]
        mu = a.smooth * el.mean(0) + (1 - a.smooth) * mu
        sigma = (a.smooth * el.std(0) + (1 - a.smooth) * sigma).clamp_min(a.sigma_min)
        h = {"gen": gen, "best": float(score[order[0]]), "elite_mean": float(score[order[:n_elite]].mean()),
             "pop_mean_success": float(r["success"].mean()), "best_success": float(r["success"][int(order[0])]),
             "best_upright_max": float(r["upright_max"][int(order[0])]),
             "best_ever": best_score, "sigma_post": float(sigma[:k * 12].mean()), "s": round(time.time() - t0, 1)}
        hist.append(h)
        print(json.dumps(h), flush=True)
    env.close()
    post, hold = unpack(best[None], k, max_seq_s(trial_s))
    mpost, mhold = unpack(mu[None], k, max_seq_s(trial_s))
    return {"start": a.start, "trial_s": trial_s, "band_deg": list(a.band),
            "scale": list(getup_scale(a.band)), "k": k, "pop": a.pop,
            "robots": a.robots, "seed": a.seed, "gens": a.gens, "current_cost": a.current_cost, "posture_seeds": len(seeds),
            "shaping": a.shaping, "init": a.init, "init_mirror": a.init_mirror,
            "best": {"postures": post[0].cpu().numpy().round(4).tolist(), "holds_s": hold[0].cpu().numpy().round(3)
                     .tolist(), "score_seed": best_score},
            "mean": {"postures": mpost[0].cpu().numpy().round(4).tolist(), "holds_s": mhold[0].cpu().numpy()
                     .round(3).tolist()},
            "history": hist}


def replay(a) -> dict:
    saved = json.loads(Path(a.replay).read_text())
    trial_s = a.trial_s or saved.get("trial_s", TRIAL_S)
    out = {"replay": a.replay, "start": a.start, "mirror": a.mirror, "robots": a.robots, "seed": a.seed,
           "trial_s": trial_s}
    for which in ("best", "mean"):
        seq = saved[which]
        k = len(seq["holds_s"])
        x = torch.tensor(np.concatenate([np.asarray(seq["postures"]).reshape(-1), seq["holds_s"]]),
                         dtype=torch.float32)[None]
        band = saved["band_deg"]
        frames = [] if (a.video and which == "best") else None
        env = build(a.robots, a.seed, band, render=frames is not None)
        meter = MotionMeter(env) if a.motion else None
        r = evaluate(env, x.cuda(), k, a.start, a.robots, a.seed, frames=frames, mirror=a.mirror,
                     trial_s=trial_s, meter=meter)
        if meter is not None:
            mres = meter.finish(r["t_success"].reshape(-1))
            meter.detach()
            out["motion"] = meter.config()
        env.close()
        t = r["t_success"][0]
        out[which] = {"success": float(r["success"][0]), "success_gyro": float(r["success_gyro"][0]),
                      "t_success_p50": float(np.median(t[t >= 0])) if (t >= 0).any() else None,
                      "time_above_0p8A_mean_s": float(r["time_above_0p8A"][0]),
                      "final_tilt_p50": float(r["final_tilt_p50"][0])}
        print(which, json.dumps(out[which]), flush=True)
        if meter is not None:
            out[which]["motion"] = mres["summary"]
            out[which]["motion_trials"] = mres["trials"]
            g = mres["summary"]["getup"]
            print("motion", which, json.dumps({k: g[k]["mean"] for k in ("rev5", "rev10", "pow3", "da2", "speed_rms")}),
                  flush=True)
        if frames is not None:
            import imageio.v2 as imageio
            imageio.mimsave(a.video, frames, fps=int(1 / DT))
            out["video"] = a.video
            if a.video.endswith(".mp4"):
                sheet = [frames[i] for i in range(0, len(frames), 25)][:12]
                rows = [np.concatenate(sheet[j:j + 4], 1) for j in range(0, len(sheet), 4)]
                w = rows[0].shape[1]
                rows = [np.pad(r_, ((0, 0), (0, w - r_.shape[1]), (0, 0))) for r_ in rows]
                imageio.imwrite(a.video[:-4] + "_sheet.png", np.concatenate(rows, 0)[::2, ::2])
                out["sheet"] = a.video[:-4] + "_sheet.png"
        if a.robots == 1:
            break
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", required=True, choices=("left", "right", "back", "nose"))
    ap.add_argument("--band", type=float, nargs=3, default=(90.0, 90.0, 90.0))
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--pop", type=int, default=64)
    ap.add_argument("--robots", type=int, default=16)
    ap.add_argument("--gens", type=int, default=25)
    ap.add_argument("--elite", type=float, default=0.125)
    ap.add_argument("--smooth", type=float, default=0.7)
    ap.add_argument("--sigma-min", type=float, default=0.05)
    ap.add_argument("--current-cost", type=float, default=0.1)
    ap.add_argument("--shaping", type=float, default=0.0,
                    help="score += this x the robots' mean best uprightness reached, (1 + max cos tilt) / 2 (partial "
                         "credit where no candidate succeeds, e.g. from the back)")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--replay", help="a search JSON: evaluate its best and mean sequences")
    ap.add_argument("--init", help="a search JSON whose best sequence is the search's mean (its best and mean seed gen 0)")
    ap.add_argument("--init-mirror", action="store_true", help="--init mirrored left <-> right")
    ap.add_argument("--mirror", action="store_true")
    ap.add_argument("--video")
    ap.add_argument("--trial-s", type=float,
                    help="s per trial (default 6; 8 for the back); the sequence may use all but the last 1.5 s "
                         "(--replay: the JSON's)")
    ap.add_argument("--json", required=True)
    ap.add_argument("--motion", action="store_true",
                    help="--replay: measure motion quality and impact (motion_meter, as getup_eval --motion)")
    ap.add_argument("--limits", action="store_true",
                    help="the measured joint ranges in the model and the get-up decoder's clamps (limits.py)")
    ap.add_argument("--filter-alpha", type=float, default=0.0, metavar="X",
                    help="the decoder's target low-pass filter alpha (ServoActionCfg.filter_alpha; 0: none)")
    a = ap.parse_args()
    LIMITS["limits"] = a.limits
    LIMITS["filter_alpha"] = a.filter_alpha
    if not a.replay:
        a.trial_s = a.trial_s or TRIAL_S
    if a.replay:
        out = replay(a)
    else:
        assert a.pop * a.robots <= 1024
        out = search(a)
    out["limits"] = bool(a.limits)
    out["filter_alpha"] = float(a.filter_alpha)
    Path(a.json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.json).write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
