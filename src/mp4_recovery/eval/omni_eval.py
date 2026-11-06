#!/usr/bin/env python3
"""Omnidirectional command tracking of a walking policy, command by command.

    python -m mp4_recovery.eval.omni_eval --policy policies/walker --seed 11 --json x.json
    python -m mp4_recovery.eval.omni_eval --policy policies/walker --ground rough --seed 11 --json x.json
    MUJOCO_GL=egl python -m mp4_recovery.eval.omni_eval --policy policies/walker --video top.mp4

Ground ``flat``: 4 m flat tiles (a sideways walk leaves a 1.6 m tile within about 8 s). ``rough``: the non-stair cells of
the rough mix (``--levels``), 1.6 m tiles: a trial's window ends at its first end (a fall, or the robot leaving its cell),
and the ended share is reported. Each command (``COMMANDS``: walks every 45 deg at 0.10 m/s, slow and fast walks, turns,
spins at +-1.0 rad/s, backward 0.05 m/s and the stand) runs on ``--per`` robots, from reset, for ``--seconds``, measured after
a 2 s settle: per command the mean body-frame speed along and across the commanded direction over the commanded speed,
the share of robots moving (along > half the command), the yaw rate, the signed heading change, the signed lateral
offset across the commanded direction in the start frame (mm, left positive), falls and early ends.

Open loop by default; ``--heading-loop``: the robot's PI loop (cap 0.4 rad/s) on the commands without a turn. ``--video``:
one robot, top-down, the commands of ``VIDEO_COMMANDS`` in sequence (4 s each), the transitions included."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import torch

from mp4_recovery.eval import terrain_eval as te
from mp4_recovery.contract import BAND_DEG, policy_band_deg
from mp4_recovery.limits import add_eval_args, eval_plant
from mp4_recovery.gpu.env_cfg import noise_profile
from mp4_recovery.heading import pi_step

DT = 0.02
SETTLE_S = 2.0
LOOP_WZ_MAX = 0.4
WALKS = tuple(
    [(round(0.10 * math.cos(math.radians(a)), 4), round(0.10 * math.sin(math.radians(a)), 4), 0.0)
     for a in range(0, 360, 45)]
    + [(0.05, 0.0, 0.0), (0.12, 0.0, 0.0), (0.0, 0.12, 0.0), (0.0, -0.12, 0.0),
       (0.0, 0.0, 0.4), (0.0, 0.0, -0.4), (0.10, 0.0, 0.4), (0.0, 0.10, -0.4)])
COMMANDS = WALKS + ((0.0, 0.0, 1.0), (0.0, 0.0, -1.0), (-0.05, 0.0, 0.0), (0.0, 0.0, 0.0))
VIDEO_COMMANDS = ((0.0, 0.0, 0.0), (0.10, 0.0, 0.0), (-0.10, 0.0, 0.0), (0.0, 0.10, 0.0), (0.0, -0.10, 0.0),
                  (0.0707, 0.0707, 0.0), (-0.0707, -0.0707, 0.0), (0.0, 0.0, 0.6), (0.0, 0.0, -0.6), (0.0, 0.0, 0.0))
ROUGH_CELL_HALF = 0.75              # m: a 1.6 m cell, less 5 cm
ROUGH_FAMILIES = ("flat", "low_poly", "raised_relief", "slope_up", "slope_down", "boxes")


def cmd_name(c) -> str:
    return f"{c[0]:+.3f},{c[1]:+.3f},{c[2]:+.2f}"


def env_kwargs(pol, n: int, ground: str, render: bool = False, tilt_limit: float = 1.0, limits: bool | None = None,
               stand_offset_abd: float | None = None):
    """The plant's options; the decoder's band is the policy's (meta ``band_deg``) when it is not the walking band;
    the joint limits and the stance offset the flags' (``limits``, ``stand_offset_abd``) or the policy's meta's
    (``limits.eval_plant``)."""
    kw = dict(num_envs=n, noisy=True, noise_ramp=False, noise_end=noise_profile(), randomise=True,
              fixed_command=(0.0, 0.0, 0.0), level=0, action_scale=pol.scale,
              history=int(pol.meta.get("history", 0) or 0), foot_solref=(0.01, 1.0), tilt_limit=float(tilt_limit),
              visuals=render)
    kw.update(eval_plant(pol.meta, limits, stand_offset_abd))
    band = policy_band_deg(pol.meta)
    if not np.array_equal(band, BAND_DEG):
        kw["band_deg"] = tuple(float(b) for b in band)
    if ground == "flat":
        kw.update(terrain="flat", rows=2, tile=4.0, terrain_seed=0)
    else:
        kw.update(terrain="rough", rows=10, terrain_seed=1)
    return kw


def build(args, pol, n: int, episode_s: float, render: bool = False):
    kw = env_kwargs(pol, n, args.ground, render, tilt_limit=getattr(args, "tilt_limit", 1.0),
                    limits=getattr(args, "limits", None), stand_offset_abd=getattr(args, "stand_offset_abd", None))
    kw["episode_s"] = episode_s
    opt = te.TerrainOptions(**kw)
    env = te.build(opt, render=render, seed=args.seed, camera=(0.9, -89.0, 90.0) if render else None)
    if args.ground == "flat":
        cells = [(r, 0) for r in range(2)]
    else:
        names = list(te.sub_terrains(opt))
        cols = [names.index(f) for f in ROUGH_FAMILIES if f in names]
        cells = [(r, c) for c in cols for r in args.levels]
    return env, cells


def pin_commands(env, C):
    """Each env's command, written at every resample (resets included) so the first observation already carries it;
    ``fixed_command`` cannot (its vy range is 0)."""
    cmd = env.command_manager.get_term("twist")
    orig = cmd._resample_command

    def rs(env_ids):
        orig(env_ids)
        cmd.vel_command_b[env_ids] = C[env_ids]
    cmd._resample_command = rs
    return cmd


def start_noise(env, seed: int):
    """The start jig's tolerance, as terrain_eval's ``--start-noise 0.01 2``: returns the start yaw."""
    robot = env.scene["robot"]
    n, dev = env.num_envs, env.device
    g = torch.Generator(device=dev).manual_seed(int(seed) + 7919)
    dy = (2 * torch.rand(n, device=dev, generator=g) - 1) * 0.01
    dpsi = (2 * torch.rand(n, device=dev, generator=g) - 1) * math.radians(2.0)
    pose = torch.cat([robot.data.root_link_pos_w, robot.data.root_link_quat_w], -1).clone()
    yaw0 = te.yaw_of(robot.data.root_link_quat_w)
    pose[:, 1] += dy
    pose[:, 3:7] = te.quat_mul(te.yaw_quat(dpsi), pose[:, 3:7])
    robot.write_root_link_pose_to_sim(pose)
    return te.wrap(yaw0 + dpsi)


@torch.no_grad()
def rollout(env, obs, pol, cmd, C_of_k, steps: int, yaw0, loop: bool, frames=None, cell_half: float | None = None):
    """Steps the policy with the command ``C_of_k(k)`` (n, 3); the heading loop (if ``loop``) on the envs whose
    commanded wz is 0, re-latched to the current heading whenever their command changes. Positions and yaw are frozen at
    each env's first termination. Returns per-step arrays and the fall / end flags."""
    robot = env.scene["robot"]
    n, dev = env.num_envs, env.device
    origin = env.scene.env_origins.clone()
    fell = torch.zeros(n, dtype=torch.bool, device=dev)
    done = torch.zeros_like(fell)
    px = torch.zeros(n, device=dev)
    py = torch.zeros(n, device=dev)
    pyaw = yaw0.clone()
    target = yaw0.clone()
    integ = torch.zeros(n, device=dev)
    prev = None
    X, Y, YAW, ALIVE = [], [], [], []
    for k in range(steps):
        C = C_of_k(k)
        yaw = te.yaw_of(robot.data.root_link_quat_w)
        if prev is not None:
            ch = (C != prev).any(-1)
            target = torch.where(ch, yaw, target)
            integ = torch.where(ch, torch.zeros_like(integ), integ)
        prev = C
        wz = C[:, 2].clone()
        if loop:
            w, integ = pi_step(te.wrap(target - yaw), integ, DT, wz_max=LOOP_WZ_MAX)
            still = C[:, 2].abs() < 1e-6
            wz = torch.where(still, w, wz)
            integ = torch.where(still, integ, torch.zeros_like(integ))
            target = torch.where(still, target, yaw)
        cmd.vel_command_b[:, :2] = C[:, :2]
        cmd.vel_command_b[:, 2] = wz
        obs, _, term, trunc, _ = env.step(pol(obs["actor"]))
        live = ~(fell | done)
        fell |= live & term & ~trunc
        done |= live & trunc
        p = robot.data.root_link_pos_w - origin
        if cell_half is not None:            # rough cells: the trial ends where the robot leaves its cell
            done |= live & ~fell & ((p[:, 0].abs() > cell_half) | (p[:, 1].abs() > cell_half))
        keep = ~(fell | done)
        px = torch.where(keep, p[:, 0], px)
        py = torch.where(keep, p[:, 1], py)
        pyaw = torch.where(keep, te.yaw_of(robot.data.root_link_quat_w), pyaw)
        X.append(px.clone())
        Y.append(py.clone())
        YAW.append(pyaw.clone())
        ALIVE.append(keep.clone())
        if frames is not None:
            frames.append(env.render())
    st = lambda L: torch.stack(L).cpu().numpy()   # noqa: E731
    return st(X), st(Y), np.unwrap(st(YAW), axis=0), st(ALIVE), fell.cpu().numpy(), done.cpu().numpy()


def body_vel(x, y, yaw):
    dx, dy = np.diff(x, axis=0), np.diff(y, axis=0)
    ym = 0.5 * (yaw[1:] + yaw[:-1])
    return (np.cos(ym) * dx + np.sin(ym) * dy) / DT, (-np.sin(ym) * dx + np.cos(ym) * dy) / DT


def command_metrics(c, ids, x, y, yaw, alive, fell, done, yaw0, k0):
    """One command's metrics over its envs, each over its live window from the settle to its first end (an env that
    ended within 1 s of the settle is left out of the means)."""
    vbx, vby = body_vel(x, y, yaw)
    live = alive[1:, :]                                          # step k's velocity uses positions k, k+1
    r = {"command": list(c), "n": int(len(ids)), "fall": float(fell[ids].mean()), "ended": float(done[ids].mean())}
    w = live[k0:, ids].astype(float)
    nlive = w.sum(0)
    ok = nlive >= int(1.0 / DT)
    r["n_measured"] = int(ok.sum())
    if not ok.any():
        return r
    ids_ok, w, nlive = ids[ok], w[:, ok], nlive[ok]
    vx = (vbx[k0:, ids_ok] * w).sum(0) / nlive
    vy = (vby[k0:, ids_ok] * w).sum(0) / nlive
    last = k0 + nlive.astype(int)                                # the window's end, per env
    dyaw = yaw[last, ids_ok] - yaw[k0, ids_ok]
    T = nlive * DT
    sp = math.hypot(c[0], c[1])
    if sp > 1e-6:
        u = np.array(c[:2]) / sp
        along = vx * u[0] + vy * u[1]
        cross = -vx * u[1] + vy * u[0]
        c0, s0 = np.cos(yaw0[ids_ok]), np.sin(yaw0[ids_ok])
        ex = x[last, ids_ok] - x[k0, ids_ok]
        ey = y[last, ids_ok] - y[k0, ids_ok]
        fx, fy = c0 * u[0] - s0 * u[1], s0 * u[0] + c0 * u[1]
        r.update(along=float(along.mean() / sp), along_mps=float(along.mean()), cross=float(cross.mean() / sp),
                 cross_mps=float(cross.mean()), moving=float((along > 0.5 * sp).mean()),
                 lat_mm=float(np.mean(-ex * fy + ey * fx) * 1000.0), lat_mm_per_m=float(
                     np.mean((-ex * fy + ey * fx) / np.maximum(np.abs(ex * fx + ey * fy), 0.05)) * 1000.0))
    else:
        r["drift_mps"] = float(np.hypot(vx, vy).mean())
    r.update(yaw_rate=float((dyaw / T).mean()), heading_deg=float(np.degrees(dyaw).mean()),
             heading_abs_deg=float(np.degrees(np.abs(dyaw)).mean()))
    return r


def run_commands(args, pol):
    m = len(COMMANDS)
    n = m * args.per
    env, cells = build(args, pol, n, args.seconds + 5.0)
    torch.manual_seed(args.seed)
    C = torch.tensor([COMMANDS[i % m] for i in range(n)], dtype=torch.float32, device=env.device)
    cmd = pin_commands(env, C)
    env.reset()
    te.assign_cells(env, [cells[(i // m) % len(cells)] for i in range(n)] if args.ground == "rough" else cells)
    obs, _ = env.reset()
    yaw0 = start_noise(env, args.seed)
    steps = int(round(args.seconds / DT))
    t0 = time.time()
    x, y, yaw, alive, fell, done = rollout(env, obs, pol, cmd, lambda k: C, steps, yaw0, args.heading_loop,
                                           cell_half=ROUGH_CELL_HALF if args.ground == "rough" else None)
    k0 = int(round(SETTLE_S / DT))
    y0 = yaw0.cpu().numpy()
    res = {cmd_name(c): command_metrics(c, np.arange(j, n, m), x, y, yaw, alive, fell, done, y0, k0)
           for j, c in enumerate(COMMANDS)}
    env.close()
    return res, time.time() - t0


def run_video(args, pol):
    import imageio
    env, cells = build(args, pol, 1, len(VIDEO_COMMANDS) * 4.0 + 6.0, render=True)
    torch.manual_seed(args.seed)
    seg = int(round(4.0 / DT))
    V = torch.tensor(VIDEO_COMMANDS, dtype=torch.float32, device=env.device)
    cmd = pin_commands(env, V[:1])
    env.reset()
    te.assign_cells(env, cells[:1])
    obs, _ = env.reset()
    yaw0 = start_noise(env, args.seed)
    frames = []
    rollout(env, obs, pol, cmd, lambda k: V[min(k // seg, len(V) - 1)].unsqueeze(0), seg * len(V), yaw0,
            args.heading_loop, frames=frames)
    env.close()
    Path(args.video).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(args.video, frames, fps=int(1 / DT))
    return {"video": args.video, "commands_4s_each": [list(c) for c in VIDEO_COMMANDS]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--policy", required=True)
    ap.add_argument("--ground", choices=("flat", "rough"), default="flat")
    ap.add_argument("--levels", type=int, nargs="+", default=[2, 4, 6], help="rough: the levels")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--per", type=int, default=24, help="robots per command")
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--heading-loop", action="store_true", help="the robot's loop on the commands without a turn")
    ap.add_argument("--video", help="one robot, top-down, VIDEO_COMMANDS 4 s each (mp4 path)")
    ap.add_argument("--tilt-limit", type=float, default=1.0, help="rad: the tilt that ends a trial as a fall")
    ap.add_argument("--json")
    add_eval_args(ap)
    args = ap.parse_args(argv)
    pol = te.TorchPolicy(Path(args.policy) / "policy.npz", "cuda:0")
    head = {"policy": str(args.policy), "seed": args.seed, "ground": args.ground, "heading_loop": args.heading_loop,
            "per": args.per, "seconds": args.seconds, "levels": args.levels if args.ground == "rough" else None,
            "tilt_limit_rad": args.tilt_limit, "band_deg": policy_band_deg(pol.meta).tolist(),
            **eval_plant(pol.meta, args.limits, args.stand_offset_abd)}
    if args.video:
        out = {**head, **run_video(args, pol)}
    else:
        r, wall = run_commands(args, pol)
        out = {**head, "mode": "commands", "commands": r, "wall_s": wall}
        for k, v in r.items():
            print(f"  {k:24s} " + " ".join(f"{a}={b:.3f}" if isinstance(b, float) else f"{a}={b}"
                                             for a, b in v.items() if a not in ("command",)))
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
