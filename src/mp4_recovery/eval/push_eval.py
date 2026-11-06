#!/usr/bin/env python3
"""Push recovery of an exported policy on flat ground: one velocity kick per trial, or repeated kicks (``--schedule``).

    python -m mp4_recovery.eval.push_eval --policy policies/walker --seed 11 --json x.json
    python -m mp4_recovery.eval.push_eval --policy policies/walker --seed 11 --schedule 0 1.2 2 5 --json x.json
    python -m mp4_recovery.eval.push_eval --policy policies/walker --seed 11 --schedule 0 0.6 2 5 0.6 1.5 8 15 --json x.json
    python -m mp4_recovery.eval.push_eval --policy policies/walker --video x.mp4 --video-cell stand:from_right:1.0

Plant and ground as ``omni_eval`` (``env_kwargs``/``build``): 4 m flat tiles, randomised
plants, the measured observation noise, foot solref (0.01, 1), tilt limit 1.0 rad (``--tilt-limit``), the policy's band (meta
``band_deg``), training's pushes off; the env is seeded before
it is built (the plants are drawn in the constructor). The robot's heading loop (``heading.pi_step``, cap 0.4 rad/s)
holds the start heading (start jig noise as ``omni_eval``); every command has wz = 0. Each env is frozen at its first
termination (a fall: tilt or body contact; or a truncation), so nothing after a fall reads the respawn.

Single kicks (default): per cell (command x direction x magnitude) ``--robots`` robots, one velocity kick: Δv added to
the base's linear velocity, horizontal, in the body's yaw frame at the angle of the direction (``from_back`` = +x, the
push comes from behind; ``from_right`` = +y; every 360/``--dirs`` degrees), at a random time U(``--kick-window``) s;
trials of ``--seconds``. Per command also a yaw-torque cell (``--yaw-torque`` N m for ``--yaw-dur`` s, random sign, at
the same random time) and an unkicked reference (the same random time is drawn and used as the origin of the recovery
clock: the recovery metric's false-negative rate). Commands: stand (0, 0, 0), fwd (0.10, 0, 0), left (0, 0.10, 0),
right (0, -0.10, 0), back (-0.10, 0, 0). Cells are packed into batches of at most ``--max-envs`` envs on one env
(the same plants in every batch: cells at the same position in their batch share plants).

Per trial: fell (a termination that is not a time-out, after the kick) and its cause (1 tilt, 2 body contact, 3 both),
a fall before the kick, the kick time, peak tilt from the kick, time to recovered from the kick (RECOVERY), the
displacement in the start frame and the heading change over the trial, the per-leg median measured abduction offset
from the stand after the settle (the stance spread D of a cell = the largest |per-leg median| over its trials' samples,
on flat ground), BAM current |τ|/kt p99 and peak per joint group (abduction, hip, knee) and the
action-rate and joint-acceleration p99 after the settle. ``reference``: the unkicked cells' action rate, joint
acceleration, D and currents pooled over samples per command and over the walking commands (the nervous-gait check).

``--schedule LO HI A B [LO2 HI2 A2 B2]``: 20 s episodes (``--sched-seconds``), kicks with |Δv| ~ U(LO, HI) in a uniform
random horizontal direction every U(A, B) s (each tier its own timer, the first kick at U(A, B) as training's push
term), ``--sched-robots`` robots per command (stand, fwd, left, right): the share of episodes with a fall and the falls
per kick.

``--video PATH --video-cell CMD:DIR:MAG``: one robot, rendered (MUJOCO_GL=egl), the kick at ``--video-kick-t`` s;
recorded until the first termination (the env respawns after it), the last frame then held 1 s."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
import warnings
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np  # noqa: E402
import torch  # noqa: E402

from mp4_recovery.eval import omni_eval as oe  # noqa: E402
from mp4_recovery.eval import terrain_eval as te  # noqa: E402
from mp4_recovery.contract import ABDUCTION  # noqa: E402
from mp4_recovery.heading import pi_step  # noqa: E402
from mp4_recovery.gpu.mdp import servo_action  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)   # all-NaN slices of ended trials

DT = 0.02
SETTLE_S = 2.0                      # s: action rate, joint acceleration and D are measured after it
LOOP_WZ_MAX = oe.LOOP_WZ_MAX        # rad/s: the runner's heading loop cap
COMMANDS = {"stand": (0.0, 0.0, 0.0), "fwd": (0.10, 0.0, 0.0), "left": (0.0, 0.10, 0.0), "right": (0.0, -0.10, 0.0),
            "back": (-0.10, 0.0, 0.0)}
WALK = ("fwd", "left", "right", "back")
SCHED_COMMANDS = ("stand", "fwd", "left", "right")
# Δv direction (deg, body yaw frame, +x forward, +y left) -> where the push comes from
DIR_LABELS = {0: "from_back", 45: "from_back_right", 90: "from_right", 135: "from_front_right", 180: "from_front",
              225: "from_front_left", 270: "from_left", 315: "from_back_left"}
LABEL_DEG = {v: k for k, v in DIR_LABELS.items()}
GROUPS = {"abd": (0, 3, 6, 9), "hip": (1, 4, 7, 10), "knee": (2, 5, 8, 11)}   # contract joint order
RR_ABD = 6
ROBOT_CUR_RATIO = 1.7               # measured: robot / BAM for the rear-right abduction p99
# RECOVERY: the first time s >= the kick at which, over [s, s + REC_HOLD_S), the robot is alive, its tilt stays below
# REC_TILT_DEG and the mean body-frame horizontal velocity is within REC_V of the command ("t_rec"). The strict variant
# ("t_rec_strict") needs the instantaneous velocity within REC_V at every step of the hold (a trot's
# within-stride velocity swing fails it while walking).
REC_V, REC_TILT_DEG, REC_HOLD_S = 0.05, 15.0, 0.5
REC_WITHIN_S = 5.0
VIDEO_HOLD_S = 1.0


def dir_angles(ndirs: int) -> list[float]:
    return [360.0 * i / ndirs for i in range(ndirs)]


def dir_label(deg: float) -> str:
    d = int(round(deg)) % 360
    return DIR_LABELS.get(d, f"dv{d:03d}") if abs(deg - round(deg)) < 1e-6 else f"dv{deg:.1f}"


def parse_dir(text: str) -> float:
    return float(LABEL_DEG[text]) if text in LABEL_DEG else float(text)


def make_cells(commands, ndirs, mags, yaw: bool):
    cells = []
    for c in commands:
        for deg in dir_angles(ndirs):
            for m in mags:
                cells.append({"command": c, "kind": "kick", "dir": dir_label(deg), "dir_deg": deg, "mag": float(m)})
        if yaw:
            cells.append({"command": c, "kind": "yaw", "dir": "yaw", "dir_deg": None, "mag": None})
        cells.append({"command": c, "kind": "none", "dir": "none", "dir_deg": None, "mag": 0.0})
    return cells


def cell_key(c) -> str:
    if c["kind"] == "kick":
        return f"{c['command']}:{c['dir']}:{c['mag']:.2f}"
    return f"{c['command']}:{c['kind']}"


def add_base_velocity(robot, ids, dv_w):
    """Add ``dv_w`` (len(ids), 3; world frame) to the free joint's linear velocity (MuJoCo's qvel[0:3] is the body
    origin's world-frame velocity, as ``write_root_link_velocity_to_sim`` writes it); read straight from qvel, so a
    kick does not depend on derived quantities from the last forward pass."""
    d = robot.data
    adr = torch.as_tensor(d.indexing.free_joint_v_adr[:3], device=dv_w.device, dtype=torch.long)
    q = d.data.qvel
    q[ids.unsqueeze(-1), adr.unsqueeze(0)] = q[ids.unsqueeze(-1), adr.unsqueeze(0)] + dv_w


def kick_world(yaw, deg, mag):
    """The kick's world-frame Δv (n, 3) for body-yaw-frame angles ``deg`` and magnitudes ``mag`` (tensors)."""
    th = torch.deg2rad(deg) + yaw
    return torch.stack([mag * torch.cos(th), mag * torch.sin(th), torch.zeros_like(mag)], -1)


def tilt_deg(robot):
    g = robot.data.projected_gravity_b
    return torch.rad2deg(torch.acos((-g[:, 2]).clamp(-1, 1)))


def make_env(args, pol, n: int, episode_s: float, render: bool = False):
    """omni_eval's flat build (its ``env_kwargs`` and cells); the video uses a side camera instead of its top view."""
    torch.manual_seed(args.seed)
    tilt = getattr(args, "tilt_limit", 1.0)
    lim = dict(limits=getattr(args, "limits", None), stand_offset_abd=getattr(args, "stand_offset_abd", None))
    ns = SimpleNamespace(ground="flat", policy=args.policy, seed=args.seed, levels=None,
                         tilt_limit=tilt, **lim)
    if not render:
        env, cells = oe.build(ns, pol, n, episode_s)
        return env, cells, oe.env_kwargs(pol, n, "flat", tilt_limit=tilt, **lim) | {
            "episode_s": episode_s}
    kw = oe.env_kwargs(pol, n, "flat", render=True, tilt_limit=tilt, **lim)
    kw["episode_s"] = episode_s
    env = te.build(te.TerrainOptions(**kw), render=True, seed=args.seed, camera=tuple(args.camera))
    return env, [(r, 0) for r in range(2)], kw


class Rollout:
    """One batch: fixed commands, a kick per env (velocity, yaw torque or none) or a schedule; frozen at the first
    termination. Records the per-step arrays the metrics need (on the GPU, stacked at the end)."""

    def __init__(self, env, pol, cmd, cells_xy, seed: int):
        self.env, self.pol, self.cmd = env, pol, cmd
        self.C = cells_xy
        self.seed = seed
        self.robot = env.scene["robot"]
        self.base_id = [self.robot.body_names.index("base")]

    @torch.no_grad()
    def run(self, cells_rc, steps: int, kick_k, kick_deg, kick_mag, yaw_on, yaw_sign, yaw_torque, yaw_n,
            sched=None, frames=None, start_seed: int = 0):
        env, robot = self.env, self.robot
        n, dev = env.num_envs, env.device
        env.reset()
        te.assign_cells(env, cells_rc)
        obs, _ = env.reset()
        yaw0 = oe.start_noise(env, start_seed)
        origin = env.scene.env_origins.clone()
        p0 = (robot.data.root_link_pos_w - origin)[:, :2].clone()
        integ = torch.zeros(n, device=dev)
        fell = torch.zeros(n, dtype=torch.bool, device=dev)
        done = torch.zeros_like(fell)
        cause = torch.zeros(n, dtype=torch.long, device=dev)
        end_k = torch.full((n,), steps, dtype=torch.long, device=dev)   # the step at which the env ended (steps: never)
        kicked = torch.zeros_like(fell)                                 # alive when its kick fired
        px, py, pyaw = p0[:, 0].clone(), p0[:, 1].clone(), yaw0.clone()
        prev_a = torch.zeros(n, 12, device=dev)
        prev_qd = robot.data.joint_vel.clone()
        zero3 = torch.zeros(n, 1, 3, device=dev)
        use_wrench = bool(yaw_on is not None and yaw_on.any())
        rec = {k: [] for k in ("tilt", "vx", "vy", "x", "y", "yaw", "alive", "cur", "abd", "da", "qacc", "qacc_pol")}
        tm = env.termination_manager
        slog = None
        if sched is not None:
            slog = {"k": [], "env": [], "mag": [], "tier": []}
            g = torch.Generator(device=dev).manual_seed(int(self.seed) * 7 + 101)
            nxt = [torch.round((A + (B - A) * torch.rand(n, device=dev, generator=g)) / DT).long()
                   for (_, _, A, B) in sched]
        for k in range(steps):
            live = ~(fell | done)
            yaw = te.yaw_of(robot.data.root_link_quat_w)
            err = te.wrap(yaw0 - yaw)
            w, integ = pi_step(err, integ, DT, wz_max=LOOP_WZ_MAX)
            self.cmd.vel_command_b[:, :2] = self.C[:, :2]
            self.cmd.vel_command_b[:, 2] = w
            if kick_k is not None:
                fire = live & (kick_k == k) & (kick_mag > 0)
                kicked |= live & (kick_k == k)
                ids = torch.nonzero(fire).flatten()
                if len(ids):
                    add_base_velocity(robot, ids, kick_world(yaw[ids], kick_deg[ids], kick_mag[ids]))
            if sched is not None:
                dv = torch.zeros(n, 3, device=dev)
                for t_i, (lo, hi, A, B) in enumerate(sched):
                    due = nxt[t_i] == k
                    fire = live & due
                    if fire.any():
                        m = lo + (hi - lo) * torch.rand(n, device=dev, generator=g)
                        th = 2 * math.pi * torch.rand(n, device=dev, generator=g)
                        dv[:, 0] += torch.where(fire, m * torch.cos(th), torch.zeros_like(m))
                        dv[:, 1] += torch.where(fire, m * torch.sin(th), torch.zeros_like(m))
                        ii = torch.nonzero(fire).flatten()
                        slog["k"] += [k] * len(ii)
                        slog["env"] += ii.tolist()
                        slog["mag"] += m[ii].tolist()
                        slog["tier"] += [t_i] * len(ii)
                    step = torch.round((A + (B - A) * torch.rand(n, device=dev, generator=g)) / DT).long()
                    nxt[t_i] = torch.where(due, k + step.clamp(min=1), nxt[t_i])
                ids = torch.nonzero(dv[:, :2].abs().sum(-1) > 0).flatten()
                if len(ids):
                    add_base_velocity(robot, ids, dv[ids])
            if use_wrench:       # held from the kick step for yaw_n steps, written every step (resets clear it)
                on = live & yaw_on & (k >= kick_k) & (k < kick_k + yaw_n)
                tq = zero3.clone()
                tq[:, 0, 2] = torch.where(on, yaw_sign * yaw_torque, torch.zeros_like(yaw_sign))
                robot.write_external_wrench_to_sim(zero3, tq, body_ids=self.base_id)
            a = self.pol(obs["actor"])
            obs, _, term, trunc, _ = env.step(a)
            new_fall = live & term & ~trunc
            c_now = tm.get_term("tilt").long() + 2 * tm.get_term("body_contact").long()
            cause = torch.where(new_fall, c_now, cause)
            fell |= new_fall
            done |= live & trunc
            end_k = torch.where(live & (term | trunc), torch.full_like(end_k, k), end_k)
            keep = ~(fell | done)
            p = robot.data.root_link_pos_w - origin
            px = torch.where(keep, p[:, 0], px)
            py = torch.where(keep, p[:, 1], py)
            pyaw = torch.where(keep, te.yaw_of(robot.data.root_link_quat_w), pyaw)
            vb = robot.data.root_link_lin_vel_b
            qd = robot.data.joint_vel
            rec["tilt"].append(tilt_deg(robot))
            rec["vx"].append(vb[:, 0].clone())
            rec["vy"].append(vb[:, 1].clone())
            rec["x"].append(px.clone())
            rec["y"].append(py.clone())
            rec["yaw"].append(pyaw.clone())
            rec["alive"].append(keep.clone())
            cur = te.bam_currents(env)
            rec["cur"].append(cur if cur is not None else torch.full((n, 12), float("nan"), device=dev))
            rec["abd"].append(servo_action(env).true_rel()[:, list(ABDUCTION)].clone())
            rec["da"].append((a - prev_a).norm(dim=-1) if k > 0 else torch.full((n,), float("nan"), device=dev))
            rec["qacc"].append(robot.data.joint_acc.abs().clone())
            rec["qacc_pol"].append(((qd - prev_qd) / DT).abs())
            prev_a, prev_qd = a.clone(), qd.clone()
            if frames is not None and bool(keep.all()):   # after a termination the env has respawned: stop there
                frames.append(env.render())
        out = {k: torch.stack(v).cpu().numpy() for k, v in rec.items()}
        out["yaw"] = np.unwrap(out["yaw"], axis=0)
        out.update(fell=fell.cpu().numpy(), done=done.cpu().numpy(), cause=cause.cpu().numpy(),
                   end_k=end_k.cpu().numpy(), kicked=kicked.cpu().numpy(), yaw0=yaw0.cpu().numpy(),
                   p0=p0.cpu().numpy())
        if slog is not None:
            out["sched_log"] = {k: np.array(v) for k, v in slog.items()}
        return out


def first_hold(ok, start, hold):
    """Per env (column), the first row s >= start[e] where ok[s:s+hold] is all True (index into ok), or -1."""
    T, n = ok.shape
    c = np.concatenate([np.zeros((1, n), int), np.cumsum(ok, axis=0)], 0)
    win = (c[hold:] - c[:-hold]) == hold                  # (T-hold+1, n): window starting at s all ok
    s_idx = np.arange(win.shape[0])[:, None]
    win &= s_idx >= start[None, :]
    any_ = win.any(0)
    first = np.where(any_, win.argmax(0), -1)
    return first


def rolling_mean(x, hold):
    c = np.concatenate([np.zeros((1,) + x.shape[1:]), np.cumsum(x, axis=0)], 0)
    return (c[hold:] - c[:-hold]) / hold


def trial_metrics(r, C, kick_k, steps):
    """Per-trial metrics from one batch's records (numpy)."""
    alive = r["alive"]
    T, n = alive.shape
    hold = int(round(REC_HOLD_S / DT))
    k0 = int(round(SETTLE_S / DT))
    kk = np.clip(kick_k, 0, T - 1)
    fall_pre = r["fell"] & (r["end_k"] < kick_k)
    fell_post = r["fell"] & ~fall_pre
    # recovery (from the kick): alive, tilt < REC_TILT_DEG, velocity within REC_V of the command over the hold
    ex, ey = r["vx"] - C[None, :, 0], r["vy"] - C[None, :, 1]
    up = alive & (r["tilt"] < REC_TILT_DEG)
    ok_strict = up & (np.hypot(ex, ey) < REC_V)
    s_strict = first_hold(ok_strict, kk, hold)
    vmx, vmy = rolling_mean(np.where(alive, r["vx"], 0.0), hold), rolling_mean(np.where(alive, r["vy"], 0.0), hold)
    c = np.concatenate([np.zeros((1, n), int), np.cumsum(up, axis=0)], 0)
    up_all = (c[hold:] - c[:-hold]) == hold
    ok_win = up_all & (np.hypot(vmx - C[None, :, 0], vmy - C[None, :, 1]) < REC_V)
    s_idx = np.arange(ok_win.shape[0])[:, None]
    ok_win &= s_idx >= kk[None, :]
    s_mean = np.where(ok_win.any(0), ok_win.argmax(0), -1)
    t_rec = np.where(s_mean >= 0, (s_mean - kk) * DT, np.nan)
    t_rec_strict = np.where(s_strict >= 0, (s_strict - kk) * DT, np.nan)
    after = (np.arange(T)[:, None] >= kk[None, :]) & alive
    tilt_pk = np.where(after.any(0), np.max(np.where(after, r["tilt"], -1.0), axis=0), np.nan)
    xe, ye, yawe = r["x"][-1], r["y"][-1], r["yaw"][-1]     # frozen at the last live step
    dx, dy = xe - r["p0"][:, 0], ye - r["p0"][:, 1]
    c0, s0 = np.cos(r["yaw0"]), np.sin(r["yaw0"])
    disp_x, disp_y = c0 * dx + s0 * dy, -s0 * dx + c0 * dy
    # unwrapped yaw: the record's first row is after step 0; the start yaw yaw0 is wrapped like it
    dyaw = np.degrees(yawe - r["yaw0"] - 2 * math.pi * np.round((r["yaw"][0] - r["yaw0"]) / (2 * math.pi)))
    settled = alive & (np.arange(T)[:, None] >= k0)
    nan = np.nan
    abd = np.where(settled[..., None], np.degrees(r["abd"]), nan)
    abd_med = np.nanmedian(abd, axis=0)                        # (n, 4)
    cur = np.where(alive[..., None], r["cur"], nan)
    cur_p99 = np.stack([np.nanpercentile(cur[:, :, list(g)].transpose(1, 0, 2).reshape(n, -1), 99, axis=1)
                        for g in GROUPS.values()], -1)
    cur_pk = np.stack([np.nanmax(cur[:, :, list(g)].transpose(1, 0, 2).reshape(n, -1), axis=1)
                       for g in GROUPS.values()], -1)
    da = np.where(settled, r["da"], nan)
    da_p99 = np.nanpercentile(da, 99, axis=0)
    qa = np.where(settled[..., None], r["qacc"], nan).transpose(1, 0, 2).reshape(n, -1)
    qa_p99 = np.nanpercentile(qa, 99, axis=1)
    qp = np.where(settled[..., None], r["qacc_pol"], nan).transpose(1, 0, 2).reshape(n, -1)
    qp_p99 = np.nanpercentile(qp, 99, axis=1)
    return {"fell": fell_post, "fell_pre_kick": fall_pre, "cause": np.where(r["fell"], r["cause"], 0),
            "ended": r["done"], "kicked": r["kicked"], "t_end_s": np.where(r["end_k"] < T, (r["end_k"] + 1) * DT, nan),
            "tilt_peak_deg": tilt_pk, "t_rec_s": t_rec, "t_rec_strict_s": t_rec_strict,
            "disp_x_m": disp_x, "disp_y_m": disp_y, "dyaw_deg": dyaw, "abd_off_med_deg": abd_med,
            "cur_p99_A": cur_p99, "cur_peak_A": cur_pk, "action_rate_p99": da_p99, "joint_acc_p99": qa_p99,
            "joint_acc_pol_p99": qp_p99}


def pooled_samples(r, ids):
    """Sample pools of the settled live steps of envs ``ids``: action rate, |joint acc|, abduction offsets, currents."""
    alive = r["alive"][:, ids]
    k0 = int(round(SETTLE_S / DT))
    st = alive.copy()
    st[:k0] = False
    return {"da": r["da"][:, ids][st], "qacc": r["qacc"][:, ids][st].ravel(), "qacc_pol": r["qacc_pol"][:, ids][st].ravel(),
            "abd": np.degrees(r["abd"][:, ids][st]), "cur": r["cur"][:, ids][alive]}


def pool_summary(P):
    if not len(P["da"]):
        return None
    da = P["da"][~np.isnan(P["da"])]
    med = np.nanmedian(P["abd"], axis=0)
    out = {"samples": int(len(P["da"])), "action_rate_p99": float(np.percentile(da, 99)),
           "action_rate_mean": float(np.mean(da)), "joint_acc_p99": float(np.percentile(P["qacc"], 99)),
           "joint_acc_pol_p99": float(np.percentile(P["qacc_pol"], 99)),
           "abd_off_med_deg": [round(float(x), 3) for x in med], "D_deg": float(np.max(np.abs(med)))}
    cur = P["cur"]
    if not np.isnan(cur).all():
        for gname, g in GROUPS.items():
            v = cur[:, list(g)].ravel()
            out[f"cur_{gname}_p99_A"] = float(np.nanpercentile(v, 99))
            out[f"cur_{gname}_peak_A"] = float(np.nanmax(v))
        out["cur_rr_abd_p99_A"] = float(np.nanpercentile(cur[:, RR_ABD], 99))
        out[f"cur_rr_abd_p99_A_x{ROBOT_CUR_RATIO}"] = out["cur_rr_abd_p99_A"] * ROBOT_CUR_RATIO
    return out


def concat_pools(ps):
    return {k: np.concatenate([p[k] for p in ps], 0) for k in ps[0]}


def cell_summary(cell, tr, P):
    """One cell's summary from its trials (dict of arrays) and its sample pool."""
    n = len(tr["fell"])
    k = tr["kicked"].astype(bool)
    nk = int(k.sum())
    fell = tr["fell"].astype(bool)
    ok = k & ~fell                                         # kicked and stood
    trec = tr["t_rec_s"]
    rec = ok & ~np.isnan(trec)
    s = {**cell, "n": n, "n_kicked": nk, "fell_pre_kick": int(tr["fell_pre_kick"].sum()),
         "fall": float(fell[k].mean()) if nk else None, "n_fall": int(fell[k].sum()),
         "fall_tilt": int((fell & (tr["cause"] == 1)).sum()), "fall_contact": int((fell & (tr["cause"] == 2)).sum()),
         "fall_both": int((fell & (tr["cause"] == 3)).sum()), "ended": float(tr["ended"].mean()),
         "tilt_peak_deg_p50": float(np.nanmedian(tr["tilt_peak_deg"][k])) if nk else None,
         "tilt_peak_deg_p90": float(np.nanpercentile(tr["tilt_peak_deg"][k], 90)) if nk else None,
         "tilt_peak_deg_max": float(np.nanmax(tr["tilt_peak_deg"][k])) if nk else None,
         "recovered": float(rec.sum() / ok.sum()) if ok.any() else None, "n_stood": int(ok.sum()),
         "rec5": float((ok & (trec <= REC_WITHIN_S)).sum() / ok.sum()) if ok.any() else None,
         "rec5_all": float((ok & (trec <= REC_WITHIN_S)).sum() / nk) if nk else None,
         "t_rec_p50_s": float(np.median(trec[rec])) if rec.any() else None,
         "t_rec_p90_s": float(np.percentile(trec[rec], 90)) if rec.any() else None,
         "recovered_strict": float((ok & ~np.isnan(tr["t_rec_strict_s"])).sum() / ok.sum()) if ok.any() else None,
         "disp_x_mm_p50": float(1000 * np.median(tr["disp_x_m"][ok])) if ok.any() else None,
         "disp_y_mm_p50": float(1000 * np.median(tr["disp_y_m"][ok])) if ok.any() else None,
         "dyaw_deg_p50": float(np.median(tr["dyaw_deg"][ok])) if ok.any() else None,
         "dyaw_abs_deg_p90": float(np.percentile(np.abs(tr["dyaw_deg"][ok]), 90)) if ok.any() else None}
    ps = pool_summary(P)
    if ps:
        s.update({k_: v for k_, v in ps.items() if k_ != "samples"})
    return s


def setting_record(args, kw, pol, env):
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                            cwd=Path(__file__).resolve().parents[3]).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True,
                           cwd=Path(__file__).resolve().parents[3]).stdout.strip()
    opt = te.TerrainOptions(**kw)
    o = {k: (list(v) if isinstance(v, tuple) else v) for k, v in opt.__dict__.items()
         if k not in ("noise_start", "noise_end")}
    return {"policy": str(args.policy), "policy_meta": {k: v for k, v in pol.meta.items() if k != "checkpoint"},
            "action_scale": list(pol.scale), "seed": args.seed, "repo_commit": commit, "repo_dirty": bool(dirty),
            "argv": sys.argv, "dt": DT, "options": o, "noise": "measured",
            "foot_solref": list(kw["foot_solref"]), "tilt_limit_rad": kw["tilt_limit"], "randomise": kw["randomise"],
            "ground": "flat 4 m tiles (omni_eval)",
            "heading_loop": {"law": "heading.pi_step", "wz_max": LOOP_WZ_MAX, "target": "the start heading"},
            "start_noise": "omni_eval.start_noise (0.01 m, 2 deg)", "settle_s": SETTLE_S,
            "recovery": {"v_mps": REC_V, "tilt_deg": REC_TILT_DEG, "hold_s": REC_HOLD_S, "within_s": REC_WITHIN_S,
                         "t_rec": "mean body velocity over the hold within v of the command, tilt below at every step",
                         "t_rec_strict": "instantaneous body velocity within v at every step of the hold"},
            "termination": "first termination freezes the env; fall = termination not a time-out; cause 1 tilt, "
                           "2 body contact, 3 both",
            "currents": "BAM |torque|/kt per joint (contract order); groups abd/hip/knee; x1.7 = the robot / BAM ratio measured on the rear-right abduction",
            "joint_acc": "|qacc| of the 12 joint dofs at the policy step (physics); joint_acc_pol: |Δqd|/0.02",
            "action_rate": "||a_t - a_t-1||_2 of the raw policy output (12)", "num_envs": env.num_envs}


def to_list(x, nd=4):
    a = np.asarray(x)
    if a.dtype == bool:
        return a.astype(int).tolist()
    if np.issubdtype(a.dtype, np.integer):
        return a.tolist()
    a = np.round(a.astype(float), nd)
    return [None if (isinstance(v, float) and math.isnan(v)) else v for v in a.ravel().tolist()] if a.ndim == 1 else \
        [[None if math.isnan(v) else v for v in row] for row in a.tolist()]


def run_kicks(args, pol):
    cmds = [c.strip() for c in args.commands.split(",")]
    cells = make_cells(cmds, args.dirs, args.mags, not args.no_yaw)
    R = args.robots
    per_batch = max(1, args.max_envs // R)
    nb = math.ceil(len(cells) / per_batch)
    per_batch = math.ceil(len(cells) / nb)
    n = per_batch * R
    steps = int(round(args.seconds / DT))
    t_build = time.time()
    env, cells_rc, kw = make_env(args, pol, n, args.seconds + 5.0)
    t_build = time.time() - t_build
    dev = env.device
    C = torch.zeros(n, 3, device=dev)
    cmd = oe.pin_commands(env, C)
    ro = Rollout(env, pol, cmd, C, args.seed)
    trials = {}
    pools = {}
    t0 = time.time()
    batch_wall = []
    for b in range(nb):
        tb = time.time()
        bc = cells[b * per_batch:(b + 1) * per_batch]
        ci = np.full(n, -1)
        for j in range(len(bc)):
            ci[j * R:(j + 1) * R] = b * per_batch + j
        C[:] = torch.tensor([COMMANDS[cells[c]["command"]] if c >= 0 else (0.0, 0.0, 0.0) for c in ci],
                            dtype=torch.float32, device=dev)
        g = torch.Generator(device=dev).manual_seed(int(args.seed) * 1000 + b)
        lo, hi = args.kick_window
        kick_k = torch.round((lo + (hi - lo) * torch.rand(n, device=dev, generator=g)) / DT).long()
        ysign = torch.where(torch.rand(n, device=dev, generator=g) < 0.5, -1.0, 1.0)
        deg = torch.tensor([cells[c]["dir_deg"] if c >= 0 and cells[c]["kind"] == "kick" else 0.0 for c in ci],
                           dtype=torch.float32, device=dev)
        mag = torch.tensor([cells[c]["mag"] if c >= 0 and cells[c]["kind"] == "kick" else 0.0 for c in ci],
                           dtype=torch.float32, device=dev)
        yaw_on = torch.tensor([c >= 0 and cells[c]["kind"] == "yaw" for c in ci], device=dev)
        r = ro.run(cells_rc, steps, kick_k, deg, mag, yaw_on, ysign, args.yaw_torque,
                   max(1, int(round(args.yaw_dur / DT))), start_seed=int(args.seed) * 100 + b)
        tm = trial_metrics(r, C.cpu().numpy(), kick_k.cpu().numpy(), steps)
        tm["t_kick_s"] = kick_k.cpu().numpy() * DT
        tm["yaw_sign"] = ysign.cpu().numpy().astype(int)
        for j in range(len(bc)):
            c = b * per_batch + j
            ids = np.arange(j * R, (j + 1) * R)
            trials[c] = {k: v[ids] for k, v in tm.items()}
            pools[c] = pooled_samples(r, ids)
        batch_wall.append(time.time() - tb)
        print(f"batch {b + 1}/{nb}: {len(bc)} cells x {R} robots, {time.time() - tb:.0f} s", flush=True)
        del r
    wall = time.time() - t0
    summ = {}
    for c, cell in enumerate(cells):
        summ[cell_key(cell)] = cell_summary(cell, trials[c], pools[c])
    ref = {}
    for cn in cmds:
        c = next(i for i, x in enumerate(cells) if x["command"] == cn and x["kind"] == "none")
        ref[cn] = pool_summary(pools[c])
    walk = [next(i for i, x in enumerate(cells) if x["command"] == cn and x["kind"] == "none") for cn in cmds if cn in WALK]
    if walk:
        ref["walk"] = pool_summary(concat_pools([pools[c] for c in walk]))
    keys = list(trials[0].keys())
    per_trial = {"cell": np.concatenate([np.full(R, c) for c in range(len(cells))]).tolist()}
    for k in keys:
        per_trial[k] = to_list(np.concatenate([trials[c][k] for c in range(len(cells))], 0))
    rec = setting_record(args, kw, pol, env)
    out = {"mode": "kicks", **rec,
           "robots_per_cell": R, "mags": list(args.mags), "dirs": args.dirs, "commands": {c: COMMANDS[c] for c in cmds},
           "kick_window_s": list(args.kick_window), "seconds": args.seconds,
           "yaw_cell": None if args.no_yaw else {"torque_Nm": args.yaw_torque, "dur_s": args.yaw_dur, "sign": "random"},
           "batches": nb, "cells_per_batch": per_batch, "build_s": t_build, "wall_s": wall, "batch_wall_s": batch_wall,
           "cell_list": [cell_key(c) for c in cells], "cells": summ, "reference": ref, "trials": per_trial}
    env.close()
    return out


def run_schedule(args, pol):
    s = args.schedule
    if len(s) not in (4, 8):
        raise SystemExit("--schedule takes LO HI A B [LO2 HI2 A2 B2]")
    tiers = [tuple(s[i:i + 4]) for i in range(0, len(s), 4)]
    cmds = [c.strip() for c in args.sched_commands.split(",")]
    R = args.sched_robots
    n = len(cmds) * R
    if n > args.max_envs:
        raise SystemExit(f"{n} envs > --max-envs {args.max_envs}")
    steps = int(round(args.sched_seconds / DT))
    t_build = time.time()
    env, cells_rc, kw = make_env(args, pol, n, args.sched_seconds + 5.0)
    t_build = time.time() - t_build
    dev = env.device
    C = torch.tensor([COMMANDS[cmds[i // R]] for i in range(n)], dtype=torch.float32, device=dev)
    cmd = oe.pin_commands(env, C)
    ro = Rollout(env, pol, cmd, C, args.seed)
    t0 = time.time()
    r = ro.run(cells_rc, steps, None, None, None, None, None, 0.0, 1, sched=tiers, start_seed=int(args.seed) * 100)
    wall = time.time() - t0
    L = r["sched_log"]
    fell, end_k, cause = r["fell"], r["end_k"], r["cause"]
    per = {}
    edges = np.arange(0.0, max(t[1] for t in tiers) + 0.25, 0.25)
    for j, cn in enumerate(cmds):
        ids = np.arange(j * R, (j + 1) * R)
        m = np.isin(L["env"], ids)
        ke, kk, km, kt = L["env"][m], L["k"][m], L["mag"][m], L["tier"][m]
        # each fall's last kick before it (the kick at the fall's step counts: it fires before the step)
        last_dt, last_mag, last_tier = [], [], []
        for e in ids[fell[ids]]:
            sel = (ke == e) & (kk <= end_k[e])
            if sel.any():
                i = np.argmax(np.where(sel, kk, -1))
                last_dt.append((end_k[e] - kk[i] + 1) * DT)
                last_mag.append(km[i])
                last_tier.append(int(kt[i]))
            else:
                last_dt.append(np.nan)
                last_mag.append(np.nan)
                last_tier.append(-1)
        last_dt, last_mag = np.array(last_dt), np.array(last_mag)
        within2 = np.isfinite(last_dt) & (last_dt <= 2.0)
        # per magnitude bin: kicks, and kicks followed by a fall within 2 s that was attributed to them
        fall_kick = set()
        for e, dtv in zip(ids[fell[ids]], last_dt):
            if np.isfinite(dtv) and dtv <= 2.0:
                sel = (ke == e) & (kk <= end_k[e])
                fall_kick.add((int(e), int(kk[sel].max())))
        bins = []
        for lo_, hi_ in zip(edges[:-1], edges[1:]):
            sel = (km >= lo_) & (km < hi_)
            nb_ = int(sel.sum())
            nf = int(sum((int(e), int(k)) in fall_kick for e, k in zip(ke[sel], kk[sel])))
            if nb_:
                bins.append({"mag_lo": float(lo_), "mag_hi": float(hi_), "kicks": nb_, "falls_within_2s": nf,
                             "rate": nf / nb_})
        alive_s = float(np.sum(np.minimum(end_k[ids] + 1, steps)) * DT)
        per[cn] = {"command": COMMANDS[cn], "n": R, "fall_share": float(fell[ids].mean()), "falls": int(fell[ids].sum()),
                   "kicks": int(m.sum()), "kicks_per_tier": [int((kt == t).sum()) for t in range(len(tiers))],
                   "falls_per_kick": float(fell[ids].sum() / max(m.sum(), 1)),
                   "falls_per_min_alive": float(fell[ids].sum() / max(alive_s / 60.0, 1e-9)),
                   "falls_within_2s_of_kick": int(within2.sum()),
                   "falls_tier": [int(sum(1 for t_ in last_tier if t_ == t)) for t in range(len(tiers))],
                   "fall_cause": {"tilt": int((fell[ids] & (cause[ids] == 1)).sum()),
                                  "contact": int((fell[ids] & (cause[ids] == 2)).sum()),
                                  "both": int((fell[ids] & (cause[ids] == 3)).sum())},
                   "ended": float(r["done"][ids].mean()),
                   "t_fall_s_p50": float(np.median((end_k[ids][fell[ids]] + 1) * DT)) if fell[ids].any() else None,
                   "fall_dt_since_kick_s": to_list(last_dt, 2), "fall_last_kick_mag": to_list(last_mag, 3),
                   "by_mag": bins}
    out = {"mode": "schedule", **setting_record(args, kw, pol, env), "tiers": [list(t) for t in tiers],
           "seconds": args.sched_seconds, "robots_per_command": R, "build_s": t_build, "wall_s": wall,
           "commands": per,
           "trials": {"command": [cmds[i // R] for i in range(n)], "fell": to_list(fell), "cause": to_list(cause),
                      "t_end_s": to_list(np.where(end_k < steps, (end_k + 1) * DT, np.nan), 2)},
           "kicks": {"env": L["env"].tolist(), "t_s": to_list(L["k"] * DT, 2), "mag": to_list(L["mag"], 3),
                     "tier": L["tier"].tolist()}}
    env.close()
    return out


def run_video(args, pol):
    import imageio.v2 as imageio
    cn, d, m = args.video_cell.split(":")
    deg, mag = parse_dir(d), float(m)
    steps = int(round(args.video_seconds / DT))
    env, cells_rc, kw = make_env(args, pol, 1, args.video_seconds + 5.0, render=True)
    dev = env.device
    C = torch.tensor([COMMANDS[cn]], dtype=torch.float32, device=dev)
    cmd = oe.pin_commands(env, C)
    ro = Rollout(env, pol, cmd, C, args.seed)
    kick_k = torch.tensor([int(round(args.video_kick_t / DT))], device=dev)
    frames = []
    r = ro.run(cells_rc[:1], steps, kick_k, torch.tensor([deg], device=dev), torch.tensor([mag], device=dev),
               None, None, 0.0, 1, frames=frames, start_seed=int(args.seed) * 100)
    tm = trial_metrics(r, C.cpu().numpy(), kick_k.cpu().numpy(), steps)
    rec = setting_record(args, kw, pol, env)
    env.close()
    n_live = len(frames)
    frames += [frames[-1]] * int(round(VIDEO_HOLD_S / DT))        # the last live frame held
    Path(args.video).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(args.video, frames, fps=int(1 / DT))
    return {"mode": "video", **rec, "video": str(args.video),
            "cell": {"command": cn, "dir": dir_label(deg), "dir_deg": deg, "mag": mag},
            "kick_t_s": args.video_kick_t, "seconds": args.video_seconds, "camera": list(args.camera),
            "frames_live": n_live, "video_note": f"recorded until the first termination (a fall ends the episode and "
            f"the env respawns), then its last frame held {VIDEO_HOLD_S} s",
            "trial": {k: to_list(v) for k, v in tm.items()}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--policy", required=True)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--json")
    ap.add_argument("--tilt-limit", type=float, default=1.0,
                    help="rad: the tilt that ends a trial as a fall (default 1.0, every result before F2)")
    ap.add_argument("--robots", type=int, default=64, help="robots per cell")
    ap.add_argument("--mags", type=float, nargs="+", default=[0.5, 0.75, 1.0, 1.25, 1.5], help="m/s")
    ap.add_argument("--dirs", type=int, default=8, help="kick directions, every 360/N degrees from +x")
    ap.add_argument("--commands", default="stand,fwd,left,right,back")
    ap.add_argument("--no-yaw", action="store_true", help="no yaw-torque cells")
    ap.add_argument("--yaw-torque", type=float, default=0.12, help="N m")
    ap.add_argument("--yaw-dur", type=float, default=0.3, help="s")
    ap.add_argument("--kick-window", type=float, nargs=2, default=[2.0, 4.0], help="s: the kick time, uniform")
    ap.add_argument("--seconds", type=float, default=10.0, help="trial length")
    ap.add_argument("--max-envs", type=int, default=1024, help="envs per batch (for the GPU's memory)")
    ap.add_argument("--schedule", type=float, nargs="+", metavar="X", help="LO HI A B [LO2 HI2 A2 B2]")
    ap.add_argument("--sched-commands", default=",".join(SCHED_COMMANDS))
    ap.add_argument("--sched-robots", type=int, default=256)
    ap.add_argument("--sched-seconds", type=float, default=20.0)
    ap.add_argument("--video", help="mp4 path: one rendered robot, --video-cell")
    ap.add_argument("--video-cell", default="stand:from_right:1.0", help="CMD:DIR:MAG (DIR a label or degrees)")
    ap.add_argument("--video-kick-t", type=float, default=3.0)
    ap.add_argument("--video-seconds", type=float, default=8.0)
    ap.add_argument("--camera", type=float, nargs=3, default=[0.45, -15.0, 160.0], metavar=("DIST", "ELEV", "AZIM"))
    from mp4_recovery.limits import add_eval_args
    add_eval_args(ap)
    args = ap.parse_args(argv)
    pol = te.TorchPolicy(Path(args.policy) / "policy.npz", "cuda:0")
    if args.video:
        out = run_video(args, pol)
        print("video", out["video"], "fell", out["trial"]["fell"], "t_rec", out["trial"]["t_rec_s"])
    elif args.schedule:
        out = run_schedule(args, pol)
        for k, v in out["commands"].items():
            print(f"  {k:6s} fall_share={v['fall_share']:.3f} kicks={v['kicks']} falls/kick={v['falls_per_kick']:.4f} "
                  f"within2s={v['falls_within_2s_of_kick']}/{v['falls']}")
        print(f"wall {out['wall_s']:.0f} s (build {out['build_s']:.0f} s)")
    else:
        out = run_kicks(args, pol)
        for k, v in out["cells"].items():
            if v["kind"] == "kick" and v["dir"] not in ("from_right", "from_left"):
                continue
            f = lambda x, p=3: "-" if x is None else f"{x:.{p}f}"  # noqa: E731
            print(f"  {k:28s} n={v['n']} fall={f(v['fall'])} rec5={f(v['rec5'], 2)} t90={f(v['t_rec_p90_s'], 2)} "
                  f"tilt50={f(v['tilt_peak_deg_p50'], 1)} D={f(v.get('D_deg'), 1)}")
        print(f"wall {out['wall_s']:.0f} s (build {out['build_s']:.0f} s, {out['batches']} batches)")
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(out))
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
