"""Score a get-up policy in simulation: from fixed fallen starts, does it stand up, how fast, and at what servo current.

    python -m mp4_recovery.eval.getup_eval --policy policies/getup --seed 12 --robots 256 --seconds 8 \
        --starts left,right,nose,back,midfall,crouch,random --motion --json runs/eval/getup_s12.json
    MUJOCO_GL=egl python -m mp4_recovery.eval.getup_eval --policy policies/getup --seed 12 --json OUT --video left.mp4 \
        --video-start left

The get-up task (``gpu.getup_task``) with its plant draws (friction, center of mass, mass, delay) seeded before the env
is built, so every start and every policy meets the same simulated robots; the observation noise at its end amplitudes;
the band and action scale from the policy's ``config.json``. Starts (one batch of ``--robots`` per start, yaw random,
joints the stand + U(+-0.5 band), base 0.06-0.10 m above the floor, drawn again where the spawn overlaps): left = roll +90
deg, right = roll -90, back = roll 180, nose = pitch +75 (nose down), random = the training's full start mix; crouch,
partial, side, midfall = those training slices, drawn as in training. Each trial runs ``--seconds`` (6).

Per trial: success (tilt < 12 deg, |gyro| < 0.8 rad/s, joint rms < 10 deg from the stand, held 0.5 s) and the time it
completed; ``success_with_height`` (the training reward's: also the base above 0.05 m); the time any servo is above 0.8 A,
the peak, and simulated guard trips (a servo above 1 A for 0.5 s) with the currents at x1 and x1.7 (the robot's currents
read up to 1.7x BAM's on one joint); the deepest floor and self penetration over the trial, after the spawn's drop has
landed (``SETTLE_S``), and sustained: the deepest penetration one geom pair holds on every step of ``SUSTAIN_S`` (0.2 s)
after ``SETTLE_S`` (impacts sink a body for a few steps). The JSON holds per-trial arrays and a per-start summary.
``--motion``: the motion quality and impact of every robot (``motion_meter.MotionMeter``), in each start's summary and per
trial; ``eval/thresholds.py`` checks them.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv

from mp4_recovery.gpu.env_cfg import noise_profile
from mp4_recovery.gpu.getup_task import (EXACT, SLICES, GetupOptions, GetupStart, SUCCESS_HOLD_S,
                                        TRACK_Z, _robot_geom_mask, base_z, contact_penetration,
                                        make_getup_env_cfg, pair_penetration, policy_filter_alpha_of,
                                        success_now, tilt_deg)
from mp4_recovery.limits import eval_plant
from mp4_recovery.eval.motion_meter import MotionMeter
from mp4_recovery.eval.terrain_eval import TorchPolicy, bam_currents

DT = 0.02
STARTS = ("left", "right", "back", "nose", "random")
SLICE_STARTS = ("crouch", "partial", "side", "midfall")   # training slices (getup_task.SLICES), drawn as in training
HIGH_A = 0.8
GUARD_A, GUARD_S = 1.0, 0.5
ROBOT_RATIO = 1.7
SETTLE_S = 0.5          # the spawn's drop (0.06-0.10 m) lands within this
SUSTAIN_S = 0.2         # one geom pair's depth held this long, after SETTLE_S


def start_term(env) -> GetupStart:
    for cfg in env.event_manager._mode_class_term_cfgs["reset"]:
        if isinstance(cfg.func, GetupStart):
            return cfg.func
    raise KeyError("no GetupStart reset term")


def set_start(env, name: str) -> None:
    """Every env starts as ``name`` at the next reset: an EXACT key (one orientation, yaw random), a training slice
    (``getup_task.SLICES`` but "random": drawn as in training), or "random" (the full mix)."""
    term = start_term(env)
    n, dev = env.num_envs, env.device
    term.fixed_slice = None
    if name in SLICE_STARTS:
        term._exact = None
        term._fixed_random = None
        term.fixed_slice = torch.full((n,), SLICES.index(name), dtype=torch.long, device=dev)
        term.start_name = [name] * n
        return
    ex = torch.full((n, 2), float("nan"), device=dev)
    if name != "random":
        ex[:] = torch.tensor(EXACT[name], device=dev)
    term._exact = ex
    term._fixed_random = torch.full((n,), name == "random", dtype=torch.bool, device=dev)
    term.start_name = [name] * n


def build(policy_dir: Path, robots: int, seed: int, render: bool = False, contact_solref="run", limits=None,
          filter_alpha=None, device: str = "cuda:0"):
    """``contact_solref``: "run" = the policy's run's.
    ``limits``: the measured joint limits (None: the run's option, else the policy's meta). ``filter_alpha``: the
    decoder's target filter (None: the policy's meta ``action_filter_alpha``, else the run's option, else 0).
    ``device``: the env's (``getup_stream`` builds one robot on the CPU). The widening of the joint limits is the
    policy's (its meta's bounds, ``limits.eval_plant``)."""
    cfg_json = json.loads((policy_dir / "config.json").read_text())
    o = cfg_json["task"]["options"]
    meta = json.loads(str(np.load(policy_dir / "policy.npz")["meta"]))
    if limits is None:
        limits = bool(o.get("limits", False) or meta.get("limits", False))
    if filter_alpha is None:
        filter_alpha = policy_filter_alpha_of(o, meta)
    opt = GetupOptions(num_envs=robots, noisy=True, noise_ramp=False, randomise=True,
                       noise_end=noise_profile(cfg_json.get("noise", "measured")), band_deg=tuple(o["band_deg"]),
                       action_scale=tuple(o["action_scale"]), masses_json=o.get("masses_json"),
                       getup_collisions=o.get("getup_collisions", True), thigh_base=o.get("thigh_base", True),
                       thigh_short_m=o.get("thigh_short_m", 0.008), fixed_starts=("random",), visuals=render,
                       contact_solref=contact_solref if contact_solref != "run" else o.get("contact_solref"),
                       episode_s=1.0e3, phase_s=o.get("phase_s", 0.0) or 0.0, limits=bool(limits),
                       widen_limits_deg=eval_plant(meta)["widen_limits_deg"], filter_alpha=float(filter_alpha))
    cfg = make_getup_env_cfg(opt)
    cfg.seed = seed
    if render:
        cfg.viewer.width, cfg.viewer.height = 960, 540
        cfg.viewer.distance, cfg.viewer.elevation, cfg.viewer.azimuth = 0.5, -20.0, 135.0
        cfg.viewer.env_idx, cfg.viewer.max_extra_envs = 0, 0
    return ManagerBasedRlEnv(cfg=cfg, device=device, render_mode="rgb_array" if render else None), opt


def window_stats(cmax_steps, after_steps, trips_win, t_succ, seconds) -> dict:
    """Current before and after the first success (a trial that never succeeds is all 'before'): the p99 of the
    per-step largest servo current over the window's samples, and the share of trials with a simulated guard trip
    inside the window (x1, x1.7)."""
    if not cmax_steps:
        return {}
    c, a = torch.stack(cmax_steps), torch.stack(after_steps)              # (steps, n)
    out = {"time_before_success_mean_s": float(torch.where(t_succ >= 0, t_succ, torch.full_like(t_succ, seconds))
                                               .mean())}
    for w, mask in (("before", ~a), ("after", a)):
        v = c[mask]
        out[f"current_p99_{w}_A"] = float(v.quantile(0.99)) if v.numel() else None
        out[f"guard_trips_x1_{w}"] = float(trips_win[(1.0, w)].float().mean())
        out[f"guard_trips_x1p7_{w}"] = float(trips_win[(ROBOT_RATIO, w)].float().mean())
    return out


def goal_criteria(env):
    """(the success criterion, with the height, without the gyro) per env: tilt, gyro, joint rms; + the base above 0.05
    m; tilt, rms and height without the gyro."""
    ok = success_now(env)
    up = base_z(env) > TRACK_Z
    return ok, ok & up, success_now(env, None) & up


def run_start(env, pol, name: str, seconds: float, seed: int, frames: list | None = None,
              noise: float = 0.0, meter: MotionMeter | None = None) -> dict:
    set_start(env, name)
    torch.manual_seed(seed)
    obs, _ = env.reset()
    if meter is not None:
        meter.begin(env)
    n, dev = env.num_envs, env.device
    steps = int(round(seconds / DT))
    hold = int(round(SUCCESS_HOLD_S / DT))
    run = torch.zeros(n, dtype=torch.long, device=dev)
    t_succ = torch.full((n,), -1.0, device=dev)
    run_h = torch.zeros(n, dtype=torch.long, device=dev)      # the training's success: + the base above 0.05 m
    t_succ_h = torch.full((n,), -1.0, device=dev)
    run_ng = torch.zeros(n, dtype=torch.long, device=dev)     # tilt, rms and height, no gyro condition
    t_succ_ng = torch.full((n,), -1.0, device=dev)
    tilt_after = torch.zeros(n, device=dev)
    high = torch.zeros(n, device=dev)
    high_after = torch.zeros(n, device=dev)
    peak = torch.zeros(n, device=dev)
    guard = {k: torch.zeros(n, 12, device=dev) for k in (1.0, ROBOT_RATIO)}       # time above 1 A, per servo
    trips = {k: torch.zeros(n, dtype=torch.bool, device=dev) for k in guard}
    trips_win = {(k, w): torch.zeros(n, dtype=torch.bool, device=dev) for k in guard for w in ("before", "after")}
    cmax_steps, after_steps = [], []                   # per step: the largest servo current, success completed
    pen_floor = torch.zeros(n, device=dev)
    pen_self = torch.zeros(n, device=dev)
    pen_late = torch.zeros(n, device=dev)          # floor or self, after the spawn's drop (SETTLE_S)
    settle = int(round(SETTLE_S / DT))
    hold_pen = int(round(SUSTAIN_S / DT))
    window = []                                    # the last hold_pen steps' per-pair penetration
    robot_g = _robot_geom_mask(env)
    selfpair = robot_g[:, None] & robot_g[None, :]
    sus = torch.zeros(n, device=dev)               # deepest penetration of one geom pair held SUSTAIN_S, after settle
    sus_self = torch.zeros(n, device=dev)
    term = start_term(env)
    tilt0 = tilt_deg(env).clone()
    gen = torch.Generator(device=dev).manual_seed(seed + 1)
    for k in range(steps):
        act = pol(obs["actor"])
        if noise > 0.0:          # the training's exploration: Gaussian action noise, std in action units
            act = act + noise * torch.randn(act.shape, device=act.device, generator=gen)
        obs, _, _, _, _ = env.step(act)
        if meter is not None:
            meter.update(env)
        if frames is not None:
            frames.append(env.render())
        ok, ok_h, ok_ng = goal_criteria(env)
        run = torch.where(ok, run + 1, torch.zeros_like(run))
        hit = (run >= hold) & (t_succ < 0)
        t_succ = torch.where(hit, torch.full_like(t_succ, (k + 1) * DT), t_succ)
        run_h = torch.where(ok_h, run_h + 1, torch.zeros_like(run_h))
        t_succ_h = torch.where((run_h >= hold) & (t_succ_h < 0), torch.full_like(t_succ_h, (k + 1) * DT), t_succ_h)
        run_ng = torch.where(ok_ng, run_ng + 1, torch.zeros_like(run_ng))
        t_succ_ng = torch.where((run_ng >= hold) & (t_succ_ng < 0), torch.full_like(t_succ_ng, (k + 1) * DT),
                                t_succ_ng)
        tilt_after = torch.where(t_succ >= 0, torch.maximum(tilt_after, tilt_deg(env)), tilt_after)
        c = bam_currents(env)
        if c is not None:
            over = (c > HIGH_A).any(-1).float() * DT
            high += over
            high_after += over * (t_succ >= 0).float()            # after the success criterion completed
            peak = torch.maximum(peak, c.max(-1).values)
            cmax_steps.append(c.max(-1).values)
            after_steps.append(t_succ >= 0)
            for r in guard:
                guard[r] = torch.where(c * r > GUARD_A, guard[r] + DT, torch.zeros_like(guard[r]))
                trip = (guard[r] >= GUARD_S - 1e-6).any(-1)
                trips[r] |= trip
                trips_win[(r, "before")] |= trip & (t_succ < 0)
                trips_win[(r, "after")] |= trip & (t_succ >= 0)
        f, s = contact_penetration(env)
        pen_floor = torch.maximum(pen_floor, f)
        pen_self = torch.maximum(pen_self, s)
        if k >= settle:
            pen_late = torch.maximum(pen_late, torch.maximum(f, s))
            window.append(pair_penetration(env))
            if len(window) > hold_pen:
                window.pop(0)
            if len(window) == hold_pen:
                held = torch.stack(window).amin(0)                     # (n, G, G): exceeded on every step
                sus = torch.maximum(sus, held.flatten(1).amax(-1))
                sus_self = torch.maximum(sus_self, (held * selfpair).flatten(1).amax(-1))
    succ = (t_succ >= 0).cpu().numpy()
    t = t_succ.cpu().numpy()
    trial = {"success": succ.tolist(), "t_success": t.tolist(),
             "success_with_height": (t_succ_h >= 0).cpu().numpy().tolist(),
             "success_no_gyro": (t_succ_ng >= 0).cpu().numpy().tolist(), "tilt0_deg": tilt0.cpu().numpy().round(2).tolist(),
             "slice": term.slice_id.cpu().numpy().tolist(), "spawn_tries": term.tries.cpu().numpy().tolist(),
             "max_tilt_after_success_deg": tilt_after.cpu().numpy().round(2).tolist(),
             "time_above_0p8A_s": high.cpu().numpy().round(3).tolist(),
             "time_above_0p8A_after_success_s": high_after.cpu().numpy().round(3).tolist(), "peak_A": peak.cpu().numpy().round(4).tolist(),
             "guard_trip_x1": trips[1.0].cpu().numpy().tolist(),
             "guard_trip_x1p7": trips[ROBOT_RATIO].cpu().numpy().tolist(),
             "pen_floor_mm": (pen_floor * 1000).cpu().numpy().round(3).tolist(),
             "pen_self_mm": (pen_self * 1000).cpu().numpy().round(3).tolist(),
             "pen_after_settle_mm": (pen_late * 1000).cpu().numpy().round(3).tolist(),
             "pen_sustained_mm": (sus * 1000).cpu().numpy().round(3).tolist(),
             "pen_sustained_self_mm": (sus_self * 1000).cpu().numpy().round(3).tolist()}
    ta = tilt_after.cpu().numpy()[succ]
    summary = {"start": name, "n": int(n), "success": float(succ.mean()),
               "success_with_height": float((t_succ_h >= 0).float().mean()),
               "success_no_gyro": float((t_succ_ng >= 0).float().mean()),
               "t_success_p50": float(np.median(t[succ])) if succ.any() else None,
               "t_success_p90": float(np.percentile(t[succ], 90)) if succ.any() else None,
               "max_tilt_after_success_p90_deg": float(np.percentile(ta, 90)) if succ.any() else None,
               "time_above_0p8A_mean_s": float(high.mean()), "time_above_0p8A_p90_s": float(high.quantile(0.9)),
               "time_above_0p8A_after_success_mean_s": float(high_after.mean()),
               "time_above_0p8A_before_success_mean_s": float((high - high_after).mean()),
               **window_stats(cmax_steps, after_steps, trips_win, t_succ, seconds),
               "time_after_success_mean_s": float(((seconds - t_succ) * (t_succ >= 0).float()).mean()),
               "peak_A_p50": float(peak.median()), "peak_A_max": float(peak.max()),
               "guard_trips_x1": float(trips[1.0].float().mean()),
               "guard_trips_x1p7": float(trips[ROBOT_RATIO].float().mean()),
               "pen_floor_mm_max": float(pen_floor.max() * 1000), "pen_self_mm_max": float(pen_self.max() * 1000),
               "pen_mm_p99": float(torch.maximum(pen_floor, pen_self).quantile(0.99) * 1000),
               "pen_after_settle_mm_max": float(pen_late.max() * 1000),
               "pen_after_settle_mm_p99": float(pen_late.quantile(0.99) * 1000),
               "pen_sustained_mm_max": float(sus.max() * 1000), "pen_sustained_mm_p99": float(sus.quantile(0.99) * 1000),
               "pen_sustained_self_mm_max": float(sus_self.max() * 1000),
               "spawn_unresolved": int(term.unresolved)}
    if meter is not None:
        m = meter.finish(t)
        summary["motion"] = m["summary"]
        trial["motion"] = m["trials"]
    return {"summary": summary, "trials": trial}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", required=True, help="an exported policy directory (policy.npz + config.json)")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--json", required=True)
    ap.add_argument("--robots", type=int, default=256, help="robots per start (<= 1024)")
    ap.add_argument("--starts", default=",".join(STARTS))
    ap.add_argument("--seconds", "--trial-s", dest="seconds", type=float, default=6.0,
                    help="s per trial")
    ap.add_argument("--action-noise", type=float, default=0.0,
                    help="add N(0, this) to the policy's actions every step (the training's exploration noise)")
    ap.add_argument("--video", help="mp4: one robot (env 0) of --video-start, rendered in its own env")
    ap.add_argument("--video-start", default="left")
    ap.add_argument("--contact-solref", nargs="+", default=["run"],
                    help="the non-foot robot contacts' solref: 'run' (the policy's run's), 'none' (MuJoCo's default) or "
                         "TIMECONST DAMPRATIO (a comparison; the policy was trained with its run's)")
    ap.add_argument("--limits", action="store_true", default=None,
                    help="the measured joint limits (default: the run's option or the policy's meta)")
    ap.add_argument("--no-limits", dest="limits", action="store_false")
    ap.add_argument("--filter-alpha", type=float, metavar="X",
                    help="the decoder's target filter alpha (default: the policy's meta action_filter_alpha, else the "
                         "run's option filter_alpha, else 0)")
    ap.add_argument("--motion", action="store_true",
                    help="measure motion quality and impact (motion_meter) into each start's summary and trials")
    a = ap.parse_args()
    starts = [s for s in a.starts.split(",") if s]
    assert all(s in STARTS + SLICE_STARTS for s in starts), starts
    assert a.robots <= 1024, "keep a job at <= 1024 envs"
    pdir = Path(a.policy)
    pol = TorchPolicy(pdir / "policy.npz", "cuda:0")
    cs = a.contact_solref
    cs = "run" if cs == ["run"] else None if cs == ["none"] else tuple(float(v) for v in cs)
    env, opt = build(pdir, a.robots, a.seed, contact_solref=cs, limits=a.limits, filter_alpha=a.filter_alpha)
    out = {"policy": str(pdir), "seed": a.seed, "robots": a.robots, "seconds": a.seconds,
           "band_deg": list(opt.band_deg), "action_scale": list(opt.action_scale),
           "contact_solref": opt.contact_solref, "limits": bool(opt.limits), "filter_alpha": opt.filter_alpha, "action_noise": a.action_noise, "sustain_s": SUSTAIN_S, "settle_s": SETTLE_S,
           "current_sampling": "per policy step (20 ms), |tau|/kt of the last physics step",
           "guard": {"amps": GUARD_A, "seconds": GUARD_S, "ratios": [1.0, ROBOT_RATIO]}, "starts": {}}
    meter = MotionMeter(env) if a.motion else None
    if meter is not None:
        out["motion"] = meter.config()
    for s in starts:
        r = run_start(env, pol, s, a.seconds, a.seed, noise=a.action_noise, meter=meter)
        out["starts"][s] = r
        print(json.dumps({k: v for k, v in r["summary"].items() if k != "motion"}), flush=True)
        if meter is not None:
            g = r["summary"]["motion"]["getup"]
            print("motion", s, json.dumps({k: g[k]["mean"] for k in ("rev5", "rev10", "pow3", "da2", "speed_rms")}),
                  flush=True)
    if meter is not None:
        meter.detach()
    env.close()
    if a.video:
        import imageio.v2 as imageio
        venv, _ = build(pdir, 1, a.seed, render=True, contact_solref=cs, limits=opt.limits,
                        filter_alpha=opt.filter_alpha)
        frames = []
        r = run_start(venv, pol, a.video_start, a.seconds, a.seed, frames=frames)
        Path(a.video).parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(a.video, frames, fps=int(1 / DT))
        out["video"] = {"path": a.video, "start": a.video_start, "summary": r["summary"], "trial": r["trials"]}
        print("video", a.video, json.dumps(r["summary"]), flush=True)
        venv.close()
    Path(a.json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.json).write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
