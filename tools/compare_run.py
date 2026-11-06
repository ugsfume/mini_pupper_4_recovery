#!/usr/bin/env python
"""Compare a real robot run with the simulation, within a minute or two per run.

    python tools/compare_run.py stream LOG.npz --stream FILE.npz --policy policies/getup [--slow] [--json OUT]
    python tools/compare_run.py floor  RUN_PREFIX --policy policies/getup [--robots 64] [--json OUT]
    python tools/compare_run.py obs    RUN_PREFIX --policy policies/getup [--json OUT]
    python tools/compare_run.py predict --stream FILE.npz --policy policies/getup [--cap 2000] [--json OUT]

Exit code: 0 clean / pass, 1 a failed check, 2 refused. A policy's ``config.json`` gives the plant (``getup_eval.build``:
noise at its end amplitudes, friction, center of mass, mass and command delay randomised; contact softness is the run's fixed
``contact_solref``, not randomised by the training).

``stream``: the log of ``mp4link stream --log``. What the robot actually sent (``sent_pos_dd``: the ramp from where the
legs were, the hold on the first frame, the frames as clipped, the ramp back; zero-order held on the 20 ms grid at its
``sent_t``, so gaps stay gaps) replayed in ``--robots`` (16) simulated robots, base held still 1 m up (feet touch nothing),
from the logged first pose, BAM plant, the log's frame cap (mode 1: duty <= kff x cap, kff 0.001). ``--stream`` is the
file played: the log's sends are checked against it (a wrong file shows). Over the playback (``t_play``), per joint group
and servo: error = |target sent at t - position at t + 20 ms| (the same function on robot and sim) p50/p95; delay by cross-correlation of target and position; current p50/p99/peak and the robot/sim
ratio. Pass: robot p95 <= 2 x sim + 5 deg (25 % speed, ``--slow`` or ``x0p25`` in the file name: p95 <= 15 deg); housing
rise <= 5 C. Ratio > 2 is flagged only.

``predict``: the same simulation with no log: from the stand, the 2 s ramp, the 1 s hold, the file, the 1 s ramp back and
1 s hold, clipped as ``--band-deg`` (default 90,180,180) and ``--step-deg`` (20) say.

``floor``: a ``runner run --log`` run. The start: the joints of the first GETUP frame, gravity the mean of its first 3
frames' observations (the IMU's noise), yaw random, dropped onto the floor with the joints held (``SETTLE_S``), then the
decoder seeded from the pose and the phase clock at 0, as at the runner's entry; a start in motion (|gyro| > 1 rad/s,
a mid-fall entry) is flagged: simulated from rest. Same policy, phase restart per the log's ``getup_phase_restart_s``
(45 deg filtered tilt; here the true tilt), for the log's ``getup_timeout_s`` (or ``--seconds``). Outcome on both
sides: the runner's hand-back criterion (tilt < 12, |gyro| < 0.8, joint rms < 10, held 0.5 s; simulated on the true
state). Reported: outcomes and times, the real tilt against the simulated 5-95 % band, tracking and current per group,
the first channel to leave its simulated band (2-98 % + a margin, held 0.2 s), and one reading:
``inside`` / ``policy`` (the sim also fails) / ``plant-gap`` (named by that channel) / ``sim-pessimistic`` / ``stopped``.

``obs``: the same run's logged observations through the numpy export must give the logged raw actions; each channel
against the 0.05-99.95 % range of a rollout from the same start kind; independent gravity checks, because that rollout
starts from the logged gravity and a sign error would carry into it: the raw accelerometer through the mounting, the
filtered tilt the step decided on, upright must read g_z ~ -1, and per axis dg/dt = -w x g (catches a one-axis flip).

Limits: robot currents are the servo's reading, sim's BAM |tau|/kt at the 5 ms sample; simulated tilt is the true one,
not the complementary filter; the simulation has no gear backlash or wear and no stand's compliance under the base; a floor
start from rest cannot reproduce a mid-fall entry's momentum.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

from mp4_recovery.contract import Contract, stand_offset_rel

STEP = 0.02
SAMPLE = 0.005
LAG = 0.02                  # the tracking error's one control step
GROUPS = {"abduction": [0, 3, 6, 9], "hip": [1, 4, 7, 10], "knee": [2, 5, 8, 11]}
NAMES = [f"{leg}_{j}" for leg in ("FR", "FL", "RR", "RL") for j in ("abd", "hip", "knee")]
KFF = 0.001                 # runner.IDENTIFIED: mode 1's duty cap = kff x the frame's cap (mA)
SLOW_P95_DEG, RISE_MAX_C, RATIO_FLAG = 15.0, 5.0, 2.0
FILE_DIFF_MAX_DD = 10.0   # rounding and the odd clipped send; more means another file, or clipping that matters
ST_GETUP = 1 << 9
SETTLE_S = 0.5              # floor: joints held while the dropped robot lands
HB_TILT, HB_GYRO, HB_RMS, HB_STEPS = 12.0, 0.8, 10.0, 25    # runner.HANDBACK_*
RESTART_TILT = 45.0         # runner.PHASE_RESTART_TILT_DEG
SUSTAIN_STEPS = 10          # a channel out of its band this long (0.2 s) has left it
OBS_NAMES = (["gyro_x", "gyro_y", "gyro_z", "gravity_x", "gravity_y", "gravity_z", "phase_phi", "phase_sin", "phase_cos"]
             + [f"pos_{n}" for n in NAMES] + [f"vel_{n}" for n in NAMES] + [f"prev_{n}" for n in NAMES])
OBS_LISTED_SHARE = 0.02     # a channel is listed when more than this share of the frames is outside its range
OBS_RANGE_MARGIN = 0.05     # ... the sim's range widened by this share of its width (one trajectory near an extreme)


# ---- pure arithmetic ------------------------------------------------------------------------------------------------
def rel_deg_from_dd(dd, stand_dd, signs) -> np.ndarray:
    """dd (a target or a reading) in the policy's coordinates about the stand, deg: rel = sign * servo_deg and
    servo_deg falls as dd rises (contract.rel_from_dd, about the log's own stand)."""
    return -np.asarray(signs, float) * (np.asarray(dd, float) - np.asarray(stand_dd, float)) / 10.0


def dd_from_rel_deg(rel_deg, stand_dd, signs) -> np.ndarray:
    return np.asarray(stand_dd, float) - 10.0 * np.asarray(signs, float) * np.asarray(rel_deg, float)


def zoh_grid(t_sent, rows, dt: float = STEP) -> tuple[np.ndarray, np.ndarray]:
    """Sends on a uniform grid from the first send: each grid point the latest row sent at or before it."""
    t_sent = np.asarray(t_sent, float)
    n = int(math.floor((t_sent[-1] - t_sent[0]) / dt + 1e-6)) + 1
    t = t_sent[0] + np.arange(n) * dt
    idx = np.searchsorted(t_sent, t + 1e-9, side="right") - 1
    return t, np.asarray(rows, float)[idx]


def tracking_errors(t_pos, pos, t_sent, target, lag: float = LAG) -> np.ndarray:
    """|target sent at t - position at t + lag| per send and servo, deg, for sends whose t + lag lies inside the
    positions (the 1-step definition); t_pos (T,), pos (T, 12), t_sent (S,), target (S, 12)."""
    t_pos, pos, t_sent, target = (np.asarray(x, float) for x in (t_pos, pos, t_sent, target))
    keep = (t_sent + lag >= t_pos[0]) & (t_sent + lag <= t_pos[-1])
    at = np.stack([np.interp(t_sent[keep] + lag, t_pos, pos[:, i]) for i in range(pos.shape[1])], axis=1)
    return np.abs(target[keep] - at)


def delay_ms(t_tgt, tgt, t_pos, pos, max_ms: float = 300.0, dt: float = SAMPLE, min_span: float = 3.0) -> float:
    """One servo's lag by cross-correlation: Pearson r of the target (zero-order held) and the position on a ``dt``
    grid over lags 0..max_ms, the peak refined by a parabola; NaN if it moves less than ``min_span`` deg."""
    t_tgt, tgt, t_pos, pos = (np.asarray(x, float) for x in (t_tgt, tgt, t_pos, pos))
    lo, hi = max(t_tgt[0], t_pos[0]), min(t_tgt[-1], t_pos[-1])
    if hi - lo < 10 * dt or np.ptp(tgt) < min_span:
        return math.nan
    t = np.arange(lo, hi, dt)
    x = tgt[np.clip(np.searchsorted(t_tgt, t + 1e-9, side="right") - 1, 0, len(tgt) - 1)]
    y = np.interp(t, t_pos, pos)
    nlag = min(int(max_ms / 1000.0 / dt), len(t) // 2)
    r = np.full(nlag + 1, -2.0)
    for k in range(nlag + 1):
        a, b = x[:len(x) - k], y[k:]
        if a.std() > 1e-9 and b.std() > 1e-9:
            r[k] = float(np.corrcoef(a, b)[0, 1])
    i = int(np.argmax(r))
    off = 0.0
    if 0 < i < nlag:
        den = r[i - 1] - 2 * r[i] + r[i + 1]
        off = 0.5 * (r[i - 1] - r[i + 1]) / den if abs(den) > 1e-12 else 0.0
    return (i + off) * dt * 1000.0


def pct(x, q) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.percentile(x, q)) if x.size else math.nan


def ratio(a: float, b: float) -> float:
    return a / b if b > 1e-9 and math.isfinite(a) else math.nan


def err_stats(e) -> dict:
    return {"p50": pct(e, 50), "p95": pct(e, 95)}


def cur_stats(c) -> dict:
    return {"p50": pct(c, 50), "p99": pct(c, 99), "peak": pct(c, 100)}


def stream_pass(robot_p95: float, sim_p95: float, slow: bool) -> tuple[bool, float]:
    """(pass, the limit): robot p95 <= 2 x sim + 5 deg; at 25 % speed p95 <= 15 deg."""
    limit = SLOW_P95_DEG if slow else 2.0 * sim_p95 + 5.0
    return bool(robot_p95 <= limit), limit


def housing_rise(temp_dc, frames: int = 20) -> np.ndarray:
    """Per servo, C: the last frames' mean less the first frames' mean."""
    t = np.asarray(temp_dc, float) / 10.0
    return t[-frames:].mean(0) - t[:frames].mean(0)


def is_slow(name: str, stream_meta: dict | None = None) -> bool:
    if "x0p25" in Path(name).name:
        return True
    sp = (stream_meta or {}).get("speed")
    return sp is not None and float(sp) <= 0.25 + 1e-9


def smoothstep(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


def planned_sends(start, stand, delta, band_deg, step_deg: float) -> tuple[np.ndarray, int, int, int]:
    """What ``mp4link stream`` would send with no log: (rows dd, first playback row, playback rows, clipped sends)."""
    start, stand, delta = np.asarray(start, float), np.asarray(stand, float), np.asarray(delta, float)
    band = np.asarray(band_deg, float) * 10.0
    step = step_deg * 10.0
    first = np.clip(stand + delta[0], stand - band, stand + band)
    rows, last, clipped = [], None, 0

    def send(t, use_band=True):
        nonlocal last, clipped
        t = np.asarray(t, float)
        c = np.clip(t, stand - band, stand + band) if use_band else t
        clipped += int(np.any(c != t))
        t = c
        if last is not None:
            c = np.clip(t, last - step, last + step)
            clipped += int(np.any(c != t))
            t = c
        last = np.clip(np.rint(t), 0, 3100)
        rows.append(last)

    for k in range(100):                    # ramp_to 2 s, only its end clipped to the band
        send(start + (first - start) * smoothstep((k + 1) / 100), use_band=False)
    clipped = 0
    for _ in range(50):
        send(first)
    i0 = len(rows)
    for r in delta:
        send(stand + r)
    n = len(rows) - i0
    frm = last.copy()
    for k in range(50):
        send(frm + (stand - frm) * smoothstep((k + 1) / 50))
    for _ in range(50):
        send(stand)
    return np.asarray(rows), i0, n, clipped


def quat_from_gravity(g, yaw: float = 0.0) -> np.ndarray:
    """(w, x, y, z), world from body, for a body-frame gravity direction ``g`` ((0, 0, -1) upright), then ``yaw``
    about the world's z: the shortest rotation taking the body's "up" (-g) to the world's z."""
    g = np.asarray(g, float)
    up = -g / np.linalg.norm(g)
    c = float(np.clip(up[2], -1.0, 1.0))
    if c < -1.0 + 1e-9:
        q = np.array([0.0, 1.0, 0.0, 0.0])
    else:
        axis = np.array([up[1], -up[0], 0.0])                    # up x e_z
        w = 1.0 + c
        q = np.array([w, axis[0], axis[1], axis[2]])
        q /= np.linalg.norm(q)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return np.array([cy * q[0] - sy * q[3], cy * q[1] - sy * q[2], cy * q[2] + sy * q[1], cy * q[3] + sy * q[0]])


def handback_steps(tilt, gyro, rms_deg, hold: int = HB_STEPS) -> int:
    """The first step (0-based) at which the criterion has held ``hold`` steps, -1 if never (1-D series)."""
    ok = (np.asarray(tilt) < HB_TILT) & (np.asarray(gyro) < HB_GYRO) & (np.asarray(rms_deg) < HB_RMS)
    run = 0
    for k, v in enumerate(ok):
        run = run + 1 if v else 0
        if run >= hold:
            return k
    return -1


def first_exit(series, lo, hi, margin: float = 0.0, sustain: int = SUSTAIN_STEPS) -> tuple[int, str] | None:
    """(the first step of the first run of ``sustain`` steps outside [lo - margin, hi + margin], "above"/"below");
    NaN samples never count."""
    series, lo, hi = np.asarray(series, float), np.asarray(lo, float), np.asarray(hi, float)
    above, below = series > hi + margin, series < lo - margin
    best = None
    for name, out in (("above", above), ("below", below)):
        run = 0
        for k, v in enumerate(out):
            run = run + 1 if v else 0
            if run >= sustain:
                s = k - sustain + 1
                if best is None or s < best[0]:
                    best = (s, name)
                break
    return best


def first_divergence(channels: dict) -> dict | None:
    """``channels``: name -> (real (T,), lo (T,), hi (T,), margin). The earliest sustained exit; a tie goes to the
    first listed (joint groups are listed before the sensors)."""
    best = None
    for name, (real, lo, hi, m) in channels.items():
        ex = first_exit(real, lo, hi, m)
        if ex is not None and (best is None or ex[0] < best["step"]):
            best = {"channel": name, "step": ex[0], "t_s": ex[0] * STEP, "direction": ex[1]}
    return best


def floor_reading(real_up: bool | None, real_t: float | None, sim_share: float, sim_times,
                  div: dict | None) -> tuple[str, str]:
    """(tag, one line). ``real_up`` None: the run was stopped before an outcome."""
    t = np.asarray(sim_times, float)
    who = "no channel left its band" if div is None else \
        f"first divergence {div['channel']} at {div['t_s']:.2f} s ({div['direction']} the sim)"
    if real_up is None:
        return "stopped", f"stopped: the run ended before an outcome; sim success {sim_share:.0%}; {who}"
    if real_up:
        if sim_share < 0.05:
            return "sim-pessimistic", f"sim-pessimistic: the robot rose, the sim only {sim_share:.0%}; {who}"
        if t.size and real_t is not None and real_t > t.max() + 0.5:
            return "plant-gap", (f"plant gap: the robot rose in {real_t:.2f} s, slower than every sim ({t.max():.2f} s); "
                                 f"{who}")
        return "inside", f"inside: the robot rose in {real_t:.2f} s among the sim's outcomes ({sim_share:.0%} rise)"
    if sim_share < 0.9:
        return "policy", f"policy: the robot stayed down and the sim also fails ({1 - sim_share:.0%} of its robots)"
    return "plant-gap", f"plant gap: the sim rises ({sim_share:.0%}), the robot does not; {who}"


def gravity_checks(g, gyro, tilt_cf=None, g_acc=None, dt: float = STEP, smooth: int = 10) -> dict:
    """Independent evidence on the logged gravity's sign. ``g`` (T, 3), ``gyro`` (T, 3) rad/s; ``tilt_cf`` (T,) deg, the
    filtered tilt logged; ``g_acc`` (T, 3), -acc/|acc| through the mounting, from the raw STATE frames."""
    g, gyro = np.asarray(g, float), np.asarray(gyro, float)
    out = {"mismatch": []}
    if g_acc is not None and len(g_acc):
        g_acc = np.asarray(g_acc, float)
        dot = np.sum(g * g_acc, -1)
        out["obs_dot_accel_p50"] = pct(dot, 50)
        corr = []
        for i, ax in enumerate("xyz"):
            if g_acc[:, i].std() > 0.05 and g[:, i].std() > 1e-6:
                corr.append(float(np.corrcoef(g[:, i], g_acc[:, i])[0, 1]))
                if corr[-1] < -0.3:
                    out["mismatch"].append(f"g_{ax} against the raw accelerometer: r {corr[-1]:+.2f}")
            elif np.median(g[:, i] * g_acc[:, i]) < -0.2:
                corr.append(math.nan)
                out["mismatch"].append(f"g_{ax} against the raw accelerometer: opposite sign")
            else:
                corr.append(math.nan)
        out["obs_accel_corr_xyz"] = corr
    if tilt_cf is not None:
        tilt_cf = np.asarray(tilt_cf, float)
        tilt_obs = np.degrees(np.arccos(np.clip(-g[:, 2], -1, 1)))
        d = pct(np.abs(tilt_obs - tilt_cf), 50)
        out["tilt_obs_vs_filter_p50_deg"] = d
        if d > 45.0:
            out["mismatch"].append(f"tilt from g_z against the filtered tilt: {d:.0f} deg apart (p50)")
        up = tilt_cf < 20.0
        if up.sum() >= 5:
            gz = pct(g[up, 2], 50)
            out["g_z_when_upright_p50"] = gz
            if gz > -0.5:
                out["mismatch"].append(f"g_z {gz:+.2f} while the filtered tilt says upright (want -1)")
    if len(g) > 3 * smooth:           # dg/dt = -w x g, per axis, smoothed (the accelerometer's noise, the shocks)
        k = np.ones(smooth) / smooth
        gs = np.stack([np.convolve(g[:, i], k, "same") for i in range(3)], 1)
        ws = np.stack([np.convolve(gyro[:, i], k, "same") for i in range(3)], 1)
        meas = np.gradient(gs, dt, axis=0)[smooth:-smooth]
        pred = -np.cross(ws, gs)[smooth:-smooth]
        kin = []
        for i, ax in enumerate("xyz"):
            if pred[:, i].std() > 0.2 and meas[:, i].std() > 1e-6:
                kin.append(float(np.corrcoef(meas[:, i], pred[:, i])[0, 1]))
                if kin[-1] < -0.3:
                    out["mismatch"].append(f"dg_{ax}/dt against -w x g: r {kin[-1]:+.2f} (one axis flipped?)")
            else:
                kin.append(math.nan)
        out["kinematic_corr_xyz"] = kin
    return out


def obs_outside(real, lo, hi) -> list[dict]:
    """Per obs channel the share of the real frames outside [lo, hi]; the listed ones sorted, worst first."""
    real = np.asarray(real, float)
    out = []
    for i, name in enumerate(OBS_NAMES):
        x = real[:, i]
        below, above = x < lo[i], x > hi[i]
        share = float((below | above).mean())
        if share > OBS_LISTED_SHARE:
            out.append({"channel": name, "share": share, "real_min": float(x.min()), "real_max": float(x.max()),
                        "sim_lo": float(lo[i]), "sim_hi": float(hi[i])})
    return sorted(out, key=lambda r: -r["share"])


# ---- logs ------------------------------------------------------------------------------------------------------------
def _meta(d) -> dict:
    return json.loads(str(d["meta"])) if "meta" in d.files else {}


def load_stream_log(path) -> dict:
    d = np.load(path, allow_pickle=False)
    m = _meta(d)
    if len(d["sent_t"]) < 2:
        raise SystemExit(f"{path}: no SET_ALL frames logged")
    if "t_play" not in m or m["t_play"][0] is None:
        raise SystemExit(f"{path}: no playback in the log (meta t_play); stop reason {m.get('stop_reason')}")
    t0, t1 = m["t_play"]
    return {"t": d["t_rx"].astype(float), "flags": d["flags"], "pos_dd": d["pos_dd"].astype(float),
            "cur_a": np.abs(d["cur_ma"].astype(float)) / 1000.0, "temp_dc": d["temp_dc"].astype(float),
            "sent_t": d["sent_t"].astype(float), "sent_dd": d["sent_pos_dd"].astype(float), "meta": m,
            "stand": np.asarray(m["stand"], float), "t0": float(t0),
            "t1": float(t1) if t1 is not None else float(d["t_rx"][-1]), "cap": int(m.get("cap", 600))}


def load_run(prefix) -> dict:
    prefix = str(prefix)
    for suf in (".policy.npz", ".npz"):
        if prefix.endswith(suf):
            prefix = prefix[: -len(suf)]
    p = np.load(prefix + ".policy.npz", allow_pickle=False)
    d = np.load(prefix + ".npz", allow_pickle=False)
    meta = _meta(p)
    steps = {k: p[k] for k in p.files if k != "meta"}
    main = {k: d[k] for k in ("t_rx", "tick", "cur_ma", "acc_mg", "gyr_ddps", "flags", "temp_dc")}
    return {"prefix": prefix, "meta": meta, "steps": steps, "main": main}


def getup_window(run: dict, n: int = 1) -> dict:
    """The run's ``n``-th GETUP: its steps [k0, k1), the event, the outcome (True hand-back, False timeout, None
    the run stopped first) and its time."""
    st, ev = run["steps"], run["meta"].get("getups") or []
    if "mode" not in st or not (st["mode"] == 1).any():
        raise SystemExit(f"{run['prefix']}: no GETUP in this run")
    if len(ev) < n:
        raise SystemExit(f"{run['prefix']}: {len(ev)} GETUP event(s), asked for {n}")
    e = ev[n - 1]
    k0 = int(e["step"])
    end = e.get("end")
    k1 = int(e["end_step"]) if e.get("end_step") is not None else len(st["mode"])
    up = True if end == "hand-back" else False if end == "timeout" else None
    t = float(e["seconds"]) if e.get("seconds") is not None else (k1 - k0) * STEP
    return {"k0": k0, "k1": k1, "event": e, "up": up, "t_s": t, "end": end}


def policy_sha(pdir) -> str:
    return hashlib.sha256((Path(pdir) / "policy.npz").read_bytes()).hexdigest()


def policy_warnings(run: dict, pdir, w: dict) -> list[str]:
    out = []
    logged = run["meta"].get("getup_policy_sha256")
    if logged and logged != policy_sha(pdir):
        out.append(f"--policy is not the run's policy ({run['meta'].get('getup_policy')})")
    return out


def group_rows(fn) -> dict:
    return {g: fn(idx) for g, idx in GROUPS.items()}


# ---- the simulation --------------------------------------------------------------------------------------------------
class Rig:
    """Wraps the physics step: holds the base still if asked, and samples positions (policy coordinates + the stand
    offset = physical, rad), BAM currents and the IMU's truth every 5 ms."""

    def __init__(self, env):
        import warp as wp
        from mp4_recovery.gpu.mdp import servo_action
        self.env, self.wp = env, wp
        self.term = servo_action(env)
        self.count, self.hold = 0, None
        self.t, self.pos, self.cur, self.grav, self.gyro = [], [], [], [], []
        self.every = int(round(SAMPLE / float(env.sim.mj_model.opt.timestep)))
        self._orig = env.sim.step
        env.sim.step = self._step

    def clear(self) -> None:
        self.count = 0
        self.t, self.pos, self.cur, self.grav, self.gyro = [], [], [], [], []

    def detach(self) -> None:
        self.env.sim.step = self._orig

    def phys_rel(self):
        return self.term.true_rel() + self.term.off if self.term._shifted else self.term.true_rel()

    def _step(self) -> None:
        self._orig()
        if self.hold is not None:
            d = self.env.sim.wp_data
            self.wp.to_torch(d.qpos)[:, :7] = self.hold
            self.wp.to_torch(d.qvel)[:, :6] = 0.0
        self.count += 1
        if self.count % self.every == 0:
            from mp4_recovery.eval.terrain_eval import bam_currents
            r = self.env.scene["robot"].data
            self.t.append(self.count * float(self.env.sim.mj_model.opt.timestep))
            self.pos.append(self.phys_rel().cpu().numpy())
            c = bam_currents(self.env)
            self.cur.append(np.zeros_like(self.pos[-1]) if c is None else c.cpu().numpy())
            self.grav.append(r.projected_gravity_b.cpu().numpy())
            self.gyro.append(r.root_link_ang_vel_b.cpu().numpy())

    def arrays(self) -> dict:
        return {"t": np.asarray(self.t), "pos_deg": np.degrees(np.stack(self.pos)), "cur_a": np.stack(self.cur),
                "grav": np.stack(self.grav), "gyro": np.stack(self.gyro)}


def set_duty_cap(env, cap_ma: int) -> float:
    """The frame cap, mode 1: duty <= min(max_pwm, kff x cap), on every BAM actuator's model."""
    duty = None
    for act in env.scene["robot"]._actuators:
        model = getattr(act, "_bam_model", None)
        if model is not None:
            model.actuator.max_pwm = min(float(model.actuator.max_pwm), KFF * float(cap_ma))
            duty = float(model.actuator.max_pwm)
    return duty


def sim_held(pdir, rows_rel_deg, start_rel_deg, cap_ma: int, robots: int, seed: int, device: str) -> dict:
    """The targets (F, 12) rel deg, one per 20 ms, on robots whose base is held still 1 m up. Returns the 5 ms samples
    (t from the first target, pos_deg (T, n, 12), cur_a (T, n, 12)), and the duty cap."""
    import torch
    import warp as wp
    from mp4_recovery.eval import getup_eval as ge

    env, _ = ge.build(Path(pdir), robots, seed, device=device)
    rig = Rig(env)
    term = rig.term
    duty = set_duty_cap(env, cap_ma)
    ge.set_start(env, "left")
    torch.manual_seed(seed)
    env.reset()
    dev = env.device
    robot = env.scene["robot"]
    pos = env.scene.env_origins.clone()
    pos[:, 2] += 1.0
    quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=dev).expand(robots, 4)
    robot.write_root_state_to_sim(torch.cat([pos, quat, torch.zeros(robots, 6, device=dev)], -1))
    start = torch.deg2rad(torch.as_tensor(start_rel_deg, dtype=torch.float32, device=dev)).expand(robots, 12)
    q = term.joint_state_from_rel(start - term.off if term._shifted else start)
    robot.write_joint_state_to_sim(q, torch.zeros_like(q), joint_ids=term._jid_all)
    env.sim.forward()
    rig.hold = wp.to_torch(env.sim.wp_data.qpos)[:, :7].clone()
    rows = torch.deg2rad(torch.as_tensor(np.asarray(rows_rel_deg), dtype=torch.float32, device=dev))
    state = {"k": 0}

    def process(_):                     # the file's targets as they are: no decoder (the robot's stream has none)
        rel = rows[state["k"]].expand(robots, 12)
        term.prev_rel = rel - term.off if term._shifted else rel
        term.target_model = term.tc.stand_phys + rel if term._shifted else term.stand + rel

    term.process_actions = process
    zero = torch.zeros(robots, 12, device=dev)
    rig.clear()
    for k in range(len(rows)):
        state["k"] = k
        env.step(zero)
    moved = float((robot.data.root_link_pos_w - pos).abs().max())
    if moved > 1e-4:
        raise RuntimeError(f"the base moved {moved:.4f} m: the hold failed")
    out = rig.arrays()
    rig.detach()
    env.close()
    out["duty_cap"] = duty
    return out


def sim_report(sim: dict, rows_rel_deg, t_rows, win: tuple[float, float]) -> dict:
    """Per group and servo over the window (s from the first target): tracking, delay, currents of the sim's robots,
    pooled; the same functions as the robot's."""
    t, P, C = sim["t"], sim["pos_deg"], sim["cur_a"]
    rows = np.asarray(rows_rel_deg, float)
    n = P.shape[1]
    keep_s = (t_rows >= win[0]) & (t_rows <= win[1])
    keep_p = (t >= win[0]) & (t <= win[1] + LAG)
    keep_c = (t >= win[0]) & (t <= win[1])
    E = np.stack([tracking_errors(t[keep_p], P[keep_p, r], t_rows[keep_s], rows[keep_s]) for r in range(n)])  # (n, S, 12)
    delays = np.array([[delay_ms(t_rows[keep_s], rows[keep_s, i], t[keep_p], P[keep_p, r, i]) for i in range(12)]
                       for r in range(n)])
    Cw = C[keep_c]
    return {"err": E, "delay": delays, "cur": Cw}


def robot_report(log: dict, signs) -> dict:
    t0, t1 = log["t0"], log["t1"]
    ks = (log["sent_t"] >= t0) & (log["sent_t"] <= t1)
    kp = (log["t"] >= t0) & (log["t"] <= t1 + LAG)
    kc = (log["t"] >= t0) & (log["t"] <= t1)
    tgt = rel_deg_from_dd(log["sent_dd"][ks], log["stand"], signs)
    pos = rel_deg_from_dd(log["pos_dd"][kp], log["stand"], signs)
    ts, tp = log["sent_t"][ks], log["t"][kp]
    E = tracking_errors(tp, pos, ts, tgt)
    D = np.array([delay_ms(ts, tgt[:, i], tp, pos[:, i]) for i in range(12)])
    return {"err": E, "delay": D, "cur": log["cur_a"][kc], "frames": int(kc.sum()), "sends": int(ks.sum())}


def compare_stream(r: dict, s: dict, slow: bool) -> dict:
    out = {"groups": {}, "servos": {}}

    def row(idx, name_of_limit=True):
        rc, sc = cur_stats(r["cur"][:, idx]), cur_stats(s["cur"][:, :, idx])
        d = {"err_robot": err_stats(r["err"][:, idx]), "err_sim": err_stats(s["err"][:, :, idx]),
             "delay_ms_robot": pct(r["delay"][idx], 50), "delay_ms_sim": pct(s["delay"][:, idx], 50),
             "cur_robot": rc, "cur_sim": sc, "ratio_p50": ratio(rc["p50"], sc["p50"]),
             "ratio_p99": ratio(rc["p99"], sc["p99"]), "ratio_peak": ratio(rc["peak"], sc["peak"])}
        d["ratio_flag"] = bool(d["ratio_p99"] > RATIO_FLAG)
        per = np.percentile(s["cur"][:, :, idx], 99, axis=(0, 2))     # the plants' spread (command delay, masses)
        d["ratio_p99_within_plants"] = [ratio(rc["p99"], float(per.max())), ratio(rc["p99"], float(per.min()))]
        ok, lim = stream_pass(d["err_robot"]["p95"], d["err_sim"]["p95"], slow)
        d["pass"], d["limit_deg"] = ok, lim
        return d

    for g, idx in GROUPS.items():
        out["groups"][g] = row(idx)
    for i, n in enumerate(NAMES):
        out["servos"][n] = row([i])
    return out


def print_stream(c: dict) -> None:
    print(f"{'':10s} {'err p50 R/S':>13s} {'err p95 R/S':>13s} {'limit':>6s} {'delay ms R/S':>13s} "
          f"{'I p50 R/S':>11s} {'I p99 R/S':>11s} {'I peak R/S':>11s} {'ratio p99':>9s} {'(plants)':>11s}")
    for name, d in list(c["groups"].items()) + list(c["servos"].items()):
        flag = " RATIO>2" if d["ratio_flag"] else ""
        print(f"{name:10s} {d['err_robot']['p50']:6.1f}/{d['err_sim']['p50']:<6.1f} "
              f"{d['err_robot']['p95']:6.1f}/{d['err_sim']['p95']:<6.1f} {d['limit_deg']:6.1f} "
              f"{d['delay_ms_robot']:6.0f}/{d['delay_ms_sim']:<6.0f} "
              f"{d['cur_robot']['p50']:5.2f}/{d['cur_sim']['p50']:<5.2f} {d['cur_robot']['p99']:5.2f}/{d['cur_sim']['p99']:<5.2f} "
              f"{d['cur_robot']['peak']:5.2f}/{d['cur_sim']['peak']:<5.2f} {d['ratio_p99']:9.2f} {d['ratio_p99_within_plants'][0]:5.2f}-{d['ratio_p99_within_plants'][1]:<5.2f}{flag}")


def mode_stream(a) -> int:
    con = Contract.load()
    log = load_stream_log(a.log)
    f = np.load(a.stream, allow_pickle=False)
    fmeta = json.loads(str(f["meta"])) if "meta" in f.files else {}
    slow = bool(a.slow) or is_slow(a.stream, fmeta)
    delta = np.asarray(f["delta_dd"], float)
    period = float(f["period_s"])
    # the sends against the file: one row per playback send, in order
    ks = (log["sent_t"] >= log["t0"]) & (log["sent_t"] <= log["t1"])
    sent = log["sent_dd"][ks]
    rep = int(log["meta"].get("repeat", 1) or 1)
    want = (log["stand"] + np.tile(delta, (rep, 1)))[:len(sent)]
    n = min(len(sent), len(want))
    diff = np.abs(sent[:n] - np.clip(np.rint(want[:n]), 0, 3100))
    file_check = {"playback_sends": int(ks.sum()), "file_frames": len(delta) * rep, "period_s": period,
                  "sends_differing_from_file": int((diff > 1).any(-1).sum()), "max_diff_dd": float(diff.max()) if n else None}
    t_grid, rows_dd = zoh_grid(log["sent_t"], log["sent_dd"])
    rows = rel_deg_from_dd(rows_dd, log["stand"], con.signs)
    start = rel_deg_from_dd(log["pos_dd"][0], log["stand"], con.signs)
    t_rows = t_grid - t_grid[0]
    win = (log["t0"] - t_grid[0], log["t1"] - t_grid[0])
    sim = sim_held(a.policy, rows, start, log["cap"], a.robots, a.seed, a.device)
    c = compare_stream(robot_report(log, con.signs), sim_report(sim, rows, t_rows, win), slow)
    rise = housing_rise(log["temp_dc"])
    play = (log["t"] >= log["t0"]) & (log["t"] <= log["t1"])
    echo = log["meta"].get("getup_echo_share")
    if echo is None and play.any():
        echo = float(((log["flags"][play] & ST_GETUP) != 0).mean())
    checks = {f"tracking_{g}": d["pass"] for g, d in c["groups"].items()}
    checks["housing_rise"] = bool(rise.max() <= RISE_MAX_C)
    checks["played_to_the_end"] = log["meta"].get("stop_reason") == "completed"     # a short window would pass on less
    checks["sends_match_the_file"] = bool(file_check["max_diff_dd"] is not None and file_check["max_diff_dd"] <= FILE_DIFF_MAX_DD)
    out = {"mode": "stream", "log": str(a.log), "stream": str(a.stream), "policy": str(a.policy), "slow": slow,
           "cap_ma": log["cap"], "sim_duty_cap": sim["duty_cap"], "robots": a.robots, "seed": a.seed,
           "stop_reason": log["meta"].get("stop_reason"), "clipped_sends": log["meta"].get("clipped_sends"),
           "getup_echo_share": echo, "housing_rise_c": rise.round(2).tolist(), "housing_rise_max_c": float(rise.max()),
           "file_check": file_check, "compare": c, "checks": checks, "pass": all(checks.values()),
           "flags": [k for k, d in {**c["groups"], **c["servos"]}.items() if d["ratio_flag"]]}
    print_stream(c)
    print(f"stop reason {out['stop_reason']}; clipped sends {out['clipped_sends']}; ST_GETUP echo "
          f"{'n/a' if echo is None else f'{echo:.0%}'}; cap {log['cap']} mA (sim duty <= {sim['duty_cap']:.2f}); housing rise "
          f"max {rise.max():.1f} C (servo {NAMES[int(np.argmax(rise))]})")
    print(f"file check: {file_check['playback_sends']} sends, {file_check['file_frames']} file frames, "
          f"{file_check['sends_differing_from_file']} differ > 1 dd (max {file_check['max_diff_dd']})")
    if file_check["max_diff_dd"] and file_check["max_diff_dd"] > 300 and not log["meta"].get("clipped_sends"):
        print("WARNING: the sends differ from --stream with no clipping logged: is it the file played?")
    print(("slow (25 %): p95 <= 15 deg" if slow else "limit: p95 <= 2 x sim + 5 deg") + "; "
          + "; ".join(f"{k} {'PASS' if v else 'FAIL'}" for k, v in checks.items()))
    if out["flags"]:
        print("ratio > 2:", ", ".join(out["flags"]))
    print("STREAM", "PASS" if out["pass"] else "FAIL", "" if out["stop_reason"] == "completed"
          else f"(stopped: {out['stop_reason']})")
    _write(a.json, out)
    return 0 if out["pass"] else 1


def mode_predict(a) -> int:
    con = Contract.load()
    f = np.load(a.stream, allow_pickle=False)
    delta = np.asarray(f["delta_dd"], float)
    stand = np.asarray(con.stand_dd, float)
    b = [float(v) for v in a.band_deg.split(",")]
    band = np.array([b[i % len(b)] for i in range(12)])
    rows_dd, i0, nplay, clipped = planned_sends(stand, stand, delta, band, a.step_deg)
    rows = rel_deg_from_dd(rows_dd, stand, con.signs)
    t_rows = np.arange(len(rows)) * STEP
    win = (t_rows[i0], t_rows[i0 + nplay - 1])
    sim = sim_held(a.policy, rows, np.zeros(12), a.cap, a.robots, a.seed, a.device)
    s = sim_report(sim, rows, t_rows, win)
    out = {"mode": "predict", "stream": str(a.stream), "policy": str(a.policy), "cap_ma": a.cap, "robots": a.robots,
           "sim_duty_cap": sim["duty_cap"], "band_deg": band.tolist(), "step_deg": a.step_deg, "seconds": nplay * STEP,
           "clipped_sends": clipped, "groups": {}}
    print(f"{a.stream}: {nplay * STEP:.2f} s, clipped sends {clipped}, cap {a.cap} mA (duty <= {sim['duty_cap']:.2f})")
    for g, idx in GROUPS.items():
        d = {"err": err_stats(s["err"][:, :, idx]), "delay_ms": pct(s["delay"][:, idx], 50),
             "cur": cur_stats(s["cur"][:, :, idx])}
        d["limit_robot_p95_deg"] = 2.0 * d["err"]["p95"] + 5.0
        out["groups"][g] = d
        print(f"{g:10s} tracking p50 {d['err']['p50']:5.1f} p95 {d['err']['p95']:5.1f} deg (robot limit "
              f"{d['limit_robot_p95_deg']:.1f}); delay {d['delay_ms']:.0f} ms; current p50 {d['cur']['p50']:.2f} "
              f"p99 {d['cur']['p99']:.2f} peak {d['cur']['peak']:.2f} A")
    _write(a.json, out)
    return 0


def floor_rollout(pdir, robots: int, seed: int, seconds: float, device: str, start=None, start_name: str = "left",
                  restart_s: float = 0.0, stop_on_handback: bool = False) -> dict:
    """``robots`` get-ups: from ``start`` ({"rel": (12,) rad, "g": (3,)}) or a named ``getup_eval`` start, settled
    on the floor. Per policy step (T, n, ...): obs, raw, target rel, the position 20 ms later, currents, tilt, gyro,
    rms; hand-back time per robot; ``rig``: the 5 ms samples. ``stop_on_handback`` ends the rollout when every robot has
    handed back (fake runs)."""
    import torch
    from mp4_recovery.eval import getup_eval as ge
    from mp4_recovery.gpu.getup_task import tilt_deg, _yaw_of  # noqa: F401
    from mp4_recovery.eval.terrain_eval import TorchPolicy, bam_currents

    pol = TorchPolicy(Path(pdir) / "policy.npz", device)
    phase_s = float(pol.meta.get("getup_phase_s", 0.0) or 0.0)
    env, _ = ge.build(Path(pdir), robots, seed, device=device)
    rig = Rig(env)
    term = rig.term
    dev = env.device
    robot = env.scene["robot"]
    ge.set_start(env, start_name)
    torch.manual_seed(seed)
    env.reset()
    settled = {}
    if start is not None:
        from mp4_recovery.gpu.getup_task import spawn_overlaps
        rng = np.random.default_rng(seed)
        quat = np.stack([quat_from_gravity(start["g"], y) for y in rng.uniform(-math.pi, math.pi, robots)])
        quat = torch.as_tensor(quat, dtype=torch.float32, device=dev)
        rel = torch.as_tensor(np.asarray(start["rel"]), dtype=torch.float32, device=dev).expand(robots, 12)
        q = term.joint_state_from_rel(rel)
        z = torch.full((robots,), 0.06, device=dev)
        for _ in range(12):                        # raised until nothing penetrates the floor
            pos = env.scene.env_origins.clone()
            pos[:, 2] += z
            robot.write_root_state_to_sim(torch.cat([pos, quat, torch.zeros(robots, 6, device=dev)], -1))
            robot.write_joint_state_to_sim(q, torch.zeros_like(q), joint_ids=term._jid_all)
            env.sim.forward()
            floor, _legs = spawn_overlaps(env)
            if not bool((floor > 0.0).any()):
                break
            z = z + torch.where(floor > 0.0, floor + 0.002, torch.zeros_like(z))
        term.reset(None)
    for _ in range(int(round(SETTLE_S / STEP))):   # the drop lands, the joints held at the pose
        a = (term.prev_rel / term.scale).clamp(-1.0, 1.0)
        env.step(a)
    settled["tilt_deg"] = tilt_deg(env).cpu().numpy()
    term.reset(None)                               # the runner's entry: decoder at the measured pose, filter restarted
    env.sim.forward()
    env.sim.sense()
    obs = env.observation_manager.compute(update_history=True)["actor"].clone()
    rig.clear()
    t_ph = torch.zeros(robots, device=dev)

    def with_phase(o):
        if phase_s > 0:
            phi = (t_ph / phase_s).clamp(max=1.0)
            o[:, 6] = phi
            o[:, 7] = torch.sin(2 * math.pi * phi)
            o[:, 8] = torch.cos(2 * math.pi * phi)
        return o

    steps = int(round(seconds / STEP))
    rec = {k: [] for k in ("obs", "raw", "target", "pos", "cur", "tilt", "gyro", "rms", "grav_true", "gyro_true")}
    hb_run = torch.zeros(robots, dtype=torch.long, device=dev)
    t_hb = torch.full((robots,), -1.0, device=dev)
    restarts = 0
    with torch.no_grad():
        for k in range(steps):
            obs = with_phase(obs)
            act = pol(obs)
            rec["obs"].append(obs.cpu().numpy())
            rec["raw"].append(act.cpu().numpy())
            o, _, _, _, _ = env.step(act)
            obs = o["actor"].clone()
            tilt = tilt_deg(env)
            gyro = robot.data.root_link_ang_vel_b.norm(dim=-1)
            true = rig.phys_rel()
            rms = torch.rad2deg(term.true_rel().pow(2).mean(-1).sqrt())
            rec["target"].append((term.prev_rel + term.off if term._shifted else term.prev_rel).cpu().numpy())
            rec["pos"].append(true.cpu().numpy())
            c = bam_currents(env)
            rec["cur"].append(c.cpu().numpy() if c is not None else np.zeros((robots, 12)))
            rec["tilt"].append(tilt.cpu().numpy())
            rec["gyro"].append(gyro.cpu().numpy())
            rec["rms"].append(rms.cpu().numpy())
            rec["grav_true"].append(robot.data.projected_gravity_b.cpu().numpy())
            rec["gyro_true"].append(robot.data.root_link_ang_vel_b.cpu().numpy())
            ok = (tilt < HB_TILT) & (gyro < HB_GYRO) & (rms < HB_RMS)
            hb_run = torch.where(ok, hb_run + 1, torch.zeros_like(hb_run))
            t_hb = torch.where((hb_run >= HB_STEPS) & (t_hb < 0), torch.full_like(t_hb, (k + 1) * STEP), t_hb)
            t_ph = t_ph + STEP
            if phase_s > 0 and restart_s > 0:     # the runner's --getup-phase-restart
                again = (t_ph >= phase_s + restart_s - 1e-6) & (tilt > RESTART_TILT)
                restarts += int(again.sum())
                t_ph = torch.where(again, torch.zeros_like(t_ph), t_ph)
            if stop_on_handback and bool((t_hb >= 0).all()):
                break
    out = {k: np.stack(v) for k, v in rec.items()}
    out.update(t_handback=t_hb.cpu().numpy(), rig=rig.arrays(), settled_tilt=settled["tilt_deg"], restarts=restarts,
               phase_s=phase_s, scale=np.asarray(pol.scale))
    rig.detach()
    env.close()
    return out


def real_series(run: dict, w: dict) -> dict:
    """The GETUP's per-step series from the log: tilt, gyro, gravity, positions, targets, tracking, currents."""
    st, m = run["steps"], run["main"]
    k0, k1 = w["k0"], w["k1"]
    obs = np.asarray(st["obs"][k0:k1], float)[:, :45]
    off = np.degrees(stand_offset_rel(float(run["meta"].get("getup_stand_offset_abd_deg", 0.0) or 0.0)))
    pos = np.degrees(obs[:, 9:21]) + off
    target = np.degrees(np.asarray(st["rel"][k0:k1], float))
    err = np.full(target.shape, np.nan)
    err[:-1] = np.abs(target[:-1] - pos[1:])      # the 1-step error: the next frame's position (20 ms later)
    idx = np.clip(np.searchsorted(m["tick"].astype(np.int64), np.asarray(st["tick"][k0:k1], dtype=np.int64)), 0,
                  len(m["tick"]) - 1)
    cur = np.abs(m["cur_ma"][idx].astype(float)) / 1000.0
    return {"obs": obs, "tilt": np.asarray(st["tilt_cf_deg"][k0:k1], float), "gyro": np.asarray(st["gyro_norm"][k0:k1], float),
            "pos": pos, "target": target, "err": err, "cur": cur, "raw": np.asarray(st["raw"][k0:k1], float)}


def floor_channels(real: dict, sim: dict, T: int) -> dict:
    """The channels for ``first_divergence`` (real, sim 2-98 % band, margin), joint groups first."""
    tgt, pos = np.degrees(sim["target"][:T]), np.degrees(sim["pos"][:T])
    ch = {}

    def band(x):                 # (T, n) -> lo, hi
        return np.percentile(x, 2, axis=1), np.percentile(x, 98, axis=1)

    for g, idx in GROUPS.items():
        lo, hi = band(np.abs(tgt - pos)[:, :, idx].mean(-1))     # both recorded at the step's end: 20 ms apart
        r = np.full(T, np.nan)
        r[:T - 1] = real["err"][:T - 1][:, idx].mean(-1)         # the last frame has no next position
        ch[f"{g} tracking"] = (r, lo, hi, 5.0)
    for g, idx in GROUPS.items():
        r = np.sqrt((real["pos"][:T][:, idx] ** 2).mean(-1))
        lo, hi = band(np.sqrt((pos[:, :, idx] ** 2).mean(-1)))
        ch[f"{g} position"] = (r, lo, hi, 10.0)
    for g, idx in GROUPS.items():
        lo, hi = band(sim["cur"][:T][:, :, idx].mean(-1))
        ch[f"{g} current"] = (real["cur"][:T][:, idx].mean(-1), lo, hi, 0.15)
    lo, hi = band(sim["gyro"][:T])
    ch["gyro"] = (real["gyro"][:T], lo, hi, 1.0)
    lo, hi = band(sim["tilt"][:T])
    ch["tilt"] = (real["tilt"][:T], lo, hi, 10.0)
    for i, ax in enumerate("xyz"):
        lo, hi = band(sim["obs"][:T, :, 3 + i])
        ch[f"gravity_{ax}"] = (real["obs"][:T, 3 + i], lo, hi, 0.2)
    return ch


def run_start(real: dict) -> dict:
    g = real["obs"][:3, 3:6].mean(0)
    return {"rel": real["obs"][0, 9:21], "g": g / np.linalg.norm(g), "gyro0": float(np.linalg.norm(real["obs"][0, :3]))}


def mode_floor(a) -> int:
    run = load_run(a.run)
    w = getup_window(run, a.getup)
    real = real_series(run, w)
    meta = run["meta"]
    restart = float(meta.get("getup_phase_restart_s", 0.0) or 0.0)
    seconds = a.seconds or float(meta.get("getup_timeout_s", 8.0) or 8.0)
    notes = policy_warnings(run, a.policy, w)
    start = run_start(real)
    if start["gyro0"] > 1.0:
        notes.append(f"the entry was in motion (|gyro| {start['gyro0']:.1f} rad/s): simulated from rest")
    sim = floor_rollout(a.policy, a.robots, a.seed, seconds, a.device, start=start, restart_s=restart)
    T = min(len(real["tilt"]), len(sim["tilt"]))
    t_hb = sim["t_handback"]
    share = float((t_hb >= 0).mean())
    times = t_hb[t_hb >= 0]
    lo, hi = np.percentile(sim["tilt"][:T], 5, axis=1), np.percentile(sim["tilt"][:T], 95, axis=1)
    ex = first_exit(real["tilt"][:T], lo, hi, 0.0)
    out_share = float(((real["tilt"][:T] > hi) | (real["tilt"][:T] < lo)).mean())
    div = first_divergence(floor_channels(real, sim, T))
    tag, line = floor_reading(w["up"], w["t_s"], share, times, div)
    tgt, pos = np.degrees(sim["target"][:T]), np.degrees(sim["pos"][:T])
    sim_err = np.abs(tgt - pos)                # the target set at the step's start, the position at its end
    groups = {g: {"err_robot": err_stats(real["err"][:T][:, idx]), "err_sim": err_stats(sim_err[:, :, idx]),
                  "cur_robot": cur_stats(real["cur"][:T][:, idx]), "cur_sim": cur_stats(sim["cur"][:T][:, :, idx]),
                  "ratio_p99": ratio(pct(real["cur"][:T][:, idx], 99), pct(sim["cur"][:T][:, :, idx], 99))}
              for g, idx in GROUPS.items()}
    real_tilt0 = float(np.degrees(np.arccos(np.clip(-start["g"][2], -1, 1))))
    out = {"mode": "floor", "run": run["prefix"], "policy": str(a.policy), "robots": a.robots, "seed": a.seed,
           "seconds": seconds, "getup": a.getup, "phase_restart_s": restart, "phase_restarts_sim": sim["restarts"],
           "phase_restart_supported": True, "real": {"up": w["up"], "t_s": w["t_s"], "end": w["end"],
                                                     "stop_reason": meta.get("stop_reason")},
           "start": {"gravity": start["g"].tolist(), "tilt_deg": real_tilt0, "gyro_rad_s": start["gyro0"],
                     "sim_settled_tilt_p50_deg": pct(sim["settled_tilt"], 50)},
           "sim": {"success": share, "t_p5": pct(times, 5), "t_p50": pct(times, 50), "t_p95": pct(times, 95),
                   "t_max": pct(times, 100)},
           "tilt": {"exit": None if ex is None else {"t_s": ex[0] * STEP, "direction": ex[1]},
                    "share_of_time_outside_5_95": out_share},
           "first_divergence": div, "groups": groups, "reading": tag, "line": line, "notes": notes}
    print(f"real: {w['end'] or 'stopped (' + str(meta.get('stop_reason')) + ')'} after {w['t_s']:.2f} s, from tilt "
          f"{real_tilt0:.0f} deg (sim settled at {pct(sim['settled_tilt'], 50):.0f} deg)")
    print(f"sim:  {share:.0%} of {a.robots} rise within {seconds:g} s; times p5/p50/p95 {pct(times, 5):.2f}/"
          f"{pct(times, 50):.2f}/{pct(times, 95):.2f} s; phase restart {restart:g} s ({sim['restarts']} restarts)")
    print("tilt: " + ("stays in the 5-95 % band" if ex is None else
                      f"leaves the 5-95 % band at {ex[0] * STEP:.2f} s, {ex[1]} the sim")
          + f"; outside {out_share:.0%} of the time")
    for g, d in groups.items():
        print(f"{g:10s} err p50/p95 robot {d['err_robot']['p50']:5.1f}/{d['err_robot']['p95']:5.1f} sim "
              f"{d['err_sim']['p50']:5.1f}/{d['err_sim']['p95']:5.1f} deg; I p99 robot {d['cur_robot']['p99']:.2f} sim "
              f"{d['cur_sim']['p99']:.2f} A (ratio {d['ratio_p99']:.2f}); peak {d['cur_robot']['peak']:.2f}/{d['cur_sim']['peak']:.2f}")
    for n in notes:
        print("NOTE:", n)
    print("READING:", line)
    _write(a.json, out)
    return 0 if tag == "inside" else 1


def mode_obs(a) -> int:
    from mp4_recovery.policy_file import NumpyPolicy
    from mp4_recovery.deploy.runner import load_mount
    run = load_run(a.run)
    w = getup_window(run, a.getup)
    real = real_series(run, w)
    pol = NumpyPolicy(Path(a.policy) / "policy.npz")
    pred = pol(real["obs"].astype(np.float32))
    d = np.abs(pred - real["raw"])
    rep = {"frames": len(d), "max_abs_diff": float(d.max()), "frames_differing": int((d > 1e-3).any(-1).sum()),
           "ok": bool(d.max() <= 1e-3)}
    # gravity's sign, independently of the rollout (which starts from the logged gravity)
    m = run["main"]
    _, R = load_mount(run["meta"].get("mount"))
    st = run["steps"]
    idx = np.clip(np.searchsorted(m["tick"].astype(np.int64), np.asarray(st["tick"][w["k0"]:w["k1"]], dtype=np.int64)),
                  0, len(m["tick"]) - 1)
    acc = (R @ m["acc_mg"][idx].astype(float).T).T
    nrm = np.linalg.norm(acc, axis=1, keepdims=True)
    g_acc = -acc / np.where(nrm > 1e-6, nrm, 1.0) if (nrm > 1e-6).all() else None
    grav = gravity_checks(real["obs"][:, 3:6], real["obs"][:, :3], real["tilt"], g_acc)
    # the channels against a short rollout
    sim = floor_rollout(a.policy, a.robots, a.seed, min(max(len(d) * STEP, 2.0), a.seconds), a.device,
                        start=run_start(real), restart_s=float(run["meta"].get("getup_phase_restart_s", 0.0) or 0.0))
    O = sim["obs"].reshape(-1, sim["obs"].shape[-1])
    lo, hi = np.percentile(O, 0.05, axis=0), np.percentile(O, 99.95, axis=0)
    pad = OBS_RANGE_MARGIN * (hi - lo) + 1e-3                # + a tolerance for a channel constant in the sim (phase)
    outside = obs_outside(real["obs"], lo - pad, hi + pad)
    out = {"mode": "obs", "run": run["prefix"], "policy": str(a.policy), "actions": rep, "gravity": grav,
           "channels_outside": outside, "notes": policy_warnings(run, a.policy, w)}
    print(f"actions: {rep['frames']} frames, max |pred - logged| {rep['max_abs_diff']:.2e}: "
          + ("reproduced" if rep["ok"] else f"NOT REPRODUCED on {rep['frames_differing']} frames"))
    print("channels outside the sim's 99.9 % range:" + ("" if outside else " none"))
    for r in outside:
        print(f"  {r['channel']:14s} {r['share']:5.0%} outside; real [{r['real_min']:+.2f}, {r['real_max']:+.2f}], sim "
              f"[{r['sim_lo']:+.2f}, {r['sim_hi']:+.2f}]")
    print("gravity:", {k: v for k, v in grav.items() if k != "mismatch"})
    for why in grav["mismatch"]:
        print("!!! GRAVITY SIGN MISMATCH:", why)
    for n in out["notes"]:
        print("NOTE:", n)
    ok = rep["ok"] and not grav["mismatch"] and not outside
    out["ok"] = ok
    print("OBS", "CLEAN" if ok else "NOT CLEAN")
    _write(a.json, out)
    return 0 if ok else 1


def _write(path, out) -> None:
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(out, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
        print("wrote", path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)

    def common(s, robots):
        s.add_argument("--policy", required=True, help="the export directory (policy.npz + config.json)")
        s.add_argument("--robots", type=int, default=robots)
        s.add_argument("--seed", type=int, default=7)
        s.add_argument("--device", default="cuda:0")
        s.add_argument("--json")

    s = sub.add_parser("stream")
    s.add_argument("log")
    s.add_argument("--stream", required=True)
    s.add_argument("--slow", action="store_true", help="the 25 % criterion (default: x0p25 in the file name)")
    common(s, 16)
    s = sub.add_parser("predict")
    s.add_argument("--stream", required=True)
    s.add_argument("--cap", type=int, default=2000, help="frame cap mA (the plan's A5: 2000; A4: 600)")
    s.add_argument("--band-deg", default="90,180,180", help="abduction, hip, knee")
    s.add_argument("--step-deg", type=float, default=20.0)
    common(s, 16)
    for name, robots, sec in (("floor", 64, 0.0), ("obs", 64, 4.0)):
        s = sub.add_parser(name)
        s.add_argument("run", help="the prefix: <run>.npz + <run>.policy.npz")
        s.add_argument("--getup", type=int, default=1, help="which GETUP of the run")
        s.add_argument("--seconds", type=float, default=sec,
                       help="simulated s" + (" (default: the log's getup_timeout_s)" if name == "floor" else
                                             " at most"))
        common(s, robots)
    a = ap.parse_args(argv)
    return {"stream": mode_stream, "predict": mode_predict, "floor": mode_floor, "obs": mode_obs}[a.mode](a)


if __name__ == "__main__":
    sys.exit(main())
