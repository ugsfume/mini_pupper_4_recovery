"""Motion quality and impact of a get-up, measured on every robot of a batch.

    meter = MotionMeter(env)          # once per env; hooks the physics step (impact) until meter.detach()
    obs, _ = env.reset(); meter.begin(env)
    for k in range(steps):
        obs, *_ = env.step(policy(obs)); meter.update(env)
    res = meter.finish(t_success_s)   # (n,) seconds, < 0 = never; {"summary": ..., "trials": ...}

Per robot, over two windows:
- ``getup``: from the start to the first policy step with tilt < ``TILT_UP_DEG`` (15 deg), that step excluded, and at
  least ``MIN_STEPS`` (25) steps long (the first 25 steps are always in it; a robot that never comes up: the whole
  trial);
- ``after``: from there to the step the success criterion completed (the trial's end if it never did); NaN where it
  is shorter than ``MIN_STEPS``.
- ``trial``: the whole trial, for the impact numbers only (a landing can straddle the window's end).

Metrics per window:
- ``rev5`` / ``rev10``: target reversals per joint per s: a local extremum of the applied target (``ServoAction.prev_rel``,
  servo order, deg; a plateau, e.g. at a clamp, is one extremum) with a swing of >= 5 / 10 deg to the farthest sample
  on both sides within +-``REV_W`` (15) steps, counted over the 12 servos / (window s x 12);
- ``pow3`` / ``pow6``: the leg joints' velocity power share at >= 3 / >= 6 Hz: the 12 joints' ``joint_vel`` (hip, thigh,
  calf), mean removed, Hann-windowed, |rfft|^2 summed over joints (DT 0.02 s: 0-25 Hz);
- ``da2``: mean over the window's steps of sum_j (applied_action_t - applied_action_{t-1})^2 (action units; the first
  step against the decoder's reset state);
- ``speed_rms`` / ``speed_p99``: the 12 leg joints' |joint_vel| (rad/s), rms and the 99th percentile over steps x joints;
- ``sat``: the share of steps x servos whose applied target moved >= 0.99 x the step limit (20 deg per step);
- ``base_force_peak_N``: the largest base-floor normal contact force (N; the sum over the base's contacts in one physics
  step, the largest over every 1 ms physics step);
- ``abd_limit_peak_Nm`` / ``abd_limit_p99_Nm``: the abduction joints' limit-constraint force (MuJoCo's
  ``efc_force`` of the joint-limit rows of the four ``*_hip`` joints, N m: the hinge's stop torque; the largest of the
  four per physics step), the largest, and the p99 over the window's policy steps of the per-step largest;
- ``touchdown_vz_mps``: the base's downward speed (m/s, the free joint's world z velocity in the physics step before
  the contact) at the first touchdown after an airborne phase: no robot geom touching the floor for >= ``AIR_MIN_MS``
  (20 ms), after the robot first touched it (the spawn's drop is not a touchdown); ``touchdown_vz_max_mps`` the largest
  over the window's touchdowns, ``airborne`` the number of airborne phases ending in the window; ``touchdown_part`` of
  the first: 1 base, 2 foot, 3 another leg body (hip can, thigh, calf); NaN / 0 where there is none.

Impact comes from ``env.sim.wp_data`` after every physics step (``efc.type/id/force``, ``contact.geom``, ``qvel``): the
meter wraps ``env.sim.step`` and only reads. The GPU simulation is not bit-reproducible from one run to the next: two
fresh envs on the same seed already differ in peak current and penetration.
Summaries per start: mean and p90 over robots (NaN robots left out; ``n``), and for the impact numbers also the p99
and the max over robots."""
from __future__ import annotations

import math

import numpy as np

DT = 0.02
TILT_UP_DEG = 15.0
MIN_STEPS = 25
REV_W = 15
PROMS = (5.0, 10.0)
SAT_FRAC = 0.99
AIR_MIN_MS = 20.0
WINDOWS = ("getup", "after")
MOTION_KEYS = ("rev5", "rev10", "pow3", "pow6", "da2", "speed_rms", "speed_p99", "sat")
IMPACT_KEYS = ("base_force_peak_N", "abd_limit_peak_Nm", "abd_limit_p99_Nm", "touchdown_vz_mps",
               "touchdown_vz_max_mps", "airborne", "touchdown_part")
PART_BASE, PART_FOOT, PART_LEG = 1, 2, 3


# ------------------------------------------------------------------------------------------------ pure numpy part
def reversals(x: np.ndarray, prom: float, w: int = REV_W, eps: float = 1e-6) -> int:
    """Local extrema of the 1-D series ``x`` with a swing >= ``prom`` to the farthest sample on both sides within
    ``w`` samples; a run of equal samples (a plateau) is one extremum, its sides taken outside the run."""
    x = np.asarray(x, dtype=np.float64)
    L = len(x)
    if L < 3:
        return 0
    d = np.diff(x)
    mv = np.nonzero(np.abs(d) > eps)[0]                      # the diffs that move
    if len(mv) < 2:
        return 0
    s = np.sign(d[mv])
    ch = np.nonzero(s[1:] != s[:-1])[0]
    if len(ch) == 0:
        return 0
    t0 = mv[ch] + 1                                          # the extremum's first sample
    t1 = mv[ch + 1]                                          # its last (the next move leaves from here)
    v = x[t0]
    pad = np.full(w, np.nan)
    xp = np.concatenate([pad, x, pad])
    win = np.lib.stride_tricks.sliding_window_view(xp, w)  # win[i] = x[i - w : i]
    left, right = win[t0], win[t1 + 1 + w]                   # x[t0-w:t0], x[t1+1:t1+1+w]
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            peak = s[ch] > 0
            lo_l, hi_l = np.nanmin(left, 1), np.nanmax(left, 1)
            lo_r, hi_r = np.nanmin(right, 1), np.nanmax(right, 1)
    ok = np.where(peak, (v - lo_l >= prom) & (v - lo_r >= prom), (hi_l - v >= prom) & (hi_r - v >= prom))
    return int(np.count_nonzero(ok))


def reversal_rate(tg_deg: np.ndarray, prom: float) -> float:
    """(L, J) targets in deg -> reversals per joint per s."""
    L, J = tg_deg.shape
    return sum(reversals(tg_deg[:, j], prom) for j in range(J)) / (L * DT) / J


def band_shares(v: np.ndarray, lows=(3.0, 6.0)) -> list[float]:
    """(L, J) velocities -> the power share at >= each of ``lows`` Hz (mean removed, Hann, summed over joints)."""
    L = v.shape[0]
    v = v - v.mean(0)
    p = (np.abs(np.fft.rfft(v * np.hanning(L)[:, None], axis=0)) ** 2).sum(1)
    f = np.fft.rfftfreq(L, DT)
    tot = p.sum()
    return [float(p[f >= lo].sum() / tot) if tot > 0 else math.nan for lo in lows]


def window_bounds(tilt: np.ndarray, step_success: np.ndarray) -> np.ndarray:
    """(T, n) tilt, (n,) the step count at which success completed (<= 0: never) -> (n, 2): the getup window's end step (exclusive), the after window's end."""
    T, n = tilt.shape
    out = np.zeros((n, 2), dtype=np.int64)
    for r in range(n):
        up = np.nonzero(tilt[MIN_STEPS:, r] < TILT_UP_DEG)[0]
        e = MIN_STEPS + int(up[0]) if len(up) else T
        out[r, 0] = e
        out[r, 1] = int(step_success[r]) if step_success[r] > 0 else T
    return out


def motion_metrics(tg_deg, tg_prev_deg0, act, act_prev0, jv, limit_deg, lo, hi) -> dict:
    """One robot, steps [lo, hi): tg (T, 12) deg, act (T, 12) action units, jv (T, 12) rad/s; the reset state
    (12,) for step 0's change."""
    if hi - lo < MIN_STEPS:
        return {k: math.nan for k in MOTION_KEYS}
    seg = slice(lo, hi)
    tg = tg_deg[seg]
    tprev = tg_deg[lo - 1] if lo > 0 else tg_prev_deg0
    aprev = act[lo - 1] if lo > 0 else act_prev0
    dtg = np.diff(np.concatenate([tprev[None], tg]), axis=0)
    da = np.diff(np.concatenate([aprev[None], act[seg]]), axis=0)
    v = jv[seg]
    p3, p6 = band_shares(v)
    return {"rev5": reversal_rate(tg, PROMS[0]), "rev10": reversal_rate(tg, PROMS[1]), "pow3": p3, "pow6": p6,
            "da2": float((da ** 2).sum(1).mean()), "speed_rms": float(np.sqrt((v ** 2).mean())),
            "speed_p99": float(np.percentile(np.abs(v), 99)),
            "sat": float((np.abs(dtg) >= SAT_FRAC * limit_deg).mean())}


def impact_metrics(base_f, abd_f, td, lo, hi) -> dict:
    """One robot, policy steps [lo, hi): per-step largest base force (T,), abduction limit force (T,); touchdowns
    ``td``: per policy step with any, (step, the first's downward speed, the largest, the first's part, count)."""
    if hi <= lo:
        return {k: (0 if k == "airborne" else math.nan) for k in IMPACT_KEYS}
    ev = [t for t in td if lo <= t[0] < hi]
    a = abd_f[lo:hi]
    return {"base_force_peak_N": float(base_f[lo:hi].max()), "abd_limit_peak_Nm": float(a.max()),
            "abd_limit_p99_Nm": float(np.percentile(a, 99)),
            "touchdown_vz_mps": float(ev[0][1]) if ev else math.nan,
            "touchdown_vz_max_mps": float(max(t[2] for t in ev)) if ev else math.nan,
            "airborne": int(sum(t[4] for t in ev)), "touchdown_part": int(ev[0][3]) if ev else 0}


def _stats(x, extra=False) -> dict:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if not len(x):
        out = {"mean": None, "p90": None, "n": 0}
        if extra:
            out.update(p99=None, max=None)
        return out
    out = {"mean": float(x.mean()), "p90": float(np.percentile(x, 90)), "n": int(len(x))}
    if extra:
        out.update(p99=float(np.percentile(x, 99)), max=float(x.max()))
    return out


def compute(series: dict, t_success_s, limit_deg: float = 20.0) -> dict:
    """``series``: tg (T, n, 12) deg, tg0 (n, 12) deg, act (T, n, 12), act0 (n, 12), jv (T, n, 12), tilt (T, n),
    and optionally base_f (T, n), abd_f (T, n), td (per robot: list of (step, vz_down, part)). ``t_success_s``: (n,)
    seconds at which success completed, < 0 never. -> {"summary": {window: {metric: stats}}, "trials": arrays}."""
    tilt = np.asarray(series["tilt"])
    T, n = tilt.shape
    ts = np.asarray(t_success_s, dtype=np.float64).reshape(-1)
    assert ts.shape == (n,), (ts.shape, n)
    step_s = np.where(ts >= 0, np.round(ts / DT), 0).astype(np.int64)
    bounds = window_bounds(tilt, step_s)
    have_imp = "base_f" in series
    per = {w: {k: [] for k in MOTION_KEYS + (IMPACT_KEYS if have_imp else ())} for w in WINDOWS}
    if have_imp:
        per["trial"] = {k: [] for k in IMPACT_KEYS}
    for r in range(n):
        e, s_end = int(bounds[r, 0]), int(bounds[r, 1])
        for w, (lo, hi) in (("getup", (0, e)), ("after", (e, max(e, s_end)))):
            m = motion_metrics(series["tg"][:, r], series["tg0"][r], series["act"][:, r], series["act0"][r],
                               series["jv"][:, r], limit_deg, lo, hi)
            for k, v in m.items():
                per[w][k].append(v)
        if have_imp:
            for w, (lo, hi) in (("getup", (0, e)), ("after", (e, max(e, s_end))), ("trial", (0, T))):
                m = impact_metrics(series["base_f"][:, r], series["abd_f"][:, r], series["td"][r], lo, hi)
                for k, v in m.items():
                    per[w][k].append(v)
    summary = {"window_steps": _stats(bounds[:, 0]), "after_steps": _stats(np.maximum(bounds[:, 1] - bounds[:, 0], 0))}
    for w, d in per.items():
        summary[w] = {}
        for k, v in d.items():
            if k == "touchdown_part":
                v = np.asarray(v)
                summary[w][k] = {"base": int((v == PART_BASE).sum()), "foot": int((v == PART_FOOT).sum()),
                                 "leg": int((v == PART_LEG).sum()), "none": int((v == 0).sum())}
            elif k == "airborne":
                v = np.asarray(v, dtype=np.float64)
                summary[w][k] = {"mean": float(v.mean()), "robots_with": int((v > 0).sum())}
            else:
                summary[w][k] = _stats(v, extra=k in IMPACT_KEYS)
    trials = {"window_end_step": bounds[:, 0].tolist(), "after_end_step": bounds[:, 1].tolist()}
    for w, d in per.items():
        for k, v in d.items():
            trials[f"{w}.{k}"] = [None if (isinstance(x, float) and not math.isfinite(x)) else
                                  (round(x, 5) if isinstance(x, float) else x) for x in v]
    return {"summary": summary, "trials": trials}


# ------------------------------------------------------------------------------------------------- the env side
class MotionMeter:
    """Records one batch's trial (see the module docstring); ``begin`` after each reset, ``update`` after each
    ``env.step``, ``finish`` at the end. ``impact=False``: no physics-step hook (motion metrics only)."""

    def __init__(self, env, impact: bool = True):
        import mujoco
        import torch
        from mp4_recovery.gpu import mdp
        self.env = env
        self.act = mdp.servo_action(env)
        self.limit_deg = math.degrees(self.act.limit)
        self.dev = env.device
        self.impact = impact
        self._torch = torch
        m = env.sim.mj_model
        n = env.num_envs
        if impact:
            assert m.opt.cone == mujoco.mjtCone.mjCONE_PYRAMIDAL, "the normal force is the pyramid rows' sum"
            cat = []
            for g in range(m.ngeom):
                body = m.body(int(m.geom_bodyid[g])).name
                if not body.startswith("robot/"):
                    cat.append(0)
                elif body == "robot/base":
                    cat.append(PART_BASE)
                elif m.geom(g).name.endswith("_foot"):
                    cat.append(PART_FOOT)
                else:
                    cat.append(PART_LEG)
            self.cat = torch.tensor(cat, device=self.dev, dtype=torch.long)
            abd = [j for j in range(m.njnt) if m.joint(j).name.endswith("_hip")]
            assert len(abd) == 4, [m.joint(j).name for j in abd]
            isabd = torch.zeros(m.njnt, dtype=torch.bool, device=self.dev)
            isabd[abd] = True
            self.isabd = isabd
            root = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
            assert len(root) == 1
            self.vz_dof = int(m.jnt_dofadr[root[0]]) + 2
            self.T_LIMIT = int(mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT)
            self.T_CONTACT = (int(mujoco.mjtConstraint.mjCNSTR_CONTACT_PYRAMIDAL),
                              int(mujoco.mjtConstraint.mjCNSTR_CONTACT_FRICTIONLESS))
            self.air_min = int(round(AIR_MIN_MS / 1000.0 / m.opt.timestep))
            self._orig_step = env.sim.step
            env.sim.step = self._physics_step
        self._n = n
        self._recording = False

    def detach(self) -> None:
        if self.impact and self.env.sim.step == self._physics_step:
            self.env.sim.step = self._orig_step

    # -- physics-step hook
    def _physics_step(self) -> None:
        self._orig_step()
        if self._recording:
            self._substep()

    def _substep(self) -> None:
        import warp as wp
        torch = self._torch
        d = self.env.sim.wp_data
        tp = wp.to_torch(d.efc.type)
        ids = wp.to_torch(d.efc.id).long()
        f = wp.to_torch(d.efc.force)
        nefc = wp.to_torch(d.nefc)
        J = tp.shape[1]
        valid = torch.arange(J, device=self.dev)[None, :] < nefc[:, None]
        cont = valid & ((tp == self.T_CONTACT[0]) | (tp == self.T_CONTACT[1]))
        geom = wp.to_torch(d.contact.geom).long()
        cid = torch.where(cont, ids, torch.zeros_like(ids)).clamp(0, geom.shape[0] - 1)
        g = geom[cid]                                                       # (n, J, 2)
        c0, c1 = self.cat[g[..., 0].clamp_min(0)], self.cat[g[..., 1].clamp_min(0)]
        rf = cont & (((c0 == 0) & (c1 > 0)) | ((c1 == 0) & (c0 > 0)))      # robot - floor rows
        part = torch.maximum(c0, c1)
        base_f = (f * (rf & (part == PART_BASE))).sum(-1)
        lim = valid & (tp == self.T_LIMIT) & self.isabd[ids.clamp(0, self.isabd.shape[0] - 1)]
        abd_f = (f.abs() * lim).amax(-1)
        self._base_f = torch.maximum(self._base_f, base_f)
        self._abd_f = torch.maximum(self._abd_f, abd_f)
        touch = rf.any(-1)
        on_base = (rf & (part == PART_BASE)).any(-1)
        on_foot = (rf & (part == PART_FOOT)).any(-1)
        vz = wp.to_torch(d.qvel)[:, self.vz_dof]
        event = touch & self._ever & (self._air >= self.air_min)
        kind = torch.where(on_base, PART_BASE, torch.where(on_foot, PART_FOOT, PART_LEG))
        self._ev_any |= event
        self._ev_vz = torch.where(event & ~self._ev_mark, -self._vz_prev, self._ev_vz)
        self._ev_part = torch.where(event & ~self._ev_mark, kind, self._ev_part)
        self._ev_mark |= event
        self._ev_count += event.long()
        self._ev_vzmax = torch.where(event, torch.maximum(self._ev_vzmax, -self._vz_prev), self._ev_vzmax)
        self._air = torch.where(touch, torch.zeros_like(self._air), self._air + 1)
        self._ever |= touch
        self._vz_prev = vz.clone()

    def _reset_step_acc(self) -> None:
        torch, n = self._torch, self._n
        self._base_f = torch.zeros(n, device=self.dev)
        self._abd_f = torch.zeros(n, device=self.dev)
        self._ev_any = torch.zeros(n, dtype=torch.bool, device=self.dev)
        self._ev_mark = torch.zeros(n, dtype=torch.bool, device=self.dev)
        self._ev_vz = torch.full((n,), math.nan, device=self.dev)
        self._ev_vzmax = torch.full((n,), -math.inf, device=self.dev)
        self._ev_part = torch.zeros(n, dtype=torch.long, device=self.dev)
        self._ev_count = torch.zeros(n, dtype=torch.long, device=self.dev)

    # -- policy-step side
    def begin(self, env=None) -> None:
        """After the reset: the decoder's state is step 0's reference."""
        import warp as wp
        torch = self._torch
        a = self.act
        self._tg0 = torch.rad2deg(a.prev_rel).clone()
        self._a0 = a.applied_action.clone()
        self._tg, self._act, self._jv, self._tilt = [], [], [], []
        self._bf, self._af, self._evs = [], [], []
        if self.impact:
            n = self._n
            self._ever = torch.zeros(n, dtype=torch.bool, device=self.dev)
            self._air = torch.zeros(n, dtype=torch.long, device=self.dev)
            self._vz_prev = wp.to_torch(self.env.sim.wp_data.qvel)[:, self.vz_dof].clone()
            self._reset_step_acc()
        self._recording = True

    def update(self, env=None) -> None:
        from mp4_recovery.gpu.getup_task import tilt_deg
        a, env = self.act, self.env
        self._tg.append(self._torch.rad2deg(a.prev_rel).clone())
        self._act.append(a.applied_action.clone())
        self._jv.append(env.scene["robot"].data.joint_vel[:, a._jid_all].clone())
        self._tilt.append(tilt_deg(env).clone())
        if self.impact:
            self._bf.append(self._base_f)
            self._af.append(self._abd_f)
            self._evs.append((self._ev_any, self._ev_vz, self._ev_vzmax, self._ev_part, self._ev_count))
            self._reset_step_acc()

    def series(self) -> dict:
        torch = self._torch
        np_ = lambda xs: torch.stack(xs).cpu().numpy()
        s = {"tg": np_(self._tg), "tg0": self._tg0.cpu().numpy(), "act": np_(self._act), "act0": self._a0.cpu().numpy(),
             "jv": np_(self._jv), "tilt": np_(self._tilt)}
        if self.impact:
            s["base_f"], s["abd_f"] = np_(self._bf), np_(self._af)
            any_, vz, vzmax, part, cnt = (np_(list(x)) for x in zip(*self._evs))   # (T, n) each
            s["td"] = [[(int(k), float(vz[k, r]), float(vzmax[k, r]), int(part[k, r]), int(cnt[k, r]))
                        for k in np.nonzero(any_[:, r])[0]] for r in range(self._n)]
        return s

    def finish(self, t_success_s) -> dict:
        self._recording = False
        return compute(self.series(), t_success_s, self.limit_deg)

    def config(self) -> dict:
        return {"dt": DT, "tilt_up_deg": TILT_UP_DEG, "min_steps": MIN_STEPS, "rev_window_steps": REV_W,
                "prominence_deg": list(PROMS), "sat_frac": SAT_FRAC, "step_limit_deg": self.limit_deg,
                "air_min_ms": AIR_MIN_MS, "impact": self.impact,
                "impact_source": "mujoco_warp efc rows after every physics step: contact rows (pyramidal: the normal "
                                 "force is the rows' sum) of the base's geoms against non-robot geoms; joint-limit "
                                 "rows of the four *_hip (abduction) joints (N m); the free joint's qvel z"}
