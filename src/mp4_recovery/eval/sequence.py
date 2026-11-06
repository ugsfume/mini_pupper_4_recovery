#!/usr/bin/env python3
"""Walking, a kick, the fall, the get-up and walking again in simulation, with a walking policy and a get-up policy and
the same state machine as ``runner run --getup-policy`` (``mp4_recovery.deploy.runner``).

    python -m mp4_recovery.eval.sequence --walker policies/walker --getup policies/getup \
        --schedule 0.5 1.5 2 4 --robots 256 --seconds 60 --seed 12 --getup-phase-restart 1.0 --json runs/eval/sequence_s12.json

One env, the walker's (``omni_eval``'s flat build: 4 m tiles, randomised plants, the measured observation noise, foot solref
(0.01, 1)), with every termination but the time-out removed (a fallen robot stays in its episode). ``--plant getup``
(default) adds the get-up task's plant to it (``getup_task._spec_fn``: its leg <-> leg, leg <-> hip can and thigh <-> base
collisions and its stiff non-foot contacts, from the get-up policy's ``config.json`` when it has one, else ``GetupOptions``'
defaults; njmax / nconmax raised); ``--plant walker`` keeps the walker's. The two policies' action configs differ (each
policy's scale and band come from the policy): the one ``ServoAction``'s scale and band are per
env and switched when an env changes mode, and its decoder is seeded from the measured pose there (``prev_rel``, the applied
action and the target), as the runner's ``ActionDecoder.reset(rel0)``; the velocity filter's six samples are refilled with
the newest (the runner resets its observation builder at a switch).

Per robot: the command (``--vx``, wz from the robot's heading loop, ``heading.pi_step`` capped at 0.4 rad/s, idle in
GETUP and re-latched at the hand-back); one kick at ``--kick-t`` (``push_eval``'s: |dv| ``--kick`` added to the
base's velocity, horizontal, at ``--kick-dir`` in the body's yaw frame); the runner's complementary filter
(``runner.GravityFilter``'s arithmetic, vectorised: the simulated gyro ``imu_ang_vel`` integrated, the simulated
accelerometer ``imu_lin_acc`` pulling with tau 1 s, the same gate) and fall detector at the policy rate (50 Hz; the
robot runs them at 200 Hz). Modes and transitions as the runner's (its constants): a fall in walk mode -> GETUP (the
get-up observation = the walker env's newest frame of each term with the command 0 (a phased get-up, meta
``getup_phase_s``: ``runner.getup_phase`` of the time since this GETUP's entry), the velocity 0 at the switch and
the previous action the seeded decoder's; the get-up policy's own history if it has one); in GETUP the hand-back
criterion (filtered tilt < 12 deg, |gyro| < 0.8 rad/s, the measured pose within 10 deg rms of the stand, held 25
steps) hands back (the walker's decoder seeded, its observation history refilled with the hand-back frame as the
runner's cleared history, the heading loop re-latched); ``--getup-timeout`` s in GETUP ends the trial (the runner
disarms: limp); a fall after ``--getup-max`` get-ups ends it (disarm). An ended robot is frozen out of the metrics
(it keeps stepping on the get-up policy).

Per robot (JSON, with a summary): fell (GETUP entered after the kick), t_fall (kick -> GETUP), t_up (GETUP ->
hand-back: time to back up), t_walk (hand-back -> walking again: the mean body-frame velocity over 0.5 s within
0.05 m/s of the command and the tilt below 15 deg at each of its steps; for a robot that did not fall, from the
kick), t_walk_from_kick, down (GETUP entry -> walking again), time in GETUP, get-ups, the end (timeout, max_getups).
Two approximations: the walker's history is refilled at the hand-back by writing mjlab's private observation history
buffers (``refill_history``: ``ObservationManager._group_obs_term_history_buffer``, ``CircularBuffer._buffer`` /
``_num_pushes``; their shapes are asserted, so an mjlab change fails loudly); and "walking again" is push_eval's loose
mean-velocity window (a trot's within-stride swing fails a per-step test), which a walker starting from the stand
can meet within a few steps of the hand-back (0.00-0.04 s in a forced check).
``--getup-phase-restart S`` (the runner's option, default off): a phased get-up's clock restarts (phi back to 0) when
phi has been 1 for S s and the filtered tilt is above ``runner.PHASE_RESTART_TILT_DEG`` (45 deg); the timeout counts
from the entry; events "phase-restart" (compared in float32, so a threshold that is a whole number of steps can come
one step later than the runner's double).
``--force-getup-t S`` enters GETUP at S s on every walking robot (a plumbing check of the switch and the hand-back
without a fall; no kick unless ``--kick`` is given). At most 256 robots; it checks ``nvidia-smi`` for 1 GB free
(``--min-free-mb``).

``--schedule LO HI A B`` (``--kick`` is then ignored): repeated
kicks on every robot that has not ended, in every mode (walking, recovering, GETUP): |dv| ~ U(LO, HI) m/s added to the
base's velocity, horizontal, at a direction ~ U(0, 360) deg in the body's yaw frame (0 = from_back, 90 = from_right:
push_eval's convention), every U(A, B) s (the first at U(A, B) s; each interval rounded to the 20 ms step, >= 1 step;
torch generator seeded from ``--seed``, ``KickSchedule``). Commands per robot from ``--sched-commands`` (push_eval's
stand, fwd, left, right, back at 0.10 m/s; contiguous blocks of robots), wz from the heading loop as above. Long
episodes (``--seconds``, e.g. 60) on 16 m flat tiles (a 60 s walk at 0.10 m/s stays on the tile; ``omni_eval``'s 4 m
otherwise); ``--getup-max`` defaults to unlimited in this mode (the runner's 3 would end
every robot within a minute of kicks).
Per kick (``kick_accounting``; steps are the 50 Hz policy steps, a kick at step k fires before the physics of step k):
- ``state``: the mode it lands in: "walk", "recovering" (walk mode, after a hand-back but before walking again) or
  "getup" (in GETUP: the untrained regime, counted, never "toppling").
- ``toppled``: a GETUP entry E is attributed to the latest kick K on that robot that landed in walk mode after the
  robot's previous entry (K < E); an entry with no such kick is "unattributed" (a fall after a hand-back without a
  fresh kick). A kick is toppling when an entry is attributed to it.
- walking again (the same criterion as a single kick): the first step s >= E from which a 0.5 s window
  (25 steps) has, at every step, walk mode, the robot not ended and its true tilt < 15 deg, and the window's mean
  body-frame velocity (vx, vy) within 0.05 m/s of the robot's command; s is the window's first step (the robot is
  confirmed 0.5 s later). The window needs walk mode, so s >= the hand-back.
- ``t_walk_s`` = (s - K) x 20 ms, from the kick to walking again; phases: ``t_detect_s`` = E - K (the fall and the
  runner's detector), ``t_getup_s`` = the GETUP steps in [E, s), ``t_handback_s`` = the rest of [E, s) (from the
  hand-back until the walking window starts, including any re-fall's detection); ``getups`` = entries in [E, s);
  ``kicks_while_down`` = later kicks in (K, s) (to the end if unresolved).
- ``within_5s`` / ``within_8s``: True if walking again by then; False if later, if the robot ended (disarmed: timeout
  or max get-ups) or if the episode left that long and it was not walking again; None (not evaluable) when the
  episode ended earlier than that after the kick.
- a non-toppling kick in walk mode: ``t_walk_s`` = the first walking window starting at or after the kick (reported).
Per robot: down intervals (a GETUP entry -> walking again; the entries before one walking-again merged), the longest
(``longest_down_s``; an interval still open at the end counts with its length so far, ``longest_down_open``; an ended
robot is down for good). Summaries per magnitude bin (0.25 m/s from LO), per direction sector (8, centred on
push_eval's labels), per command and overall; BAM currents per servo by mode at x1 and x1.7 (time above 0.8 A, peak)."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np  # noqa: E402
import torch  # noqa: E402

from mp4_recovery.deploy import runner as rn  # noqa: E402
from mp4_recovery.contract import (has_heading_obs, obs_terms, policy_band_deg, policy_filter_alpha,  # noqa: E402
                                      policy_kind, policy_stand_offset_deg, stand_offset_rel)
from mp4_recovery.heading import pi_step  # noqa: E402

DT = 0.02
LOOP_WZ_MAX = 0.4                  # rad/s: the runner's heading loop cap (omni_eval, push_eval)
WALK_V, WALK_TILT_DEG, WALK_HOLD_S = 0.05, 15.0, 0.5
G_MPS2 = 9.81
MAX_ROBOTS = 256
SCHED_BIN = 0.25                  # m/s: the magnitude bins of the summaries
SECTORS = 8
SCHED_TILE_M = 16.0
HIGH_A, ROBOT_RATIO = 0.8, 1.7     # getup_eval's current threshold and robot/BAM ratio
NO_MAX = 10 ** 6                  # --getup-max in schedule mode: unlimited


# ---------------------------------------------------------------------------------------------- the IMU, vectorised
class TorchGravityFilter:
    """``runner.GravityFilter`` for n robots (torch): ``update(gyro (n, 3) rad/s, acc (n, 3) mg, dt)`` -> tilt deg."""

    def __init__(self, n: int, device, tau_s: float = rn.CF_TAU_S, gate_g: tuple = rn.CF_GATE_G):
        self.tau_s, self.gate_g = tau_s, gate_g
        self.g = torch.full((n, 3), float("nan"), device=device)

    def update(self, gyro, acc, dt: float) -> torch.Tensor:
        self.g = self.g.to(acc.dtype)
        n_a = acc.norm(dim=-1, keepdim=True)
        valid = n_a > rn.ACC_DROPOUT_MG
        a_dir = -acc / n_a.clamp_min(1e-12)
        init = torch.isnan(self.g[:, :1])
        g = torch.where(init, torch.zeros_like(self.g), self.g)
        wn = gyro.norm(dim=-1, keepdim=True)
        th = wn * dt
        k = gyro / wn.clamp_min(1e-12)
        c, s = torch.cos(th), torch.sin(th)
        rot = g * c - torch.linalg.cross(k, g) * s + k * (k * g).sum(-1, keepdim=True) * (1.0 - c)
        g = torch.where(th > 0, rot, g)                       # g turned by -th about the gyro's axis
        lo, hi = self.gate_g
        wt = ((hi - (n_a / 1000.0 - 1.0).abs()) / (hi - lo)).clamp(0.0, 1.0) * valid
        g = g + min(1.0, dt / self.tau_s) * wt * (a_dir - g)
        g = g / g.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        g = torch.where(init, a_dir, g)                       # the first frame within the gate starts it
        self.g = torch.where(init & ~(valid & ((n_a / 1000.0 - 1.0).abs() < hi)), torch.full_like(g, float("nan")), g)
        return self.tilt()

    def tilt(self) -> torch.Tensor:
        return torch.rad2deg(torch.acos((-self.g[:, 2]).clamp(-1.0, 1.0)))


def phase_obs(t_s: torch.Tensor, phase_s: float) -> torch.Tensor:
    """``runner.getup_phase`` for n robots: (phi, sin 2 pi phi, cos 2 pi phi), phi = t_s / phase_s held at 1."""
    phi = (t_s / phase_s).clamp(0.0, 1.0)
    return torch.stack([phi, torch.sin(2 * math.pi * phi), torch.cos(2 * math.pi * phi)], -1)


class KickSchedule:
    """``--schedule LO HI A B`` for n robots: each robot's next kick step (the first at U(A, B) s), and at a due step
    its |dv| ~ U(LO, HI) m/s and its body-yaw-frame direction ~ U(0, 360) deg; the next interval U(A, B) s rounded
    to the step (>= 1). One torch generator, seeded: the same draws on every run with the same seed and n."""

    def __init__(self, n: int, lo: float, hi: float, a: float, b: float, seed: int, device, dt: float = DT):
        assert 0.0 <= lo <= hi and 0.0 < a <= b, (lo, hi, a, b)
        self.lo, self.hi, self.a, self.b, self.dt = float(lo), float(hi), float(a), float(b), dt
        self.g = torch.Generator(device=device).manual_seed(int(seed) * 7 + 103)
        self.n, self.device = n, device
        self.next_k = self._interval()

    def _interval(self) -> torch.Tensor:
        u = torch.rand(self.n, device=self.device, generator=self.g)
        return torch.round((self.a + (self.b - self.a) * u) / self.dt).long().clamp(min=1)

    def due(self, k: int, live: torch.Tensor):
        """(ids, mag m/s, deg) of the robots kicked at step ``k`` (due and live); every due robot is rescheduled."""
        due = self.next_k == k
        mag = self.lo + (self.hi - self.lo) * torch.rand(self.n, device=self.device, generator=self.g)
        deg = 360.0 * torch.rand(self.n, device=self.device, generator=self.g)
        self.next_k = torch.where(due, k + self._interval(), self.next_k)
        ids = torch.nonzero(due & live).flatten()
        return ids, mag[ids], deg[ids]


def robot_commands(names: list[str], n: int) -> tuple[np.ndarray, list[str]]:
    """The command (n, 3) of each robot: ``names`` (push_eval's COMMANDS) in contiguous blocks."""
    from mp4_recovery.eval import push_eval as pe
    per = [names[i * len(names) // n] for i in range(n)]
    return np.array([pe.COMMANDS[c] for c in per], dtype=np.float32), per


def command_array(args, n: int) -> np.ndarray:
    """Every robot's command: the schedule's per-robot commands, else (``--vx``, 0, 0)."""
    if getattr(args, "schedule", None):
        return robot_commands(sched_command_names(args), n)[0]
    return np.tile(np.array([[args.vx, 0.0, 0.0]], dtype=np.float32), (n, 1))


def sched_command_names(args) -> list[str]:
    return [c.strip() for c in args.sched_commands.split(",") if c.strip()]


def sector_of(deg) -> np.ndarray:
    """The direction sector (0..SECTORS-1) of body-frame angles (deg), centred on push_eval's labels (0 = from_back)."""
    w = 360.0 / SECTORS
    return (np.floor((np.asarray(deg, float) % 360.0 + w / 2) / w) % SECTORS).astype(int)


def mag_bins(lo: float, hi: float, width: float = SCHED_BIN) -> list[tuple[float, float]]:
    """Magnitude bins of ``width`` covering [lo, hi] (lo rounded down, hi up to the width)."""
    a = math.floor(lo / width + 1e-9) * width
    b = math.ceil(hi / width - 1e-9) * width
    nb = max(1, int(round((b - a) / width)))
    return [(round(a + i * width, 6), round(a + (i + 1) * width, 6)) for i in range(nb)]


# ---------------------------------------------------------------------------------- the observation's layout
def obs_layout(heading: bool, history: int) -> list[tuple[str, slice, slice]]:
    """Per term: (name, its slice in one frame, its block in the actor's vector). The vector is mjlab's (each term's
    frames oldest to newest, the terms in order; ``contract.HistoryStacker``); history 0 = one frame."""
    L = max(int(history), 1)
    out, pos = [], 0
    for name, sl in obs_terms(heading):
        d = sl.stop - sl.start
        out.append((name, sl, slice(pos, pos + L * d)))
        pos += L * d
    return out


def newest_frame(obs: torch.Tensor, layout) -> torch.Tensor:
    """The newest frame (n, 45 or 46) of a stacked actor vector."""
    frame = torch.empty(obs.shape[0], layout[-1][1].stop, device=obs.device, dtype=obs.dtype)
    for _, sl, blk in layout:
        frame[:, sl] = obs[:, blk.stop - (sl.stop - sl.start):blk.stop]
    return frame


def tiled(frame: torch.Tensor, layout) -> torch.Tensor:
    """The stacked vector whose every slot is ``frame`` (a cleared history's first observation)."""
    L = (layout[0][2].stop - layout[0][2].start) // (layout[0][1].stop - layout[0][1].start)
    return torch.cat([frame[:, sl].repeat(1, L) for _, sl, _ in layout], dim=-1)


class TorchStacker:
    """``contract.HistoryStacker`` for n robots, cleared per robot (a get-up policy with a history)."""

    def __init__(self, length: int, heading: bool, n: int, device):
        self.L = int(length)
        self.layout = obs_layout(heading, length)
        self.frames = None
        self.fresh = torch.ones(n, dtype=torch.bool, device=device)

    def reset(self, ids) -> None:
        self.fresh[ids] = True

    def __call__(self, frame):
        if self.frames is None:
            self.frames = frame[:, None, :].repeat(1, self.L, 1)
        else:
            self.frames = torch.cat([self.frames[:, 1:], frame[:, None, :]], dim=1)
        f = self.fresh
        self.frames[f] = frame[f, None, :].expand(-1, self.L, -1)
        self.fresh = torch.zeros_like(f)
        return torch.cat([self.frames[:, :, sl].reshape(frame.shape[0], -1) for _, sl, _ in self.layout], dim=-1)


def refill_history(env, ids, frame, layout) -> None:
    """The actor group's history buffers of robots ``ids``: every slot = ``frame``'s term (the runner clears the
    walker's history at a hand-back; its first observation then fills every slot). Private mjlab state, checked."""
    om = env.observation_manager
    bufs = om._group_obs_term_history_buffer["actor"]
    for name, sl, _ in layout:
        cb = bufs.get(name)
        if cb is None:
            continue
        assert cb._buffer is not None and cb._buffer.shape[-1] == sl.stop - sl.start, (name, cb._buffer.shape)
        cb._buffer[:, ids] = frame[ids][None, :, sl]
        cb._num_pushes[ids] = cb.max_length
    om._obs_buffer = None


# ------------------------------------------------------------------------------------------------- the env
def gpu_free_mb() -> int | None:
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=20)
        return int(r.stdout.split()[0])
    except Exception:
        return None


def build(args, wpol, gdir: Path):
    from mp4_recovery.eval import omni_eval as oe
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.managers.termination_manager import TerminationTermCfg
    from mjlab.tasks.velocity import mdp as vel_mdp
    from mp4_recovery.gpu.getup_task import CONTACT_SOLREF, THIGH_SHORT_M, _spec_fn
    from mp4_recovery.gpu.terrain_task import TerrainOptions, make_terrain_env_cfg

    kw = oe.env_kwargs(wpol, args.robots, "flat", limits=getattr(args, "limits", None),
                       stand_offset_abd=getattr(args, "stand_offset_abd", None))
    kw["episode_s"] = args.seconds + 5.0
    if getattr(args, "schedule", None):
        kw["tile"] = SCHED_TILE_M
    opt = TerrainOptions(**kw)
    cfg = make_terrain_env_cfg(opt)
    cfg.seed = args.seed
    cfg.terminations = {"time_out": TerminationTermCfg(func=vel_mdp.time_out, time_out=True)}
    plant = {"plant": args.plant}
    if args.plant == "getup":
        o = {}
        cj = gdir / "config.json"
        if cj.is_file():
            o = json.loads(cj.read_text())["task"]["options"]
        coll, tb = o.get("getup_collisions", True), o.get("thigh_base", True)
        short = o.get("thigh_short_m", THIGH_SHORT_M)
        solref = o.get("contact_solref", CONTACT_SOLREF) if cj.is_file() else CONTACT_SOLREF
        robot = cfg.scene.entities["robot"]
        robot.spec_fn = _spec_fn(robot.spec_fn, coll, tb, short, None if solref is None else tuple(solref))
        cfg.sim.njmax = int(o.get("njmax", 1000))
        cfg.sim.nconmax = int(o.get("nconmax", 128))
        plant.update(getup_collisions=coll, thigh_base=tb, thigh_short_m=short,
                     contact_solref=None if solref is None else list(solref), njmax=cfg.sim.njmax,
                     nconmax=cfg.sim.nconmax, from_config=cj.is_file())
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
    return env, kw, plant


class ModeBounds:
    """The decoder's bounds per robot by mode: the walker's (``walk``) or, for robots in GETUP (``getup`` True), the
    get-up's (``limits.decoder_bounds("getup")``)."""

    def __init__(self, walk, getup, n, dev):
        self.w, self.g = walk, getup
        self.getup = torch.zeros(n, dtype=torch.bool, device=dev)

    def clamp_rel(self, rel, ids=None):
        g = self.getup if ids is None else self.getup[ids]
        return torch.where(g[:, None], self.g.clamp_rel(rel), self.w.clamp_rel(rel))


@torch.no_grad()
def run_sequence(env, wpol, gpol, args, step_hook=None) -> dict:
    """The simulation (module docstring). ``step_hook(k, env, mode, entry_k, live)``, if given, is called at every step
    just before ``env.step`` (the state the step's actions act on, after its kicks): ``mode`` (n,) 0 walk / 1 GETUP,
    ``entry_k`` the step of each robot's latest GETUP entry, ``live`` not ended (tools/record_fall_states.py)."""
    from mp4_recovery.eval import omni_eval as oe
    from mp4_recovery.eval import push_eval as pe
    from mp4_recovery.eval import terrain_eval as te
    from mp4_recovery.gpu.mdp import _sensor, servo_action

    n, dev = env.num_envs, env.device
    robot = env.scene["robot"]
    servo = servo_action(env)
    w_head = has_heading_obs(wpol.meta)
    w_layout = obs_layout(w_head, int(wpol.meta.get("history", 0) or 0))
    g_hist = int(gpol.meta.get("history", 0) or 0)
    g_head = has_heading_obs(gpol.meta)
    g_stack = TorchStacker(g_hist, g_head, n, dev) if g_hist else None
    g_phase = float(gpol.meta.get("getup_phase_s", 0.0) or 0.0)                 # runner.getup_phase
    g_scale = torch.tensor(gpol.scale, dtype=torch.float32, device=dev).expand(12)
    g_band = torch.deg2rad(torch.tensor(policy_band_deg(gpol.meta), dtype=torch.float32, device=dev))
    g_off = torch.tensor(stand_offset_rel(policy_stand_offset_deg(gpol.meta)), dtype=torch.float32, device=dev)
    g_alpha = policy_filter_alpha(gpol.meta)                                    # the target filter (0: none)
    C = torch.tensor(command_array(args, n), device=dev)
    cmd = oe.pin_commands(env, C)
    torch.manual_seed(args.seed)
    env.reset()
    te.assign_cells(env, [(r, 0) for r in range(2)])          # omni_eval's flat cells (4 m tiles)
    obs, _ = env.reset()
    yaw0 = oe.start_noise(env, args.seed)
    target = yaw0.clone()
    integ = torch.zeros(n, device=dev)

    # the per-env action configs: the walker's, switched to the get-up's in GETUP
    w_scale = torch.tensor(wpol.scale, dtype=torch.float32, device=dev).expand(12)
    w_band = torch.deg2rad(torch.tensor(policy_band_deg(wpol.meta), dtype=torch.float32, device=dev))
    assert torch.allclose(servo.scale, w_scale) and torch.allclose(servo.band, w_band), "the env is not the walker's"
    servo.scale = w_scale.expand(n, 12).clone()
    servo.band = w_band.expand(n, 12).clone()
    # the target filter per robot (each policy's meta action_filter_alpha; the runner's decoders): only when a policy
    # has one
    w_alpha = policy_filter_alpha(wpol.meta)
    per_env_alpha = bool(w_alpha) or bool(g_alpha)
    if per_env_alpha:
        servo.set_filter_alpha(torch.full((n, 1), w_alpha, device=dev))
    # the limits (the plant's, from the walker's meta): the walker's decoder bounds ("walk", the env's) and the
    # get-up's ("getup"), per robot by mode; the stance offsets per robot (the walker's, the get-up's: their metas)
    w_off = servo.off.clone()
    mb = None
    if servo.bounds is not None:
        from mp4_recovery.contract import Bounds
        from mp4_recovery.limits import decoder_bounds
        from mp4_recovery.gpu.mdp import TorchBounds
        mb = ModeBounds(servo.bounds, TorchBounds(Bounds.from_meta(decoder_bounds("getup").to_meta()),
                                                  servo.tc.stand_phys), n, dev)
        servo.bounds = mb
    per_env_off = bool(w_off.any()) or bool(g_off.any())
    if per_env_off:
        servo.off = w_off.expand(n, 12).clone()
        servo.stand = (servo.tc.stand_phys + servo.off).clone()
        servo._shifted = True

    def switch(ids, scale, band, off=None, getup=False, alpha=0.0):
        """Mode change of robots ``ids``: their scale and band (and bounds, stance offset and target filter), the
        decoder (and so the filter's state) seeded from the measured pose, the velocity filter refilled with the
        newest sample."""
        servo.scale[ids] = scale
        servo.band[ids] = band
        if per_env_alpha:
            servo.alpha[ids] = float(alpha)
        if per_env_off:
            d = servo.off[ids] - off                   # the positions observed in the new policy's coordinates
            servo.hist[ids] = servo.hist[ids] + d[:, None, :]
            servo.off[ids] = off
            servo.stand[ids] = servo.tc.stand_phys + off
        if mb is not None:
            mb.getup[ids] = getup
        meas = servo.pos_rel[ids]
        rel = torch.maximum(torch.minimum(meas, band), -band)
        if mb is not None:
            rel = servo.physical(rel, ids) - servo._sel(servo.off, ids)
        servo.prev_rel[ids] = rel
        servo.applied_action[ids] = rel / scale
        servo._raw[ids] = rel / scale
        servo.target_model[ids] = servo._sel(servo.stand, ids) + rel
        servo.hist[ids] = servo.hist[ids, -1:].expand(-1, servo.hist.shape[1], -1).clone()

    gf = TorchGravityFilter(n, dev)
    gf.g = robot.data.projected_gravity_b.clone()                 # started from a still robot's, as on the robot
    mode = torch.zeros(n, dtype=torch.long, device=dev)          # 0 walk, 1 GETUP
    ended = torch.zeros(n, dtype=torch.bool, device=dev)
    end_code = torch.zeros(n, dtype=torch.long, device=dev)      # 1 timeout, 2 max get-ups
    end_k = torch.full((n,), -1, dtype=torch.long, device=dev)
    n_up = torch.zeros(n, dtype=torch.long, device=dev)
    fallen = torch.zeros(n, dtype=torch.bool, device=dev)
    since = torch.full((n,), -1, dtype=torch.long, device=dev)
    entry_k = torch.full((n,), -1, dtype=torch.long, device=dev)
    phase_k = torch.full((n,), -1, dtype=torch.long, device=dev)  # the phase clock's origin (the entry or a restart)
    restart_s = float(getattr(args, "getup_phase_restart", 0.0) or 0.0)
    phase_on = torch.ones(n, dtype=torch.bool, device=dev)
    first_entry = torch.full((n,), -1, dtype=torch.long, device=dev)
    first_hb = torch.full((n,), -1, dtype=torch.long, device=dev)
    hb_run = torch.zeros(n, dtype=torch.long, device=dev)
    tilt_cf = torch.zeros(n, device=dev)
    gyro_n = torch.zeros(n, device=dev)
    steps = int(round(args.seconds / DT))
    sched = None
    if getattr(args, "schedule", None):
        sched = KickSchedule(n, *args.schedule, seed=args.seed, device=dev)
    k_kick = int(round(args.kick_t / DT)) if (args.kick > 0 and sched is None) else -1
    k_force = int(round(args.force_getup_t / DT)) if args.force_getup_t is not None else -1
    fall_steps = int(round(rn.FALL_HOLD_S / DT))
    deg = torch.full((n,), pe.parse_dir(args.kick_dir), device=dev)
    mag = torch.full((n,), float(args.kick), device=dev)
    rec = {k: [] for k in ("mode", "tilt", "tilt_cf", "vx", "vy", "alive", "first_rel_err")}
    events = []
    klog = {k: [] for k in ("k", "robot", "mag", "deg", "mode")}
    # BAM currents per robot and servo, by mode (0 walk, 1 GETUP), not ended: time above HIGH_A at x1 and x1.7, peak
    cur_hi = {(m, r_): torch.zeros(n, 12, device=dev) for m in (0, 1) for r_ in (1.0, ROBOT_RATIO)}
    cur_pk = {m: torch.zeros(n, 12, device=dev) for m in (0, 1)}
    cur_t = {m: torch.zeros(n, device=dev) for m in (0, 1)}
    cur_any = {(m, r_): torch.zeros(n, device=dev) for m in (0, 1) for r_ in (1.0, ROBOT_RATIO)}
    has_cur = False
    for k in range(steps):
        live = ~ended
        # ---- decisions on the last step's state (the runner: frames -> filter -> mode -> act)
        if k == k_force:
            fallen |= live & (mode == 0)
        enter = live & (mode == 0) & fallen
        over = enter & (n_up >= args.getup_max)
        ended |= over
        end_code = torch.where(over, torch.full_like(end_code, 2), end_code)
        end_k = torch.where(over, torch.full_like(end_k, k), end_k)
        enter &= ~over
        ids_in = torch.nonzero(enter).flatten()
        if len(ids_in):
            mode[ids_in] = 1
            n_up[ids_in] += 1
            entry_k[ids_in] = k
            phase_k[ids_in] = k
            first_entry[ids_in] = torch.where(first_entry[ids_in] < 0, torch.full_like(ids_in, k), first_entry[ids_in])
            hb_run[ids_in] = 0
            fallen[ids_in] = False
            since[ids_in] = -1
            g = gf.g[ids_in]
            switch(ids_in, g_scale, g_band, g_off, True, g_alpha)
            if g_stack is not None:
                g_stack.reset(ids_in)
            roll = torch.rad2deg(torch.atan2(-g[:, 1], -g[:, 2]))
            events += [{"k": k, "robot": int(i), "event": "getup", "n": int(n_up[i]),
                        "roll_deg": round(float(r_), 2),
                        "gravity": [round(v, 5) for v in gg]}
                       for i, r_, gg in zip(ids_in.tolist(), roll.tolist(), g.tolist())]
        rms = torch.rad2deg(servo.pos_rel.pow(2).mean(-1).sqrt())
        ok = (tilt_cf < rn.HANDBACK_TILT_DEG) & (gyro_n < rn.HANDBACK_GYRO) & (rms < rn.HANDBACK_RMS_DEG)
        in_g = live & (mode == 1) & ~enter
        hb_run = torch.where(in_g & ok, hb_run + 1, torch.zeros_like(hb_run))
        hb = in_g & (hb_run >= rn.HANDBACK_STEPS)
        t_in = (k - entry_k).float() * DT
        timeout = in_g & ~hb & (t_in > args.getup_timeout)
        ended |= timeout
        end_code = torch.where(timeout, torch.ones_like(end_code), end_code)
        end_k = torch.where(timeout, torch.full_like(end_k, k), end_k)
        ids_hb = torch.nonzero(hb).flatten()
        if restart_s > 0:                                         # the runner's --getup-phase-restart
            rs = (in_g & ~hb & ~timeout & phase_on & (g_phase > 0)
                  & ((k - phase_k).float() * DT >= g_phase + restart_s - 1e-9)
                  & (tilt_cf > rn.PHASE_RESTART_TILT_DEG))
            if bool(rs.any()):
                phase_k = torch.where(rs, torch.full_like(phase_k, k), phase_k)
                events += [{"k": k, "robot": int(i), "event": "phase-restart", "t_in_s": round((k - int(entry_k[i])) * DT, 3)}
                           for i in torch.nonzero(rs).flatten().tolist()]
        yaw = te.yaw_of(robot.data.root_link_quat_w)
        if len(ids_hb):
            mode[ids_hb] = 0
            first_hb[ids_hb] = torch.where(first_hb[ids_hb] < 0, torch.full_like(ids_hb, k), first_hb[ids_hb])
            switch(ids_hb, w_scale, w_band, w_off, False, w_alpha)
            target[ids_hb] = yaw[ids_hb]                          # the heading loop re-latched
            integ[ids_hb] = 0.0
            events += [{"k": k, "robot": int(i), "event": "hand-back"} for i in ids_hb.tolist()]
        # ---- the command: the heading loop on walking robots
        err = te.wrap(target - yaw)
        w, integ_new = pi_step(err, integ, DT, wz_max=LOOP_WZ_MAX)
        walking = mode == 0
        integ = torch.where(walking, integ_new, integ)
        cmd.vel_command_b[:, :2] = C[:, :2]
        cmd.vel_command_b[:, 2] = torch.where(walking, w, torch.zeros_like(w))
        if k == k_kick:
            ids = torch.nonzero(live).flatten()
            pe.add_base_velocity(robot, ids, pe.kick_world(yaw[ids], deg[ids], mag[ids]))
        if sched is not None:
            ids, m_k, d_k = sched.due(k, live)
            if len(ids):
                pe.add_base_velocity(robot, ids, pe.kick_world(yaw[ids], d_k, m_k))
                klog["k"].append(torch.full_like(ids, k))
                klog["robot"].append(ids)
                klog["mag"].append(m_k)
                klog["deg"].append(d_k)
                klog["mode"].append(mode[ids].clone())
        # ---- the observations and the actions
        o_w = obs["actor"].clone()
        frame = newest_frame(o_w, w_layout)
        fresh = torch.zeros(n, dtype=torch.bool, device=dev)
        fresh[ids_in] = True
        fresh[ids_hb] = True
        if len(ids_in) or len(ids_hb):                            # the switch's frame: velocity 0, the seeded action
            sl = dict((nm, s) for nm, s, _ in w_layout)
            frame[fresh, sl["vel"]] = 0.0
            frame[fresh, sl["prev_action"]] = servo.applied_action[fresh]
        if len(ids_hb):
            frame[ids_hb, sl["command"]] = cmd.vel_command_b[ids_hb]
            o_w[ids_hb] = tiled(frame[ids_hb], w_layout)
            refill_history(env, ids_hb, frame, w_layout)
        o_g = frame[:, :45].clone()
        o_g[:, 6:9] = 0.0                                         # the get-up observes the command 0
        o_g[:, 33:45] = servo.applied_action
        a = wpol(o_w)
        if g_phase > 0:                                           # ... or its phase since this GETUP's entry
            t_ph = (k - phase_k).float() * DT
            o_g[:, 6:9] = phase_obs(torch.where(phase_on, t_ph, torch.zeros_like(t_ph)), g_phase)
        if g_head:
            o_g = torch.cat([o_g, torch.zeros(n, 1, device=dev)], -1)
        a_g = gpol(g_stack(o_g) if g_stack is not None else o_g)
        a = torch.where((mode == 1)[:, None], a_g, a)
        prev_rel = servo.prev_rel.clone()
        meas = servo.pos_rel.clone()
        if step_hook is not None:
            step_hook(k, env, mode, entry_k, ~ended)
        obs, _, _, _, _ = env.step(a)
        # the first target after a switch moves at most the step limit from the measured pose
        err_first = torch.where(fresh[:, None], (servo.prev_rel - meas).abs(), torch.zeros_like(meas)).amax(-1)
        if fresh.any():                                           # the switch's seed: band, then the bounds
            fids = fresh.nonzero().squeeze(-1)
            want = torch.maximum(torch.minimum(meas[fresh], servo.band[fresh]), -servo.band[fresh])
            if servo.bounds is not None:
                want = servo.physical(want, fids) - servo._sel(servo.off, fids)
            assert torch.allclose(prev_rel[fresh], want, atol=1e-5)
        # ---- the simulated IMU -> the filter and the fall detector (walking robots)
        gyro = _sensor(env, "robot/imu_ang_vel")
        acc = _sensor(env, "robot/imu_lin_acc") / G_MPS2 * 1000.0
        tilt_cf = gf.update(gyro, acc, DT)
        gyro_n = gyro.norm(dim=-1)
        cur = te.bam_currents(env)
        if cur is not None:
            has_cur = True
            for m in (0, 1):
                on = ((mode == m) & ~ended).float()
                cur_t[m] += on * DT
                for r_ in (1.0, ROBOT_RATIO):
                    hi_ = cur * r_ > HIGH_A
                    cur_hi[(m, r_)] += hi_.float() * on[:, None] * DT
                    cur_any[(m, r_)] += hi_.any(-1).float() * on * DT
                cur_pk[m] = torch.maximum(cur_pk[m], cur * on[:, None])
        walking = (mode == 0) & ~ended
        hi = walking & (tilt_cf > rn.FALL_TILT_DEG)
        since = torch.where(hi, torch.where(since < 0, torch.full_like(since, k), since), torch.full_like(since, -1))
        fallen |= hi & ((k - since) >= fall_steps)
        fallen |= walking & (tilt_cf > rn.FALL_FAST_TILT_DEG) & (gyro_n > rn.FALL_FAST_GYRO)
        vb = robot.data.root_link_lin_vel_b
        rec["mode"].append(mode.clone())
        rec["tilt"].append(pe.tilt_deg(robot))
        rec["tilt_cf"].append(tilt_cf.clone())
        rec["vx"].append(vb[:, 0].clone())
        rec["vy"].append(vb[:, 1].clone())
        rec["alive"].append(~ended)
        rec["first_rel_err"].append(err_first)
    r = {k: torch.stack(v).cpu().numpy() for k, v in rec.items()}
    r.update(n_up=n_up.cpu().numpy(), end_code=end_code.cpu().numpy(), end_k=end_k.cpu().numpy(),
             first_entry=first_entry.cpu().numpy(), first_hb=first_hb.cpu().numpy(), events=events,
             k_kick=k_kick, scale_end=servo.scale.cpu().numpy(), band_end=servo.band.cpu().numpy(),
             C=C.cpu().numpy())
    r["kicks"] = {k: (torch.cat(v).cpu().numpy() if v else np.zeros(0)) for k, v in klog.items()}
    for k in ("k", "robot", "mode"):
        r["kicks"][k] = r["kicks"][k].astype(int)
    r["currents"] = None if not has_cur else {
        "time_s": {m: cur_t[m].cpu().numpy() for m in (0, 1)},
        "above": {(m, r_): cur_hi[(m, r_)].cpu().numpy() for m in (0, 1) for r_ in (1.0, ROBOT_RATIO)},
        "any": {(m, r_): cur_any[(m, r_)].cpu().numpy() for m in (0, 1) for r_ in (1.0, ROBOT_RATIO)},
        "peak": {m: cur_pk[m].cpu().numpy() for m in (0, 1)}}
    return r


def walk_ok(r, cmd) -> np.ndarray:
    """(T - hold + 1, n): the 0.5 s walking window starting at each step qualifies: walk mode, not ended and the tilt
    below WALK_TILT_DEG at every step, and the window's mean body velocity within WALK_V of the command (``cmd``: the
    forward speed, or (n, 3) / (n, 2) per robot)."""
    T, n = r["mode"].shape
    hold = int(round(WALK_HOLD_S / DT))
    cmd = np.asarray(cmd, float)
    cx, cy = (cmd[:, 0], cmd[:, 1]) if cmd.ndim == 2 else (np.full(n, float(cmd)), np.zeros(n))
    up = (r["mode"] == 0) & r["alive"] & (r["tilt"] < WALK_TILT_DEG)
    c = np.concatenate([np.zeros((1, n), int), np.cumsum(up, axis=0)], 0)
    up_all = (c[hold:] - c[:-hold]) == hold
    cv = lambda x: np.concatenate([np.zeros((1, n)), np.cumsum(np.where(up, x, 0.0), axis=0)], 0)  # noqa: E731
    vx_m, vy_m = (cv(r["vx"])[hold:] - cv(r["vx"])[:-hold]) / hold, (cv(r["vy"])[hold:] - cv(r["vy"])[:-hold]) / hold
    return up_all & (np.hypot(vx_m - cx[None], vy_m - cy[None]) < WALK_V)


def first_walk(r, start, vx) -> np.ndarray:
    """Per robot, the first step s >= start[e] (-1: none asked) from which, over WALK_HOLD_S, it is walking (mode 0,
    not ended), its tilt below WALK_TILT_DEG at every step and its mean body velocity within WALK_V of the command
    (``walk_ok``)."""
    ok = walk_ok(r, vx)
    s = np.arange(ok.shape[0])[:, None]
    ok &= (s >= start[None, :]) & (start[None, :] >= 0)
    return np.where(ok.any(0), ok.argmax(0), -1)


def next_true(ok: np.ndarray, T: int) -> np.ndarray:
    """(T + 1, n): the smallest s' >= s with ok[s'] (rows of ``ok`` beyond its length count as False), else -1."""
    n = ok.shape[1]
    out = np.full((T + 1, n), -1, dtype=np.int64)
    for s in range(T - 1, -1, -1):
        out[s] = np.where(ok[s], s, out[s + 1]) if s < ok.shape[0] else out[s + 1]
    return out


def kick_accounting(mode, ok, kicks: dict, entries, end_k, end_code, T: int, hold: int) -> dict:
    """Per-kick and per-robot accounting of a schedule run (the module's docstring states the rules).

    mode (T, n) 0 walk / 1 GETUP after the step's decisions; ok: ``walk_ok`` (window starts); kicks: arrays k, robot,
    mag, deg, mode (the mode it landed in); entries: [(k, robot)] GETUP entries; end_k / end_code (n,): the step a
    robot ended (-1: never) and why (1 timeout, 2 max get-ups). Returns {"kicks": per-kick lists, "robots": ...}."""
    n = mode.shape[1]
    nxt = next_true(ok, T)
    ent = [sorted(k for k, e in entries if e == r_) for r_ in range(n)]
    kk_all, kr_all = np.asarray(kicks["k"], int), np.asarray(kicks["robot"], int)
    order = np.lexsort((kk_all, kr_all))
    nk = len(kk_all)
    rec = {f: [None] * nk for f in ("state", "toppled", "t_detect_s", "t_getup_s", "t_handback_s", "t_walk_s",
                                     "getups", "kicks_while_down", "resolved", "ended", "down_so_far_s",
                                     "within_5s", "within_8s", "entry_k", "walk_k")}
    per_robot_kicks = [[] for _ in range(n)]
    for j in order:
        per_robot_kicks[kr_all[j]].append(int(j))
    attributed = {}                                         # (robot, entry k) -> kick index
    unattributed = []
    for e in range(n):
        prev = -1
        for E in ent[e]:
            cand = [j for j in per_robot_kicks[e] if prev < kk_all[j] < E and int(kicks["mode"][j]) == 0]
            if cand:
                attributed[(e, E)] = cand[-1]
            else:
                unattributed.append((e, E))
            prev = E
    top_of = {j: E for (e, E), j in attributed.items()}
    t_end = np.where(end_k >= 0, np.minimum(end_k, T), T)
    win_last = T - hold                                     # the last step a window can start
    for j in range(nk):
        e, K, mk = int(kr_all[j]), int(kk_all[j]), int(kicks["mode"][j])
        if mk == 1:
            state = "getup"
        else:
            past = [E for E in ent[e] if E < K]
            state = "walk"
            if past:
                w = nxt[past[-1], e]
                if w < 0 or w > K:
                    state = "recovering"
        rec["state"][j] = state
        rec["toppled"][j] = j in top_of
        rec["ended"][j] = bool(end_k[e] >= 0)
        if j in top_of:
            E = top_of[j]
            w = int(nxt[E, e])
            later = [i for i in per_robot_kicks[e] if kk_all[i] > K and (w < 0 or kk_all[i] < w)]
            rec["entry_k"][j] = E
            rec["t_detect_s"][j] = round((E - K) * DT, 3)
            rec["kicks_while_down"][j] = len(later)
            if w >= 0:
                g = int((mode[E:w, e] == 1).sum())
                rec["walk_k"][j] = w
                rec["resolved"][j] = True
                rec["t_walk_s"][j] = round((w - K) * DT, 3)
                rec["t_getup_s"][j] = round(g * DT, 3)
                rec["t_handback_s"][j] = round((w - E - g) * DT, 3)
                rec["getups"][j] = sum(1 for x in ent[e] if E <= x < w)
            else:
                rec["resolved"][j] = False
                rec["getups"][j] = sum(1 for x in ent[e] if x >= E)
                rec["down_so_far_s"][j] = round((t_end[e] - K) * DT, 3)
            avail = (win_last - K) * DT
            for lim, f in ((5.0, "within_5s"), (8.0, "within_8s")):
                if w >= 0:
                    rec[f][j] = (w - K) * DT <= lim + 1e-9
                elif end_k[e] >= 0 or avail >= lim - 1e-9:
                    rec[f][j] = False
        elif mk == 0:
            w = int(nxt[K, e]) if K < T else -1
            if w >= 0:
                rec["walk_k"][j] = w
                rec["t_walk_s"][j] = round((w - K) * DT, 3)
    robots = {f: [] for f in ("kicks", "kicks_walk", "toppled", "getups", "unattributed_falls", "longest_down_s",
                              "longest_down_open", "down_intervals", "end")}
    for e in range(n):
        groups = {}
        for E in ent[e]:
            w = int(nxt[E, e])
            groups.setdefault(w if w >= 0 else -1, []).append(E)
        ivs = []
        for w, Es in groups.items():
            st = min(Es)
            ivs.append(((w - st) * DT, False) if w >= 0 else ((t_end[e] - st) * DT, True))
        longest = max(ivs, key=lambda x: x[0]) if ivs else (0.0, False)
        ks = per_robot_kicks[e]
        robots["kicks"].append(len(ks))
        robots["kicks_walk"].append(sum(1 for j in ks if int(kicks["mode"][j]) == 0))
        robots["toppled"].append(sum(1 for j in ks if j in top_of))
        robots["getups"].append(len(ent[e]))
        robots["unattributed_falls"].append(sum(1 for (x, _) in unattributed if x == e))
        robots["longest_down_s"].append(round(float(longest[0]), 3))
        robots["longest_down_open"].append(bool(longest[1]))
        robots["down_intervals"].append(len(ivs))
        robots["end"].append(["", "timeout", "max_getups"][int(end_code[e])])
    return {"kicks": rec, "robots": robots, "unattributed": [[int(e), int(E)] for e, E in unattributed]}


def _share(xs):
    v = [x for x in xs if x is not None]
    return (float(np.mean(v)) if v else None), len(v)


def _pct(xs, q):
    v = [x for x in xs if x is not None]
    return float(np.percentile(v, q)) if v else None


def kick_group_summary(sel: list[int], K: dict) -> dict:
    """The summary of the kicks ``sel`` (indices into ``K``, the per-kick lists)."""
    walk = [j for j in sel if K["state"][j] != "getup"]
    top = [j for j in walk if K["toppled"][j]]
    w8, n8 = _share([K["within_8s"][j] for j in top])
    w5, n5 = _share([K["within_5s"][j] for j in top])
    out = {"kicks": len(sel), "kicks_walk_mode": len(walk), "kicks_in_getup": len(sel) - len(walk),
           "kicks_recovering": sum(1 for j in sel if K["state"][j] == "recovering"),
           "toppled": len(top), "topple_rate": (len(top) / len(walk)) if walk else None,
           "within_8s": w8, "within_5s": w5, "evaluable_8s": n8, "evaluable_5s": n5,
           "not_evaluable_8s": len(top) - n8, "unresolved": sum(1 for j in top if not K["resolved"][j]),
           "ended": sum(1 for j in top if K["ended"][j] and not K["resolved"][j]),
           "with_kicks_while_down": sum(1 for j in top if (K["kicks_while_down"][j] or 0) > 0)}
    for f in ("t_walk_s", "t_detect_s", "t_getup_s", "t_handback_s"):
        v = [K[f][j] for j in top if K["resolved"][j]]
        out[f + "_p50"], out[f + "_p90"] = _pct(v, 50), _pct(v, 90)
        out[f + "_max"] = float(max(v)) if v else None
    nt = [K["t_walk_s"][j] for j in walk if not K["toppled"][j]]
    out["not_toppled_t_walk_s_p50"], out["not_toppled_t_walk_s_p90"] = _pct(nt, 50), _pct(nt, 90)
    return out


def schedule_metrics(r, cmd_names: list[str], lo: float, hi: float) -> dict:
    """The schedule run's per-kick records, per-robot records and summaries (by magnitude bin, direction sector,
    command, overall), and the currents."""
    from mp4_recovery.eval import push_eval as pe
    T, n = r["mode"].shape
    hold = int(round(WALK_HOLD_S / DT))
    ok = walk_ok(r, r["C"])
    entries = [(e["k"], e["robot"]) for e in r["events"] if e["event"] == "getup"]
    acc = kick_accounting(r["mode"], ok, r["kicks"], entries, r["end_k"], r["end_code"], T, hold)
    kd = r["kicks"]
    K = acc["kicks"]
    sec = sector_of(kd["deg"]) if len(kd["k"]) else np.zeros(0, int)
    labels = [pe.dir_label(360.0 * i / SECTORS) for i in range(SECTORS)]
    K = {"k": kd["k"].tolist(), "t_s": [round(k * DT, 3) for k in kd["k"].tolist()], "robot": kd["robot"].tolist(),
         "command": [cmd_names[e] for e in kd["robot"].tolist()],
         "mag_mps": [round(float(x), 4) for x in kd["mag"]], "dir_deg": [round(float(x), 2) for x in kd["deg"]],
         "sector": [labels[i] for i in sec], **K}
    idx = list(range(len(K["k"])))
    bins = mag_bins(lo, hi)
    by_mag = {}
    for i, (a, b) in enumerate(bins):
        last = i == len(bins) - 1
        sel = [j for j in idx if a <= K["mag_mps"][j] < b or (last and K["mag_mps"][j] == b)]
        by_mag[f"{a:.2f}-{b:.2f}"] = kick_group_summary(sel, K)
    by_dir = {lab: kick_group_summary([j for j in idx if K["sector"][j] == lab], K) for lab in labels}
    by_cmd = {c: kick_group_summary([j for j in idx if K["command"][j] == c], K) for c in dict.fromkeys(cmd_names)}
    R = acc["robots"]
    lon = np.array(R["longest_down_s"])
    opn = np.array(R["longest_down_open"])
    ended = np.array([x != "" for x in R["end"]])
    over15 = (lon > 15.0) | (ended & (np.array(R["getups"]) > 0))   # an ended robot is down for good
    summary = {"n": int(n), "seconds": T * DT, "overall": kick_group_summary(idx, K),
               "longest_down_s_max": float(lon.max()) if n else None,
               "longest_down_s_p90": float(np.percentile(lon, 90)) if n else None,
               "robots_down_over_15s": int(over15.sum()),
               "robots_down_open_at_end": int((opn & ~ended).sum()),
               "robots_ended": int(ended.sum()), "ended_timeout": int(sum(x == "timeout" for x in R["end"])),
               "ended_max_getups": int(sum(x == "max_getups" for x in R["end"])),
               "unattributed_falls": len(acc["unattributed"]), "getups_total": int(sum(R["getups"]))}
    cur = None
    if r.get("currents"):
        c = r["currents"]
        cur = {}
        for m, name in ((0, "walk"), (1, "getup")):
            t = c["time_s"][m]
            d = {"robot_time_s": float(t.sum()), "peak_A_per_servo": np.round(c["peak"][m].max(0), 4).tolist(),
                 "peak_A": float(c["peak"][m].max())}
            for r_, lab in ((1.0, "x1"), (ROBOT_RATIO, "x1p7")):
                a = c["above"][(m, r_)]
                d[f"time_above_0p8A_{lab}_per_servo_s"] = np.round(a.sum(0), 3).tolist()
                y = c["any"][(m, r_)]
                d[f"time_above_0p8A_{lab}_any_servo_share"] = float(y.sum() / max(t.sum(), 1e-9))
                d[f"time_above_0p8A_{lab}_any_servo_robot_max_s"] = float(y.max())
            cur[name] = d
        cur["note"] = ("BAM |tau|/kt per servo sampled per policy step, by mode, robots not ended; x1.7: the current "
                       "x 1.7 (the robot / BAM ratio); per_servo_s: summed over robots; any_servo_share: the time any servo is "
                       "above 0.8 A over the robots' time in the mode; robot_max_s: the largest robot's total")
    return {"summary": summary, "by_mag": by_mag, "by_dir": by_dir, "by_command": by_cmd, "kicks": K,
            "robots": R, "unattributed": acc["unattributed"], "currents": cur,
            "bins": [list(b) for b in bins], "sectors": labels}


def metrics(r, vx: float) -> dict:
    n = r["mode"].shape[1]
    kk = r["k_kick"]
    fe, hb = r["first_entry"], r["first_hb"]
    fell = fe >= max(kk, 0)
    start = np.where(fell, hb, kk if kk >= 0 else -1)
    sw = first_walk(r, start, vx)
    nan = np.nan
    t = lambda a, b, cond: np.where(cond & (a >= 0) & (b >= 0), (a - b) * DT, nan)  # noqa: E731
    out = {"fell": fell, "t_fall_s": t(fe, kk, fell), "t_up_s": t(hb, fe, fell),
           "t_walk_s": t(sw, start, start >= 0), "t_walk_from_kick_s": t(sw, kk, (sw >= 0) & (kk >= 0)),
           "down_s": t(sw, fe, fell), "getup_time_s": ((r["mode"] == 1) & r["alive"]).sum(0) * DT, "getups": r["n_up"],
           "end": np.array(["", "timeout", "max_getups"])[r["end_code"]],
           "t_end_s": np.where(r["end_k"] >= 0, r["end_k"] * DT, nan),
           "first_target_err_deg_max": np.degrees(r["first_rel_err"].max(0))}
    fin = lambda x: [None if (isinstance(v, float) and math.isnan(v)) else v for v in x]  # noqa: E731
    per = {k: (v.astype(int).tolist() if v.dtype == bool else fin(np.round(v.astype(float), 3).tolist())
               if v.dtype.kind == "f" else v.tolist()) for k, v in out.items()}
    f = out["fell"]
    med = lambda x: float(np.nanmedian(x)) if np.isfinite(x).any() else None  # noqa: E731
    summary = {"n": int(n), "fell": float(f.mean()), "up": float((out["t_up_s"] >= 0).sum() / max(f.sum(), 1)),
               "walking_again": float(np.isfinite(out["t_walk_s"]).mean()),
               "t_fall_s_p50": med(out["t_fall_s"]), "t_up_s_p50": med(out["t_up_s"]),
               "t_walk_s_p50": med(out["t_walk_s"]), "down_s_p50": med(out["down_s"]),
               "gave_up": int((out["end"] == "timeout").sum()), "max_getups_ended": int((out["end"] == "max_getups").sum()),
               "getups_total": int(out["getups"].sum()),
               "first_target_err_deg_max": float(np.max(out["first_target_err_deg_max"]))}
    return {"summary": summary, "robots": per}


def load_policy(path: str):
    from mp4_recovery.eval.terrain_eval import TorchPolicy
    return TorchPolicy(Path(path) / "policy.npz", "cuda:0")


def main(argv=None, step_hook=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--walker", required=True, help="a walking policy directory (policy.npz [+ config.json])")
    ap.add_argument("--getup", required=True, help="a get-up policy directory (meta kind 'getup')")
    ap.add_argument("--robots", type=int, default=8)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--seconds", type=float, default=16.0)
    ap.add_argument("--vx", type=float, default=0.10)
    ap.add_argument("--kick", type=float, default=1.5, help="m/s (0: none)")
    ap.add_argument("--kick-dir", default="from_right", help="push_eval's labels (from_back, from_right, ...) or deg")
    ap.add_argument("--kick-t", type=float, default=3.0)
    ap.add_argument("--getup-max", type=int, default=None, help="get-ups before a fall ends the trial (default 3; "
                                                                     "unlimited with --schedule)")
    ap.add_argument("--getup-timeout", type=float, default=10.0)
    ap.add_argument("--getup-phase-restart", type=float, default=0.0, metavar="S",
                    help="the runner's option: a phased get-up's clock restarts (phi 0) when phi has been 1 for S s "
                         "and the filtered tilt is above 45 deg (0: off)")
    ap.add_argument("--plant", choices=("getup", "walker"), default="getup")
    ap.add_argument("--force-getup-t", type=float, help="s: enter GETUP on every walking robot (a plumbing check)")
    ap.add_argument("--json", required=True)
    ap.add_argument("--trace", help="npz: the per-step arrays (mode, 0 walk, 1 get-up; tilt, tilt_cf, vx, vy, alive), steps "
                                    "x robots")
    ap.add_argument("--min-free-mb", type=int, default=1024)
    from mp4_recovery.limits import add_eval_args
    add_eval_args(ap)
    ap.add_argument("--schedule", type=float, nargs=4, metavar=("LO", "HI", "A", "B"),
                    help="repeated kicks |dv| ~ U(LO, HI) m/s, any direction, every U(A, B) s ")
    ap.add_argument("--sched-commands", default="stand,fwd,left,right",
                    help="--schedule: the robots' commands (push_eval's names), in contiguous blocks")
    a = ap.parse_args(argv)
    if a.getup_max is None:
        a.getup_max = NO_MAX if a.schedule else 3
    if a.schedule:
        from mp4_recovery.eval import push_eval as pe
        bad = [c for c in sched_command_names(a) if c not in pe.COMMANDS]
        if bad or not sched_command_names(a):
            print(f"refused: --sched-commands {a.sched_commands!r} (known: {', '.join(pe.COMMANDS)})")
            return 2
    assert 1 <= a.robots <= MAX_ROBOTS, f"keep to <= {MAX_ROBOTS} robots"
    dirs = [a.walker, a.getup]
    metas = [json.loads(str(np.load(Path(d) / "policy.npz")["meta"])) for d in dirs]
    kinds = [policy_kind(m) for m in metas]
    if kinds[0] != "walk" or any(k_ != "getup" for k_ in kinds[1:]):
        print(f"refused: kinds {' / '.join(map(repr, kinds))}, need 'walk' / 'getup'")
        return 2
    for why in (rn.band_refusal(m) for m in metas):
        if why:
            print(f"refused: {why}")
            return 2
    free = gpu_free_mb()
    if free is not None and free < a.min_free_mb:
        print(f"refused: {free} MB free on the GPU (< {a.min_free_mb})", flush=True)
        return 2
    wpol, gpol = load_policy(a.walker), load_policy(a.getup)
    gdir = Path(a.getup)
    env, kw, plant = build(a, wpol, gdir)
    r = run_sequence(env, wpol, gpol, a, step_hook=step_hook)
    env.close()
    if a.schedule:
        m = schedule_metrics(r, robot_commands(sched_command_names(a), a.robots)[1], a.schedule[0], a.schedule[1])
        m["mode"] = "schedule"
        m["schedule"] = {"lo_mps": a.schedule[0], "hi_mps": a.schedule[1], "interval_s": list(a.schedule[2:]),
                         "commands": sched_command_names(a), "tile_m": SCHED_TILE_M,
                         "direction": "U(0, 360) deg in the body's yaw frame (0 = from_back, 90 = from_right)",
                         "kicks_in": "every mode (walk, recovering, GETUP); none on ended robots"}
        m["robot_commands"] = robot_commands(sched_command_names(a), a.robots)[1]
    else:
        m = metrics(r, a.vx)
        m["mode"] = "single"
    m["limits_on"] = kw.get("limits", False)
    m["stand_offset_abd"] = kw.get("stand_offset_abd")
    if a.trace:
        Path(a.trace).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(a.trace, **{k: r[k] for k in ("mode", "tilt", "tilt_cf", "vx", "vy", "alive", "first_rel_err")})
    root = Path(__file__).resolve().parents[3]
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=root).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True,
                           cwd=root).stdout.strip()
    out = {"walker": a.walker, "getup": a.getup, "walker_meta": {k: v for k, v in wpol.meta.items() if k != "checkpoint"},
           "getup_meta": {k: v for k, v in gpol.meta.items() if k != "checkpoint"},
           "walker_scale": list(wpol.scale), "getup_scale": list(gpol.scale), "argv": sys.argv if argv is None else argv,
           "repo_commit": commit, "repo_dirty": bool(dirty), "seed": a.seed, "robots": a.robots, "seconds": a.seconds, "vx": a.vx,
           "kick": {"mps": a.kick, "dir": a.kick_dir, "t_s": a.kick_t}, "getup_max": None if a.getup_max >= NO_MAX else a.getup_max,
           "getup_timeout_s": a.getup_timeout, "getup_phase_restart_s": a.getup_phase_restart,
           "force_getup_t": a.force_getup_t, "plant": plant,
           "rules": {"cf_tau_s": rn.CF_TAU_S, "cf_gate_g": list(rn.CF_GATE_G), "imu_rate_hz": 1 / DT,
                     "fall": [rn.FALL_TILT_DEG, rn.FALL_HOLD_S, rn.FALL_FAST_TILT_DEG, rn.FALL_FAST_GYRO],
                     "handback": [rn.HANDBACK_TILT_DEG, rn.HANDBACK_GYRO, rn.HANDBACK_RMS_DEG, rn.HANDBACK_STEPS],
                     "walking_again": {"v_mps": WALK_V, "tilt_deg": WALK_TILT_DEG, "hold_s": WALK_HOLD_S}},
           "getup_phase_s": float(gpol.meta.get("getup_phase_s", 0.0) or 0.0),
           "walker_band_deg": policy_band_deg(wpol.meta).tolist(), "getup_band_deg": policy_band_deg(gpol.meta).tolist(),
           "limits": bool(kw.get("limits")), "walker_stand_offset_abd_deg": float(kw.get("stand_offset_abd_deg", 0.0)),
           "getup_stand_offset_abd_deg": policy_stand_offset_deg(gpol.meta),
           "walker_filter_alpha": policy_filter_alpha(wpol.meta), "getup_filter_alpha": policy_filter_alpha(gpol.meta),
           "events": r["events"], **m}
    Path(a.json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.json).write_text(json.dumps(out, indent=1))
    if a.schedule:
        print(json.dumps({k: v for k, v in m["summary"].items() if k != "overall"}), flush=True)
        for b, v in m["by_mag"].items():
            print(f"  {b}: kicks {v['kicks']} toppled {v['toppled']} within8 {v['within_8s']} "
                  f"within5 {v['within_5s']} t_walk p50 {v['t_walk_s_p50']}", flush=True)
    else:
        print(json.dumps(m["summary"]), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
