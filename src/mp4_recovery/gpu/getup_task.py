"""The get-up task: an episodic policy that rights the fallen robot and stands it at the stand pose.

Built on the walking task's plant (``make_terrain_env_cfg``: the fitted servo model, the masses, the feet's contact, the
friction, center-of-mass, mass and delay draws, the observation noise and its ramp) on one flat tile, and changed:

- **Episodes** of 6 s by default. The only terminations are the time-out, a guard against a physics blow-up
  (``physics_blowup``), a finiteness guard (``NonfiniteGuard``) and optionally the success (``end_on_success``). A
  fallen robot is not a failure here: there is nothing else to do but get up.
- **Observations**: the walker's 45 floats (``contract.py``) with the command replaced by the get-up's phase
  (``obs_phase``) and no history. The critic adds the true base velocity, the base height and the foot contacts.
- **Action**: stand + clip(a, +-1) x 1.57 rad on every joint (the band's width where that is wider: ``getup_scale``),
  inside a band about the stand per joint (``band_deg``), at most 20 deg per step. The decoder is seeded from the
  measured pose at every reset, as the robot's decoder is on a mode switch.
- **Starts** (``GetupStart``, every reset): the base origin 0.06-0.10 m above the floor, the joints the stand + U(+-0.5
  band), the orientation drawn per slice from a mix ramped by iteration from ``MIX_START`` to ``MIX_END`` at
  ``curriculum_iters`` (a reverse curriculum): crouch (tilt < 30 deg), partial roll (|roll| 30-80 deg), side (roll +-90
  +- 20 deg), nose or tail down (pitch +-60-90 deg), back (roll 180 +- 30 deg), uniform random, and mid-fall (near the
  stand, tilted 40-70 deg about a random horizontal axis and rotating 2-6 rad/s further about it). A spawn whose robot
  overlaps the floor or itself by more than 2 mm is drawn again.
- **Rewards** (mjlab multiplies every term by dt, so a weight is per second of the term's value; the progress terms are
  signed differences divided by dt, so an episode's sum telescopes and oscillating earns nothing):
  ``upright_progress`` 5 x d cos(tilt); ``height_progress`` 30 x d min(z, 0.0725 m); ``fallen_tax`` -0.5 per second
  while tilt > 25 deg or z < 0.05 m; ``success`` +10 once per episode when the criterion has held 0.5 s (tilt < 12 deg,
  joint rms < 10 deg from the stand, base above 0.05 m, and optionally |gyro| < 0.8 rad/s; without the height a robot on
  its belly in the stand pose passes); ``stand_track`` exp(-rms^2 / 0.1) gated on tilt < 30 deg and z > 0.05 m;
  ``action_rate`` (``action_rate_weight``, x ``action_rate_tilted_scale`` while tilt > 40 deg); ``torques`` -1e-4;
  ``joint_vel`` (``joint_vel_weight``).
- **Collisions** (``getup_collisions``, this task only): leg <-> leg (thigh and calf capsules and the feet of different
  legs) and leg <-> hip cans, on top of the walker's shin and foot <-> base. MuJoCo filters parent-child pairs only, so a
  calf against its own leg's hip can stays a pair. Without the measured limits thigh <-> base is on too, each thigh
  capsule shortened at its hip end (the model's thighs overlap the base box at the stand). With the limits that pair is
  off: its inward contact comes at 2-7 deg against the robot's measured 14-22, and the decoder's hip envelope covers it.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass

import mujoco
import numpy as np
import torch

from rsl_rl.modules.distribution import GaussianDistribution

from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.manager_base import ManagerTermBase
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.velocity import mdp as vel_mdp
from mjlab.utils.lab_api.math import quat_apply, quat_from_angle_axis, quat_mul, random_orientation

from mp4_recovery.gpu import mdp
from mp4_recovery.gpu.env_cfg import STEPS_PER_ITER
from mp4_recovery.gpu.terrain_task import TerrainOptions, make_terrain_env_cfg, servo_limits
from mp4_recovery.robot.sku2 import INIT_BASE_Z, LEGS

GETUP_SCALE = 1.57                  # rad per unit action, every joint
GETUP_BAND_DEG = (90.0, 90.0, 90.0)  # abduction, hip, knee about the stand
SLICES = ("crouch", "partial", "side", "nose", "back", "random", "midfall")
MIX_START = (0.40, 0.30, 0.20, 0.10, 0.00, 0.00, 0.00)    # iteration 0
MIX_END = (0.15, 0.15, 0.30, 0.15, 0.10, 0.15, 0.00)      # from curriculum_iters on
MIDFALL_TILT_DEG = (40.0, 70.0)     # mid-fall starts: tilt about a random horizontal axis ...
MIDFALL_OMEGA = (2.0, 6.0)          # ... rotating further (rad/s) about that axis
MIDFALL_JOINT_DEG = 8.0             # ... the joints the stand + U(+-this)
MIDFALL_Z0 = 0.03                   # m: drawn low, then raised until no geom touches the floor (+ 2 mm)
# fixed evaluation starts (yaw random): roll +90 ("left"), roll -90, roll 180, pitch +75 (nose down)
EXACT = {"left": (math.pi / 2, 0.0), "right": (-math.pi / 2, 0.0), "back": (math.pi, 0.0),
         "nose": (0.0, math.radians(75.0))}
SUCCESS_TILT_DEG = 12.0
SUCCESS_GYRO = 0.8
SUCCESS_RMS_DEG = 10.0
SUCCESS_HOLD_S = 0.5
STAND_Z = INIT_BASE_Z               # 0.0725 m: the height progress is capped here
FALLEN_TILT_DEG, FALLEN_Z = 25.0, 0.05
TRACK_TILT_DEG, TRACK_Z = 30.0, 0.05
OVERLAP_M = 0.002                   # a spawn penetrating the floor or itself by more than this is drawn again
LEG_BIT, CAN_BIT, BASE_BIT = 8, 16, 4
BLOWUP_LIN_MPS, BLOWUP_ANG_RADPS, BLOWUP_JOINT_RADPS = 10.0, 100.0, 300.0
CONTACT_SOLREF = (0.005, 1.0)       # (timeconst, dampratio) of the body's and legs' contacts (MuJoCo's default 0.02)
THIGH_SHORT_M = 0.008               # 6 mm clears every thigh at the stand (FR by 0.01 mm); 8 mm leaves 0.16 mm


@dataclass
class GetupOptions(TerrainOptions):
    terrain: str = "flat"
    rows: int = 1
    episode_s: float = 6.0
    action_scale: tuple | None = None       # None: getup_scale(band_deg), 1.57 rad on every joint at the 90 deg band
    push_mps: float = 0.0
    fixed_command: tuple | None = (0.0, 0.0, 0.0)
    max_init_level: int = 0
    band_deg: tuple = GETUP_BAND_DEG        # abduction, hip, knee (deg about the stand) or 12 values; stored as 12
    curriculum_iters: int = 1000            # the start mix reaches its end here
    getup_collisions: bool = True           # leg <-> leg, leg <-> hip cans, thigh <-> base
    thigh_base: bool = True                 # thigh <-> base (the capsule shortened by thigh_short_m), without limits
    thigh_short_m: float = THIGH_SHORT_M
    success_gyro: float | None = SUCCESS_GYRO   # the success reward's gyro limit (rad/s); None: no gyro condition
    track_tilt_deg: float = TRACK_TILT_DEG  # stand_track's gate: tilt below this (and the base above TRACK_Z)
    contact_solref: tuple | None = CONTACT_SOLREF   # the non-foot robot geoms' contacts (stiff_contacts); None: MuJoCo's
    spawn_z: tuple = (0.06, 0.10)           # m: the base origin above the floor at a spawn
    start_joints: str = "random"            # "random": stand + U(+-0.5 band); "stand"
    fixed_starts: tuple | None = None       # evaluation: env i starts as fixed_starts[i % len] (EXACT keys or "random")
    fixed_iter: int | None = None           # the start mix at this iteration instead of the step counter's
    njmax: int = 1000
    nconmax: int = 128

    kind: str = "getup"                     # the export meta's policy kind
    mix_start: tuple | None = None          # the start mix (SLICES order) at iteration 0; None: MIX_START
    mix_end: tuple | None = None            # ... from curriculum_iters on; None: MIX_END (6 values: midfall 0)
    midfall: float = 0.0                    # > 0: the mid-fall slice's share of both mixes (the others scaled down)
    end_on_success: bool = False            # the episode ends when the success criterion has held 0.5 s
    blowup_guard: bool = True               # reset a world whose physics goes non-finite or implausibly fast
    nonfinite_guard: bool = True            # per-term finiteness of observations, rewards, sensors (NonfiniteGuard)
    phase_s: float = 0.0                    # > 0: the command floats carry the get-up's phase over this many s
    action_rate_weight: float = -0.02       # the action-rate cost's weight
    action_rate_tilted_scale: float = 0.1   # ... its factor while tilt > 40 deg
    joint_vel_weight: float = 0.0           # < 0: a cost on the legs' squared joint velocities
    filter_alpha: float = 0.0               # the target filter's final alpha; > 0: the action-rate cost is on the
                                            # applied target's change
    filter_alpha_start: float | None = None # training: the alpha at this run's first iteration (None: filter_alpha)
    filter_ramp_iters: int = 0              # training: start -> filter_alpha over this many of the run's iterations

    def __post_init__(self):
        self.band_deg = band12(self.band_deg)   # 12 floats, servo order: what export_policy writes for the runner
        if self.action_scale is None:
            self.action_scale = getup_scale(self.band_deg)
        if self.midfall:
            self.mix_start = with_midfall(self.mix_start or MIX_START, self.midfall)
            self.mix_end = with_midfall(self.mix_end or MIX_END, self.midfall)


def mirror_joints() -> tuple[list[int], list[float]]:
    """Left-right mirror of a joint vector in servo order: mirrored = x[perm] * sign (legs FR<->FL, RR<->RL, abduction and
    hip negated)."""
    perm, sign = [], []
    for leg in (1, 0, 3, 2):
        for joint, s in enumerate((-1.0, -1.0, 1.0)):
            perm.append(3 * leg + joint)
            sign.append(s)
    return perm, sign


def mix7(mix) -> tuple:
    """A start mix over SLICES (6 values: the mid-fall slice 0)."""
    mix = tuple(float(w) for w in mix)
    assert len(mix) in (6, 7), mix
    return mix + (0.0,) if len(mix) == 6 else mix


def with_midfall(mix, share: float) -> tuple:
    """The mix with the mid-fall slice at ``share`` and the other slices scaled to 1 - share."""
    m = mix7(mix)[:6]
    k = (1.0 - share) / max(sum(m), 1e-12)
    return tuple(w * k for w in m) + (float(share),)


def getup_scale(band) -> tuple:
    """rad per unit action: 1.57 on every joint, or the band where it is wider (a 150 deg band needs 2.62 rad)."""
    return tuple(GETUP_SCALE if b <= 90.0 else math.radians(b) for b in band12(band))

def band12(band) -> tuple:
    band = tuple(float(b) for b in band)
    return band * 4 if len(band) == 3 else band


# ---------------------------------------------------------------------------------------------------- collisions
def getup_collisions(spec: mujoco.MjSpec, thigh_base: bool = True, thigh_short_m: float = THIGH_SHORT_M) -> dict:
    """Self-collision pairs of the get-up task, by contype / conaffinity bits (robot geoms are contype 1 / conaffinity
    2, so they never collide with each other otherwise; the terrain is 1 / 1 and sees none of the new bits):
    LEG_BIT on every thigh and calf geom (the feet included) both ways: leg <-> leg; CAN_BIT on the legs' contype and
    the hip cans' conaffinity: leg <-> hip can; BASE_BIT (``sku2.leg_base_collisions``: the base's conaffinity) on the
    thighs' contype, each thigh capsule shortened at its hip end by ``thigh_short_m``. Returns what was changed."""
    out = {"thigh_capsule_m": {}}
    for leg in LEGS:
        for part in ("thigh", "calf"):
            for g in spec.body(f"{leg}_{part}").geoms:
                if g.contype or g.conaffinity:
                    g.contype |= LEG_BIT | CAN_BIT
                    g.conaffinity |= LEG_BIT
        for g in spec.body(f"{leg}_hip").geoms:
            if g.contype or g.conaffinity:
                g.conaffinity |= CAN_BIT
        if thigh_base:
            for g in spec.body(f"{leg}_thigh").geoms:
                if g.name == f"{leg}_thigh_collision":
                    ft = np.array(g.fromto, dtype=float)
                    a, b = ft[:3], ft[3:]
                    length = float(np.linalg.norm(b - a))
                    assert length - thigh_short_m >= 0.030, "under 30 mm of thigh capsule left: drop thigh <-> base"
                    g.fromto = np.r_[a + (b - a) / length * thigh_short_m, b]
                    g.contype |= BASE_BIT
                    out["thigh_capsule_m"][leg] = length - thigh_short_m
    for g in spec.body("base").geoms:          # the walker's leg_base_collisions sets it already (self_collision)
        if g.contype or g.conaffinity:
            g.conaffinity |= BASE_BIT
    return out


def stiff_contacts(spec: mujoco.MjSpec, solref: tuple = CONTACT_SOLREF) -> int:
    """Every robot collision geom but the feet: contact priority 2 and ``solref``. MuJoCo takes a contact's parameters
    from the geom of higher priority (equal: mixed by solmix), so these govern the body's and the legs' contacts with
    the floor (priority 0), with each other and with a foot (priority 1); the feet keep the walker's own (priority 1,
    the terrain task's foot solref) against the floor and each other. Returns the geoms changed."""
    k = 0
    for b in spec.bodies:
        for g in b.geoms:
            if (g.contype or g.conaffinity) and not g.name.endswith("_foot"):
                g.priority = 2
                g.solref = list(solref)
                k += 1
    return k


def _spec_fn(prev, collisions: bool, thigh_base: bool, thigh_short_m: float, solref):
    def fn():
        spec = prev()
        if collisions:
            getup_collisions(spec, thigh_base, thigh_short_m)
        if solref is not None:
            stiff_contacts(spec, solref)
        return spec
    return fn


# ---------------------------------------------------------------------------------------------------- state helpers
def tilt_cos(env) -> torch.Tensor:
    return (-env.scene["robot"].data.projected_gravity_b[:, 2]).clamp(-1.0, 1.0)


def tilt_deg(env) -> torch.Tensor:
    return torch.rad2deg(torch.acos(tilt_cos(env)))


def base_z(env) -> torch.Tensor:
    return env.scene["robot"].data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]


def joint_rms(env) -> torch.Tensor:
    """The servo coordinates' rms distance from the stand (rad; the knee via its tendon)."""
    return mdp.servo_action(env).true_rel().pow(2).mean(-1).sqrt()


def success_now(env, gyro_max: float | None = SUCCESS_GYRO) -> torch.Tensor:
    """Tilt and joint rms within the success limits (to be held ``SUCCESS_HOLD_S``). ``gyro_max`` None drops the gyro
    condition: under exploration noise of std 0.05 a standing robot wobbles at 1-3 rad/s."""
    ok = (tilt_deg(env) < SUCCESS_TILT_DEG) & (joint_rms(env) < math.radians(SUCCESS_RMS_DEG))
    if gyro_max is not None:
        ok &= env.scene["robot"].data.root_link_ang_vel_b.norm(dim=-1) < gyro_max
    return ok


def goal_now(env, gyro_max: float | None = SUCCESS_GYRO) -> torch.Tensor:
    """The instantaneous success criterion: ``success_now`` and the base above ``TRACK_Z``."""
    return success_now(env, gyro_max) & (base_z(env) > TRACK_Z)


def contact_penetration(env) -> tuple[torch.Tensor, torch.Tensor]:
    """Per env, the deepest penetration (m, >= 0) of the robot into the floor and into itself, from the contacts of
    the last forward pass."""
    import warp as wp
    d = env.sim.wp_data
    n = env.num_envs
    floor = torch.zeros(n, device=env.device)
    self_ = torch.zeros(n, device=env.device)
    k = int(wp.to_torch(d.nacon)[0])
    if k == 0:
        return floor, self_
    dist = wp.to_torch(d.contact.dist)[:k]
    world = wp.to_torch(d.contact.worldid)[:k].long()
    geom = wp.to_torch(d.contact.geom)[:k].long()
    robot = _robot_geom_mask(env)
    both = robot[geom[:, 0]] & robot[geom[:, 1]]
    pen = (-dist).clamp_min(0.0)
    floor.scatter_reduce_(0, world, torch.where(both, 0.0, pen), reduce="amax")
    self_.scatter_reduce_(0, world, torch.where(both, pen, 0.0), reduce="amax")
    return floor, self_


def pair_penetration(env) -> torch.Tensor:
    """(n, G, G): per env and geom pair (i <= j), the deepest penetration (m, >= 0) in the last forward pass."""
    import warp as wp
    d = env.sim.wp_data
    n, ng = env.num_envs, env.sim.mj_model.ngeom
    out = torch.zeros(n * ng * ng, device=env.device)
    k = int(wp.to_torch(d.nacon)[0])
    if k:
        pen = (-wp.to_torch(d.contact.dist)[:k]).clamp_min(0.0)
        g = wp.to_torch(d.contact.geom)[:k].long()
        lo, hi = g.min(-1).values, g.max(-1).values
        idx = (wp.to_torch(d.contact.worldid)[:k].long() * ng + lo) * ng + hi
        out.scatter_reduce_(0, idx, pen, reduce="amax")
    return out.view(n, ng, ng)


def _robot_geom_mask(env) -> torch.Tensor:
    if not hasattr(env, "_robot_geoms"):
        m = env.sim.mj_model
        mask = [m.body(int(m.geom_bodyid[i])).name.startswith("robot/") for i in range(m.ngeom)]
        env._robot_geoms = torch.tensor(mask, device=env.device)
    return env._robot_geoms


# ---------------------------------------------------------------------------------------------------- starts
def _ids(env, env_ids) -> torch.Tensor:
    if env_ids is None or isinstance(env_ids, slice):
        return torch.arange(env.num_envs, device=env.device)
    return env_ids.to(env.device).long()


def start_mix(it: int, iters: int, mix0=None, mix1=None) -> torch.Tensor:
    f = min(1.0, max(0.0, it / max(iters, 1)))
    m0 = torch.tensor(mix7(MIX_START if mix0 is None else mix0), dtype=torch.float32)
    m1 = torch.tensor(mix7(MIX_END if mix1 is None else mix1), dtype=torch.float32)
    return m0 * (1.0 - f) + m1 * f


def _quat_rp(roll, pitch):
    """R = Ry(pitch) Rx(roll) as (w, x, y, z)."""
    cr, sr, cp, sp = torch.cos(roll / 2), torch.sin(roll / 2), torch.cos(pitch / 2), torch.sin(pitch / 2)
    return torch.stack([cp * cr, cp * sr, sp * cr, -sp * sr], -1)


def _sign(n, dev):
    return torch.where(torch.rand(n, device=dev) < 0.5, -1.0, 1.0)


def _u(n, lo, hi, dev):
    return lo + (hi - lo) * torch.rand(n, device=dev)


def sample_orientation(slice_id: torch.Tensor, exact: torch.Tensor | None = None, return_axis: bool = False):
    """(w, x, y, z) per env for slice indices into SLICES (yaw not included). ``exact`` (n, 2): (roll, pitch) used
    where not NaN (the evaluation's fixed starts). ``return_axis``: also the mid-fall slice's tilt axis (n, 3;
    horizontal, zeros for the other slices): the tilt is a positive rotation about it."""
    n, dev = len(slice_id), slice_id.device
    d2r = math.radians
    zero = torch.zeros(n, device=dev)
    q = torch.zeros(n, 4, device=dev)
    q[:, 0] = 1.0
    # crouch: tilt U(0, 30 deg) about a random horizontal axis
    phi = _u(n, -math.pi, math.pi, dev)
    axis = torch.stack([torch.cos(phi), torch.sin(phi), zero], -1)
    sel = slice_id == 0
    q[sel] = quat_from_angle_axis(_u(n, 0.0, d2r(30.0), dev), axis)[sel]
    sel = slice_id == 1                                    # partial roll |roll| 30-80 deg
    rs = _sign(n, dev)
    q[sel] = _quat_rp(rs * _u(n, d2r(30), d2r(80), dev), zero)[sel]
    sel = slice_id == 2                                    # side: roll +-(90 +- 20) deg
    q[sel] = _quat_rp(rs * _u(n, d2r(70), d2r(110), dev), zero)[sel]
    sel = slice_id == 3                                    # nose- / tail-down: pitch +-(60-90) deg
    q[sel] = _quat_rp(zero, _sign(n, dev) * _u(n, d2r(60), d2r(90), dev))[sel]
    sel = slice_id == 4                                    # back: roll 180 +- 30 deg
    q[sel] = _quat_rp(_u(n, d2r(150), d2r(210), dev), zero)[sel]
    sel = slice_id == 5                                    # uniform random orientation
    q[sel] = random_orientation(n, str(dev))[sel]
    sel = slice_id == 6                                    # mid-fall: tilt 40-70 deg about a random horizontal axis
    phi2 = _u(n, -math.pi, math.pi, dev)
    axis2 = torch.stack([torch.cos(phi2), torch.sin(phi2), zero], -1)
    q[sel] = quat_from_angle_axis(_u(n, d2r(MIDFALL_TILT_DEG[0]), d2r(MIDFALL_TILT_DEG[1]), dev), axis2)[sel]
    axis2 = axis2 * sel[:, None].float()
    if exact is not None:
        sel = ~torch.isnan(exact[:, 0])
        if bool(sel.any()):
            q[sel] = _quat_rp(exact[sel, 0], exact[sel, 1])
            axis2[sel] = 0.0
    return (q, axis2) if return_axis else q


# ---------------------------------------------------------------------------------------------------- the start bank


def _yaw_of(q: torch.Tensor) -> torch.Tensor:
    """The heading (rad) of (w, x, y, z): the body x axis's direction in the horizontal plane (ZYX yaw)."""
    w, x, y, z = q.unbind(-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class GetupStart(ManagerTermBase):
    """Reset event: a fallen start per env (see the module docstring). The spawn is checked against the contacts of
    a forward pass: a leg penetrating its own robot by more than ``OVERLAP_M`` (another leg, a hip can, the base) has
    its three joints drawn again (rejection sampling, leg by leg), and a robot penetrating the floor is raised by the
    penetration + 2 mm; up to ``max_tries`` passes (in the last, a leg still overlapping is put at the stand). Keeps per
    env the slice drawn (``slice_id``: an index into SLICES, -1 for a fixed evaluation start), the passes the spawn took
    (``tries``) and the spawn's height above the floor (``spawn_height``); ``unresolved`` counts spawns still
    overlapping."""

    def __init__(self, cfg, env):
        super().__init__(env)
        n, dev = env.num_envs, env.device
        self.slice_id = torch.zeros(n, dtype=torch.long, device=dev)
        self.tries = torch.zeros(n, dtype=torch.long, device=dev)
        self.spawn_height = torch.zeros(n, device=dev)
        self.unresolved = 0                       # spawns still overlapping after max_tries (cumulative)
        self.spawned = 0
        p = cfg.params
        fixed = p.get("fixed_starts")
        self._exact = None
        self._fixed_random = None
        self.start_name = None
        self.fixed_slice = None                   # evaluation: (n,) slice index per env (SLICES), drawn per slice
        if fixed:
            names = [fixed[i % len(fixed)] for i in range(n)]
            ex = torch.full((n, 2), float("nan"))
            for i, s in enumerate(names):
                if s != "random":
                    ex[i] = torch.tensor(EXACT[s])
            self._exact = ex.to(dev)
            self._fixed_random = torch.tensor([s == "random" for s in names], device=dev)
            self.start_name = names

    def __call__(self, env, env_ids, band_deg: tuple, spawn_z: tuple, curriculum_iters: int,
                 start_joints: str = "random", fixed_starts=None, fixed_iter=None, max_tries: int = 12,
                 mix_start=None, mix_end=None):
        ids = _ids(env, env_ids)
        m = len(ids)
        if m == 0:
            return
        dev = env.device
        it = int(env.common_step_counter) // STEPS_PER_ITER if fixed_iter is None else int(fixed_iter)
        robot = env.scene["robot"]
        act = mdp.servo_action(env)
        half = 0.5 * torch.deg2rad(torch.tensor(band12(band_deg), device=dev))
        if self._exact is not None:          # evaluation's "random": the default full mix
            mix = start_mix(curriculum_iters, curriculum_iters).to(dev)
        else:
            mix = start_mix(it, curriculum_iters, mix_start, mix_end).to(dev)
        sid = torch.multinomial(mix, m, replacement=True)
        exact = None
        if self._exact is not None:   # evaluation: exact poses, the full mix for "random"
            exact = self._exact[ids]
            sid = torch.where(self._fixed_random[ids], sid, torch.full_like(sid, -1))
        if self.fixed_slice is not None:
            sid = self.fixed_slice[ids]
        self.slice_id[ids] = sid
        yaw = _u(m, -math.pi, math.pi, dev)
        qyaw = torch.stack([torch.cos(yaw / 2), torch.zeros_like(yaw), torch.zeros_like(yaw), torch.sin(yaw / 2)], -1)
        q_tilt, axis = sample_orientation(sid, exact, return_axis=True)
        quat = quat_mul(qyaw, q_tilt)
        mid = sid == SLICES.index("midfall")
        z = torch.where(mid, MIDFALL_Z0, _u(m, spawn_z[0], spawn_z[1], dev))
        amp = torch.where(mid[:, None], math.radians(MIDFALL_JOINT_DEG), half[None, :])
        draw0 = (lambda: (2.0 * torch.rand(m, 12, device=dev) - 1.0) * amp) if start_joints == "random" else \
            (lambda: torch.zeros(m, 12, device=dev))
        if act.bounds is not None:          # limits: the draw clamped into the decoder's bounds (inside the ranges)
            draw = lambda: act.physical(draw0(), ids) - act._sel(act.off, ids)  # noqa: E731
        else:
            draw = draw0
        rel = draw()
        bad = torch.ones(m, dtype=torch.bool, device=dev)
        for k in range(max_tries):
            pos = env.scene.env_origins[ids].clone()
            pos[:, 2] += z
            robot.write_root_state_to_sim(torch.cat([pos, quat, torch.zeros(m, 6, device=dev)], -1), env_ids=ids)
            q_j = act.joint_state_from_rel(rel, ids)
            robot.write_joint_state_to_sim(q_j, torch.zeros_like(q_j), env_ids=ids, joint_ids=act._jid_all)
            env.sim.forward()
            floor, legs = spawn_overlaps(env)
            floor, legs = floor[ids], legs[ids]
            self.tries[ids] = torch.where(bad, k + 1, self.tries[ids])
            lift = (floor > OVERLAP_M) | (mid & (floor > 0.0))        # mid-fall: no floor contact at all
            bad = lift | legs.any(-1)
            if not bool(bad.any()):
                break
            if k == max_tries - 1:
                break
            # the last pass puts a still-overlapping leg at the stand (overlap-free on its own) instead of a new draw
            new = draw() if k < max_tries - 2 else (torch.zeros_like(rel) if act.bounds is None
                                                     else act.physical(torch.zeros_like(rel), ids)
                                                     - act._sel(act.off, ids))
            rel = torch.where(legs.repeat_interleave(3, dim=1), new, rel)
            z = z + torch.where(lift, floor + 0.002, 0.0)
        if bool(mid.any()):          # mid-fall: rotating further about the tilt axis, about the feet
            w_axis = quat_apply(qyaw, axis)                                       # the axis in the world (yawed)
            omega = w_axis * _u(m, MIDFALL_OMEGA[0], MIDFALL_OMEGA[1], dev)[:, None]
            r = quat_apply(quat, torch.tensor([0.0, 0.0, STAND_Z], device=dev).expand(m, 3))
            vel = torch.cat([torch.cross(omega, r, dim=-1), omega], -1)
            robot.write_root_link_velocity_to_sim(vel[mid], env_ids=ids[mid])
        self.spawn_height[ids] = z
        self.unresolved += int(bad.sum())
        self.spawned += m

def _geom_leg(env) -> torch.Tensor:
    """Per geom: the leg index (0-3, LEGS order) of a thigh or calf geom (the feet included); -2 another robot geom
    (base, hip cans); -1 not the robot (the floor)."""
    if not hasattr(env, "_geom_leg_table"):
        m = env.sim.mj_model
        out = []
        for i in range(m.ngeom):
            b = m.body(int(m.geom_bodyid[i])).name
            if not b.startswith("robot/"):
                out.append(-1)
                continue
            part = b.split("/", 1)[1]
            leg = part.split("_")[0]
            out.append(LEGS.index(leg) if part.endswith(("_thigh", "_calf")) else -2)
        env._geom_leg_table = torch.tensor(out, device=env.device)
    return env._geom_leg_table


def spawn_overlaps(env, tol: float = OVERLAP_M) -> tuple[torch.Tensor, torch.Tensor]:
    """From the last forward pass's contacts: per env the deepest floor penetration (m) and, per leg, whether a geom
    of that leg penetrates the robot itself by more than ``tol`` ((n, 4) bool)."""
    import warp as wp
    d = env.sim.wp_data
    n = env.num_envs
    floor = torch.zeros(n, device=env.device)
    legs = torch.zeros(n, 4, dtype=torch.bool, device=env.device)
    k = int(wp.to_torch(d.nacon)[0])
    if k == 0:
        return floor, legs
    pen = (-wp.to_torch(d.contact.dist)[:k]).clamp_min(0.0)
    world = wp.to_torch(d.contact.worldid)[:k].long()
    gl = _geom_leg(env)[wp.to_torch(d.contact.geom)[:k].long()]          # (k, 2)
    is_floor = (gl == -1).any(-1)
    floor.scatter_reduce_(0, world, torch.where(is_floor, pen, 0.0), reduce="amax")
    hit = ~is_floor & (pen > tol)
    for side in range(2):
        sel = hit & (gl[:, side] >= 0)
        if bool(sel.any()):
            legs[world[sel], gl[sel, side]] = True
    return floor, legs


# ---------------------------------------------------------------------------------------------------- rewards
class UprightProgress(ManagerTermBase):
    """(cos tilt_t - cos tilt_{t-1}) / dt; 0 on an episode's first step."""

    def __init__(self, cfg, env):
        super().__init__(env)
        self._prev = torch.zeros(env.num_envs, device=env.device)
        self._fresh = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids=None):
        self._fresh[slice(None) if env_ids is None else env_ids] = True

    def value(self, env):
        return tilt_cos(env)

    def __call__(self, env):
        v = self.value(env)
        prev = torch.where(self._fresh, v, self._prev)
        self._prev, self._fresh[:] = v, False
        return (v - prev) / env.step_dt


class HeightProgress(UprightProgress):
    """(min(z_t, 0.0725) - min(z_{t-1}, 0.0725)) / dt, signed; 0 on an episode's first step."""

    def value(self, env):
        return base_z(env).clamp(max=STAND_Z)


def fallen(env):
    return ((tilt_deg(env) > FALLEN_TILT_DEG) | (base_z(env) < FALLEN_Z)).float()


def stand_track(env, tilt_max_deg: float = TRACK_TILT_DEG):
    gate = (tilt_deg(env) < tilt_max_deg) & (base_z(env) > TRACK_Z)
    return torch.exp(-joint_rms(env) ** 2 / 0.1) * gate.float()


class GetupSuccess(ManagerTermBase):
    """1 / dt at the step the success criterion (``success_now`` and the base above ``TRACK_Z``) has held
    ``SUCCESS_HOLD_S``, once per episode (x dt x weight 10 = +10 per episode). ``done`` and ``t_success`` (s from the
    episode's start, -1: not yet) per env."""

    def __init__(self, cfg, env):
        super().__init__(env)
        n, dev = env.num_envs, env.device
        self.run = torch.zeros(n, dtype=torch.long, device=dev)
        self.done = torch.zeros(n, dtype=torch.bool, device=dev)
        self.t_success = torch.full((n,), -1.0, device=dev)
        self.hold = int(round(SUCCESS_HOLD_S / env.step_dt))

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.run[ids] = 0
        self.done[ids] = False
        self.t_success[ids] = -1.0

    def __call__(self, env, gyro_max: float | None = SUCCESS_GYRO):
        ok = success_now(env, gyro_max) & (base_z(env) > TRACK_Z)
        self.run = torch.where(ok, self.run + 1, torch.zeros_like(self.run))
        fire = (self.run >= self.hold) & ~self.done
        self.done |= fire
        self.t_success = torch.where(fire, env.episode_length_buf.float() * env.step_dt, self.t_success)
        return fire.float() / env.step_dt


class GetupActionRate(ManagerTermBase):
    """Squared change of the (clipped) action, x 0.1 while tilt > 40 deg; the previous action after a reset is the
    seeded decoder's (``ServoAction.applied_action``). ``applied`` (the target filter on): the change of the applied
    action (the filtered, clamped target / scale) instead of the raw action's, so the cost sees the motion."""

    def __init__(self, cfg, env):
        super().__init__(env)
        self._prev = torch.zeros(env.num_envs, 12, device=env.device)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self._prev[ids] = mdp.servo_action(self._env).applied_action[ids]

    def __call__(self, env, tilted_scale: float = 0.1, applied: bool = False):
        a = mdp.servo_action(env).applied_action if applied else env.action_manager.action.clamp(-1.0, 1.0)
        r = ((a - self._prev) ** 2).sum(-1)
        self._prev = a.clone()
        return r * torch.where(tilt_deg(env) > 40.0, tilted_scale, 1.0)


def getup_joint_vel(env) -> torch.Tensor:
    """Sum of the squared joint velocities (rad/s) of the twelve leg joints: the smoothness cost on flailing legs."""
    act = mdp.servo_action(env)
    return env.scene["robot"].data.joint_vel[:, act._jid_all].pow(2).sum(-1)


class SuccessReached(ManagerTermBase):
    """A termination: the success criterion (as the success reward's: ``success_now`` with ``gyro_max`` and the base
    above ``TRACK_Z``) has held ``SUCCESS_HOLD_S`` -- the runner's hand-back to the walker. The success reward fires on
    the same step, so the episode ends with its +10 and nothing is learnt about standing on afterwards."""

    def __init__(self, cfg, env):
        super().__init__(env)
        self.run = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.hold = int(round(SUCCESS_HOLD_S / env.step_dt))

    def reset(self, env_ids=None):
        self.run[slice(None) if env_ids is None else env_ids] = 0

    def __call__(self, env, gyro_max: float | None = SUCCESS_GYRO):
        ok = success_now(env, gyro_max) & (base_z(env) > TRACK_Z)
        self.run = torch.where(ok, self.run + 1, torch.zeros_like(self.run))
        return self.run >= self.hold


def physics_blowup(env, max_lin: float = BLOWUP_LIN_MPS, max_ang: float = BLOWUP_ANG_RADPS,
                   max_joint: float = BLOWUP_JOINT_RADPS) -> torch.Tensor:
    """A termination (logged as a time-out): the world's physics state is non-finite, or the base or a joint moves
    implausibly fast (an unstable contact solve; in normal training the base stays under 4 m/s and a joint under 36
    rad/s). The env is reset before the observations are computed, so one bad world cannot reach the policy;
    ``env.blowups`` counts them."""
    qpos, qvel = torch.as_tensor(env.sim.data.qpos), torch.as_tensor(env.sim.data.qvel)
    bad = ~(torch.isfinite(qpos).all(-1) & torch.isfinite(qvel).all(-1))
    v = torch.nan_to_num(qvel, nan=0.0, posinf=1e9, neginf=-1e9)
    bad |= v[:, :3].norm(dim=-1) > max_lin
    bad |= v[:, 3:6].norm(dim=-1) > max_ang
    bad |= v[:, 6:].abs().amax(-1) > max_joint
    env.blowups = getattr(env, "blowups", 0) + int(bad.sum())
    return bad


class _CheckedTerm:
    """A manager term's callable (a function or a ``ManagerTermBase`` instance) whose output is checked for
    non-finite values per env: those rows are returned as 0 and reported to the ``NonfiniteGuard``. Every other
    attribute is the wrapped term's (``reset``, a class term's state)."""

    def __init__(self, inner, guard: "NonfiniteGuard", k: int):
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_guard", guard)
        object.__setattr__(self, "_k", k)

    def __call__(self, env, **params):
        x = self._inner(env, **params)
        bad = ~torch.isfinite(x.reshape(x.shape[0], -1)).all(-1)
        self._guard.note(self._k, bad)
        return torch.where(bad.view(-1, *([1] * (x.dim() - 1))), torch.zeros_like(x), x)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_inner"), name)

    def __setattr__(self, name, value):
        setattr(self._inner, name, value)


class NonfiniteGuard(ManagerTermBase):
    """A termination (logged as a time-out, ``Episode_Termination/nonfinite``): a per-term
    finiteness check on every observation term of every group, on every reward term and on the scene's sensor buffers
    (each float field of each sensor's data), cheap enough to leave on (a few element-wise kernels per term, no host
    sync but one per step here).

    The observation and reward terms are wrapped (``_CheckedTerm``): a non-finite row is returned as 0, so a NaN
    never reaches the policy, the critic or PPO's returns, and the env is marked; at the next step this term resets
    it. The sensor buffers are checked here, before the step's reset, and such an env is reset in the same step. For
    each event the first offending term, the env id and the env's qpos / qvel at that moment (snapshotted on the GPU
    when the term fired) are kept in ``log`` (the first ``max_log``), printed, and appended to ``log_path`` (JSON
    lines) if set; ``env.nonfinite_resets`` counts the envs reset."""

    def __init__(self, cfg, env):
        super().__init__(env)
        n, dev = env.num_envs, env.device
        self.log_path = cfg.params.get("log_path")
        self.max_log = int(cfg.params.get("max_log", 50))
        self.names: list[str] = []
        self.pending = torch.zeros(n, dtype=torch.bool, device=dev)
        self.first = torch.full((n,), -1, dtype=torch.long, device=dev)
        qpos, qvel = self._state()
        self.qpos, self.qvel = torch.zeros_like(qpos), torch.zeros_like(qvel)
        self.log: list[dict] = []
        self.count = 0
        self._rewards_wrapped = False
        om = env.observation_manager
        for g, cfgs in om._group_obs_term_cfgs.items():
            for name, tcfg in zip(om._group_obs_term_names[g], cfgs):
                tcfg.func = self.wrap(tcfg.func, f"obs/{g}/{name}")
        env.nonfinite_resets = 0

    def _state(self):
        d = self._env.sim.data
        return torch.as_tensor(d.qpos), torch.as_tensor(d.qvel)

    def wrap(self, func, name: str) -> _CheckedTerm:
        self.names.append(name)
        return _CheckedTerm(func, self, len(self.names) - 1)

    def note(self, k: int, bad: torch.Tensor) -> None:
        """Term ``k`` produced a non-finite value in the envs ``bad`` (n,) bool."""
        new = bad & (self.first < 0)
        self.first = torch.where(new, k, self.first)
        qpos, qvel = self._state()
        self.qpos = torch.where(new[:, None], qpos, self.qpos)
        self.qvel = torch.where(new[:, None], qvel, self.qvel)
        self.pending |= bad

    def reset(self, env_ids=None):
        pass

    def _sensor_fields(self, env):
        import dataclasses
        out = []
        for sname, sensor in env.scene.sensors.items():
            data = sensor.data
            if not dataclasses.is_dataclass(data):
                continue
            for f in dataclasses.fields(data):
                v = getattr(data, f.name)
                if isinstance(v, torch.Tensor) and v.is_floating_point() and v.shape[:1] == (env.num_envs,):
                    out.append((sname, f.name))
        return out

    def __call__(self, env, log_path: str | None = None, max_log: int = 50):
        if not self._rewards_wrapped:          # the reward manager is built after the terminations
            rm = env.reward_manager
            for name, tcfg in zip(rm._term_names, rm._term_cfgs):
                tcfg.func = self.wrap(tcfg.func, f"reward/{name}")
            self._sensors = []
            for sname, field in self._sensor_fields(env):
                self.names.append(f"sensor/{sname}.{field}")
                self._sensors.append((sname, field, len(self.names) - 1))
            self._rewards_wrapped = True
        for sname, field, k in self._sensors:
            v = getattr(env.scene[sname].data, field)
            self.note(k, ~torch.isfinite(v.reshape(v.shape[0], -1)).all(-1))
        bad = self.pending.clone()
        self.pending.zero_()
        if bool(bad.any()):
            ids = torch.nonzero(bad).flatten()
            first, qpos, qvel = self.first[ids].cpu(), self.qpos[ids].cpu(), self.qvel[ids].cpu()
            ep = env.episode_length_buf[ids].cpu()
            for j, i in enumerate(ids.cpu().tolist()):
                rec = {"env_id": i, "term": self.names[int(first[j])], "step": int(env.common_step_counter),
                       "episode_step": int(ep[j]), "qpos": qpos[j].tolist(), "qvel": qvel[j].tolist()}
                if len(self.log) < self.max_log:
                    self.log.append(rec)
                    print("NONFINITE", json.dumps(rec), flush=True)
                    if self.log_path:
                        with open(self.log_path, "a") as fh:
                            fh.write(json.dumps(rec) + "\n")
            self.count += len(ids)
            env.nonfinite_resets = self.count
            self.first[ids] = -1
        return bad


def obs_zero_command(env):
    return torch.zeros(env.num_envs, 3, device=env.device)


def obs_phase(env, phase_s: float):
    """The get-up's phase in the command's three floats: (phi, sin 2 pi phi, cos 2 pi phi), phi = the time since the
    episode's start / ``phase_s``, held at 1 after it."""
    phi = (env.episode_length_buf.float() * env.step_dt / phase_s).clamp(max=1.0)
    return torch.stack([phi, torch.sin(2 * math.pi * phi), torch.cos(2 * math.pi * phi)], dim=-1)


# ---------------------------------------------------------------------------------------------------- the task
def filter_on(opt) -> bool:
    """Whether a get-up run's target filter is on at any point (its final or its starting alpha > 0)."""
    return bool(opt.filter_alpha) or bool(opt.filter_alpha_start)


def policy_filter_alpha_of(options: dict | None, meta: dict | None = None) -> float:
    """A trained get-up's target filter alpha, for tools that rebuild its ``GetupOptions``: the exported meta's
    ``action_filter_alpha`` when it has the key (what the robot runs), else the run's option ``filter_alpha`` (its
    final alpha), else 0."""
    if meta is not None and "action_filter_alpha" in meta:
        return float(meta["action_filter_alpha"] or 0.0)
    return float((options or {}).get("filter_alpha", 0.0) or 0.0)


def filter_alpha_at(k: int, alpha: float, start: float | None = None, ramp_iters: int = 0) -> float:
    """The target filter's alpha at iteration ``k`` of a run (0 at its first, a resumed run's too):
    ``start + (alpha - start) min(1, k / ramp_iters)`` (``start`` None: ``alpha``; ``ramp_iters`` 0: ``alpha``)."""
    a0 = float(alpha if start is None else start)
    if ramp_iters <= 0:
        return float(alpha)
    return a0 + (float(alpha) - a0) * min(1.0, max(int(k), 0) / int(ramp_iters))


def install_filter_ramp(runner, servo, alpha: float, start: float | None = None, ramp_iters: int = 0,
                        tag: str = "Getup/filter_alpha") -> dict:
    """The target filter's schedule on an rsl_rl ``OnPolicyRunner``: ``servo`` (``mdp.ServoAction``) gets
    ``filter_alpha_at(0)`` now and, after each iteration's update (the runner's ``logger.log``, called once per
    iteration with ``it`` and ``start_it``), the alpha of the next; each iteration's alpha goes to the writer as
    ``tag`` and is printed (``FILTER_ALPHA it alpha``). Returns the state (``alpha``: the current, ``log``: the
    (it, alpha) of every iteration)."""
    st = {"alpha": filter_alpha_at(0, alpha, start, ramp_iters), "log": []}
    servo.set_filter_alpha(st["alpha"])
    logger = runner.logger
    orig = logger.log

    def log(*args, **kw):
        it, start_it = int(kw["it"]), int(kw["start_it"])
        used = st["alpha"]
        st["log"].append((it, used))
        out = orig(*args, **kw)
        if getattr(logger, "writer", None) is not None:
            logger.writer.add_scalar(tag, used, it)
        print(f"FILTER_ALPHA {it} {used:.6f}", flush=True)
        st["alpha"] = filter_alpha_at(it - start_it + 1, alpha, start, ramp_iters)
        servo.set_filter_alpha(st["alpha"])
        return out

    logger.log = log
    return st


def make_getup_env_cfg(opt: GetupOptions | None = None):
    opt = opt or GetupOptions()
    assert opt.terrain == "flat" and opt.rows == 1 and not opt.history, (opt.terrain, opt.rows, opt.history)
    cfg = make_terrain_env_cfg(opt)
    band = band12(opt.band_deg)

    robot = cfg.scene.entities["robot"]
    # with the measured limits the abduction stops are joint ranges and the hip-abduction coupling the decoder's
    # envelope: the thigh <-> base pair's inward contact at 2-7 deg would stop it short of the robot's 14-22
    thigh_base = opt.thigh_base and not opt.limits
    robot.spec_fn = _spec_fn(robot.spec_fn, opt.getup_collisions, thigh_base, opt.thigh_short_m,
                             None if opt.contact_solref is None else tuple(opt.contact_solref))

    # the feet's sensor for the critic; no terrain rays, no leg-terrain contact term
    cfg.scene.sensors = tuple(s for s in cfg.scene.sensors if s.name == "feet_contact")

    # observations: the walker's 45 with the command zeroed, no history; the critic + lin vel, height, contacts
    cfg.observations.pop("anchor", None)
    for g in ("actor", "critic"):
        cfg.observations[g].terms["command"] = (
            ObservationTermCfg(func=obs_phase, params={"phase_s": opt.phase_s})
            if opt.phase_s else ObservationTermCfg(func=obs_zero_command))
    for t in ("body_scan", "foot_heights"):
        cfg.observations["critic"].terms.pop(t, None)
    assert list(cfg.observations["actor"].terms) == ["ang_vel", "gravity", "command", "pos", "vel", "prev_action"]
    cfg.observations["actor"].history_length = 0

    # action: the get-up scale and band, the decoder seeded from the measured pose
    cfg.actions["servo"] = mdp.ServoActionCfg(scale=tuple(opt.action_scale), band_deg=band, seed_from_measured=True,
                                              filter_alpha=float(opt.filter_alpha))
    servo_limits(cfg.actions["servo"], opt, "getup")

    # events: the fallen starts replace the stand resets; no pushes
    for k in ("reset_base", "reset_joints", "push"):
        cfg.events.pop(k, None)
    cfg.events["getup_start"] = EventTermCfg(func=GetupStart, mode="reset", params={
        "band_deg": band, "spawn_z": tuple(opt.spawn_z), "curriculum_iters": opt.curriculum_iters,
        "start_joints": opt.start_joints, "fixed_starts": tuple(opt.fixed_starts) if opt.fixed_starts else None,
        "fixed_iter": opt.fixed_iter, "mix_start": opt.mix_start, "mix_end": opt.mix_end})

    # rewards (weights per second: mjlab multiplies by dt)
    cfg.rewards = {
        "upright_progress": RewardTermCfg(func=UprightProgress, weight=5.0, params={}),
        "height_progress": RewardTermCfg(func=HeightProgress, weight=30.0, params={}),
        "fallen_tax": RewardTermCfg(func=fallen, weight=-0.5),
        "success": RewardTermCfg(func=GetupSuccess, weight=10.0, params={"gyro_max": opt.success_gyro}),
        "stand_track": RewardTermCfg(func=stand_track, weight=1.0, params={"tilt_max_deg": opt.track_tilt_deg}),
        "action_rate": RewardTermCfg(func=GetupActionRate, weight=opt.action_rate_weight,
                                     params={"tilted_scale": opt.action_rate_tilted_scale,
                                             "applied": filter_on(opt)}),
        "torques": RewardTermCfg(func=mdp.torques_l2, weight=-1.0e-4),
        **({"joint_vel": RewardTermCfg(func=getup_joint_vel, weight=opt.joint_vel_weight)} if opt.joint_vel_weight else {}),
    }

    # terminations: the time-out (and the blow-up guard); curriculum: the noise ramp only
    cfg.terminations = {"time_out": TerminationTermCfg(func=vel_mdp.time_out, time_out=True)}
    if opt.blowup_guard:
        cfg.terminations["blowup"] = TerminationTermCfg(func=physics_blowup, time_out=True)
    if opt.nonfinite_guard:
        cfg.terminations["nonfinite"] = TerminationTermCfg(func=NonfiniteGuard, time_out=True,
                                                           params={"log_path": None, "max_log": 50})
    if opt.end_on_success:
        cfg.terminations["success"] = TerminationTermCfg(func=SuccessReached, params={"gyro_max": opt.success_gyro})
    cfg.curriculum = {k: v for k, v in cfg.curriculum.items() if k == "noise_ramp"}
    cfg.sim.njmax = opt.njmax
    cfg.sim.nconmax = opt.nconmax
    cfg.episode_length_s = opt.episode_s
    return cfg


class BoundedGaussianDistribution(GaussianDistribution):
    """rsl_rl's Gaussian with the standard deviation clamped to [``std_min``, ``std_max``] (the parameter keeps its
    value; outside the range its gradient is 0): with actions clipped to +-1 the entropy bonus otherwise inflates the
    std without limit."""

    def __init__(self, output_dim: int, init_std: float = 1.0, std_type: str = "scalar", std_min: float = 0.0,
                 std_max: float = 1.0e9):
        super().__init__(output_dim, init_std=init_std, std_type=std_type)
        self.std_min, self.std_max = float(std_min), float(std_max)

    def update(self, mlp_output: torch.Tensor) -> None:
        from torch.distributions import Normal
        std = self.std_param if self.std_type == "scalar" else torch.exp(self.log_std_param)
        self._distribution = Normal(mlp_output, std.clamp(self.std_min, self.std_max).expand_as(mlp_output))



