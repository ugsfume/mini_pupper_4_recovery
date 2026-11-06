"""The robot's MDP terms for mjlab: the servo action, the robot-side observations, delays, terminations.

The action and the observations implement ``contract.py`` in torch (the same arithmetic; ``tests/gpu/test_contract_torch.py``
compares them). Observation noise is applied here, from a per-environment amplitude table (``NoiseTable``) that the
noise ramp updates: mjlab caches a noise config's bounds as tensors on first use, so changing them later would do nothing.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from mjlab.managers.action_manager import ActionTerm, ActionTermCfg

from mp4_recovery.contract import (BAND_DEG, DD0, STATE_DT, STEP_LIMIT_DEG, Bounds, Contract, stand_offset_rel)
from mp4_recovery.robot.sku2 import LEGS

if TYPE_CHECKING:
    pass

F32 = torch.float32
SAMPLE_EVERY = 5            # physics steps per STATE frame (5 ms at 1 ms physics)


# ------------------------------------------------------------------------------------------------ the contract, torch
class TorchContract:
    """``stand``: the policy's stand in the model's coordinates (the calibrated stand ``stand_phys`` + the abduction
    stand offset ``offset``, ``stand_offset_abd_deg`` outward; contract.py). ``quantise_rel`` and ``measure_rel`` take
    physical rel (relative to ``stand_phys``), as the firmware does."""

    def __init__(self, device, stand_offset_abd_deg: float = 0.0):
        c = Contract.load()
        self.c = c
        self.signs = torch.tensor(c.signs, dtype=F32, device=device)
        self.offsets = torch.tensor(c.offsets_deg, dtype=F32, device=device)
        self.stand_phys = torch.tensor(c.stand_rel_rad, dtype=F32, device=device)
        self.stand_offset_abd_deg = float(stand_offset_abd_deg)
        self.offset = torch.tensor(stand_offset_rel(self.stand_offset_abd_deg), dtype=F32, device=device)
        self.stand = self.stand_phys + self.offset if self.stand_offset_abd_deg else self.stand_phys.clone()

    def ticks_from_servo_deg(self, sdeg):
        ang = sdeg.to(F32) + self.offsets
        sig = 511 + torch.trunc(ang * 10240.0 / 3100.0)
        return sig.clamp(0, 1023)

    @staticmethod
    def dd_from_ticks(ticks):
        return torch.trunc(3100.0 - ticks.to(F32) * 3100.0 / 1024.0)

    def servo_deg_from_dd(self, dd):
        return (DD0 - dd) / 10.0 - self.offsets

    def quantise_rel(self, rel):
        sdeg = torch.rad2deg(rel) * self.signs
        dd = self.dd_from_ticks(self.ticks_from_servo_deg(sdeg))
        return dd, torch.deg2rad(self.servo_deg_from_dd(dd)) * self.signs

    def measure_rel(self, rel):
        """A true position as the AT32 reports it: an integer deci-degree on the servo's track."""
        sdeg = torch.rad2deg(rel) * self.signs
        dd = torch.round(DD0 - 10.0 * (sdeg + self.offsets))
        return torch.deg2rad(self.servo_deg_from_dd(dd)) * self.signs


class TorchBounds:
    """``contract.Bounds`` in torch: ``clamp_rel(rel)`` (physical rel, (n, 12)) into the bounds, the same arithmetic
    (a value already inside is unchanged)."""

    def __init__(self, b: Bounds, stand_phys: torch.Tensor):
        dev = stand_phys.device
        t = lambda x: torch.as_tensor(np.asarray(x, dtype=np.float64), dtype=F32, device=dev)  # noqa: E731
        st = stand_phys.view(4, 3)
        self.st = st
        self.lo = t(b.lo) - stand_phys
        self.hi = t(b.hi) - stand_phys
        self.kc = t(b.kc)
        self.calf_lo, self.calf_hi = t(b.calf_lo), t(b.calf_hi)
        self.env = {}
        for li, leg in enumerate(LEGS):
            e = (b.hip_envelope or {}).get(leg)
            if e is not None:
                self.env[li] = (t(e["abd"]), t(e["lo"]), t(e["hi"]))

    @staticmethod
    def _interp(x, xp, fp):
        """np.interp (constant beyond the ends) for x (n,), xp increasing."""
        i = torch.searchsorted(xp, x.contiguous()).clamp(1, len(xp) - 1)
        x0, x1, f0, f1 = xp[i - 1], xp[i], fp[i - 1], fp[i]
        w = ((x - x0) / (x1 - x0)).clamp(0.0, 1.0)
        return f0 + w * (f1 - f0)

    def clamp_rel(self, rel: torch.Tensor, ids=None) -> torch.Tensor:
        r = torch.maximum(torch.minimum(rel, self.hi), self.lo)
        v = r.view(-1, 4, 3).clone()
        lo3, hi3 = self.lo.view(4, 3), self.hi.view(4, 3)
        for li, (xa, flo, fhi) in self.env.items():
            abd = self.st[li, 0] + v[:, li, 0]
            elo = self._interp(abd, xa, flo) - self.st[li, 1]
            ehi = self._interp(abd, xa, fhi) - self.st[li, 1]
            h = torch.maximum(torch.minimum(v[:, li, 1], ehi), elo)
            v[:, li, 1] = h.clamp(float(lo3[li, 1]), float(hi3[li, 1]))
        thigh = self.st[None, :, 1] + v[:, :, 1]
        a = self.kc[:, 0] * thigh + self.kc[:, 1] * self.calf_lo - self.st[None, :, 2]
        b = self.kc[:, 0] * thigh + self.kc[:, 1] * self.calf_hi - self.st[None, :, 2]
        k = torch.maximum(torch.minimum(v[:, :, 2], torch.maximum(a, b)), torch.minimum(a, b))
        v[:, :, 2] = torch.maximum(torch.minimum(k, hi3[:, 2]), lo3[:, 2])
        return v.view_as(rel)


# ------------------------------------------------------------------------------------------------------ the action
@dataclass(kw_only=True)
class ServoActionCfg(ActionTermCfg):
    entity_name: str = "robot"
    scale: tuple = (0.10, 0.40, 0.42) * 4          # rad per unit action, servo order (abduction, hip, knee) x 4
    step_limit_deg: float = STEP_LIMIT_DEG
    quantise: bool = True
    band_deg: tuple | None = None                  # 12 values (servo order) about the stand; None: contract.BAND_DEG
    seed_from_measured: bool = False               # reset: the decoder starts at the measured pose, not the stand
    bounds: dict | None = None                     # contract.Bounds.to_meta() (limits.decoder_bounds): the targets
                                                   # clamped into the measured limits; None: the band only
    stand_offset_abd_deg: float = 0.0              # the abductions' stand shifted outward (contract.py)
    filter_alpha: float = 0.0                      # > 0: a first-order low-pass on the target (contract.py's
                                                   # ActionDecoder): rel = prev_rel + (1 - alpha) (rel - prev_rel)
                                                   # before the band

    def build(self, env):
        return ServoAction(self, env)


class ServoAction(ActionTerm):
    """Policy action (12, servo order) -> targets on the abduction and hip joints and the knee tendons.

    Also samples the positions every 5 physics steps, as the robot's 200 Hz STATE stream, quantised to the AT32's
    deci-degree, for the position and velocity observations. Positions are computed from the joint angles (the knee:
    the tendon's coefficients), which are valid right after a reset write, before any forward pass."""

    cfg: ServoActionCfg

    def __init__(self, cfg: ServoActionCfg, env):
        super().__init__(cfg, env)
        dev = self.device
        self.tc = TorchContract(dev, cfg.stand_offset_abd_deg)
        c = self.tc.c
        self.bounds = None if cfg.bounds is None else TorchBounds(Bounds.from_meta(cfg.bounds), self.tc.stand_phys)
        self._shifted = bool(cfg.stand_offset_abd_deg)
        # the policy's stand offset (physical rel = policy rel + off) and stand (model): (12,), or (n, 12) when a tool
        # sets per-env offsets (sequence: the walker's and the get-up's); _shifted False: no offset arithmetic at all
        self.off = self.tc.offset
        self.stand = self.tc.stand
        names = []
        for leg in LEGS:
            names += [f"{leg}_hip", f"{leg}_thigh", f"{leg}_calf"]
        jid, jn = self._entity.find_joints(names, preserve_order=True)
        assert list(jn) == names, jn
        self._jid_all = torch.tensor(jid, device=dev, dtype=torch.long)
        target_j = [n for n in names if not n.endswith("_calf")]
        tj, _ = self._entity.find_joints(target_j, preserve_order=True)
        self._tj = torch.tensor(tj, device=dev, dtype=torch.long)
        tt, tn = self._entity.find_tendons([f"{leg}_knee_drive" for leg in LEGS], preserve_order=True)
        self._tt = torch.tensor(tt, device=dev, dtype=torch.long)
        self._jidx = torch.tensor([i for i in range(12) if i % 3 != 2], device=dev)   # servo index of each joint target
        self._kidx = torch.tensor([2, 5, 8, 11], device=dev)
        # knee tendon = coef_thigh * q_thigh + coef_calf * q_calf
        co = c.meta["tendon_coefs"]
        self._kc = torch.tensor([[co[leg][f"{leg}_thigh"], co[leg][f"{leg}_calf"]] for leg in LEGS], dtype=F32,
                                device=dev)
        self.scale = torch.tensor(cfg.scale, dtype=F32, device=dev).expand(12).clone()
        band = BAND_DEG if cfg.band_deg is None else cfg.band_deg
        assert len(band) == 12, band
        self.band = torch.deg2rad(torch.tensor(band, dtype=F32, device=dev))
        self.limit = math.radians(cfg.step_limit_deg)
        self.set_filter_alpha(cfg.filter_alpha)
        n = self.num_envs
        self._raw = torch.zeros(n, 12, device=dev)
        self.prev_rel = torch.zeros(n, 12, device=dev)
        self.applied_action = torch.zeros(n, 12, device=dev)
        self.target_model = self.tc.stand.expand(n, 12).clone()
        self.hist = torch.zeros(n, 6, 12, device=dev)
        self._substep = 0
        self._decimation = int(env.cfg.decimation)
        self._synced_at = -1
        self.reset(None)

    @property
    def action_dim(self) -> int:
        return 12

    @property
    def raw_action(self) -> torch.Tensor:
        return self._raw

    def true_rel(self) -> torch.Tensor:
        """The true positions (servo order, rel to the stand, rad), from the joint angles."""
        q = self._entity.data.joint_pos[:, self._jid_all].view(-1, 4, 3)
        knee = (q[:, :, 1:3] * self._kc).sum(-1)
        model = torch.stack([q[:, :, 0], q[:, :, 1], knee], dim=-1).view(-1, 12)
        return model - self.stand

    def joint_state_from_rel(self, rel: torch.Tensor, ids=None) -> torch.Tensor:
        """Joint angles (hip, thigh, calf per leg) that put the servo coordinates at ``rel`` (the knee via its
        tendon; envs ``ids``, None: all or a (12,) stand); for tools that pose the robot directly."""
        model = (self._sel(self.stand, ids) + rel).view(-1, 4, 3)
        q = torch.empty_like(model)
        q[..., 0] = model[..., 0]
        q[..., 1] = model[..., 1]
        q[..., 2] = (model[..., 2] - self._kc[:, 0] * model[..., 1]) / self._kc[:, 1]
        return q.view(-1, 12)

    @staticmethod
    def _sel(x: torch.Tensor, ids):
        """A (12,) or per-env (n, 12) tensor for envs ``ids`` (None: all)."""
        return x if (ids is None or x.dim() == 1) else x[ids]

    def physical(self, rel: torch.Tensor, ids=None) -> torch.Tensor:
        """A policy rel (envs ``ids``; None: all) as the physical target (rel to the calibrated stand): the offset
        added, into the bounds."""
        phys = rel + self._sel(self.off, ids) if self._shifted else rel
        return phys if self.bounds is None else self.bounds.clamp_rel(phys, ids)

    def measured_rel(self) -> torch.Tensor:
        """The positions as the AT32 reports them, in the policy's coordinates (physical, quantised, less the offset)."""
        if not self._shifted:
            return self.tc.measure_rel(self.true_rel())
        return self.tc.measure_rel(self.true_rel() + self.off) - self.off

    def set_filter_alpha(self, alpha) -> None:
        """The target filter's alpha at run time (a training ramp, sequence's per-robot decoders): a float, or a
        per-env (n, 1) tensor (0 where a robot's decoder does not filter). A float 0 skips the filter (the decoder
        before it, bit for bit)."""
        if isinstance(alpha, torch.Tensor):
            assert alpha.shape == (self.num_envs, 1), alpha.shape
            self.alpha = alpha.to(device=self.device, dtype=F32)
        else:
            assert 0.0 <= float(alpha) < 1.0, alpha
            self.alpha = float(alpha)

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw[:] = actions
        a = actions.clamp(-1.0, 1.0)
        rel = self.scale * a
        if isinstance(self.alpha, torch.Tensor) or self.alpha:
            # prev_rel + (1 - alpha) (rel - prev_rel), written so that alpha 0 leaves rel exactly as it was; the state
            # is the previous applied target (after the band, step limit, bounds and quantisation), which the
            # observation's previous action shows
            rel = rel + self.alpha * (self.prev_rel - rel)
        rel = torch.maximum(torch.minimum(rel, self.band), -self.band)
        rel = torch.maximum(torch.minimum(rel, self.prev_rel + self.limit), self.prev_rel - self.limit)
        phys = self.physical(rel)
        if self.cfg.quantise:
            _, phys = self.tc.quantise_rel(phys)
        rel = phys - self.off if self._shifted else phys
        self.prev_rel = rel
        self.applied_action = rel / self.scale
        self.target_model = self.tc.stand_phys + phys if self._shifted else self.tc.stand + rel

    def apply_actions(self) -> None:
        # Within a policy step, apply_actions sees the state at 0, 1, ..., decimation-1 ms. The 200 Hz samples are the
        # states at 5, 10 and 15 ms here, and at the observation's own instant (20 ms) in ``sync``, so the newest
        # position is as fresh as the IMU (the command delay carries the robot's sensing age).
        i = self._substep % self._decimation
        if i and i % SAMPLE_EVERY == 0:
            self._push(self.measured_rel())
        self._substep += 1
        self._entity.set_joint_position_target(self.target_model[:, self._jidx], joint_ids=self._tj)
        self._entity.set_tendon_len_target(self.target_model[:, self._kidx], tendon_ids=self._tt)

    def _push(self, x, env_ids=None):
        if env_ids is None:
            self.hist = torch.cat([self.hist[:, 1:], x[:, None, :]], dim=1)
        else:
            self.hist[env_ids] = x[env_ids, None, :].expand(-1, 6, -1)

    def reset(self, env_ids=None) -> None:
        """At the stand (``prev_rel`` 0, the target the stand), or with ``seed_from_measured`` at the measured pose
        (the reset events run before this, so it is the new episode's): ``prev_rel`` = the measured position (the
        AT32's deci-degree) clipped to the band, the target there, and the previous action ``prev_rel / scale``, as
        the robot's ``contract.ActionDecoder.reset(rel0)``, so the step limit acts from where the joints are."""
        ids = slice(None) if env_ids is None else env_ids
        all_ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids
        measured = self.measured_rel()
        if self.cfg.seed_from_measured:
            rel = torch.maximum(torch.minimum(measured[ids], self.band), -self.band)
            if self.bounds is not None:
                rel = self.physical(rel, ids) - self._sel(self.off, ids)
            self.prev_rel[ids] = rel
            self.applied_action[ids] = rel / self.scale
            self._raw[ids] = rel / self.scale
            self.target_model[ids] = self._sel(self.stand, ids) + rel
        elif self.bounds is not None:
            rel = self.physical(torch.zeros_like(self.prev_rel[ids]), ids) - self._sel(self.off, ids)
            self._raw[ids] = rel / self.scale
            self.prev_rel[ids] = rel
            self.applied_action[ids] = rel / self.scale
            self.target_model[ids] = self._sel(self.stand, ids) + rel
        else:
            self._raw[ids] = 0.0
            self.prev_rel[ids] = 0.0
            self.applied_action[ids] = 0.0
            self.target_model[ids] = self._sel(self.stand, ids)
        self._push(measured, all_ids)

    def sync(self) -> None:
        """Push the state at the observation's instant, once per policy step."""
        step = int(self._env.common_step_counter)
        if step != self._synced_at:
            self._synced_at = step
            self._push(self.measured_rel())

    # the robot's own estimators -----------------------------------------------------------------------------------
    @property
    def pos_rel(self) -> torch.Tensor:
        return self.hist[:, -1]

    @property
    def velocity(self) -> torch.Tensor:
        return velocity_from_hist(self.hist)


def velocity_from_hist(h: torch.Tensor) -> torch.Tensor:
    """contract.VelocityFilter in torch: a median of 3, then the 4-sample least-squares slope (h: (n, 6, 12))."""
    med = torch.stack([h[:, k:k + 3].median(dim=1).values for k in range(4)], dim=1)       # (n, 4, 12)
    w = torch.tensor([-1.5, -0.5, 0.5, 1.5], device=h.device, dtype=h.dtype) / (5.0 * STATE_DT)
    return (w[None, :, None] * med).sum(1)


def servo_action(env) -> ServoAction:
    return env.action_manager.get_term("servo")


# ------------------------------------------------------------------------------------------------- observation noise
NOISY = ("ang_vel", "gravity", "pos", "vel")


class NoiseTable:
    """Uniform noise half-widths per observation channel, a scalar or one value per element; the ramp writes
    ``amp``. One per env object."""

    def __init__(self, env, start: dict, end: dict):
        dev = env.device
        self.start = {k: torch.as_tensor(start[k], dtype=torch.float32, device=dev) for k in NOISY}
        self.end = {k: torch.as_tensor(end[k], dtype=torch.float32, device=dev) for k in NOISY}
        self.amp = dict(self.start)

    def apply(self, name: str, x: torch.Tensor) -> torch.Tensor:
        a = self.amp[name]
        if not bool((a > 0.0).any()):
            return x
        return x + (torch.rand_like(x) * 2.0 - 1.0) * a


def _noise(env) -> NoiseTable | None:
    return getattr(env, "obs_noise", None)


def _maybe_noise(env, name, x, noisy):
    nt = _noise(env)
    return nt.apply(name, x) if (noisy and nt is not None) else x


# ------------------------------------------------------------------------------------------------- observations
def obs_ang_vel(env, noisy: bool = True):
    x = _sensor(env, "robot/imu_ang_vel")
    return _maybe_noise(env, "ang_vel", x, noisy)


def obs_gravity(env, noisy: bool = True):
    acc = _sensor(env, "robot/imu_lin_acc")
    g = -acc / acc.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return _maybe_noise(env, "gravity", g, noisy)


def obs_command(env, command_name: str = "twist"):
    return env.command_manager.get_command(command_name).clone()


def obs_pos(env, noisy: bool = True):
    term = servo_action(env)
    term.sync()
    return _maybe_noise(env, "pos", term.pos_rel, noisy)


def obs_vel(env, noisy: bool = True):
    term = servo_action(env)
    term.sync()
    return _maybe_noise(env, "vel", term.velocity, noisy)


def obs_prev_action(env):
    return servo_action(env).applied_action


def obs_true_lin_vel(env):
    return env.scene["robot"].data.root_link_lin_vel_b


def obs_base_height(env):
    return env.scene["robot"].data.root_link_pos_w[:, 2:3]


def obs_foot_contact(env, sensor_name: str = "feet_contact"):
    return (env.scene[sensor_name].data.found > 0).float()


# ------------------------------------------------------------------------------------------------------ events
def _sensor(env, name):
    from mjlab.envs.mdp import builtin_sensor
    return builtin_sensor(env, sensor_name=name)


# ------------------------------------------------------------------------------------------------------ events


def reset_command_delay(env, env_ids, latency_steps: tuple, chain_ms: tuple = (8.0, 19.0)):
    """One sensing-to-command delay per environment and episode (ms, drawn from ``chain_ms``, the whole chain from sensor
    to servo), added to each actuator's own latency."""
    robot = env.scene["robot"]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    lo, hi = int(round(chain_ms[0])), int(round(chain_ms[1]))
    chain = torch.randint(lo, hi + 1, (len(env_ids),), device=env.device)
    acts = robot.actuators
    assert len(acts) == len(latency_steps), (len(acts), latency_steps)
    for act, lat in zip(acts, latency_steps):
        act.set_lags(chain + int(lat), env_ids)
    if not hasattr(env, "chain_ms"):
        env.chain_ms = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    env.chain_ms[env_ids] = chain




def torques_l2(env):
    return (env.scene["robot"].data.actuator_force ** 2).sum(-1)




# -------------------------------------------------------------------------------------------------- terminations
def body_contact(env, sensor_name: str = "body_contact"):
    return (env.scene[sensor_name].data.found > 0).any(dim=-1)


def tilt_beyond(env, limit_rad: float):
    g = env.scene["robot"].data.projected_gravity_b
    return torch.acos((-g[:, 2]).clamp(-1.0, 1.0)) > limit_rad


# --------------------------------------------------------------------------------------------------- curriculum
def ramp_noise(env, env_ids, hold_iters: int, ramp_iters: int, steps_per_iter: int = 24):
    """Hold the start amplitudes, then move linearly to the end amplitudes."""
    nt = _noise(env)
    if nt is None:
        return 0.0
    it = env.common_step_counter // steps_per_iter
    frac = 0.0 if it <= hold_iters else min(1.0, (it - hold_iters) / max(ramp_iters, 1))
    for k in NOISY:
        nt.amp[k] = nt.start[k] + frac * (nt.end[k] - nt.start[k])
    return frac
