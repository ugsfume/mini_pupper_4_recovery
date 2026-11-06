"""The policy contract for SKU2: what the network sees and what its output does, in numpy.

This module is the robot side's reference (the Pi runner imports it; it needs numpy only). The training environment
implements the same arithmetic in torch (`mdp.py`); `tests/gpu/test_contract_torch.py` checks that the two agree.

Coordinates. Per servo id 1..12 (FR, FL, RR, RL x abduction, hip, knee), the policy works in the servo's own angle in
radians, relative to the stand: ``rel_i = sign_i * radians(servo_deg_i)``, where ``servo_deg`` is the angle
the firmware's ``servo_write`` takes without the calibration offset (0 = the stand). In the MuJoCo model the abduction
and hip are the ``*_hip`` and ``*_thigh`` joints and the knee is the ``*_knee_drive`` tendon (the shin's angle); the
signs and the stand are in ``data/sku2_contract.json``.

The firmware's units:
``tick = 511 + (int)((servo_deg + offset) * 10240 / 3100)``, ``dd = (int)(3100 - tick * 3100 / 1024)``, both in
float32. The exact inverse for a measured AT32 angle is ``servo_deg = (DD0 - dd) / 10 - offset``, DD0 the dd of tick
511 (convert with 1024, not 1023).

Observation (45, float32): base angular velocity (3, rad/s, body frame, the gyro bias removed), gravity direction (3,
-acc/|acc| in the body frame: (0, 0, -1) upright), command (3: vx, 0, 0 m/s; an omni policy: vx, vy m/s, wz rad/s, body
frame), position (12, rel, rad), velocity (12,
rad/s, ``VelocityFilter``), previous applied action (12).

A policy exported with ``heading_error`` in its meta's ``obs_terms_extra`` (``has_heading_obs``) observes 46: the
45, then the heading loop's error (``heading_obs``: wrap(target - heading), clipped to +-HEADING_CLIP rad, over
HEADING_CLIP: [-1, 1], + = turn left to correct; 0 without the loop). Every other policy's layout is unchanged.

Action (12): ``a`` -> clip to [-1, 1] -> ``scale * a`` (rel, rad) -> the band about the stand -> at most
``STEP_LIMIT_DEG`` from the previous applied target -> firmware ticks -> the dd sent. The previous action observed is
the applied target in action units. The band is ``BAND_DEG`` (the walking band) unless the policy's meta names its own
``band_deg`` (``policy_band_deg``; a get-up policy, ``kind`` "getup"). A reset puts the previous applied target at the
stand, or at a measured pose (``ActionDecoder.reset(rel0)``).

Joint limits (``limits.py``). A policy exported with the measured limits has ``bounds`` in its meta (``Bounds``): per
servo a lower and upper target in the model's coordinates (``stand_rel_rad`` + rel: the abduction and hip joints, the knee
tendon), and the knee's coupling (the knee servo is clamped so that the knee-thigh angle, the ``*_calf`` joint, stays in
its range given the hip's target). The decoder clamps after the step limit, in the calibrated stand's coordinates, then
quantises; the previous applied target is the clamped one.

Stance offset. A policy's meta ``stand_offset_abd_deg`` D shifts its abductions' stand outward by D deg
(``stand_offset_rel``: model / servo-rel sign ``ABD_OUT_SIGN``, + on FR and RL, - on FL and RR): the policy's rel 0 is
the calibrated stand + the offset, so its position observation is the measured rel minus the offset and its targets
are its rel plus the offset (``ActionDecoder.obs_offset``). The band and the step limit act in the policy's
coordinates, the bounds in the calibrated stand's.

Calibration. A policy's bounds hold the potentiometer tracks' limits measured from the stand, so they belong to one
calibration: meta ``bounds.info.contract_stand_dd`` records that stand, and the runner refuses a policy whose stand is not the
contract's (``calibration_refusal``).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

CONTRACT_JSON = Path(__file__).resolve().parent / "data" / "sku2_contract.json"

DD0 = 3100.0 - 511.0 * 3100.0 / 1024.0          # 1553.02734375, the dd of tick 511
STATE_DT = 0.005                                # the ESP32's tick: one STATE frame per 5 ms
CONTROL_DT = 0.020                              # the policy's step (50 Hz)
STEP_LIMIT_DEG = 20.0                           # per policy step, the same in simulation and on the robot
ABDUCTION = (0, 3, 6, 9)                        # indices of the abductions in servo order
BAND_DEG = np.array([20.0 if i in ABDUCTION else 60.0 for i in range(12)])   
ROBOT_BAND_MAX_DEG = 90.0                       # the widest band the robot runs without decoder bounds
ROBOT_GETUP_BAND_MAX_DEG = 180.0                # the widest band of a get-up policy with decoder bounds
POLICY_KINDS = ("walk", "getup")
ABD_OUT_SIGN = (1.0, -1.0, -1.0, 1.0)           # per leg (FR, FL, RR, RL): the abduction's model / rel sign of "outward"
TRACK_DD = (0.0, 3100.0)                        # the potentiometer's electrical track (dd); a ~50 deg dead zone beyond
OBS_SIZE = 45
ACTION_SIZE = 12
# The firmware trot's 360 ms cycle (24 ticks of 15 ms, stanford_gait.h) is 18 policy steps.
GAIT_CYCLE_STEPS = 18
F32 = np.float32


@dataclass
class Contract:
    signs: np.ndarray
    stand_rel_rad: np.ndarray            # the stand in the model's coordinates (joint angles, tendon lengths)
    stand_joint_rad: np.ndarray          # all twelve model joints at the stand (hip, thigh, calf per leg)
    offsets_deg: np.ndarray
    stand_dd: np.ndarray
    stand_ticks: np.ndarray
    joint_order: list
    policy_targets: list
    meta: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path = CONTRACT_JSON) -> "Contract":
        d = json.loads(Path(path).read_text())
        c = cls(signs=np.array(d["signs"], dtype=float), stand_rel_rad=np.array(d["stand_policy_rad"]),
                stand_joint_rad=np.array(d["stand_joint_rad"]), offsets_deg=np.array(d["offsets_deg"]),
                stand_dd=np.array(d["stand_dd"]), stand_ticks=np.array(d["stand_ticks"]),
                joint_order=d["joint_order"], policy_targets=d["policy_targets"], meta=d)
        return c

    # ---- firmware units -------------------------------------------------------------------------------------------
    def ticks_from_servo_deg(self, servo_deg) -> np.ndarray:
        ang = (np.asarray(servo_deg, dtype=F32) + self.offsets_deg.astype(F32)).astype(F32)
        sig = 511 + np.trunc(ang * F32(10240.0) / F32(3100.0)).astype(np.int64)
        return np.clip(sig, 0, 1023)

    @staticmethod
    def dd_from_ticks(ticks) -> np.ndarray:
        t = np.asarray(ticks, dtype=F32)
        return np.trunc(F32(3100.0) - t * F32(3100.0) / F32(1024.0)).astype(np.int64)

    def servo_deg_from_dd(self, dd) -> np.ndarray:
        """A measured AT32 angle (dd) in servo degrees. Not a target's inverse: the dd of tick t reads up to 0.1 deg
        above t's angle, and C's truncation towards zero turns that into t + 1 for ticks below 511."""
        return (DD0 - np.asarray(dd, dtype=float)) / 10.0 - self.offsets_deg

    def servo_deg_of_tick(self, ticks) -> np.ndarray:
        """The target angle (servo degrees) whose firmware tick is exactly ``ticks``: the centre of the tick's
        truncation interval (k = tick - 511: [k, k+1) for k > 0, (k-1, k] for k < 0, (-1, 1) for k = 0)."""
        k = np.asarray(ticks, dtype=float) - 511.0
        return (k + 0.5 * np.sign(k)) * 3100.0 / 10240.0 - self.offsets_deg

    # ---- policy coordinates ---------------------------------------------------------------------------------------
    def rel_from_servo_deg(self, servo_deg) -> np.ndarray:
        return self.signs * np.radians(np.asarray(servo_deg, dtype=float))

    def servo_deg_from_rel(self, rel) -> np.ndarray:
        return np.degrees(np.asarray(rel, dtype=float)) * self.signs

    def rel_from_ticks(self, ticks) -> np.ndarray:
        """A policy target that the firmware path turns into exactly ``ticks`` (the teacher's targets)."""
        return self.rel_from_servo_deg(self.servo_deg_of_tick(ticks))

    def rel_from_dd(self, dd) -> np.ndarray:
        """A measured AT32 angle (STATE ``pos_dd``) as a policy position (rad, relative to the stand)."""
        return self.rel_from_servo_deg(self.servo_deg_from_dd(dd))

    def quantise_rel(self, rel) -> tuple[np.ndarray, np.ndarray]:
        """(the dd the firmware path would send for a target, that dd back in policy coordinates)."""
        dd = self.dd_from_ticks(self.ticks_from_servo_deg(self.servo_deg_from_rel(rel)))
        return dd, self.rel_from_dd(dd)


def calibration_refusal(meta: dict, contract: "Contract") -> str | None:
    """Why a policy's bounds don't fit the contract's calibration, or None (no bounds, or the same stand)."""
    stand = ((meta.get("bounds") or {}).get("info") or {}).get("contract_stand_dd")
    if stand is not None and [int(v) for v in stand] != [int(v) for v in contract.stand_dd]:
        return "bounds made for another calibration (stand dd differ): re-export the policy for this contract"
    return None


def band_deg_array(spec=None) -> np.ndarray:
    """A band spec as 12 floats (deg about the stand, servo order): None = ``BAND_DEG``; one value for all twelve;
    three (abduction, hip, knee) for every leg; or twelve."""
    if spec is None:
        return BAND_DEG.copy()
    b = np.asarray(spec, dtype=float).reshape(-1)
    if b.size == 1:
        b = np.full(ACTION_SIZE, b[0])
    elif b.size == 3:
        b = np.tile(b, 4)
    if b.shape != (ACTION_SIZE,) or not np.all(np.isfinite(b)) or np.any(b <= 0):
        raise ValueError(f"band_deg: 1, 3 or 12 positive values expected, got {spec!r}")
    return b


def policy_band_deg(meta: dict) -> np.ndarray:
    """An exported policy's band (its meta's ``band_deg``); a policy exported without one has the walking band."""
    return band_deg_array(meta.get("band_deg"))


def policy_filter_alpha(meta: dict) -> float:
    """An exported policy's target filter (meta ``action_filter_alpha``): 0 when absent (no filter)."""
    alpha = float(meta.get("action_filter_alpha", 0.0) or 0.0)
    if not 0.0 <= alpha < 1.0:
        raise ValueError(f"action_filter_alpha {alpha!r}: in [0, 1)")
    return alpha


def stand_offset_rel(deg: float) -> np.ndarray:
    """The abductions' stand shifted outward by ``deg`` (rel, rad, servo order; 0 on the hips and knees)."""
    off = np.zeros(ACTION_SIZE)
    for leg, i in enumerate(ABDUCTION):
        off[i] = ABD_OUT_SIGN[leg] * np.radians(float(deg))
    return off


def policy_stand_offset_deg(meta: dict) -> float:
    """An exported policy's abduction stand offset (meta ``stand_offset_abd_deg``, deg outward): 0 when absent."""
    return float(meta.get("stand_offset_abd_deg", 0.0) or 0.0)


@dataclass
class Bounds:
    """Per-servo target bounds in the model's coordinates (servo order: abduction and hip joint angles, knee tendon;
    rad), and the knee coupling: per leg the knee tendon ``kc[0] * thigh + kc[1] * calf`` and the calf (knee - thigh)
    range, so the knee target is clamped into ``kc[0] * thigh_target + kc[1] * [calf_lo, calf_hi]`` after the hip's
    clamp. ``hip_envelope`` (optional, get-up): per leg the hip target's range as a piecewise-linear function of the
    abduction target (``abd``, ``lo``, ``hi``, model rad; constant beyond the ends). The pot's track ends win over the
    coupling where the two disagree (a position the pot cannot read is worse than a target on a joint stop)."""
    lo: np.ndarray
    hi: np.ndarray
    kc: np.ndarray                      # (4, 2)
    calf_lo: np.ndarray                 # (4,)
    calf_hi: np.ndarray                 # (4,)
    hip_envelope: dict | None = None    # {leg: {"abd": [...], "lo": [...], "hi": [...]}}
    mode: str = ""
    info: dict = field(default_factory=dict)

    FRAME = "model: contract stand_policy_rad + rel (servo order; abduction and hip joints, knee tendon), rad"

    def to_meta(self) -> dict:
        return {"frame": self.FRAME, "mode": self.mode, "lo": [float(v) for v in self.lo],
                "hi": [float(v) for v in self.hi], "tendon_coefs": [[float(a), float(b)] for a, b in self.kc],
                "calf_range": [[float(a), float(b)] for a, b in zip(self.calf_lo, self.calf_hi)],
                "hip_envelope": self.hip_envelope, "info": self.info}

    @classmethod
    def from_meta(cls, d: dict | None) -> "Bounds | None":
        if d is None:
            return None
        if d.get("frame") != cls.FRAME:
            raise ValueError(f"bounds: unknown frame {d.get('frame')!r}")
        b = cls(lo=np.array(d["lo"], dtype=float), hi=np.array(d["hi"], dtype=float),
                kc=np.array(d["tendon_coefs"], dtype=float), calf_lo=np.array([r[0] for r in d["calf_range"]], float),
                calf_hi=np.array([r[1] for r in d["calf_range"]], float), hip_envelope=d.get("hip_envelope"),
                mode=str(d.get("mode", "")), info=dict(d.get("info") or {}))
        b.check()
        return b

    def check(self) -> None:
        ok = (self.lo.shape == (ACTION_SIZE,) and self.hi.shape == (ACTION_SIZE,) and self.kc.shape == (4, 2)
              and self.calf_lo.shape == (4,) and self.calf_hi.shape == (4,))
        if not ok:
            raise ValueError("bounds: 12 lo, 12 hi, 4 x 2 tendon coefficients and 4 calf ranges expected")
        vals = np.concatenate([self.lo, self.hi, self.kc.ravel(), self.calf_lo, self.calf_hi])
        if not np.all(np.isfinite(vals)) or np.any(self.lo >= self.hi) or np.any(self.calf_lo >= self.calf_hi):
            raise ValueError("bounds: finite values with lo < hi expected")
        if np.any(self.kc[:, 1] == 0):
            raise ValueError("bounds: a knee tendon without its calf term")
        for leg, e in (self.hip_envelope or {}).items():
            a = np.asarray(e["abd"], dtype=float)
            if (leg not in ("FR", "FL", "RR", "RL") or a.ndim != 1 or len(a) < 2 or np.any(np.diff(a) <= 0)
                    or len(e["lo"]) != len(a) or len(e["hi"]) != len(a)
                    or np.any(np.asarray(e["lo"], dtype=float) > np.asarray(e["hi"], dtype=float))):
                raise ValueError(f"bounds: bad hip envelope for {leg!r}")

    def violation_rel(self, rel, stand) -> float:
        """How far (rad, the largest joint) physical rel (shape (..., 12)) lies outside the bounds, the knee coupling
        and the hip envelope included: 0 inside (``clamp_rel`` returns an inside value unchanged)."""
        rel = np.asarray(rel, dtype=float)
        return float(np.max(np.abs(self.clamp_rel(rel, stand) - rel))) if rel.size else 0.0

    def clamp_rel(self, rel, stand) -> np.ndarray:
        """Physical rel (rad: the model's coordinates minus the calibrated ``stand``), shape (..., 12), clamped into
        the bounds; a value already inside comes back unchanged (bit for bit)."""
        stand = np.asarray(stand, dtype=float)
        lo, hi = self.lo - stand, self.hi - stand
        shape = np.shape(rel)
        v = np.clip(np.asarray(rel, dtype=float), lo, hi).reshape(-1, 4, 3)
        st, lo3, hi3 = stand.reshape(4, 3), lo.reshape(4, 3), hi.reshape(4, 3)
        if self.hip_envelope:
            for li, leg in enumerate(("FR", "FL", "RR", "RL")):
                e = self.hip_envelope.get(leg)
                if e is None:
                    continue
                abd = st[li, 0] + v[:, li, 0]
                elo = np.interp(abd, e["abd"], e["lo"]) - st[li, 1]
                ehi = np.interp(abd, e["abd"], e["hi"]) - st[li, 1]
                v[:, li, 1] = np.clip(np.clip(v[:, li, 1], elo, ehi), lo3[li, 1], hi3[li, 1])
        thigh = st[None, :, 1] + v[:, :, 1]
        a = self.kc[:, 0] * thigh + self.kc[:, 1] * self.calf_lo - st[None, :, 2]
        b = self.kc[:, 0] * thigh + self.kc[:, 1] * self.calf_hi - st[None, :, 2]
        v[:, :, 2] = np.clip(np.clip(v[:, :, 2], np.minimum(a, b), np.maximum(a, b)), lo3[:, 2], hi3[:, 2])
        return v.reshape(shape)


def policy_bounds(meta: dict) -> Bounds | None:
    """An exported policy's decoder bounds (meta ``bounds``); None for a policy exported without them."""
    return Bounds.from_meta(meta.get("bounds"))


def contract_tendon_coefs(contract: "Contract") -> np.ndarray:
    co = contract.meta["tendon_coefs"]
    return np.array([[co[leg][f"{leg}_thigh"], co[leg][f"{leg}_calf"]] for leg in ("FR", "FL", "RR", "RL")], float)


def policy_kind(meta: dict) -> str:
    """An exported policy's kind (its meta's ``kind``): "walk" when absent."""
    return str(meta.get("kind") or "walk")


class ActionDecoder:
    """Stateful: the step limit acts against the previous applied target. ``band_deg``: the band about the stand
    (``band_deg_array``; default ``BAND_DEG``). ``reset()`` at the stand; ``reset(rel0)`` at a measured pose.
    ``bounds`` (``Bounds``, a policy's meta ``bounds``): the targets clamped into the measured limits after the step
    limit. ``stand_offset_abd_deg``: the policy's abduction stand offset (outward). ``rel0`` and the returned ``rel``
    are physical (relative to the calibrated stand: what the dd encode); ``prev_rel``, ``rel_policy`` and ``action``
    are the policy's (rel minus the offset). Without bounds and offset: only the band and the step limit.
    ``filter_alpha`` (a policy's meta ``action_filter_alpha``: ``policy_filter_alpha``): a first-order low-pass
    on the target before the band, ``rel = prev_rel + (1 - alpha) (rel - prev_rel)``, whose state is the previous
    applied target (``reset(rel0)`` seeds it); 0: exactly the decoder before it. ``mdp.ServoAction`` is the same."""

    def __init__(self, contract: Contract, scale, quantise: bool = True, band_deg=None, bounds: Bounds | None = None,
                 stand_offset_abd_deg: float = 0.0, filter_alpha: float = 0.0):
        self.c = contract
        self.scale = np.broadcast_to(np.asarray(scale, dtype=float), (ACTION_SIZE,)).copy()
        self.quantise = quantise
        self.band_deg = band_deg_array(band_deg)
        self.band = np.radians(self.band_deg)
        self.limit = np.radians(STEP_LIMIT_DEG)
        self.bounds = bounds
        if bounds is not None and not np.array_equal(contract_tendon_coefs(contract), bounds.kc):
            raise ValueError("bounds: tendon coefficients differ from the contract's")
        self.stand_offset_abd_deg = float(stand_offset_abd_deg)
        self.offset = stand_offset_rel(self.stand_offset_abd_deg)
        self.filter_alpha = float(filter_alpha)
        if not 0.0 <= self.filter_alpha < 1.0:
            raise ValueError(f"filter_alpha {filter_alpha!r}: in [0, 1)")
        self.reset()

    @property
    def obs_offset(self) -> np.ndarray:
        """Subtracted from a measured (physical) rel: the policy's position observation."""
        return self.offset

    def physical(self, rel_policy) -> np.ndarray:
        """A policy rel as the physical target: the offset added, clamped into the bounds."""
        phys = np.asarray(rel_policy, dtype=float) + self.offset
        if self.bounds is not None:
            phys = self.bounds.clamp_rel(phys, self.c.stand_rel_rad)
        return phys

    def stand_dd(self) -> np.ndarray:
        """The dd of the policy's stand (the contract's stand dd without an offset or bounds)."""
        if not self.offset.any() and self.bounds is None:
            return np.asarray(self.c.stand_dd, dtype=np.int64)
        return self.c.quantise_rel(self.physical(np.zeros(ACTION_SIZE)))[0]

    def reset(self, rel0=None) -> None:
        """The previous applied target: the stand (``rel0`` None), or ``rel0`` (a measured pose, physical rel, rad)
        in the policy's coordinates clipped to the band (and into the bounds), so that the first target moves from
        where the joints are under the step limit. The previous action observed follows it (``prev_rel / scale``, not
        clipped to [-1, 1]: the applied target in action units)."""
        if rel0 is None:
            prev = np.zeros(ACTION_SIZE)
        else:
            rel0 = np.asarray(rel0, dtype=float).reshape(ACTION_SIZE)
            if not np.all(np.isfinite(rel0)):
                raise ValueError("reset: non-finite rel0")
            prev = np.clip(rel0 - self.offset, -self.band, self.band)
        if self.bounds is not None:
            prev = self.physical(prev) - self.offset
        self.prev_rel = prev
        self.prev_action = self.prev_rel / self.scale

    def __call__(self, raw_action) -> dict:
        a = np.clip(np.asarray(raw_action, dtype=float), -1.0, 1.0)
        rel = self.scale * a
        if self.filter_alpha:
            rel = rel + self.filter_alpha * (self.prev_rel - rel)      # = prev + (1 - alpha) (rel - prev)
        rel = np.clip(rel, -self.band, self.band)
        rel = np.clip(rel, self.prev_rel - self.limit, self.prev_rel + self.limit)
        phys = self.physical(rel)
        exact = phys
        if self.quantise:
            dd, phys = self.c.quantise_rel(phys)
        else:
            dd = None
        rel = phys - self.offset
        self.prev_rel = rel
        self.prev_action = rel / self.scale
        return {"dd": dd, "rel": phys, "rel_policy": rel, "action": self.prev_action.copy(), "rel_exact": exact}


class VelocityFilter:
    """Joint velocity from the 200 Hz positions: a median of 3, then the least-squares slope over the last 4.

    Fed every STATE frame (5 ms). Until six samples have arrived, the history is filled with the first one."""

    WEIGHTS = np.array([-1.5, -0.5, 0.5, 1.5]) / (5.0 * STATE_DT)

    def __init__(self, n: int = ACTION_SIZE):
        self.hist = None
        self.n = n

    def reset(self) -> None:
        self.hist = None

    def push(self, x) -> None:
        x = np.asarray(x, dtype=float)
        if self.hist is None:
            self.hist = np.repeat(x[None, :], 6, axis=0)
        else:
            self.hist = np.concatenate([self.hist[1:], x[None, :]], axis=0)

    def velocity(self) -> np.ndarray:
        if self.hist is None:
            return np.zeros(self.n)
        h = self.hist
        med = np.stack([np.median(h[k:k + 3], axis=0) for k in range(4)])
        return self.WEIGHTS @ med


def gravity_from_accel(acc_body) -> np.ndarray:
    acc = np.asarray(acc_body, dtype=float)
    return -acc / np.linalg.norm(acc)


# The observation's terms (name, slice), in order: what a history stacks per term (mjlab's observation manager).
OBS_TERMS = (("ang_vel", slice(0, 3)), ("gravity", slice(3, 6)), ("command", slice(6, 9)), ("pos", slice(9, 21)),
             ("vel", slice(21, 33)), ("prev_action", slice(33, 45)))
HEADING_TERM = "heading_error"          # the optional 46th float, appended last (its own block in a history)
HEADING_CLIP = 0.8                      # rad: the observed error saturates here (the loop's wz from 0.2: kp 2, 0.4)


def has_heading_obs(meta: dict) -> bool:
    """Does an exported policy observe the heading error (its meta's ``obs_terms_extra``)?"""
    return HEADING_TERM in (meta.get("obs_terms_extra") or ())


def obs_terms(heading: bool = False) -> tuple:
    return OBS_TERMS + ((HEADING_TERM, slice(OBS_SIZE, OBS_SIZE + 1)),) if heading else OBS_TERMS


def obs_size(heading: bool = False) -> int:
    return OBS_SIZE + 1 if heading else OBS_SIZE


def heading_obs(err) -> float:
    """The observed heading error: ``err`` (rad, + = turn left to correct) clipped to +-HEADING_CLIP, over it."""
    return float(np.clip(err, -HEADING_CLIP, HEADING_CLIP)) / HEADING_CLIP


class HistoryStacker:
    """The actor's observation with a history of ``length`` steps, as mjlab builds it: each term's frames oldest to
    newest, the terms concatenated in order; the first observation after a reset fills every slot. ``heading``: the
    observation has the heading error (46 floats), its frames the last block."""

    def __init__(self, length: int, heading: bool = False):
        self.length = int(length)
        self.terms = obs_terms(heading)
        self.frames: list | None = None

    def reset(self) -> None:
        self.frames = None

    def __call__(self, obs) -> np.ndarray:
        obs = np.asarray(obs, dtype=F32)
        if self.frames is None:
            self.frames = [obs] * self.length
        else:
            self.frames = self.frames[1:] + [obs]
        h = np.stack(self.frames)                                   # (length, 45 or 46), oldest first
        assert h.shape[1] == self.terms[-1][1].stop, h.shape
        return np.concatenate([h[:, sl].reshape(-1) for _, sl in self.terms]).astype(F32)


def assemble_observation(ang_vel, gravity, command, pos_rel, vel, prev_action, heading_error=None) -> np.ndarray:
    """``command``: vx (a forward policy) or (vx, vy, wz) (an omni policy). ``heading_error``: the observed heading
    term (``heading_obs``) of a policy that has it, appended as the 46th float; None: the 45."""
    obs = np.empty(obs_size(heading_error is not None), dtype=F32)
    obs[0:3] = ang_vel
    obs[3:6] = gravity
    obs[6:9] = (command, 0.0, 0.0) if np.ndim(command) == 0 else command
    obs[9:21] = pos_rel
    obs[21:33] = vel
    obs[33:45] = prev_action
    if heading_error is not None:
        obs[45] = heading_error
    return obs
