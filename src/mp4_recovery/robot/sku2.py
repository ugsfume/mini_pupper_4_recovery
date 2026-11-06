"""The Mini Pupper v4 SKU2 as an mjlab entity: the model in ``robot/assets`` with the weighed masses and corrected stand.

Changes, made on the MjSpec at load time (the file is never edited):

- the ghost robot (``g_base``), the ``stand`` keyframe, the twelve XML position actuators (the servo model's replace
  them) and the XML sensor are removed; with ``strip_visuals`` the mesh geoms (``contype = conaffinity = 0``) and the
  meshes too (masses are explicit ``<inertial>``s, so the dynamics do not change);
- an IMU at site ``imu``: gyro ``imu_ang_vel``, velocimeter ``imu_lin_vel``, accelerometer ``imu_lin_acc``;
- the weighed masses (``data/sku2_masses*.json``: each module 66.82 g, each leg's thigh and calf scaled together to 6.5 g,
  the base takes the rest, inertias scale with mass, the base's center of mass x set to the value solved for the measured
  56.6 % rear share);
- the feet get contact priority 1, so that their friction (randomised) is the one MuJoCo uses against the ground
  (equal priorities take the larger of the two, and the ground's is 1.0).

With ``limits`` the abduction (``*_hip``) and knee - thigh (``*_calf``) joints get the measured ranges (``limits.py``).

The stand (all twelve joints) comes from ``data/sku2_contract.json``, with the stand geometry corrected to the firmware's
neutral pose (hip axes 70 mm above the feet).
"""
from __future__ import annotations

import json

import mujoco
import numpy as np

from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

from mp4_recovery import paths
from mp4_recovery.contract import Contract

LEGS = ("FR", "FL", "RR", "RL")
FOOT_GEOMS = tuple(f"{leg}_foot" for leg in LEGS)
FOOT_SITES = tuple(f"{leg}_foot" for leg in LEGS)
LEG_COLLISION_GEOMS = tuple(f"{leg}_{part}_collision" for leg in ("FR", "FL", "RR", "RL") for part in ("thigh", "calf"))
BODY_CONTACT_BODIES = ("base",) + tuple(f"{leg}_hip" for leg in LEGS)   # the trunk and the motor cans
# The base origin's height at the settled stand (71.9 mm); the start sits just above it.
INIT_BASE_Z = 0.0725


def masses(path=None) -> dict:
    return json.loads((paths.MASSES_JSON if path is None else paths.DATA / path).read_text())


def apply_masses(spec: mujoco.MjSpec, m: dict) -> dict:
    new = {}
    for leg in LEGS:
        new[f"{leg}_hip"] = m["module_kg"]
        t, c = spec.body(f"{leg}_thigh").mass, spec.body(f"{leg}_calf").mass
        k = m["leg_kg"] / (t + c)
        new[f"{leg}_thigh"], new[f"{leg}_calf"] = t * k, c * k
    others = sum(b.mass for b in spec.bodies if b.name and b.name not in new and b.name not in ("base", "world"))
    new["base"] = m["total_kg"] - sum(new.values()) - others
    for name, mass in new.items():
        b = spec.body(name)
        b.fullinertia = np.asarray(b.fullinertia) * (mass / b.mass)
        b.mass = mass
    ipos = np.array(spec.body("base").ipos)
    ipos[0] = m["base_com_x_m"]
    spec.body("base").ipos = ipos
    return new


def leg_base_collisions(spec: mujoco.MjSpec) -> None:
    """Let each shin and foot collide with the base boxes (the model has no self-collision: robot geoms are contype 1,
    conaffinity 2). Bit 4 on the base's conaffinity and the calves' contype pairs exactly those; the thighs start on
    the base box and stay excluded. Terrain contacts are unchanged."""
    for g in spec.body("base").geoms:
        if g.contype or g.conaffinity:
            g.conaffinity |= 4
    for leg in LEGS:
        for g in spec.body(f"{leg}_calf").geoms:
            if g.contype or g.conaffinity:
                g.contype |= 4


def get_spec(strip_visuals: bool = True, masses_json: str | None = None,
             self_collision: bool = False, foot_solref: tuple | None = None, limits: bool = False) -> mujoco.MjSpec:
    """``limits``: the measured abduction and knee - thigh ranges as joint ranges (``limits.apply_limits``)."""
    spec = mujoco.MjSpec.from_file(str(paths.SKU2_XML))
    spec.delete(spec.body("g_base"))
    for key in list(spec.keys):
        spec.delete(key)
    for act in list(spec.actuators):
        spec.delete(act)
    for s in list(spec.sensors):
        spec.delete(s)
    if strip_visuals:
        for g in list(spec.geoms):
            if g.contype == 0 and g.conaffinity == 0:
                spec.delete(g)
        for mesh in list(spec.meshes):
            spec.delete(mesh)
    for name, kind in (("imu_ang_vel", mujoco.mjtSensor.mjSENS_GYRO),
                       ("imu_lin_vel", mujoco.mjtSensor.mjSENS_VELOCIMETER),
                       ("imu_lin_acc", mujoco.mjtSensor.mjSENS_ACCELEROMETER)):
        spec.add_sensor(name=name, type=kind, objtype=mujoco.mjtObj.mjOBJ_SITE, objname="imu")
    for leg in LEGS:                         # name the legs' collision capsules (feet are named in the MJCF)
        for part in ("thigh", "calf"):
            for k, g in enumerate(g for g in spec.body(f"{leg}_{part}").geoms
                                  if (g.contype or g.conaffinity) and g.name not in FOOT_GEOMS):
                g.name = f"{leg}_{part}_collision{k or ''}"
    for g in FOOT_GEOMS:
        spec.geom(g).priority = 1
        if foot_solref is not None:          # the feet's priority makes their contact parameters the ones used
            spec.geom(g).solref = list(foot_solref)
    apply_masses(spec, masses(masses_json))
    if self_collision:
        leg_base_collisions(spec)
    if limits:
        from mp4_recovery.limits import apply_limits
        apply_limits(spec)
    return spec


def robot_cfg(actuators: tuple, strip_visuals: bool = True, masses_json: str | None = None,
              self_collision: bool = False, foot_solref: tuple | None = None, limits: bool = False) -> EntityCfg:
    c = Contract.load()
    joint_pos = {name: float(v) for name, v in zip(c.joint_order, c.stand_joint_rad)}
    return EntityCfg(
        init_state=EntityCfg.InitialStateCfg(pos=(0.0, 0.0, INIT_BASE_Z), joint_pos=joint_pos,
                                             joint_vel={".*": 0.0}),
        spec_fn=lambda: get_spec(strip_visuals, masses_json, self_collision, foot_solref, limits),
        articulation=EntityArticulationInfoCfg(actuators=actuators),
    )


def sources() -> dict:
    """Hashes of every input the model is built from (written into each run's config)."""
    return {str(paths.SKU2_XML.relative_to(paths.REPO)): paths.sha256(paths.SKU2_XML),
            str(paths.MASSES_JSON.relative_to(paths.REPO)): paths.sha256(paths.MASSES_JSON)}
