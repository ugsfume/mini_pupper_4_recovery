"""The Mini Pupper v4 SKU2 base environment in mjlab, which the walking and get-up tasks both extend.

One policy step is 20 ms (50 Hz) of twenty 1 ms physics steps. It holds the robot (fitted servo model, measured masses),
the observations of ``contract.py``, the observation noise and its ramp, the command delay, the plant randomisation
(the feet's friction, the base's center of mass), and the fall terminations.

Observation noise is uniform with per-element half-widths. The ramp starts from a clean floor and ends at a profile
measured on the robot (``data/noise_measured.json``: the sensor noise plus the broadband excess of the logs over the
simulation in motion).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity import mdp as vel_mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

from mp4_recovery import paths
from mp4_recovery.actuators import servo_model
from mp4_recovery.actuators.bam_servo import bam_servo_init
from mp4_recovery.gpu import mdp
from mp4_recovery.robot.sku2 import BODY_CONTACT_BODIES, FOOT_GEOMS, robot_cfg

DECIMATION = 20
STEPS_PER_ITER = 24         # env steps per PPO iteration
CHAIN_MS = (8.0, 19.0)      # the sensing-to-command delay's range
NOISE_START = {"ang_vel": 0.02, "gravity": 0.01, "pos": 0.002, "vel": 0.05}     # rad/s, unit vector, rad, rad/s
NOISE_HOLD_ITERS, NOISE_RAMP_ITERS = 150, 300
NOISE_JSON = paths.DATA / "noise_measured.json"


def noise_profile(name: str = "measured") -> dict:
    """The noise's end amplitudes per channel and element, as measured on the robot."""
    if name != "measured":
        raise ValueError(f"unknown noise profile {name!r}")
    prof = json.loads(NOISE_JSON.read_text())["profile_uniform_half_width"]
    return {ch: [float(v) for v in vals] for ch, vals in prof.items()}


@dataclass
class TaskOptions:
    num_envs: int = 4096
    noise_start: dict = field(default_factory=lambda: dict(NOISE_START))
    noise_end: dict = field(default_factory=noise_profile)
    noise_ramp: bool = True                 # False: the end amplitudes from the start (evaluation under noise)
    noisy: bool = True                      # False: clean observations (evaluation, play)
    randomise: bool = True                  # friction, center of mass, delay drawn per env
    chain_ms: tuple = CHAIN_MS              # the sensing-to-command delay's range (ms), one draw per episode
    action_scale: tuple = (0.25, 0.50, 0.50) * 4
    episode_s: float = 20.0
    visuals: bool = False                   # keep the meshes (rendering only)


def install_noise(env, env_ids, start: dict, end: dict, ramp: bool):
    if not hasattr(env, "obs_noise"):
        env.obs_noise = mdp.NoiseTable(env, start, end)
        if not ramp:
            env.obs_noise.amp = dict(env.obs_noise.end)


def _obs_terms(noisy: bool) -> dict:
    return {
        "ang_vel": ObservationTermCfg(func=mdp.obs_ang_vel, params={"noisy": noisy}),
        "gravity": ObservationTermCfg(func=mdp.obs_gravity, params={"noisy": noisy}),
        "command": ObservationTermCfg(func=mdp.obs_command),
        "pos": ObservationTermCfg(func=mdp.obs_pos, params={"noisy": noisy}),
        "vel": ObservationTermCfg(func=mdp.obs_vel, params={"noisy": noisy}),
        "prev_action": ObservationTermCfg(func=mdp.obs_prev_action),
    }


def make_env_cfg(opt: TaskOptions | None = None) -> ManagerBasedRlEnvCfg:
    opt = opt or TaskOptions()
    model = servo_model.md01()
    critic = _obs_terms(False)
    critic.update({
        "lin_vel": ObservationTermCfg(func=mdp.obs_true_lin_vel),
        "height": ObservationTermCfg(func=mdp.obs_base_height),
        "contact": ObservationTermCfg(func=mdp.obs_foot_contact),
    })
    observations = {
        "actor": ObservationGroupCfg(terms=_obs_terms(opt.noisy), concatenate_terms=True, enable_corruption=False),
        "critic": ObservationGroupCfg(terms=critic, concatenate_terms=True, enable_corruption=False),
    }
    sensors = (
        ContactSensorCfg(name="feet_contact",
                         primary=ContactMatch(mode="geom", pattern=FOOT_GEOMS, entity="robot"),
                         secondary=ContactMatch(mode="body", pattern="terrain"),
                         fields=("found", "force"), reduce="netforce", num_slots=1, track_air_time=True),
        ContactSensorCfg(name="body_contact",
                         primary=ContactMatch(mode="body", pattern=BODY_CONTACT_BODIES, entity="robot"),
                         secondary=ContactMatch(mode="body", pattern="terrain"),
                         fields=("found",), reduce="none", num_slots=1),
    )
    events = {
        "noise": EventTermCfg(func=install_noise, mode="startup",
                              params={"start": opt.noise_start, "end": opt.noise_end, "ramp": opt.noise_ramp}),
        "reset_base": EventTermCfg(func=vel_mdp.reset_root_state_uniform, mode="reset",
                                   params={"pose_range": {}, "velocity_range": {}}),
        "reset_joints": EventTermCfg(func=vel_mdp.reset_joints_by_offset, mode="reset",
                                     params={"position_range": (0.0, 0.0), "velocity_range": (0.0, 0.0),
                                             "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))}),
        "command_delay": EventTermCfg(func=mdp.reset_command_delay, mode="reset",
                                      params={"latency_steps": model.latency_steps,
                                              "chain_ms": opt.chain_ms if opt.randomise else
                                              (sum(opt.chain_ms) / 2,) * 2}),
        "bam_init": EventTermCfg(func=bam_servo_init, mode="startup"),
    }
    if opt.randomise:
        events["foot_friction"] = EventTermCfg(
            func=envs_mdp.dr.geom_friction, mode="startup",
            params={"asset_cfg": SceneEntityCfg("robot", geom_names=FOOT_GEOMS), "operation": "abs",
                    "axes": [0], "ranges": (0.4, 1.2), "shared_random": True})
        events["base_com"] = EventTermCfg(
            func=envs_mdp.dr.body_com_offset, mode="startup",
            params={"asset_cfg": SceneEntityCfg("robot", body_names=("base",)), "operation": "add",
                    "ranges": {0: (-0.005, 0.005), 1: (-0.005, 0.005), 2: (-0.003, 0.003)}})
    terminations = {
        "time_out": TerminationTermCfg(func=vel_mdp.time_out, time_out=True),
        "tilt": TerminationTermCfg(func=mdp.tilt_beyond, params={"limit_rad": 0.7}),
        "body_contact": TerminationTermCfg(func=mdp.body_contact),
    }
    curriculum = {}
    if opt.noise_ramp and opt.noisy:
        curriculum["noise_ramp"] = CurriculumTermCfg(
            func=mdp.ramp_noise, params={"hold_iters": NOISE_HOLD_ITERS, "ramp_iters": NOISE_RAMP_ITERS})
    commands = {"twist": UniformVelocityCommandCfg(
        entity_name="robot", resampling_time_range=(1.0e6, 1.0e6), heading_command=False,
        rel_standing_envs=0.0, rel_heading_envs=0.0, debug_vis=False,
        ranges=UniformVelocityCommandCfg.Ranges(lin_vel_x=(0.0, 0.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0)))}
    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(terrain=TerrainEntityCfg(terrain_type="plane"), num_envs=opt.num_envs, env_spacing=1.0,
                       entities={"robot": robot_cfg(model.actuators, strip_visuals=not opt.visuals)}, sensors=sensors),
        observations=observations,
        actions={"servo": mdp.ServoActionCfg(scale=tuple(opt.action_scale))},
        commands=commands,
        events=events,
        rewards={},
        terminations=terminations,
        curriculum=curriculum,
        viewer=ViewerConfig(origin_type=ViewerConfig.OriginType.ASSET_BODY, entity_name="robot", body_name="base",
                            distance=0.6, elevation=-10.0, azimuth=90.0),
        sim=SimulationCfg(njmax=300, nconmax=None,
                          mujoco=MujocoCfg(timestep=servo_model.PHYSICS_DT, iterations=10, ls_iterations=20)),
        decimation=DECIMATION,
        episode_length_s=opt.episode_s,
    )


def describe(opt: TaskOptions) -> dict:
    """What a run's saved config must name: the options, the servo model and the sources' hashes."""
    from mp4_recovery import limits
    from mp4_recovery.robot.sku2 import sources
    model = servo_model.md01()
    out = {"options": {k: (list(v) if isinstance(v, tuple) else v) for k, v in opt.__dict__.items()},
           "servo_model": model.name, "servo_params": model.params, "latency_steps": list(model.latency_steps),
           "decimation": DECIMATION, "physics_dt": servo_model.PHYSICS_DT, "sources": sources()}
    if getattr(opt, "limits", False):
        out["limits"] = limits.describe(True, getattr(opt, "kind", "walk"), getattr(opt, "stand_offset_abd_deg", 0.0))
        out["sources"][str(limits.LIMITS_JSON.relative_to(paths.REPO))] = limits.LIMITS_SHA256
    return out
