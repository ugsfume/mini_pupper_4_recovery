"""The walking task's environment: the base environment (``env_cfg``) on terrain, for evaluating the walker.

- **terrain:** mjlab's generator with the sub-terrains of ``terrain.py``, a spawn per difficulty row and family.
- **commands:** one fixed (vx, vy, wz) per env, written by the evaluation tools, observed whole (45 floats, or stacked
  frames for a policy with history).
- **robot:** shins and feet collide with the base, a stiffer foot contact, the base's mass drawn per env.
- **actions:** the policy's action scale and band; with ``limits`` the measured joint limits in the model and the
  decoder (``limits.py``); ``stand_offset_abd_deg`` shifts the abductions' stand outward.
- **terminations:** a tilt beyond ``tilt_limit``, a body touching the ground, the time-out, leaving the terrain.
- 20 s episodes by default.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from mjlab.envs import mdp as envs_mdp
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sensor.builtin_sensor import ObjRef
from mjlab.sensor.raycast_sensor import RingPatternCfg
from mjlab.sensor.terrain_height_sensor import TerrainHeightSensorCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.terrains.primitive_terrains import BoxFlatTerrainCfg
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg

from mp4_recovery.actuators import servo_model
from mp4_recovery.gpu import terrain as terrain_mod
from mp4_recovery.gpu.env_cfg import TaskOptions, make_env_cfg
from mp4_recovery.robot.sku2 import FOOT_SITES, LEG_COLLISION_GEOMS, robot_cfg

FOOT_R = 0.0049
BORDER = 0.5
SINGLE_ROW = ("square_walls", "stairs_built")


def _fresh_raycasts() -> None:
    """mjlab 1.3.0 caches a sensor's data until ``reset`` or ``update``, but raycasts are refreshed later, in
    ``sense()``, so a reader between the two gets the previous step's rays (upstream issue 998, fixed in 1.5.3).
    Invalidate the cache when ``sense()`` writes new rays."""
    from mjlab.sensor.raycast_sensor import RayCastSensor
    if getattr(RayCastSensor, "_fresh_rays", False):
        return
    original = RayCastSensor.postprocess_rays

    def postprocess_rays(self):
        original(self)
        self._invalidate_cache()

    RayCastSensor.postprocess_rays = postprocess_rays
    RayCastSensor._fresh_rays = True


_fresh_raycasts()


@dataclass
class TerrainOptions(TaskOptions):
    terrain: str = "rough"                  # "rough", "flat", "square_walls", "stairs_built"
    rows: int = 10                          # difficulty levels
    tile: float = terrain_mod.TILE_SIZE
    terrain_seed: int = 0
    max_init_level: int = 2
    band_deg: tuple | None = None           # the decoder's band about the stand, 12 values (servo order; deg); None:
                                            # the walking band 20/60/60 (contract.BAND_DEG)
    history: int = 0                        # actor observation history (0: 45 floats)
    masses_json: str | None = "sku2_masses_478g.json"   # the robot with its Raspberry Pi heatsink
    self_collision: bool = True             # shins and feet collide with the base
    friction_range: tuple = (0.5, 1.3)
    # the feet's contact (timeconst, dampratio). MuJoCo's (0.02, 1) lets a walking foot sink 3.3 mm (p99); (0.01, 1)
    # 1.7 mm, (0.005, 1) 0.6 mm
    foot_solref: tuple | None = (0.01, 1.0)
    fixed_command: tuple = (0.0, 0.0, 0.0)  # (vx, 0, wz) for every env
    level: int | None = 0                   # every env on this difficulty row
    stair_layout: tuple | None = None       # "stairs_built": ((riser, tread), ...) in m, a measured staircase
    stair_width: float | None = None        # "stairs_built": the treads' width (m; None: the whole tile)
    stair_landing: float | None = None      # "stairs_built": the top tread's length (m; None: to the tile's edge)
    tilt_limit: float = 0.7                 # rad: the body's tilt from vertical that ends an episode as a fall
    limits: bool = False                    # the measured joint limits (limits.py), in the model and the decoder
    widen_limits_deg: float | None = None   # hip and knee targets may go this far past their track ends (limits.py)
    stand_offset_abd_deg: float = 0.0       # the abductions' stand shifted outward; the clamps stay in the calibrated
                                            # stand's coordinates


def out_of_bounds(env, half_x: float, half_y: float, margin: float = 0.3):
    p = env.scene["robot"].data.root_link_pos_w
    return (p[:, 0].abs() > half_x - margin) | (p[:, 1].abs() > half_y - margin)


def foot_heights_centre(env, sensor_name: str = "foot_height") -> torch.Tensor:
    """Each foot's sphere centre above the terrain straight under it (the ring's centre ray; the site's current height,
    the ray from the previous ``sense()``); 0 when the ray missed."""
    d = env.scene[sensor_name].data
    B, F = d.heights.shape[0], d.heights.shape[1]
    hit_z = d.hit_pos_w[..., 2].view(B, F, -1)[..., 0]
    hit = d.distances.view(B, F, -1)[..., 0] >= 0
    site = env.scene["robot"].data.site_pos_w[:, _foot_site_ids(env), 2]
    return torch.where(hit, site - hit_z, torch.zeros_like(site))


def _foot_site_ids(env):
    robot = env.scene["robot"]
    if not hasattr(env, "_foot_sites"):
        env._foot_sites = [robot.site_names.index(s) for s in FOOT_SITES]
    return env._foot_sites


def sub_terrains(opt: TerrainOptions) -> dict:
    if opt.terrain == "rough":
        return terrain_mod.rough_sub_terrains(opt.tile)
    if opt.terrain == "square_walls":  # four blocks of the printed tile, every edge a mirror line, with vertical sides
        return {"square_walls": terrain_mod.TileStripCfg(size=(opt.tile, opt.tile), n_long=2, n_wide=2, walls=True)}
    if opt.terrain == "stairs_built":  # one measured staircase
        return {"built": terrain_mod.StairGridCfg(size=(opt.tile, opt.tile), layout=tuple(opt.stair_layout),
                                                  width=opt.stair_width, landing=opt.stair_landing, tag="built")}
    if opt.terrain == "flat":
        return {"flat": BoxFlatTerrainCfg(size=(opt.tile, opt.tile))}
    raise ValueError(f"unknown terrain {opt.terrain!r}")


def servo_limits(servo_cfg, opt, mode: str) -> None:
    """The decoder's clamps (``opt.limits``: ``limits.decoder_bounds(mode)``) and stand offset on a ServoActionCfg."""
    if opt.limits:
        from mp4_recovery.limits import decoder_bounds
        servo_cfg.bounds = decoder_bounds(mode, widen_limits_deg=opt.widen_limits_deg).to_meta()
    servo_cfg.stand_offset_abd_deg = float(opt.stand_offset_abd_deg)


def make_terrain_env_cfg(opt: TerrainOptions | None = None):
    opt = opt or TerrainOptions()
    cfg = make_env_cfg(opt)
    subs = sub_terrains(opt)
    rows = 1 if opt.terrain in SINGLE_ROW else opt.rows
    gen = TerrainGeneratorCfg(size=(opt.tile, opt.tile), sub_terrains=subs, num_rows=rows, curriculum=True,
                              border_width=BORDER, seed=opt.terrain_seed)
    cfg.scene.terrain = TerrainEntityCfg(terrain_type="generator", terrain_generator=gen,
                                         max_init_terrain_level=opt.max_init_level if opt.level is None else 0)
    cfg.scene.entities = {"robot": robot_cfg(servo_model.md01().actuators, strip_visuals=not opt.visuals,
                                             masses_json=opt.masses_json, self_collision=opt.self_collision,
                                             foot_solref=opt.foot_solref, limits=opt.limits)}
    half_x = rows * opt.tile / 2 + BORDER
    half_y = len(subs) * opt.tile / 2 + BORDER

    foot_height = TerrainHeightSensorCfg(name="foot_height",
                                         frame=tuple(ObjRef(type="site", name=s, entity="robot") for s in FOOT_SITES),
                                         pattern=RingPatternCfg.single_ring(radius=0.006, num_samples=4),
                                         ray_alignment="yaw", max_distance=0.5, exclude_parent_body=True,
                                         include_geom_groups=(0,), reduction="min")
    leg_contact = ContactSensorCfg(name="leg_contact",                     # thigh and shin capsules, not the feet
                                   primary=ContactMatch(mode="geom", pattern=LEG_COLLISION_GEOMS, entity="robot"),
                                   secondary=ContactMatch(mode="body", pattern="terrain"),
                                   fields=("found",), reduce="none", num_slots=1)
    cfg.scene.sensors = tuple(cfg.scene.sensors) + (foot_height, leg_contact)

    if opt.history:
        cfg.observations["actor"].history_length = opt.history

    fixed = opt.fixed_command
    cfg.commands = {"twist": UniformVelocityCommandCfg(
        entity_name="robot", resampling_time_range=(1.0e6, 1.0e6), heading_command=False,
        heading_control_stiffness=2.0, rel_standing_envs=0.0, rel_heading_envs=0.0, debug_vis=False,
        ranges=UniformVelocityCommandCfg.Ranges(lin_vel_x=(fixed[0], fixed[0]), lin_vel_y=(0.0, 0.0),
                                                ang_vel_z=(fixed[2], fixed[2])))}

    if opt.randomise:
        cfg.events["foot_friction"].params["ranges"] = tuple(opt.friction_range)
        cfg.events["base_mass"] = EventTermCfg(
            func=envs_mdp.dr.body_mass, mode="startup",
            params={"asset_cfg": SceneEntityCfg("robot", body_names=("base",)), "operation": "scale",
                    "ranges": (0.9, 1.1)})

    cfg.terminations["out_of_bounds"] = TerminationTermCfg(func=out_of_bounds, time_out=True,
                                                           params={"half_x": half_x, "half_y": half_y})
    cfg.terminations["tilt"].params["limit_rad"] = opt.tilt_limit
    if opt.band_deg is not None:
        band = tuple(float(b) for b in opt.band_deg)
        assert len(band) == 12 and all(b > 0 for b in band), band
        cfg.actions["servo"].band_deg = band
    servo_limits(cfg.actions["servo"], opt, "walk")
    cfg.sim.njmax = 600
    cfg.sim.nconmax = 64
    cfg.episode_length_s = opt.episode_s
    return cfg
