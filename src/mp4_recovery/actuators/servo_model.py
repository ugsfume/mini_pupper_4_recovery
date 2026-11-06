"""The fitted MD01 servo model: what turns a policy target into a joint torque.

The model is BAM (``actuators/bam``, class md01vp, model m6), refitted on bench data with the armature pinned at 0.6 of a
bound measured on the bare output arm (knee 2.2e-4, hip 2.7e-4 kg m^2) and kd_ratio at 1.0, which predicts the robot's
joints better than the published parameters at its light loads. The knee and abduction servos use one parameter file,
the hip another. At kp 80, kd 400, kff 0.001, max_pwm 0.99, 12 V.

Each set's fitted command delay is applied on the 1 ms physics grid, rounded up. The sensing-to-command delay, the same
for every servo, is added at reset by ``mdp.reset_command_delay``; every actuator holds its lag for the whole episode
(``delay_hold_prob = 1``: mjlab otherwise resamples a lag at every physics step, and actuator groups with different
transmissions keep separate buffers).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from mjlab.actuator.actuator import TransmissionType

from mp4_recovery import paths
from mp4_recovery.actuators.bam_servo import BamServoCfg

PHYSICS_DT = 0.001
MAX_LAG_STEPS = 60          # the delay buffers' capacity (ms at 1 ms physics); lags are set per episode below it
NAME = "md01"
ROLES = (("abd", (r".*_hip",), paths.DATA / "md01_knee_abduction.json", TransmissionType.JOINT),
         ("hip", (r".*_thigh",), paths.DATA / "md01_hip.json", TransmissionType.JOINT),
         ("knee", (r".*_knee_drive",), paths.DATA / "md01_knee_abduction.json", TransmissionType.TENDON))


@dataclass(frozen=True)
class ServoModel:
    name: str
    actuators: tuple            # mjlab ActuatorCfg, in the entity's order
    latency_steps: tuple        # each actuator config's own command latency, physics steps
    params: dict                # every parameter, echoed into the run config


def command_delay_steps(params_path) -> int:
    """A parameter file's ``command_delay`` in physics steps: a command takes effect at the first 1 ms step at or after
    its time + delay, i.e. the delay rounded up."""
    ms = float(json.loads(Path(params_path).read_text())["command_delay"]) * 1000.0
    return int(math.ceil(ms / (PHYSICS_DT * 1000.0) - 1e-9))


def md01() -> ServoModel:
    cfgs, lat, roles = [], [], {}
    for role, targets, path, transmission in ROLES:
        delay = command_delay_steps(path)
        cfgs.append(BamServoCfg(target_names_expr=targets, transmission_type=transmission, vin_range=None,
                                stiff_frictionloss=False, delay_min_lag=0, delay_max_lag=MAX_LAG_STEPS,
                                delay_hold_prob=1.0, json_path=str(path)))
        lat.append(delay)
        roles[role] = {"params": str(path.relative_to(paths.REPO)), "sha256": paths.sha256(path),
                       "command_delay_steps": delay}
    return ServoModel(NAME, tuple(cfgs), tuple(lat), {"roles": roles})
