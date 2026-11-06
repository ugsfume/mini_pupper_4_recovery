"""Every public simulation module imports (imports alone: no GPU use)."""
import importlib

import pytest

MODULES = ("actuators.servo_model", "actuators.bam_servo", "gpu.algorithms", "gpu.env_cfg", "gpu.getup_bc", "gpu.getup_search",
           "gpu.getup_task", "gpu.mdp", "gpu.patch_warp", "gpu.terrain", "gpu.terrain_task", "gpu.train", "eval.getup_eval",
           "eval.motion_meter", "eval.omni_eval", "eval.push_eval", "eval.sequence", "eval.terrain_eval", "robot.sku2")


@pytest.mark.parametrize("name", MODULES)
def test_imports(name):
    importlib.import_module(f"mp4_recovery.{name}")
