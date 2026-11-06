"""Every module the CPU install needs imports without a GPU or the GPU packages."""
import importlib

import pytest

MODULES = ("contract", "limits", "paths", "heading", "obs_norm", "policy_file", "export_policy", "gpu.actor", "eval.thresholds")


@pytest.mark.parametrize("name", MODULES)
def test_imports(name):
    importlib.import_module(f"mp4_recovery.{name}")
