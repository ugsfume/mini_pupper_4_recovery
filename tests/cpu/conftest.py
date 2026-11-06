import pytest
from robot_helpers import robot  # noqa: F401

from mp4_recovery.deploy import runner


@pytest.fixture(autouse=True)
def ignore_late_steps(monkeypatch):
    """The simulated board runs in real time, on a machine that may be busy: late steps are not what these tests check."""
    monkeypatch.setattr(runner, "LATE_STEPS_STOP", 10**6)
