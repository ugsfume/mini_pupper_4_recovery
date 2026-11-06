# Copyright 2025 Marc Duclusaud & Grégoire Passault

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Trimmed to the Mangdang MD01 family. Upstream also registers Dynamixel,
# Feetech, eRob, Waveshare and Unitree actuators here.

from .mangdang.actuator import (
    MD01PwmActuator,
    MD01PwmPinnedActuator,
)
from .testbench import Pendulum

actuators = {
    # Mangdang MD01
    "md01v": lambda: MD01PwmActuator(Pendulum),
    "md01vp": lambda: MD01PwmPinnedActuator(Pendulum),
}

__all__ = ["actuators"]
