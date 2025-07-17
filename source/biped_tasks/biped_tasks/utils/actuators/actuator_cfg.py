# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from dataclasses import MISSING

from . import actuator_pd
from isaaclab.actuators.actuator_cfg import ActuatorBaseCfg


"""
Explicit Actuator Models.
"""

@configclass
class FixedPDActuatorCfg(ActuatorBaseCfg):
    """Configuration for an fixed PD actuator."""

    class_type: type = actuator_pd.FixedPDActuator
    
    q_ref: float = MISSING