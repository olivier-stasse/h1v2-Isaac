# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import omni.log

from isaaclab.utils.types import ArticulationActions

from isaaclab.actuators.actuator_base import ActuatorBase

if TYPE_CHECKING:
    from .actuator_cfg import FixedPDActuatorCfg


"""
Explicit Actuator Models.
"""


class FixedPDActuator(ActuatorBase):
    def __init__(
        self,
        cfg: ActuatorBaseCfg,
        joint_names: list[str],
        joint_ids: slice | torch.Tensor,
        num_envs: int,
        device: str,
        stiffness: torch.Tensor | float = 0.0,
        damping: torch.Tensor | float = 0.0,
        q_ref: torch.Tensor | float = 0.0,
        armature: torch.Tensor | float = 0.0,
        friction: torch.Tensor | float = 0.0,
        effort_limit: torch.Tensor | float = torch.inf,
        velocity_limit: torch.Tensor | float = torch.inf,
    ):
        # save parameters
        self.cfg = cfg
        self._num_envs = num_envs
        self._device = device
        self._joint_names = joint_names
        self._joint_indices = joint_ids

        # For explicit models, we do not want to enforce the effort limit through the solver
        # (unless it is explicitly set)
        if not self.is_implicit_model and self.cfg.effort_limit_sim is None:
            self.cfg.effort_limit_sim = self._DEFAULT_MAX_EFFORT_SIM

        # parse joint stiffness and damping
        self.stiffness = self._parse_joint_parameter(self.cfg.stiffness, stiffness)
        self.damping = self._parse_joint_parameter(self.cfg.damping, damping)
        # parse joint armature and friction
        self.armature = self._parse_joint_parameter(self.cfg.armature, armature)
        self.friction = self._parse_joint_parameter(self.cfg.friction, friction)
        # parse reference
        self.q_ref = self._parse_joint_parameter(self.cfg.q_ref, q_ref)
        # parse joint limits
        # -- velocity
        self.velocity_limit_sim = self._parse_joint_parameter(self.cfg.velocity_limit_sim, velocity_limit)
        self.velocity_limit = self._parse_joint_parameter(self.cfg.velocity_limit, self.velocity_limit_sim)
        # -- effort
        self.effort_limit_sim = self._parse_joint_parameter(self.cfg.effort_limit_sim, effort_limit)
        self.effort_limit = self._parse_joint_parameter(self.cfg.effort_limit, self.effort_limit_sim)

        # create commands buffers for allocation
        self.computed_effort = torch.zeros(self._num_envs, self.num_joints, device=self._device)
        self.applied_effort = torch.zeros_like(self.computed_effort)

    def reset(self, env_ids: Sequence[int]):
        pass

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        # compute errors
        error_pos = control_action.joint_positions - self.q_ref
        error_vel = control_action.joint_velocities
        # calculate the desired joint torques
        self.computed_effort = self.stiffness * error_pos + self.damping * error_vel
        # clip the torques based on the motor limits
        self.applied_effort = self._clip_effort(self.computed_effort)
        # set the computed actions back into the control action
        control_action.joint_efforts = self.applied_effort
        control_action.joint_positions = None
        control_action.joint_velocities = None
        return control_action
