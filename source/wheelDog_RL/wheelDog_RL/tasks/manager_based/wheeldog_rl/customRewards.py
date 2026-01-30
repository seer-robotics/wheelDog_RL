# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to define rewards for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.envs import mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def feet_ground_time(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 0.04,
) -> torch.Tensor:
    """Rewards long continuous ground contact periods using L2-norm.

    The reward is accumulated only when ground contact has been maintained
    longer than the specified threshold.
    If the command norm is very small (agent is not supposed to move), the reward is zeroed out to avoid incentivizing freezing in place when motion is expected.
    """

    # Extract the contact sensor
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # Time since last loss of contact  →  how long the foot has been continuously on ground
    last_ground_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]

    # Whether a new contact has been established in this very small time window
    # (helps detect when contact is actively continuing right now)
    new_or_ongoing_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]

    # Reward grows with how long the foot has already been in stable contact
    reward = torch.sum(
        torch.clamp(last_ground_time - threshold, min=0.0) * new_or_ongoing_contact,
        dim=1
    )

    # No reward for zero command.
    command = env.command_manager.get_command(command_name)[:, :2]
    command_norm = torch.norm(command, dim=1)
    # Zero command threshold — tune as needed (m/s)
    moving = command_norm > 0.1     
    reward = reward * moving.float()

    return reward


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward
