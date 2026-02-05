# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.utils.math import wrap_to_pi

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


def base_height_threshold(
    env: ManagerBasedRLEnv,
    height_threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalize asset height beneath threshold.

    Note:
        For flat terrain, target height is in the world frame. For rough terrain,
        sensor readings can adjust the target height to account for the terrain.
    """
    # Enable type-hints.
    # asset: RigidObject = env.scene[asset_cfg.name]

    # Rectify threshold with sensor data.
    asset: RigidObject = env.scene[asset_cfg.name]
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        # Adjust the threshold height using the sensor data
        adjusted_threshold_height = height_threshold + torch.mean(sensor.data.ray_hits_w[..., 2], dim=1)
    else:
        # Use the provided threshold height directly for flat terrain
        adjusted_threshold_height = height_threshold

    # Compute penalty.
    retval = torch.nan_to_num(
        adjusted_threshold_height - asset.data.root_pos_w[:, 2],
        nan=0.0, posinf=10.0, neginf=-10.0)
    retval = torch.relu(retval)
    return retval


def joint_pos_target_l2(env: ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint position deviation from a target value."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # wrap the joint positions to (-pi, pi)
    joint_pos = wrap_to_pi(asset.data.joint_pos[:, asset_cfg.joint_ids])
    # compute the reward
    return torch.sum(torch.square(joint_pos - target), dim=1)
