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
from isaaclab.utils.math import wrap_to_pi, quat_apply_inverse, normalize

# Import custom modules.
from .observations import terrain_normals

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import ActionTerm
    from typing import List


def default_joint_pos(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    std: float,
) -> torch.Tensor:
    """Reward for abdomen (hip abduction/adduction) joints being close to their default positions.

    Returns:
        torch.Tensor: Shape (num_envs,) reward in [0, 1] range.
    """
    # Retrieve the robot articulation
    asset: Articulation = env.scene[asset_cfg.name]

    # Current joint positions (num_envs, num_joints)
    current_pos = wrap_to_pi(asset.data.joint_pos[:, asset_cfg.joint_ids])

    # Target positions as default positions
    target_pos = asset.data.default_joint_pos[:, asset_cfg.joint_ids]

    # Squared L2 error per environment (sum over joints)
    squared_error = torch.mean(torch.square(current_pos - target_pos), dim=1)

    # Exponential reward: exp(-error / std**2)
    # → 1 when error=0, decays to ~0 for large error
    reward = torch.exp(-squared_error / std**2)
    return reward


def joint_deviation_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint positions that deviate from the default one."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.square(angle), dim=1)


def joint_energy_l1(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Penalizes joint energy exertion.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    torques = asset.data.applied_torque[:, asset_cfg.joint_ids]
    velocities = asset.data.joint_vel[:, asset_cfg.joint_ids]
    energy = torch.sum(torch.abs(torques * velocities))
    return energy


def terrain_orientation_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner"),
    reward_temperature: float = 5.0,
) -> torch.Tensor:
    """
    Reward base alignment with terrain normal, where terrain normal is scanned with ray-caster sensor.
    """
    # Enable type hints.
    # robot: RigidObject = env.scene[asset_cfg.name]

    # Get current terrain normal under the robot, in robot base frame.
    terrain_normal_b = terrain_normals(env, sensor_cfg)
    base_up_b = torch.zeros_like(terrain_normal_b, device=env.device)
    base_up_b[:, 2] = 1.0

    # Calculate cosine similarity.
    cos_sim = torch.sum(base_up_b * terrain_normal_b, dim=1)
    cos_sim = torch.clamp(cos_sim, -1.0, 1.0)

    # Penalize using L2 squared kernel.
    reward = torch.exp(-reward_temperature * (1.0 - cos_sim))
    return reward


def terrain_orientation(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner"),
) -> torch.Tensor:
    """Penalize deviation between base projected gravity and the negative terrain normal.
    """
    # Enable type hints.
    # robot: RigidObject = env.scene[asset_cfg.name]

    # Get current terrain normal under the robot, in robot base frame.
    terrain_normal_b = terrain_normals(env, sensor_cfg)

    # Penalize using L2 squared kernel.
    penalty = torch.sum(torch.square(terrain_normal_b[:, :2]), dim=1)
    return penalty


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


def actionTerm_rate_l2(
    env: ManagerBasedRLEnv, 
    term_names: List[str],
) -> torch.Tensor:
    """Penalize the rate of change of specified action terms using L2 squared kernel."""
    total_penalty = torch.zeros(env.num_envs, device=env.device)
    
    offset = 0
    active_terms_list = env.action_manager.active_terms
    
    # Loop through all the terms, update if term is a target.
    for name in active_terms_list:
        term = env.action_manager.get_term(name)
        dim = term.action_dim
        
        if name in term_names:
            current_slice = env.action_manager.action[:, offset : offset + dim]
            previous_slice = env.action_manager.prev_action[:, offset : offset + dim]
            term_penalty = torch.sum(torch.square(current_slice - previous_slice), dim=1)
            total_penalty += term_penalty
        
        offset += dim
    
    # Safety check for missing terms (helpful during configuration/debugging)
    missing_terms = [n for n in term_names if n not in active_terms_list]
    if missing_terms:
        raise ValueError(
            f"One or more action terms not found: {missing_terms}. "
            f"Available terms: {active_terms_list}"
        )
    
    return total_penalty


def ang_vel_xy_l2_clipped(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    Penalize xy-axis base angular velocity using L2 squared kernel.

    Output clipped at 2.0
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    penalty = torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)
    penalty_clipped = torch.clamp(penalty, max=2.0)
    return penalty_clipped


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


def joint_pos_target_l2(
        env: ManagerBasedRLEnv,
        target: torch.Tensor,
        std: float,
        asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward joint position adherence to a target value."""
    # Extract the used quantities (to enable type-hinting).
    asset: Articulation = env.scene[asset_cfg.name]
    # Wrap the joint positions to (-pi, pi).
    joint_pos = wrap_to_pi(asset.data.joint_pos[:, asset_cfg.joint_ids])
    # Compute the reward.
    squared_error = torch.sum(torch.square(joint_pos - target.unsqueeze(0)), dim=1)
    return torch.exp(-squared_error / std**2)


def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]

    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward
