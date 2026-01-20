# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg, ManagerBase
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# Manager class that records cumulative velocity error.
class VelocityErrorRecorder():
    def __init__(self, config, env):
        self._env = env
        self._num_envs = env.num_envs
        self.device = env.device

        # Integration time step.
        self.dt = env.step_dt

        # Angular velocity error scaling factor.
        self.angular_scale: dict[str, float] = config.get("angular_scale", 1.0)

        # Initialize cumulative error buffer. 
        self._episode_cum_error = torch.zeros(self._num_envs, device=self.device)

    def reset(self, env_ids: Sequence[int]):
        # Clear buffer on env reset.
        # print(f"[INFO]: Recorder manager resetting cumulative error: {self._episode_cum_error}")
        self._episode_cum_error[env_ids] = 0.0

    def post_physics_step(self):
        # Record cumulative error after physics update.
        robot_data = self._env.scene["robot"].data

        # Get current actual velocities. 
        actual_lin_vel = robot_data.root_lin_vel_b[:, :2]
        actual_ang_vel = robot_data.root_ang_vel_b[:, 2]

        # Get current commands.
        cmd_vel = self._env.command_manager.get_command("base_velocity")

        # Compute instantaneous error
        lin_error = torch.norm(cmd_vel[:, :2] - actual_lin_vel, dim=1)
        ang_error = torch.abs(cmd_vel[:, 2] - actual_ang_vel)
        inst_error = lin_error + ang_error * self.angular_scale

        # Accumulate (integrate over time)
        self._episode_cum_error += inst_error * self.dt

    def get_episode_cum_error(self, env_ids: Sequence[int] = None) -> torch.Tensor:
        if env_ids is None:
            return self._episode_cum_error.clone()
        return self._episode_cum_error[env_ids].clone()


# Distance based curriculum function. 
def terrain_levels_velocityError(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    error_threshold_up: float = 0.5,
    error_threshold_down: float = 2.0
) -> torch.Tensor:
    """Curriculum based on the integrated velocity error the robot walked when commanded to move at a desired velocity.

    This term is used to increase the difficulty of the terrain when the robot walks far enough and decrease the
    difficulty when the robot walks less than half of the distance required by the commanded velocity.

    .. note::
        It is only possible to use this term with the terrain type ``generator``. For further information
        on different terrain types, check the :class:`isaaclab.terrains.TerrainImporter` class.

    Returns:
        The mean terrain level for the given environment ids.
    """
    # Access the recorder manager.
    # error_recorder = env.managers["velocity_error_recorder"]
    error_recorder = env.velocity_error_recorder
    cum_errors = error_recorder.get_episode_cum_error(env_ids)

    # Normalize error by episode duration.
    episode_durations = env.max_episode_length_s * torch.ones_like(cum_errors)  # Approximate; refine if needed.
    norm_errors = cum_errors / episode_durations

    # Terrain management.
    terrain: TerrainImporter = env.scene.terrain

    # Difficulty updates based on normalized error.
    move_up = norm_errors < error_threshold_up
    move_down = norm_errors > error_threshold_down
    move_down *= ~move_up

    # Update terrain origins.
    terrain.update_env_origins(env_ids, move_up, move_down)

    # Return the mean terrain level.
    print(f"[INFO]: Updated terrain levels, mean levels: {terrain.terrain_levels.float()}")
    return torch.mean(terrain.terrain_levels.float())
    


def terrain_levels_vel(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Curriculum based on the distance the robot walked when commanded to move at a desired velocity.

    This term is used to increase the difficulty of the terrain when the robot walks far enough and decrease the
    difficulty when the robot walks less than half of the distance required by the commanded velocity.

    .. note::
        It is only possible to use this term with the terrain type ``generator``. For further information
        on different terrain types, check the :class:`isaaclab.terrains.TerrainImporter` class.

    Returns:
        The mean terrain level for the given environment ids.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")
    # compute the distance the robot walked
    distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    # robots that walked far enough progress to harder terrains
    move_up = distance > terrain.cfg.terrain_generator.size[0] / 2
    # robots that walked less than half of their required distance go to simpler terrains
    move_down = distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    move_down *= ~move_up
    # update terrain levels
    terrain.update_env_origins(env_ids, move_up, move_down)
    # return the mean terrain level
    return torch.mean(terrain.terrain_levels.float())
