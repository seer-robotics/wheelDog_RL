"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.assets import Articulation
    from isaaclab.terrains import TerrainImporter


# Manager class that records cumulative velocity error.
class VelocityErrorRecorder():
    def __init__(self, config: dict, env: ManagerBasedRLEnv):
        self._env = env
        self._num_envs = env.num_envs
        self.device = env.device

        # Integration time step.
        # Remember that step_dt is sim.dt*decimation
        self.step_dt = env.step_dt

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
        robot: Articulation = self._env.scene["robot"]

        # Get current actual velocities. 
        actual_lin_vel = robot.data.root_lin_vel_b[:, :2]
        actual_ang_vel = robot.data.root_ang_vel_b[:, 2]

        # Get current commands.
        cmd_vel = self._env.command_manager.get_command("base_velocity")

        # Compute instantaneous error
        lin_error = torch.norm(cmd_vel[:, :2] - actual_lin_vel, dim=1)
        ang_error = torch.abs(cmd_vel[:, 2] - actual_ang_vel)
        inst_error = lin_error + ang_error * self.angular_scale

        # Accumulate (integrate over time)
        self._episode_cum_error += inst_error * self.step_dt

    def get_episode_cum_error(self, env_ids: Sequence[int] = None) -> torch.Tensor:
        if env_ids is None:
            return self._episode_cum_error.clone()
        return self._episode_cum_error[env_ids].clone()


# Velocity error based terrain curriculum function. 
def terrain_levels_velocityError(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "base_velocity",
    error_threshold_up: float = 0.4,
    error_threshold_down: float = 1.6
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
    # Access the velocity error recorder manager and acquire the data from specified envs.
    error_recorder: VelocityErrorRecorder = env.velocity_error_recorder
    cum_errors = error_recorder.get_episode_cum_error(env_ids)

    # Normalize error by maximum episode duration.
    norm_errors = cum_errors / env.max_episode_length_s

    # Difficulty updates based on episode duration normalized error.
    move_up = norm_errors < error_threshold_up
    move_down = norm_errors > error_threshold_down
    move_down *= ~move_up

    # Filter to only update terrain levels for envs at episode timeout.
    episode_lengths = env.episode_length_buf[env_ids]
    non_timeout = (episode_lengths - env.max_episode_length) != 0
    move_up[non_timeout] = False
    move_down[non_timeout] = False

    # Filter to only update terrain levels for moving commands.
    commands = env.command_manager.get_command(command_name)
    cmd_subset = commands[env_ids]
    cmd_norm = torch.norm(cmd_subset[:, :2], dim=1)
    is_moving_cmd = cmd_norm > 0.1
    move_up   *= is_moving_cmd
    move_down *= is_moving_cmd

    # Update terrain levels.
    terrain: TerrainImporter = env.scene.terrain
    terrain.update_env_origins(env_ids, move_up, move_down)

    # Debug prints.
    # if torch.any(move_up) or torch.any(move_down):
    #     print(f"[INFO]: Env IDs: \n{env_ids}")
    #     print(f"[INFO]: Move up: \n{norm_errors}")
    #     print(f"[INFO]: Move down: \n{norm_errors}")
    # print(f"[INFO]: Normalized errors: \n{norm_errors}")
    # print(f"[INFO]: Current mean terrain levels: \n{terrain.terrain_levels.float()}")

    # Return the mean terrain level.
    return torch.mean(terrain.terrain_levels.float())


# Terrain levels based action scale curriculum.
def action_scale_terrainLevels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    action_levels: int = 8
) -> torch.Tensor:
    """Curriculum based on the rounded mean terrain levels of all envs.

    This term is used to increase the action scale of the robots as the mean terrain levels progress to higher difficulties.

    .. note::
        It is only possible to use this term when there is a terrain levels curriculum in place.

    Returns:
        
    """
    terrain: TerrainImporter = env.scene.terrain
    mean_levels = torch.mean(terrain.terrain_levels.float())
    action_manager = env.action_manager

    stage = ((torch.floor(mean_levels)) * action_levels) // 20
    stage = torch.clamp(stage, min=0, max = action_levels-1)

    # print(f"stage: {stage}")
    # print(f"action_manager._terms.items(): {action_manager._terms.items()}")

    # import torch

    # action_levels = 8

    # # mean_levels = torch.arange(0, 20.5, 0.5)
    # mean_levels = torch.mean(torch.Tensor([10, 11]))

    # stage = ((torch.floor(mean_levels)) * action_levels) // 20
    # stage = torch.clamp(stage, min=0, max = action_levels-1)

    # print(f"stages: \n{stage}")

    for term_name, action_term in action_manager._terms.items():
        base_scale = action_term.cfg.scale

        current_scale = base_scale + base_scale * stage
        # print(f"action_term._scale: {action_term._scale}")
        action_term._scale = current_scale

        # print(f"current_scale: {current_scale}")

    # Placeholder return value.
    return current_scale
