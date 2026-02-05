"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

# Local mdp module inherited from Isaac.
from wheelDog_RL.tasks.manager_based.wheeldog_rl import mdp


if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.terrains import TerrainImporter
    from wheelDog_RL.tasks.manager_based.wheeldog_rl.envEntry import WheelDog_BlindLocomotionEnv


# Manager class that records cumulative velocity error.
class VelocityErrorRecorder():
    def __init__(self, config: dict, env: WheelDog_BlindLocomotionEnv):
        self._env = env
        self._num_envs = env.num_envs
        self.device = env.device

        # Integration time step.
        # Remember that step_dt is sim.dt*decimation
        self.step_dt = env.step_dt

        # Angular velocity error scaling factor.
        self.angular_scale: dict[str, float] = config.get("angular_scale", 1.0)

        # Initialize cumulative buffers. 
        self._episode_cum_error = torch.zeros(self._num_envs, device=self.device)
        self._episode_cum_command = torch.zeros(self._num_envs, device=self.device)

    def reset(self, env_ids: Sequence[int]):
        # Clear buffers on env reset.
        # print(f"[INFO]: Recorder manager resetting cumulative error: {self._episode_cum_error}")
        self._episode_cum_error[env_ids] = 0.0
        self._episode_cum_command[env_ids] = 0.0

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

        # Integrate error.
        self._episode_cum_error += inst_error * self.step_dt

        # Integrate command.
        self._episode_cum_command += \
            torch.norm(cmd_vel[:, :2], dim=1) * self.step_dt + \
            torch.abs(cmd_vel[:, 2]) * self.angular_scale * self.step_dt

    def get_episode_cum_error(self, env_ids: Sequence[int] = None) -> torch.Tensor:
        """
        Acquire the per-episode error from command value integrated over time.
        
        :return: Time-integrated error from command of all the environments.
        :rtype: Tensor
        """
        if env_ids is None:
            return self._episode_cum_error.clone()
        return self._episode_cum_error[env_ids].clone()
    
    def get_episode_cum_command(self, env_ids: Sequence[int] = None) -> torch.Tensor:
        """
        Acquire the per-episode command value integrated over time.
        
        :return: Time-integrated command values of all the environments.
        :rtype: Tensor
        """
        if env_ids is None:
            return self._episode_cum_command.clone()
        return self._episode_cum_command[env_ids].clone()


# Velocity error based terrain curriculum function. 
def terrain_levels_velocityError(
    env: WheelDog_BlindLocomotionEnv,
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

    :return: The current mean terrain levels.
    :rtype: Tensor
    """
    # Access the velocity error recorder manager and acquire the data from specified envs.
    error_recorder: VelocityErrorRecorder = env.velocity_error_recorder
    cum_errors = error_recorder.get_episode_cum_error(env_ids)
    cum_command = error_recorder.get_episode_cum_command(env_ids)

    # Normalize error by maximum episode duration.
    # Also prevent divide by zero for standing still environments (they won't be considered in the final output anyway).
    norm_errors = cum_errors / torch.where(cum_command.abs() < 1e-6, 1e-2, cum_command)

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
    # Essentially envs where cum_command is greater than a threshold.
    is_moving_cmd = cum_command > 1e-1
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


# Terrain levels based joint position deviation penalty curriculum.
def joint_deviation_penalty_levels(
    env: WheelDog_BlindLocomotionEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_term_name: str = "dof_pos_deviate",
    scale_levels: int = 8,
    min_factor: float = 0.5,
) -> torch.Tensor:
    """
    Curriculum function that scales down joint deviation penalty as robot learns to stay alive.
    
    :param target_term_name: Name for the penalty term whose weight will be modified.
    :type target_term_name: str
    :param scale_levels: Number of stages of the scaling.
    :type scale_levels: int
    :param min_factor: Minimum factor that the original weight will be multiplied by.
    :type min_factor: float
    :return: The weight of the penalty of joint deviations from default position (defined in ArticulationCfg).
    :rtype: Tensor
    """
    terrain: TerrainImporter = env.scene.terrain
    mean_levels = torch.mean(terrain.terrain_levels.float())
    original_term_cfg = getattr(env.env_cfg_at_startup.rewards, target_term_name)
    original_weight = original_term_cfg.weight

    stage = ((torch.floor(mean_levels)) * scale_levels) // 20
    stage = torch.clamp(stage, min=0, max = scale_levels-1)

    factor = 1.0 + (min_factor - 1.0) * (stage/(scale_levels-1))

    current_weight: torch.Tensor = original_weight * factor

    # Modify the penalty term's weight.
    term_cfg_new = env.reward_manager.get_term_cfg(term_name=target_term_name)
    term_cfg_new.weight = current_weight.item()
    env.reward_manager.set_term_cfg(term_name=target_term_name, cfg=term_cfg_new)

    
    # print(f"[INFO] current_weight: {current_weight}")
    # print(f"[INFO] term_cfg_new.weight: {term_cfg_new.weight}")

    # Return the current weight for logging.
    return current_weight


# # Terrain levels based action scale curriculum.
# def action_scale_terrainLevels(
#     env: ManagerBasedRLEnv,
#     env_ids: Sequence[int],
#     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
#     action_levels: int = 8,
#     max_scale: float = 2.0,
# ) -> torch.Tensor:
#     """Curriculum based on the rounded mean terrain levels of all envs.

#     This term is used to increase the action scale of the robots as the mean terrain levels progress to higher difficulties.

#     .. note::
#         It is only possible to use this term when there is a terrain levels curriculum in place.

#     :return: 
#     :rtype: Tensor
        
#     """
#     terrain: TerrainImporter = env.scene.terrain
#     mean_levels = torch.mean(terrain.terrain_levels.float())
#     action_manager = env.action_manager

#     stage = ((torch.floor(mean_levels)) * action_levels) // 20
#     stage = torch.clamp(stage, min=0, max = action_levels-1)

#     factor = 1.0 + (max_scale - 1.0) * (stage/(action_levels-1))

#     # print(f"stage: {stage}")
#     # print(f"action_manager._terms.items(): {action_manager._terms.items()}")

#     # import torch

#     # action_levels = 8

#     # # mean_levels = torch.arange(0, 20.5, 0.5)
#     # mean_levels = torch.mean(torch.Tensor([10, 11]))

#     # stage = ((torch.floor(mean_levels)) * action_levels) // 20
#     # stage = torch.clamp(stage, min=0, max = action_levels-1)

#     # print(f"stages: \n{stage}")

#     for term_name, action_term in action_manager._terms.items():
#         base_scale = action_term.cfg.scale

#         current_scale = base_scale + base_scale * factor
#         # print(f"action_term._scale: {action_term._scale}")
#         action_term._scale = current_scale

#         # print(f"current_scale: {current_scale}")

#     # Placeholder return value.
#     return stage
