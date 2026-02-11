"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

# Library imports
import torch
import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Dict, Any

from isaaclab.managers import SceneEntityCfg

# Local mdp module inherited from Isaac.
from wheelDog_RL.tasks.manager_based.wheeldog_rl import mdp


if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.terrains import TerrainImporter
    from isaaclab.envs import mdp as isaac_mdp
    from wheelDog_RL.tasks.manager_based.wheeldog_rl.envEntry import WheelDog_BlindLocomotionEnv

# Manager class that handles the command curriculum.
class CommandCurriculumManager:
    """
    Custom manager for the staged command curriculum (vx → ωz → restricted vy).
    Handles persistent state and performance tracking.
    """

    def __init__(self,
        env: WheelDog_BlindLocomotionEnv,
        cfg: Dict[str, Any],
    ):
        self.env = env
        self.cfg = cfg
        self.num_envs = env.num_envs
        self.device = env.device

        # EMA parameters - tune these.
        # The closer alpha is to 1, the longer the memory is.
        self.ema_alpha = cfg.get("ema_alpha", 0.99)
        self.min_steps_per_stage = cfg.get("min_steps_per_stage", 1_500_000)

        # Per-environment EMA of instantaneous mae (updated every step)
        self.ema_vx_mae   = torch.zeros(self.num_envs, device=self.device)
        self.ema_omega_mae = torch.zeros(self.num_envs, device=self.device)
        self.ema_vy_mae   = torch.zeros(self.num_envs, device=self.device)

        # Global (batch-averaged) values used for curriculum decisions.
        self.avg_vx_mae   = 0.0
        self.avg_omega_mae = 0.0
        self.avg_vy_mae   = 0.0

        # Stage tracking.
        self.current_stage = 0
        self.stage_start_step = 0
        self.total_steps = 0

    def reset(self, env_ids: torch.Tensor):
        # Reset episode statistics buffers for specified environments.
        if env_ids is not None:
            self.ema_vx_mae[env_ids]   = 0.0
            self.ema_vy_mae[env_ids] = 0.0
            self.ema_omega_mae[env_ids]   = 0.0

    def step(self):
        """
        Update EMA tracking errors and global averages.
        Called after every physics step.
        """
        self.total_steps = self.env.common_step_counter

        # Get current commands and velocities.
        robot: Articulation = self.env.scene["robot"]
        commands = self.env.command_manager.get_command("base_velocity")
        base_lin_vel = robot.data.root_lin_vel_b[:, :2]
        base_ang_vel = robot.data.root_ang_vel_b[:, 2]
        vx_cmd = commands[:, 0]
        vy_cmd = commands[:, 1]
        omega_cmd = commands[:, 2]
        vx_act = base_lin_vel[:, 0]
        vy_act = base_lin_vel[:, 1]
        omega_act = base_ang_vel[:, 2]

        # Instantaneous mean absolute errors.
        curr_vx_mae = torch.abs(vx_cmd - vx_act)
        curr_vy_mae = torch.abs(vy_cmd - vy_act)
        curr_omega_mae = torch.abs(omega_cmd - omega_act)

        # Per environment EMA update.
        self.ema_vx_mae = self.ema_alpha * self.ema_vx_mae + (1 - self.ema_alpha) * curr_vx_mae
        self.ema_omega_mae = self.ema_alpha * self.ema_omega_mae + (1 - self.ema_alpha) * curr_omega_mae
        self.ema_vy_mae = self.ema_alpha * self.ema_vy_mae + (1 - self.ema_alpha) * curr_vy_mae

        # Global averages.
        self.avg_vx_mae   = self.ema_vx_mae.mean().item()
        self.avg_omega_mae = self.ema_omega_mae.mean().item()
        self.avg_vy_mae   = self.ema_vy_mae.mean().item()

    def should_advance_stage(self) -> bool:
        if self.total_steps - self.stage_start_step < self.min_steps_per_stage:
            return False

        if self.current_stage == 0:   # vx only
            return self.avg_vx_mae < self.cfg.get("stage0_vx_threshold", 0.18)
        elif self.current_stage == 1: # + yaw
            retval = (
                self.avg_vx_mae < self.cfg.get("stage1_vx_threshold", 0.15) and
                self.avg_omega_mae < self.cfg.get("stage1_omega_threshold", 0.18)
            )
            return retval
        elif self.current_stage == 2:
            return self.avg_vy_mae < self.cfg.get("stage2_vy_threshold", 0.30)
        
        return False

    def advance_stage(self):
        if self.current_stage >= 2:
            return
        self.current_stage += 1
        self.stage_start_step = self.total_steps
        print(f"[Curriculum] → Advanced to Stage {self.current_stage} at step {self.total_steps:,}")

    def get_current_command_ranges(self) -> Dict[str, Any]:
        """
        Returns current min/max ranges for velocity commands.
        
        Behavior:
        - During the first min_steps_per_stage of a stage → returns the narrowest (initial) range
        - After min_steps_per_stage → linearly interpolates toward the maximum range
        based on current average tracking error (progress = 1 - mae / target)
        """
        steps_in_stage = self.total_steps - self.stage_start_step
        force_min_range = steps_in_stage < self.min_steps_per_stage

        if self.current_stage == 0:
            # Use vx error to drive vx range expansion
            target_vx_mae = self.cfg.get("stage0_vx_threshold", 0.18)

            if force_min_range:
                vx_min = -0.3
                vx_max = 0.4
            else:
                progress = max(0.0, min(1.0,
                    1.0 - (self.avg_vx_mae / target_vx_mae)
                ))
                vx_min = -0.3 + (-0.7 * progress)
                vx_max =  0.4 + ( 1.6 * progress)

            vy_min = vy_max = 0.0
            omega_min = omega_max = 0.0

        elif self.current_stage == 1:
            target_omega_mae = self.cfg.get("stage1_omega_threshold", 0.18)
            
            if force_min_range:
                omega_range = 0.4
            else:
                progress = max(0.0, min(1.0,
                    1.0 - (self.avg_omega_mae / target_omega_mae)
                ))
                omega_range = 0.4 + 1.2 * progress

            omega_min = -omega_range
            omega_max = omega_range
            vx_min, vx_max = -1.0, 2.0
            vy_min = vy_max = 0.0

        else:  # stage 2 — vy uses vy_mae or combined metric
            target_vy_mae = self.cfg.get("stage2_vy_threshold", 0.30)
            
            if force_min_range:
                vy_range = 0.08
            else:
                progress = max(0.0, min(1.0,
                    1.0 - (self.avg_vy_mae / target_vy_mae)
                ))
                vy_range = 0.08 + 0.22 * progress

            vx_min, vx_max = -1.0, 2.0
            omega_min, omega_max = -1.6, 1.6
            vy_min = -vy_range
            vy_max =  vy_range

        return {
            "vx": (vx_min, vx_max),
            "vy": (vy_min, vy_max),
            "omega": (omega_min, omega_max),
        }


# Command curriculum function callable.
def command_staged_curriculum(
    env: WheelDog_BlindLocomotionEnv,
    env_ids: torch.Tensor | None = None,
    **kwargs
) -> torch.Tensor:
    """
    Curriculum term callable executed by CurriculumManager.
    Updates the command sampler ranges based on current curriculum state.
    """
    # if not hasattr(env, "command_curriculum_manager"):
    #     return

    # Check for stage updates.
    mgr: CommandCurriculumManager = env.command_curriculum_manager
    if mgr.should_advance_stage():
        mgr.advance_stage()

    # Get the current ranges from the manager.
    ranges = mgr.get_current_command_ranges()

    # Apply command curriculum.
    try:
        cmd_term = env.command_manager.get_term("base_velocity")
        cmd_term.cfg.ranges.lin_vel_x = ranges["vx"]
        cmd_term.cfg.ranges.lin_vel_y = ranges["vy"]
        cmd_term.cfg.ranges.ang_vel_z = ranges["omega"]
        # cmd_term.cfg.ranges=isaac_mdp.UniformVelocityCommandCfg.Ranges(
        #     lin_vel_x=ranges["vx"],
        #     lin_vel_y=ranges["vy"],
        #     ang_vel_z=ranges["omega"],
        # ),

    except (AttributeError, KeyError) as e:
        print(f"Warning: Could not update command ranges: {e}")


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

    def step(self):
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
        
        # print(f"[INFO]: self._episode_cum_error: \n{self._episode_cum_error}")
        # print(f"[INFO]: self._episode_cum_command: \n{self._episode_cum_command}")

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

    ##
    # Do something here to cooperate with the command curriculum.
    """
    Such as only begin to increment terrain levels after a certain stage in the command curriculum.
    """
    ##

    # Update terrain levels.
    terrain: TerrainImporter = env.scene.terrain
    terrain.update_env_origins(env_ids, move_up, move_down)

    # Debug prints.
    # if torch.any(move_up) or torch.any(move_down):
    #     print(f"[INFO]: Env IDs: \n{env_ids}")
    #     print(f"[INFO]: Move up: \n{norm_errors}")
    #     print(f"[INFO]: Move down: \n{norm_errors}")
    # print(f"[INFO]: norm_errors: \n{norm_errors}")
    # print(f"[INFO]: cum_command: \n{cum_command}")
    # print(f"[INFO]: Current mean terrain levels: \n{terrain.terrain_levels.float()}")

    # Return the mean terrain level.
    return torch.mean(terrain.terrain_levels.float())


# Terrain levels based penalty weight curriculum.
def penalty_levels_meanTerrain(
    env: WheelDog_BlindLocomotionEnv,
    env_ids: Sequence[int],
    target_term_name: str,
    scale_levels: int,
    min_factor: float,
    min_factor_terrainLevel: int,
) -> torch.Tensor:
    """
    Curriculum function that scales down weight of specified penalty as robot learns to stay alive.
    
    :param target_term_name: Name for the penalty term whose weight will be modified.
    :type target_term_name: str
    :param scale_levels: Number of stages of the scaling.
    :type scale_levels: int
    :param min_factor: Minimum factor that the original weight will be multiplied by.
    :type min_factor: float
    :return: The weight of the specified penalty.
    :rtype: Tensor
    """
    terrain: TerrainImporter = env.scene.terrain
    mean_levels = torch.mean(terrain.terrain_levels.float())
    original_term_cfg = getattr(env.env_cfg_at_startup.rewards, target_term_name)
    original_weight = original_term_cfg.weight

    stage = (torch.clamp(torch.floor(mean_levels), min=0, max=min_factor_terrainLevel) * scale_levels) // min_factor_terrainLevel
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


def base_contact_threshold_decay(
        env: WheelDog_BlindLocomotionEnv,
        env_ids,
        old_value,
        **kwargs
) -> float:
    """
    Curriculum schedule:
      1. Flat high threshold for the first 'flat_steps'
      2. Linear decay from initial_threshold → target_threshold over the next 'decay_steps'
      3. Fixed at target_threshold thereafter
    """
    total_warmup_steps = kwargs["flat_steps"] + kwargs["decay_steps"]
    current_step = env.common_step_counter

    # Phase 1: keep threshold fixed at initial value
    if current_step < kwargs["flat_steps"]:
        return kwargs["initial_threshold"]

    # Phase 2: linear interpolation during decay phase
    if current_step < total_warmup_steps:
        progress = (current_step - kwargs["flat_steps"]) / float(kwargs["decay_steps"])
        progress = min(1.0, progress)  # safeguard
        return (
            kwargs["initial_threshold"] + progress * (kwargs["target_threshold"] - kwargs["initial_threshold"])
        )

    # Phase 3: stay at final target value
    return kwargs["target_threshold"]


def command_curriculum(
    env: WheelDog_BlindLocomotionEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "base_velocity",
) -> torch.Tensor:
    """
    Docstring for command_curriculum
    
    :param env: Description
    :type env: WheelDog_BlindLocomotionEnv
    :param env_ids: Description
    :type env_ids: Sequence[int]
    :param asset_cfg: Description
    :type asset_cfg: SceneEntityCfg
    :param command_name: Description
    :type command_name: str
    :return: Description
    :rtype: Tensor
    """
