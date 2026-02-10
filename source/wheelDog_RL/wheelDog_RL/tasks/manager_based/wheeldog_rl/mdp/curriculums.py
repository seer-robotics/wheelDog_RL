"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

# Library imports
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING, Dict, Any

from isaaclab.managers import SceneEntityCfg

# Local mdp module inherited from Isaac.
from wheelDog_RL.tasks.manager_based.wheeldog_rl import mdp


if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.terrains import TerrainImporter
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

        # EMA parameters - tune these
        self.ema_alpha = cfg.get("ema_alpha", 0.995)          # 0.99 = fast, 0.999 = very smooth
        self.min_steps_per_stage = cfg.get("min_steps_per_stage", 1_500_000)  # safety floor

        # Per-environment EMA of instantaneous RMSE (updated every step)
        self.ema_vx_rmse   = torch.zeros(self.num_envs, device=self.device)
        self.ema_omega_rmse = torch.zeros(self.num_envs, device=self.device)
        self.ema_vy_rmse   = torch.zeros(self.num_envs, device=self.device)

        # Global (batch-averaged) values used for curriculum decisions
        self.avg_vx_rmse   = 0.0
        self.avg_omega_rmse = 0.0
        self.avg_vy_rmse   = 0.0
        self.avg_success   = 0.0

        # Stage tracking
        self.current_stage = 0
        self.stage_start_step = 0
        self.total_steps = self.env.common_step_counter

        # Still keep accumulators for optional diagnostics / logging
        self.episode_vx_error_sq   = torch.zeros(self.num_envs, device=self.device)
        self.episode_omega_error_sq = torch.zeros(self.num_envs, device=self.device)
        self.episode_vy_error_sq   = torch.zeros(self.num_envs, device=self.device)

    def reset(self, env_ids: torch.Tensor):
        # Reset episode statistics buffers.
        if env_ids is not None:
            self.episode_vx_error_sq[env_ids]   = 0.0
            self.episode_omega_error_sq[env_ids] = 0.0
            self.episode_vy_error_sq[env_ids]   = 0.0

    def post_physics_step(self):
        # Get current commands and velocities
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

        # Accumulate for diagnostics (optional)
        self.episode_vx_error_sq   += (vx_cmd - vx_act) ** 2
        self.episode_omega_error_sq += (omega_cmd - omega_act) ** 2
        self.episode_vy_error_sq   += (vy_cmd - vy_act) ** 2

        # === EMA update every step ===
        curr_vx_rmse   = torch.sqrt((vx_cmd - vx_act) ** 2)
        curr_omega_rmse = torch.sqrt((omega_cmd - omega_act) ** 2)
        curr_vy_rmse   = torch.sqrt((vy_cmd - vy_act) ** 2)

        self.ema_vx_rmse   = self.ema_alpha * self.ema_vx_rmse   + (1 - self.ema_alpha) * curr_vx_rmse
        self.ema_omega_rmse = self.ema_alpha * self.ema_omega_rmse + (1 - self.ema_alpha) * curr_omega_rmse
        self.ema_vy_rmse   = self.ema_alpha * self.ema_vy_rmse   + (1 - self.ema_alpha) * curr_vy_rmse

        # Global averages for curriculum
        self.avg_vx_rmse   = self.ema_vx_rmse.mean().item()
        self.avg_omega_rmse = self.ema_omega_rmse.mean().item()
        self.avg_vy_rmse   = self.ema_vy_rmse.mean().item()

        # Success proxy (fraction of envs with decent tracking this step)
        good = (curr_vx_rmse < 0.25) & (curr_omega_rmse < 0.25)
        self.avg_success = good.float().mean().item()

    def should_advance_stage(self) -> bool:
        if self.total_steps - self.stage_start_step < self.min_steps_per_stage:
            return False

        if self.current_stage == 0:   # vx only
            return self.avg_vx_rmse < self.cfg.get("stage0_vx_threshold", 0.18)
        elif self.current_stage == 1: # + yaw
            retval = (self.avg_vx_rmse < self.cfg.get("stage1_vx_threshold", 0.15) and self.avg_omega_rmse < self.cfg.get("stage1_omega_threshold", 0.18))
            return retval
        return False

    def advance_stage(self):
        if self.current_stage >= 2:
            return
        self.current_stage += 1
        self.stage_start_step = self.total_steps
        print(f"[Curriculum] → Advanced to Stage {self.current_stage} at step {self.total_steps:,}")

        # Optional: speed up adaptation after stage change
        # self.ema_alpha = max(0.98, self.ema_alpha * 0.9)  # temporarily more responsive

    def get_current_command_ranges(self) -> Dict[str, Any]:
        """
        Returns current min/max and sampling params.
        Includes gradual within-stage ramping.
        """
        progress = min(1.0, (self.total_steps - self.stage_start_step) / self.cfg.get("stage_duration_steps", 5_000_000))

        if self.current_stage == 0:  # vx ramp
            vx_min = -0.3 * progress
            vx_max = (0.6 + 1.6 * progress)
            omega_min = omega_max = 0.0
            vy_min = vy_max = 0.0
            vy_prob = 0.0

        elif self.current_stage == 1:  # ωz ramp, vx full
            vx_min, vx_max = -1.0, 2.2
            omega_range = 0.3 + 1.4 * progress
            omega_min = -omega_range
            omega_max = omega_range
            vy_min = vy_max = 0.0
            vy_prob = 0.0

        else:  # stage 2: limited vy
            vx_min, vx_max = -1.0, 2.2
            omega_min, omega_max = -1.6, 1.6
            vy_range = 0.08 + 0.25 * progress   # keep "little"
            vy_min = -vy_range
            vy_max = vy_range
            vy_prob = 0.12 + 0.18 * progress    # low probability of sampling non-zero vy

        return {
            "vx": (vx_min, vx_max),
            "vy": (vy_min, vy_max),
            "omega": (omega_min, omega_max),
            "vy_sample_prob": vy_prob,
        }
    

# Command curriculum function callable.
def command_staged_curriculum(
    env,
    env_ids: torch.Tensor | None = None,  # sometimes passed, often ignored here
    **kwargs
):
    """
    Curriculum term callable executed by CurriculumManager.
    Updates the command sampler ranges based on current curriculum state.
    """
    if not hasattr(env, "command_curriculum_manager"):
        return

    mgr: CommandCurriculumManager = env.command_curriculum_manager

    # Optional: only update every N steps to avoid unnecessary writes
    if env.common_step_counter % 200 != 0:
        return

    # Get the current ranges from the manager
    ranges = mgr.get_current_command_ranges()

    # Apply them to your actual command term
    # Adjust the attribute names / path according to your exact CommandTerm
    try:
        cmd_term = env.command_manager.command_terms["base_velocity"]  # or "velocity_command", etc.

        cmd_term.cfg.ranges.lin_vel_x = [ranges["vx"][0], ranges["vx"][1]]
        cmd_term.cfg.ranges.lin_vel_y = [ranges["vy"][0], ranges["vy"][1]]
        cmd_term.cfg.ranges.ang_vel_z = [ranges["omega"][0], ranges["omega"][1]]

        # If your command term supports probabilistic lateral sampling
        if hasattr(cmd_term.cfg, "vy_sample_prob"):
            cmd_term.cfg.vy_sample_prob = ranges["vy_sample_prob"]
        elif hasattr(cmd_term.cfg, "lateral_sample_prob"):
            cmd_term.cfg.lateral_sample_prob = ranges["vy_sample_prob"]

        # Optional: force resample if you want immediate effect (careful with stability)
        # env.command_manager.resample(env_ids=None)

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
