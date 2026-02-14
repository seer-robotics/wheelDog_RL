# Library imports.
import torch
import copy
from isaaclab.envs import ManagerBasedRLEnv
from collections.abc import Sequence
from isaaclab.envs.common import VecEnvStepReturn

# Import custom manager.
from wheelDog_RL.tasks.manager_based.wheeldog_rl.mdp import VelocityErrorRecorder, CommandCurriculumManager, TiltDetectionManager
from wheelDog_RL.tasks.manager_based.wheeldog_rl import watchDogs

# Import settings.
from wheelDog_RL.tasks.manager_based.wheeldog_rl.settings import \
    ANGULAR_ERROR_SCALE, \
    CMD_CURRICULUM_RANGE_PROGRESS_SCALES, \
    CMD_CURRICULUM_TARGET_MAX_RANGES, \
    FALL_GRACE_STEPS, \
    FALL_TILT_DEGREES, \
    FALL_TILT_DURATION

# # Import settings. 
# from wheelDog_RL.tasks.manager_based.wheeldog_rl.settings import \
#     CMD_CURRICULUM_INIT_MIN_RANGES


class WheelDog_BlindLocomotionEnv(ManagerBasedRLEnv):
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
        # Keep a copy of the env startup MDP configurations.
        self.env_cfg_at_startup = copy.deepcopy(self.cfg)

        # Initialize custom managers.
        self.tilt_detection_manager = TiltDetectionManager(
            env=self,
            grace_steps=FALL_GRACE_STEPS,
            tilt_threshold_degrees=FALL_TILT_DEGREES,
            tilt_duration_seconds=FALL_TILT_DURATION,
        )
        print("[INFO]: Added tilt_detection_manager manager.")
        self.velocity_error_recorder = VelocityErrorRecorder(
            config={"angular_scale": ANGULAR_ERROR_SCALE},
            env=self,
        )
        print("[INFO]: Added velocity_error_recorder manager.")
        # self.command_curriculum_manager = CommandCurriculumManager(
        #     env=self,
        #     cfg={
        #         "range_progress_scales": CMD_CURRICULUM_RANGE_PROGRESS_SCALES,
        #         "target_max_ranges": CMD_CURRICULUM_TARGET_MAX_RANGES,
        #     },
        # )
        # print("[INFO]: Added command_curriculum_manager manager.")

    def step(self, actions: torch.Tensor) -> VecEnvStepReturn:
        # Compute step returns.
        # Add pre-processing before this step if necessary.
        obs, rew, terminated, truncated, info = super().step(actions)

        # Iterate custom managers.
        self.tilt_detection_manager.step()
        self.velocity_error_recorder.step()
        # self.command_curriculum_manager.step()

        # Observations' numerical corruption detection.
        watchDogs.check_observations_for_nans_infs(obs, self.common_step_counter, self.num_envs)

        # Rewards' numerical corruption detection and clipping.
        rewClipped = watchDogs.check_rewards_for_nan_infs(rew, self.common_step_counter, info)

        return obs, rewClipped, terminated, truncated, info
    
    def _reset_idx(self, env_ids: Sequence[int]):
        # Reset specified environments.
        super()._reset_idx(env_ids)

        # Reset custom managers.
        self.tilt_detection_manager.reset(env_ids)
        self.velocity_error_recorder.reset(env_ids)
        # self.command_curriculum_manager.reset(env_ids)


class CrippleDog_BlindLocomotionEnv(ManagerBasedRLEnv):
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
        # Keep a copy of the env startup MDP configurations.
        self.env_cfg_at_startup = copy.deepcopy(self.cfg)

        # Initialize custom managers.
        self.tilt_detection_manager = TiltDetectionManager(
            env=self,
            grace_steps=FALL_GRACE_STEPS,
            tilt_threshold_degrees=FALL_TILT_DEGREES,
            tilt_duration_seconds=FALL_TILT_DURATION,
        )
        print("[INFO]: Added tilt_detection_manager manager.")
        # self.velocity_error_recorder = VelocityErrorRecorder(
        #     config={"angular_scale": ANGULAR_ERROR_SCALE},
        #     env=self,
        # )
        # print("[INFO]: Added velocity_error_recorder manager.")
        # self.command_curriculum_manager = CommandCurriculumManager(
        #     env=self,
        #     cfg={
        #         "range_progress_scales": CMD_CURRICULUM_RANGE_PROGRESS_SCALES,
        #         "target_max_ranges": CMD_CURRICULUM_TARGET_MAX_RANGES,
        #     },
        # )
        # print("[INFO]: Added command_curriculum_manager manager.")

    def step(self, actions: torch.Tensor) -> VecEnvStepReturn:
        # Compute step returns.
        # Add pre-processing before this step if necessary.
        obs, rew, terminated, truncated, info = super().step(actions)

        # Iterate custom managers.
        self.tilt_detection_manager.step()
        # self.velocity_error_recorder.step()
        # self.command_curriculum_manager.step()

        # Observations' numerical corruption detection.
        watchDogs.check_observations_for_nans_infs(obs, self.common_step_counter, self.num_envs)

        # Rewards' numerical corruption detection and clipping.
        rewClipped = watchDogs.check_rewards_for_nan_infs(rew, self.common_step_counter, info)

        return obs, rewClipped, terminated, truncated, info
    
    def _reset_idx(self, env_ids: Sequence[int]):
        # Reset specified environments.
        super()._reset_idx(env_ids)

        # Reset custom managers.
        self.tilt_detection_manager.reset(env_ids)
        # self.velocity_error_recorder.reset(env_ids)
        # self.command_curriculum_manager.reset(env_ids)
