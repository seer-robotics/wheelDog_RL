# Library imports.
import torch
import copy
from isaaclab.envs import ManagerBasedRLEnv
from collections.abc import Sequence
from isaaclab.envs.common import VecEnvStepReturn

# Import custom manager.
from wheelDog_RL.tasks.manager_based.wheeldog_rl.mdp import VelocityErrorRecorder, CommandCurriculumManager
from wheelDog_RL.tasks.manager_based.wheeldog_rl import watchDogs

# Import settings.
from wheelDog_RL.tasks.manager_based.wheeldog_rl.settings import ANGULAR_ERROR_SCALE


class WheelDog_BlindLocomotionEnv(ManagerBasedRLEnv):
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
        # Keep a copy of the env startup MDP configurations.
        self.env_cfg_at_startup = copy.deepcopy(self.cfg)

        self.velocity_error_recorder = VelocityErrorRecorder(
            config={"angular_scale": ANGULAR_ERROR_SCALE},
            env=self
        )
        print("[INFO]: Added velocity_error_recorder manager.")

        self.command_curriculum_manager = CommandCurriculumManager(
            
        )

    def step(self, actions: torch.Tensor) -> VecEnvStepReturn:
        # Compute step returns.
        obs, rew, terminated, truncated, info = super().step(actions)

        # Iterate custom curriculum recorder manager.
        self.velocity_error_recorder.post_physics_step()

        # Observations numerical corruption detection.
        watchDogs.check_observations_for_nans_infs(obs, self.common_step_counter, self.num_envs)

        # Rewards numerical corruption detection.
        rewClipped = watchDogs.check_rewards_for_nan_infs(rew, self.common_step_counter, info)

        return obs, rewClipped, terminated, truncated, info
    
    def _reset_idx(self, env_ids: Sequence[int]):
        super()._reset_idx(env_ids)
        self.velocity_error_recorder.reset(env_ids)
