# Library imports.
import torch
from isaaclab.envs import ManagerBasedRLEnv
from collections.abc import Sequence
from isaaclab.envs.common import VecEnvStepReturn

# Import custom manager.
from wheelDog_RL.tasks.manager_based.wheeldog_rl.customCurriculum import VelocityErrorRecorder


class WheelDog_BlindLocomotionEnv(ManagerBasedRLEnv):
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
        # Initialize per-environment cumulative error tensor (shape: num_envs)
        # self._cumulative_vel_error = torch.zeros(self.num_envs, device=self.device)
        self.velocity_error_recorder = VelocityErrorRecorder(
            config={"angular_scale": 1.0},
            env=self
        )
        print("[INFO]: Added velocity_error_recorder manager.")

    def step(self, actions: torch.Tensor) -> VecEnvStepReturn:
        obs, rew, terminated, truncated, info = super().step(actions)
        self.velocity_error_recorder.post_physics_step()
        return obs, rew, terminated, truncated, info
    
    def _reset_idx(self, env_ids: Sequence[int]):
        super()._reset_idx(env_ids)
        self.velocity_error_recorder.reset(env_ids)
