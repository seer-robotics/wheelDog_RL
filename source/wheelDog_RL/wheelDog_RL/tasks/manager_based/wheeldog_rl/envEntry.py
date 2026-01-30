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

        # Insert after self.obs_buf is fully computed
        obs_buf = self.obs_buf["critic"]
        if not torch.isfinite(obs_buf).all():
            bad_mask = ~torch.isfinite(obs_buf).any(dim=-1)  # per-env
            bad_env_ids = torch.nonzero(bad_mask).squeeze(-1)
            print(f"[CRITICAL] Non-finite values in obs_buf at step {self.common_step_counter}")
            print(f"Number of bad envs: {len(bad_env_ids)} / {self.num_envs}")
            if len(bad_env_ids) > 0:
                # Inspect first few bad envs
                for eid in bad_env_ids[:3]:
                    print(f"Env {eid.item()}: min={obs_buf[eid].min():.4f}, max={obs_buf[eid].max():.4f}")
                    print(f"  obs: {obs_buf[eid]}")
            # Temporary mitigation to continue running (debug only):
            # self.obs_buf.nan_to_num_(0.0)

        return obs, rew, terminated, truncated, info
    
    def _reset_idx(self, env_ids: Sequence[int]):
        super()._reset_idx(env_ids)
        self.velocity_error_recorder.reset(env_ids)
