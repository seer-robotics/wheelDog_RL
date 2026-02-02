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
        # Compute step returns.
        obs, rew, terminated, truncated, info = super().step(actions)

        # Iterate custom curriculum recorder manager.
        self.velocity_error_recorder.post_physics_step()

        # Observations numerical corruption detection.
        check_observations_for_nans_infs(obs, self.common_step_counter, self.num_envs)

        # Rewards numerical corruption detection.
        if not torch.isfinite(rew).all():
            bad_mask = ~torch.isfinite(rew).all(dim=-1)
            bad_env_ids = torch.nonzero(bad_mask).squeeze(-1)
            print(f"[CRITICAL]: Non-finite values in rewards at step {self.common_step_counter}")
            print(f"Number of bad envs: {len(bad_env_ids)} / {self.num_envs}")
            if len(bad_env_ids) > 0:
                # Inspect first few bad envs
                for eid in bad_env_ids[:3]:
                    print(f"Env {eid.item()}: min={rew[eid].min():.4f}, max={rew[eid].max():.4f}")
                    print(f"  obs: {rew[eid]}")

        return obs, rew, terminated, truncated, info
    
    def _reset_idx(self, env_ids: Sequence[int]):
        super()._reset_idx(env_ids)
        self.velocity_error_recorder.reset(env_ids)


def check_observations_for_nans_infs(obs, step_counter: int, num_envs: int) -> None:
    """
    Recursively check all tensor leaves in the observations dict for non-finite values.
    Prints detailed information when corruption is detected.
    """
    bad_envs_per_key = {}

    def recurse(key_path: str, tensor: torch.Tensor):
        if not torch.isfinite(tensor).all():
            # Per-environment mask: true if the whole row is finite → we want the inverse
            row_finite = torch.isfinite(tensor).all(dim=-1)     # shape (num_envs,)
            bad_mask     = ~row_finite                           # true = at least one NaN/Inf
            bad_env_ids  = torch.nonzero(bad_mask).squeeze(-1)

            if len(bad_env_ids) > 0:
                bad_envs_per_key[key_path] = (bad_env_ids, tensor)

    # Traverse the observation dict recursively
    def traverse(d, prefix: str = ""):
        for k, v in d.items():
            current_key = f"{prefix}{k}" if prefix == "" else f"{prefix}.{k}"
            if isinstance(v, torch.Tensor):
                recurse(current_key, v)
            elif isinstance(v, dict):
                traverse(v, prefix=current_key + ".")
            # You can add elif isinstance(v, (list, tuple)) if needed

    traverse(obs)

    # ── Reporting ────────────────────────────────────────────────────────────────
    if bad_envs_per_key:
        total_unique_bad_envs = set()
        for bad_ids, _ in bad_envs_per_key.values():
            total_unique_bad_envs.update(bad_ids.tolist())

        print(f"[CRITICAL] Non-finite values detected in observations at step {step_counter}")
        print(f"Total unique bad environments: {len(total_unique_bad_envs)} / {num_envs}")
        print(f"Affected observation keys: {len(bad_envs_per_key)}")

        # Show worst offenders first (most environments affected)
        sorted_items = sorted(
            bad_envs_per_key.items(),
            key=lambda x: len(x[1][0]), reverse=True
        )

        for obs_key, (bad_ids, tensor) in sorted_items[:4]:  # limit to 4 most severe keys
            print(f"\n  → Key: {obs_key}   ({len(bad_ids)} bad envs)")
            # Show first few problematic environments
            for eid in bad_ids[:3]:
                row = tensor[eid]
                print(f"    env {eid.item():3d} | min={row.min(): .4e}  max={row.max(): .4e}  "
                      f"has_nan={torch.isnan(row).any()}  has_inf={torch.isinf(row).any()}")
                # Optional: print full vector only when very few envs are bad
                if len(bad_ids) <= 5:
                    print(f"         obs = {row.tolist()}")
                else:
                    print(f"         (vector too long – {row.shape[-1]} dims)")


if __name__ == "__main__":
    torch.manual_seed(42)
    device = torch.device("cpu")          # change to "cuda" if desired
    num_envs = 32
    obs_dim = 17                              # typical small observation vector size

    print("=== Test 1: Clean observations (should report OK) ===")
    clean_obs = {
        "proprio": torch.randn(num_envs, obs_dim, device=device),
        "commands": torch.randn(num_envs, 12, device=device),
        "privileged": {
            "terrain": torch.randn(num_envs, 88, device=device),
            "dynamics": torch.randn(num_envs, 6, device=device),
        }
    }
    check_observations_for_nans_infs(clean_obs, step_counter=100, num_envs=num_envs)
    print("\n")

    print("=== Test 2: NaN in proprio of envs 3,7,8 ===")
    dirty_obs = {
        "proprio": torch.randn(num_envs, obs_dim, device=device),
        "commands": torch.randn(num_envs, 12, device=device),
        "privileged": {
            "terrain": torch.randn(num_envs, 88, device=device),
            "dynamics": torch.randn(num_envs, 6, device=device),
        }
    }
    dirty_obs["proprio"][[3, 7, 8], 4] = float("nan")
    check_observations_for_nans_infs(dirty_obs, step_counter=101, num_envs=num_envs)
    print("\n")

    print("=== Test 3: Inf in privileged.terrain of envs 0,1,30 + NaN in commands ===")
    dirty_obs_2 = {
        "proprio": torch.randn(num_envs, obs_dim, device=device),
        "commands": torch.randn(num_envs, 12, device=device),
        "privileged": {
            "terrain": torch.randn(num_envs, 88, device=device),
            "dynamics": torch.randn(num_envs, 6, device=device),
        }
    }
    dirty_obs_2["commands"][[5, 14], :] = float("inf")
    dirty_obs_2["privileged"]["terrain"][[0, 1, 30], 42] = float("nan")
    check_observations_for_nans_infs(dirty_obs_2, step_counter=102, num_envs=num_envs)
    print("\n")

    print("=== Test 4: Very few envs affected, should print full vectors ===")
    small_dirty = {
        "joint_pos": torch.randn(8, 12, device=device),   # only 8 envs
        "imu": torch.randn(8, 6, device=device),
    }
    small_dirty["imu"][[2, 6], [1, 4]] = float("nan")
    small_dirty["joint_pos"][0, :] = float("inf")
    check_observations_for_nans_infs(small_dirty, step_counter=200, num_envs=8)
    print("\n")

    print("=== All tests completed. ===")