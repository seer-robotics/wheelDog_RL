from __future__ import annotations

# Library imports.
import torch
import math
from typing import TYPE_CHECKING, Dict, Any
from collections.abc import Sequence
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.assets import Articulation

if TYPE_CHECKING:
    from wheelDog_RL.tasks.manager_based.wheeldog_rl.envEntry import WheelDog_BlindLocomotionEnv


# Manager class that handles the command curriculum.
class TiltDetectionManager:
    """
    Custom manager that detects and arbitrates whether a robot should be terminated.
    """
    def __init__(
        self,
        env: WheelDog_BlindLocomotionEnv,
        grace_steps: int = 12000,
        tilt_duration_seconds: float = 4.0,
        tilt_threshold_degrees: float = 90.0,
    ):
        self.env = env
        self.grace_steps = grace_steps
        self.tilt_threshold_steps = int(tilt_duration_seconds / self.env.step_dt)
        self.cos_threshold = math.cos(math.radians(tilt_threshold_degrees))

        # Per-environment state buffer.
        self.consecutive_bad_tilt = torch.zeros(
            env.num_envs, dtype=torch.int64, device=self.env.device
        )

    def reset(self, env_ids: Sequence[int]):
        """Called from env._reset_idx(env_ids)."""
        if env_ids is not None:
            self.consecutive_bad_tilt[env_ids] = 0

    def step(self):
        """Call this every environment step (after physics step, before termination computation)."""
        # Get current tilt condition.
        robot: Articulation = self.env.scene["robot"]
        projected_gravity = robot.data.projected_gravity_b  # (num_envs, 3)

        # Update consecutive counter (resets when condition is false).
        self.consecutive_bad_tilt = torch.where(
            self.is_bad_tilt(projected_gravity),
            self.consecutive_bad_tilt + 1,
            torch.zeros_like(self.consecutive_bad_tilt, device=self.env.device),
        )

    def is_bad_tilt(self, projected_gravity):
        return -projected_gravity[:, 2] < self.cos_threshold

    def is_grace_period_over(self) -> bool:
        """Global grace period based on common_step_counter."""
        return self.env.common_step_counter >= self.grace_steps

def terminate_fallen(
    env: WheelDog_BlindLocomotionEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float,
    term_name: str = "fallen",
) -> torch.Tensor:
    """Terminate when the contact force on the sensor exceeds the force threshold."""
    # Enable type-hints.
    manager: TiltDetectionManager = env.tilt_detection_manager
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # Check for base contact.
    net_contact_forces = contact_sensor.data.net_forces_w_history
    has_illegal_contact = torch.any(
        torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold, dim=1
    )

    if manager.is_grace_period_over():
        # After grace: immediate termination on base contact or any bad tilt.
        # Check for bad tilt.
        robot: Articulation = env.scene["robot"]
        projected_gravity = robot.data.projected_gravity_b
        is_bad_tilt_now = manager.is_bad_tilt(projected_gravity)
        markedForTermination = has_illegal_contact | is_bad_tilt_now

        # fall_termination_cfg = env.termination_manager.get_term_cfg(term_name)
        # fall_termination_cfg.time_out = False
        # env.termination_manager.set_term_cfg(term_name=term_name, cfg=fall_termination_cfg)
    else:
        # During grace: only terminate on prolonged bad tilt (4 seconds)
        markedForTermination = manager.consecutive_bad_tilt >= manager.tilt_threshold_steps

        # fall_termination_cfg = env.termination_manager.get_term_cfg(term_name)
        # fall_termination_cfg.time_out = True
        # env.termination_manager.set_term_cfg(term_name=term_name, cfg=fall_termination_cfg)

    return markedForTermination
