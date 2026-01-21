"""Custom functions that can be used to define observations for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.ObservationTermCfg` object to
specify the observation function and its parameters.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def net_contact_force(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Extract the total contact forces on specified bodies."""
    # Enable type-hinting.
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    # Compute 3-axis contact forces.
    net_force = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0]
    # Compute 3-axis sum, clip 0, and return.
    return torch.sum(net_force.clip(min=0.0), dim=1)
