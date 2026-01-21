"""Custom functions that can be used to define observations for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.ObservationTermCfg` object to
specify the observation function and its parameters.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def contact_forces(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Extract the normal contact forces on specified bodies."""
    # Enable type-hinting.
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    robot: Articulation = env.scene["robot"]

    # Normal forces in world frame.
    net_forces_w = contact_sensor.data.net_forces_w
    
    # Isolate data from specified bodies.
    wheel_ids = sensor_cfg.body_ids
    wheel_forces_w = net_forces_w[:, wheel_ids]
    
    # Acquire robot base frame quarternions. 
    base_quats_w = robot.data.root_link_quat_w

    # Transform forces to robot base frame and return.
    return quat_apply_inverse(base_quats_w.unsqueeze(1), wheel_forces_w)

    
