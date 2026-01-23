"""Custom functions that can be used to define observations for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.ObservationTermCfg` object to
specify the observation function and its parameters.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.utils.math import quat_apply_inverse

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg
    from isaaclab.assets import Articulation
    from isaaclab.sensors import ContactSensor, RayCaster


def contact_states(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    """Feet binary contact states.

    Determined by feet z-axis normal contact forces in robot base frame.

    ``threshold``: Base frame feet z-axis normal force above which contact is considered true. 
    """
    # Enable type-hinting.
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    robot: Articulation = env.scene["robot"]

    # Normal forces in the world frame.
    net_forces_w = contact_sensor.data.net_forces_w
    
    # Isolate data from specified bodies.
    wheel_ids = sensor_cfg.body_ids
    wheel_forces_w = net_forces_w[:, wheel_ids]
    
    # Acquire robot base frame quarternions and match shape with forces tensor.
    base_quat_w = robot.data.root_link_quat_w
    base_quat_w = base_quat_w.unsqueeze(dim=1)
    base_quat_w = base_quat_w.repeat(1, (wheel_forces_w.shape[1] // base_quat_w.shape[1]), 1)

    # Transform forces to robot base frame.
    wheel_forces_b = quat_apply_inverse(base_quat_w, wheel_forces_w)
    
    # Determine contact state and return.
    return (wheel_forces_b[..., 2] > threshold).float()


def contact_forces(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Extract the total contact forces (normal and tangential) on the specified body."""
    # Enable type-hinting.
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    robot: Articulation = env.scene["robot"]

    # Normal and tangential forces in the world frame.
    net_forces_w = contact_sensor.data.net_forces_w
    friction_forces_w = torch.sum(contact_sensor.data.friction_forces_w, dim=2)
    
    # Isolate data from specified bodies.
    # This doesn't really do anything here, because the sensor currently only supports filtered sensing for one explicit body.
    wheel_ids = sensor_cfg.body_ids
    wheel_forces_w = torch.sum(
        net_forces_w[:, wheel_ids] + friction_forces_w[:, wheel_ids],
        dim=1)
    
    # Acquire robot base frame quarternions. 
    base_quat_w = robot.data.root_link_quat_w
    base_quat_w = base_quat_w

    # Transform forces to robot base frame and return.
    wheel_forces_b = quat_apply_inverse(base_quat_w, wheel_forces_w)
    return wheel_forces_b


def terrain_normals(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Extract the body frame terrain normals with the specified ray-caster."""
    # Enable type-hinting.
    leg_rays: RayCaster = env.scene.sensors[sensor_cfg.name]
    robot: Articulation = env.scene["robot"]

    # Ray-cast hit points in the world frame.
    leg_scans_w = leg_rays.data.ray_hits_w

    # Compute world frame normal vector of point cloud surface.
    B = leg_scans_w.shape[1]
    centroids = leg_scans_w.mean(dim=1)
    centered_points = leg_scans_w - centroids.unsqueeze(1)
    cov_matrices = (centered_points .transpose(1, 2) @ centered_points) / (B - 1)
    eigenvecs = torch.linalg.eigh(cov_matrices)[1]
    normals_w = eigenvecs[:, :, 0]
    normals_w = torch.nn.functional.normalize(normals_w, dim=1)

    # Acquire robot base frame quarternions. 
    base_quat_w = robot.data.root_link_quat_w

    # Transform normals to robot base frame.
    normals_b = quat_apply_inverse(base_quat_w, normals_w)
    return normals_b


def normal_forces(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Extract the normal contact forces on specified bodies."""
    # Enable type-hinting.
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    robot: Articulation = env.scene["robot"]

    # Normal forces in the world frame.
    net_forces_w = contact_sensor.data.net_forces_w
    
    # Isolate data from specified bodies.
    wheel_ids = sensor_cfg.body_ids
    wheel_forces_w = net_forces_w[:, wheel_ids]
    
    # Acquire robot base frame quarternions. 
    base_quat_w = robot.data.root_link_quat_w
    base_quat_w = base_quat_w.unsqueeze(dim=1)
    base_quat_w = base_quat_w.repeat(1, (wheel_forces_w.shape[1] // base_quat_w.shape[1]), 1)

    # Transform forces to robot base frame and return.
    wheel_forces_b = quat_apply_inverse(base_quat_w, wheel_forces_w)
    return wheel_forces_b
