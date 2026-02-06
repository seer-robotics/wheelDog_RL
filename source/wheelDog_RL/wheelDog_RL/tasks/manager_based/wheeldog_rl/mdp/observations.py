"""Custom functions that can be used to define observations for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.ObservationTermCfg` object to
specify the observation function and its parameters.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.utils.math import quat_apply_inverse, normalize

from isaaclab.assets import Articulation

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg
    from isaaclab.sensors import ContactSensor, RayCaster


def contact_states(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    """Feet binary contact states.

    Determined by feet z-axis normal contact forces in robot base frame.

    ``threshold``: Base frame feet z-axis normal force above which contact is considered true. 
    """
    # Enable type-hinting.
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    
    # Net forces in world frame: shape (num_envs, num_bodies_with_sensor, 3)
    net_forces_w = contact_sensor.data.net_forces_w
    
    # Select only the wheel bodies.
    wheel_ids = sensor_cfg.body_ids
    wheel_forces_w = net_forces_w[:, wheel_ids]
    wheel_normal_forces_z = wheel_forces_w[..., 2]
    
    # Determine contact state and return.
    wheel_normal_forces_z = torch.nan_to_num(wheel_normal_forces_z, nan=0.0, posinf=0.0, neginf=0.0)
    binary_states = (wheel_normal_forces_z > threshold).float()
    return binary_states


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
    base_quat_w = normalize(robot.data.root_link_quat_w.clone())

    # Transform forces to robot base frame and return.
    wheel_forces_b = quat_apply_inverse(base_quat_w, wheel_forces_w)
    return wheel_forces_b


def terrain_normals(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Extract the body frame terrain normals with the specified ray-caster."""
    # Enable type-hinting.
    ray_caster: RayCaster = env.scene.sensors[sensor_cfg.name]
    robot: Articulation = env.scene["robot"]

    # Ray-cast hit points in the world frame.
    scans_w = ray_caster.data.ray_hits_w
    scans_w = torch.nan_to_num(scans_w, nan=0.0, posinf=100.0, neginf=-100.0)

    # Compute world frame normal vector of point cloud surface.
    B = scans_w.shape[1]
    centroids = scans_w.mean(dim=1)
    centered_points = scans_w - centroids.unsqueeze(1)
    cov_matrices = (centered_points .transpose(1, 2) @ centered_points) / (B - 1)
    eigenvecs = torch.linalg.eigh(cov_matrices)[1]
    normals_w = eigenvecs[:, :, 0]
    normals_w = torch.nn.functional.normalize(normals_w, dim=-1)

    # Account for flat terrain.
    z_var     = torch.var(scans_w[:, :, 2], dim=1)
    cos_theta = torch.abs(normals_w[:, 2])
    VAR_THRESHOLD  = 1e-5       # m² – variance below which we distrust PCA
    MIN_COS_THRESH = 0.40       # cos(66°) ≈ 0.40 → start blending below ~66°
    MAX_COS_THRESH = 0.70       # cos(45°) ≈ 0.71 → full confidence above ~45°
    slope_confidence = torch.clamp(
        (cos_theta - MIN_COS_THRESH) / (MAX_COS_THRESH - MIN_COS_THRESH),
        0.0, 1.0
    )
    flat_confidence = torch.clamp(
        z_var / VAR_THRESHOLD,
        0.0, 1.0
    )
    confidence = torch.minimum(slope_confidence, flat_confidence)

    # Blend toward world +z.
    world_up = torch.tensor([0.0, 0.0, 1.0], device=normals_w.device, dtype=normals_w.dtype)
    normals_w = confidence.unsqueeze(-1) * normals_w + (1.0 - confidence).unsqueeze(-1) * world_up

    # Re-normalize (necessary after blending).
    normals_w = torch.nn.functional.normalize(normals_w, dim=-1)

    # Now apply sign correction (only meaningful when confidence is high)
    flip_mask = (normals_w[:, 2] < 0.0) & (confidence > 0.2)
    normals_w[flip_mask] = -normals_w[flip_mask]
    # print(f"normals_w: {normals_w}")

    # Acquire robot base frame quarternions. 
    base_quat_w = normalize(robot.data.root_link_quat_w.clone())

    # Transform normals to robot base frame.
    normals_b = quat_apply_inverse(base_quat_w, normals_w)
    return normals_b


def contact_friction(env: ManagerBasedRLEnv, link_names: list[str]) -> torch.Tensor:
    """Extract the friction coefficients of the specified bodies.

    ``materials``: Union[np.ndarray, torch.Tensor, wp.array]: An array of material properties with shape (count, max_shapes, 3) where count is the number of rigid objects in the view and max_shapes is the maximum number of shapes in all the rigid objects in the view. The 3 elements of the last dimension are the static friction, dynamic friction, and restitution respectively.
    """
    # Enable type-hinting.
    robot: Articulation = env.scene["robot"]

    # Extract specified body indeces.
    # Note that currently, there is no collision body for .*_ABAD_LINK.
    # Obtain number of shapes per body (needed for indexing the material properties correctly).
    # Note: this is a workaround since the Articulation does not provide a direct way to obtain the number of shapes per body.
    # We use the physics simulation view to obtain the number of shapes per body.
    if isinstance(robot, Articulation):
        num_shapes_per_body = []
        for link_path in robot.root_physx_view.link_paths[0]:
            link_physx_view = robot._physics_sim_view.create_rigid_body_view(link_path)  # type: ignore
            num_shapes_per_body.append(link_physx_view.max_shapes)
        # ensure the parsing is correct
        num_shapes = sum(num_shapes_per_body)
        expected_shapes = robot.root_physx_view.max_shapes
        if num_shapes != expected_shapes:
            raise ValueError(
                "Randomization term 'randomize_rigid_body_material' failed to parse the number of shapes per body."
                f" Expected total shapes: {expected_shapes}, but got: {num_shapes}."
            )
    else:
        # in this case, we don't need to do special indexing
        num_shapes_per_body = None

    if num_shapes_per_body != None:
        shape_mapping: dict[str, list[int]] = {}
        current_index = 0
        for name, num_shapes in zip(robot.body_names, num_shapes_per_body):
            shape_indices = list(range(current_index, current_index + num_shapes))
            shape_mapping[name] = shape_indices
            current_index += num_shapes
        wheel_physx_indeces = [shape_mapping[key] for key in link_names]
    else:
        wheel_physx_indeces = [robot.body_names.index(target) for target in link_names]

    # Reference physx indeces info:
    # num_shapes_per_body: [1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    # body_names: ['BASE_LINK', 'FAR_ABAD_LINK', 'FBL_ABAD_LINK', 'RAR_ABAD_LINK', 'RBL_ABAD_LINK', 'FAR_HIP_LINK', 'FBL_HIP_LINK', 'RAR_HIP_LINK', 'RBL_HIP_LINK', 'FAR_KNEE_LINK', 'FBL_KNEE_LINK', 'RAR_KNEE_LINK', 'RBL_KNEE_LINK', 'FAR_FOOT_LINK', 'FBL_FOOT_LINK', 'RAR_FOOT_LINK', 'RBL_FOOT_LINK']
    
    # Extract material properties, shape (N, count, max_shapes, 3) where N is the number of parallel environments, count is the number of rigid objects in the view, and max_shapes is the maximum number of shapes in all the rigid objects in the view.
    materials: torch.Tensor = robot.root_physx_view.get_material_properties()
    materials = materials.to(env.device)
    materials = materials[:, wheel_physx_indeces]
    materials = materials.flatten(start_dim=1)
    return materials


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
    base_quat_w = normalize(robot.data.root_link_quat_w.clone())
    base_quat_w = base_quat_w.unsqueeze(dim=1)
    base_quat_w = base_quat_w.repeat(1, (wheel_forces_w.shape[1] // base_quat_w.shape[1]), 1)

    # Transform forces to robot base frame and return.
    wheel_forces_b = quat_apply_inverse(base_quat_w, wheel_forces_w)
    wheel_forces_b = torch.nan_to_num(wheel_forces_b, nan=0.0, posinf=100.0, neginf=-100.0)
    return wheel_forces_b
