from __future__ import annotations
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch
import math
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, ImuCfg, RayCasterCfg, patterns
from collections.abc import Callable
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

if TYPE_CHECKING:
    from isaaclab.assets import ArticulationCfg

# Import terrain generator configurations. 
from wheelDog_RL.tasks.manager_based.wheeldog_rl.terrainCfg import allTerrain_config


def generate_point_circle_pattern(
    cfg,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate ray starts and directions: 1 center + desired num of points on circle, all pointing straight down.

    Returns:
        starts: (N, 3) tensor of ray starting positions (usually all zeros)
        directions: (N, 3) unit vectors pointing straight down (-Z in local frame)
    """
    radius = cfg.radius
    n_circle = cfg.num_circle_points
    total_rays = 1 + n_circle

    # All rays start at the sensor origin in local coordinates
    starts = torch.zeros((total_rays, 3), device=device)

    # All directions point straight down: local -Z axis
    directions = torch.zeros((total_rays, 3), device=device)
    directions[:, 2] = -1.0  # straight down

    if n_circle > 0 and radius > 0.0:
        # Angles for the circle points.
        # Excludes the closing 2π point.
        angles = torch.linspace(
            0.0,
            2.0 * math.pi,
            n_circle + 1,
            device=device,
        )[:-1]

        # Place circle rays at (r cos θ, r sin θ, 0) offset in XY plane
        # (directions remain -Z, only the origin is offset horizontally)
        offsets = torch.zeros((n_circle, 3), device=device)
        offsets[:, 0] = radius * torch.cos(angles)
        offsets[:, 1] = radius * torch.sin(angles)

        # Assign to rays 1 through n_circle
        starts[1:] = offsets

    return starts, directions


@configclass
class PointCirclePatternCfg(patterns.PatternBaseCfg):
    """Configuration for a manual 9-ray pattern: 1 center + 8 on a circle."""
    func: Callable = generate_point_circle_pattern
    """Function to generate the pattern.

    The function should take in the configuration and the device name as arguments. It should return
    the pattern's starting positions and directions as a tuple of torch.Tensor.
    """

    radius: float = 0.25
    """Horizontal radius of the circle (meters)."""

    num_circle_points: int = 8
    """Number of points on the circle."""


@configclass
class wheelDog_RL_sceneCfg(InteractiveSceneCfg):
    # Lighting. 
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    # Terrain configurations. 
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=allTerrain_config,
        max_init_terrain_level=1,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=True,
    )

    # Sensors.
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/BASE_LINK",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.4, size=[1.2, 0.8]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    base_IMU = ImuCfg(
        prim_path="{ENV_REGEX_NS}/Robot/BASE_LINK",
        gravity_bias=(0.00, 0.00, 9.81), # Remember to account for gravity bias
        update_period=0.02,
        history_length=0,
        offset=ImuCfg.OffsetCfg(pos=(0.10, 0.00, 0.05)),
        debug_vis=False, 
    )
    contact_forces = ContactSensorCfg(
        # Contact sensors on all robot links.
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=6,
        track_air_time=True,
        force_threshold=1,
        debug_vis=False,
    )
    # fl_foot_contacts = ContactSensorCfg(
    #     # Front left foot filtered contact sensor.
    #     prim_path="{ENV_REGEX_NS}/Robot/FBL_FOOT_LINK",
    #     filter_prim_paths_expr=["/World/ground"],
    #     history_length=6,
    #     track_friction_forces=True,
    #     max_contact_data_count_per_prim=16,
    #     debug_vis=False,
    # )
    # rl_foot_contacts = ContactSensorCfg(
    #     # Rear left foot filtered contact sensor.
    #     prim_path="{ENV_REGEX_NS}/Robot/RBL_FOOT_LINK",
    #     filter_prim_paths_expr=["/World/ground"],
    #     history_length=6,
    #     track_friction_forces=True,
    #     max_contact_data_count_per_prim=16,
    #     debug_vis=False,
    # )
    # fr_foot_contacts = ContactSensorCfg(
    #     # Front right foot filtered contact sensor.
    #     prim_path="{ENV_REGEX_NS}/Robot/FAR_FOOT_LINK",
    #     filter_prim_paths_expr=["/World/ground"],
    #     history_length=6,
    #     track_friction_forces=True,
    #     max_contact_data_count_per_prim=16,
    #     debug_vis=False,
    # )
    # rr_foot_contacts = ContactSensorCfg(
    #     # Rear right foot filtered contact sensor.
    #     prim_path="{ENV_REGEX_NS}/Robot/RAR_FOOT_LINK",
    #     filter_prim_paths_expr=["/World/ground"],
    #     history_length=6,
    #     track_friction_forces=True,
    #     max_contact_data_count_per_prim=16,
    #     debug_vis=False,
    # )
    fl_leg_ray = RayCasterCfg(
        # Front left foot terrain scanner.
        prim_path="{ENV_REGEX_NS}/Robot/FBL_FOOT_LINK",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 4.8e-2, 0.0)),
        ray_alignment="yaw",
        pattern_cfg=PointCirclePatternCfg(
            radius=8.0e-2, num_circle_points=8,
        ),
        mesh_prim_paths=["/World/ground"],
    )
    rl_leg_ray = RayCasterCfg(
        # Rear left foot terrain scanner.
        prim_path="{ENV_REGEX_NS}/Robot/RBL_FOOT_LINK",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 4.8e-2, 0.0)),
        ray_alignment="yaw",
        pattern_cfg=PointCirclePatternCfg(
            radius=8.0e-2, num_circle_points=8,
        ),
        mesh_prim_paths=["/World/ground"],
    )
    fr_leg_ray = RayCasterCfg(
        # Front right foot terrain scanner.
        prim_path="{ENV_REGEX_NS}/Robot/FAR_FOOT_LINK",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, -4.8e-2, 0.0)),
        ray_alignment="yaw",
        pattern_cfg=PointCirclePatternCfg(
            radius=8.0e-2, num_circle_points=8,
        ),
        mesh_prim_paths=["/World/ground"],
    )
    rr_leg_ray = RayCasterCfg(
        # Rear right foot terrain scanner.
        prim_path="{ENV_REGEX_NS}/Robot/RAR_FOOT_LINK",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, -4.8e-2, 0.0)),
        ray_alignment="yaw",
        pattern_cfg=PointCirclePatternCfg(
            radius=8.0e-2, num_circle_points=8,
        ),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    # Here, the specific robot is left abstracted. 
    # Robot assignment and tuning are done in the robot configurations. 
    robot: ArticulationCfg = MISSING
