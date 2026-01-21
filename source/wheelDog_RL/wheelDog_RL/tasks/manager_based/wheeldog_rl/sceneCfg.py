from __future__ import annotations
from dataclasses import MISSING
from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, ImuCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

if TYPE_CHECKING:
    from isaaclab.assets import ArticulationCfg

# Import terrain generator configurations. 
from wheelDog_RL.tasks.manager_based.wheeldog_rl.terrainCfg import allTerrain_config

# Import settings file. 
from wheelDog_RL.tasks.manager_based.wheeldog_rl.settings import OBS_HISTORY_LEN

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
        max_init_terrain_level=3,
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
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
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
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=6,
        track_air_time=True,
        debug_vis=False,
    )

    # Here, the specific robot is left abstracted. 
    # Robot assignment and tuning are done in the robot configurations. 
    robot: ArticulationCfg = MISSING
