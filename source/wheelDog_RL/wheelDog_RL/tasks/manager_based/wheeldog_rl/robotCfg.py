# Library imports. 
from isaaclab.utils import configclass

# Import robot asset configuration. 
from wheelDog_RL.assets.configs.xg_wheel_su import XG_WHEEL_SU_CFG

# Import training environment definition. 
from wheelDog_RL.tasks.manager_based.wheeldog_rl.wheeldog_rl_env_cfg import BlindLocomotionCfg


@configclass
class WheelDog_BlindLocomotionEnvCfg(BlindLocomotionCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # Assign robot asset. 
        self.scene.robot = XG_WHEEL_SU_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


# Example play configuration from official repositories. 
# Smaller scene and no randomization.
# @configclass
# class UnitreeGo1RoughEnvCfg_PLAY(UnitreeGo1RoughEnvCfg):
#     def __post_init__(self):
#         # post init of parent
#         super().__post_init__()

#         # make a smaller scene for play
#         self.scene.num_envs = 50
#         self.scene.env_spacing = 2.5
#         # spawn the robot randomly in the grid (instead of their terrain levels)
#         self.scene.terrain.max_init_terrain_level = None
#         # reduce the number of terrains to save memory
#         if self.scene.terrain.terrain_generator is not None:
#             self.scene.terrain.terrain_generator.num_rows = 5
#             self.scene.terrain.terrain_generator.num_cols = 5
#             self.scene.terrain.terrain_generator.curriculum = False

#         # disable randomization for play
#         self.observations.policy.enable_corruption = False
#         # remove random pushing event
#         self.events.base_external_force_torque = None
#         self.events.push_robot = None