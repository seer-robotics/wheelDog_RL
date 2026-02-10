# Library imports. 
from isaaclab.utils import configclass

# Import robot asset configuration. 
from wheelDog_RL.assets.configs.xg_wheel_su import XG_WHEEL_SU_CFG

# Import training environment definition. 
from wheelDog_RL.tasks.manager_based.wheeldog_rl.wheeldog_rl_env_cfg import BlindLocomotionCfg


@configclass
class WheelDog_BlindLocomotionEnvCfg(BlindLocomotionCfg):
    def __post_init__(self):
        # Finish parent post init first.
        super().__post_init__()
        # Assign robot asset. 
        self.scene.robot = XG_WHEEL_SU_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Tick priviledged sensors based on the smallest update period (physics update period).
        self.scene.fl_leg_ray.update_period = self.sim.dt
        self.scene.fr_leg_ray.update_period = self.sim.dt
        self.scene.rl_leg_ray.update_period = self.sim.dt
        self.scene.rr_leg_ray.update_period = self.sim.dt
        self.scene.height_scanner.update_period = self.sim.dt
        self.scene.contact_forces.update_period = self.sim.dt


@configclass
class WheelDog_BlindLocomotionEnvPlayCfg(WheelDog_BlindLocomotionEnvCfg):
    def __post_init__(self):
        # Finish parent post init first.
        super().__post_init__()

        self.scene.num_envs = 32
        self.scene.env_spacing = 8

        # Spawn the robot randomly in the grid.
        self.scene.terrain.max_init_terrain_level = 5

        # # Reduce the number of terrains to save memory.
        # if self.scene.terrain.terrain_generator is not None:
        #     self.scene.terrain.terrain_generator.num_rows = 5
        #     self.scene.terrain.terrain_generator.num_cols = 10
        #     self.scene.terrain.terrain_generator.curriculum = False

        # Remove random events.
        self.events.base_external_force_torque = None
        self.events.push_robot = None
