# Library imports. 
import math

# Isaac Lab imports
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

# Import scene definition. 
from wheelDog_RL.tasks.manager_based.wheeldog_rl.sceneCfg import wheelDog_RL_sceneCfg

# Local mdp module inherited from Isaac.
from wheelDog_RL.tasks.manager_based.wheeldog_rl import mdp

# Import custom modules.
from .commandSpaceCfg import CommandsCfg
from .actionSpaceCfg import ActionsCfg
from .observationSpaceCfg import ObservationsCfg
from .rewardCfg import RewardsCfg

# Import settings. 
from wheelDog_RL.tasks.manager_based.wheeldog_rl.settings import \
    CPU_POOL_BUCKET_SIZE, \
    CURRICULUM_ERROR_THRESHOLD_UP, \
    CURRICULUM_ERROR_THRESHOLD_DOWN, \
    BASE_CONTACT_INIT_THRESHOLD, \
    BASE_CONTACT_TARGET_THRESHOLD, \
    BASE_CONTACT_FLAT_STEPS, \
    BASE_CONTACT_DECAY_STEPS, \
    ABD_POS_DEVIATE_SCALE_LEVELS, \
    ABD_POS_DEVIATE_MIN_FACTOR, \
    ABD_POS_DEVIATE_MIN_FACTOR_TERRAIN_LEVEL, \
    LEG_POS_DEVIATE_SCALE_LEVELS, \
    LEG_POS_DEVIATE_MIN_FACTOR, \
    LEG_POS_DEVIATE_MIN_FACTOR_TERRAIN_LEVEL


##
# Events and curriculum settings.
##


@configclass
class EventCfg:
    """Configuration for events."""

    # Startup events
    # All these are CPU intensive tasks. 
    # Do these once on env initialization. 
    # Make sure the number of envs is high enough for these to show effect. 
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "make_consistent": True,
            "static_friction_range": (0.8, 1.2),
            "dynamic_friction_range": (0.7, 0.9),
            "restitution_range": (0.0, 0.1),
            "num_buckets": CPU_POOL_BUCKET_SIZE,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="BASE_LINK"),
            "mass_distribution_params": (0.8, 1.3),
            "recompute_inertia": True,
            "operation": "scale",
        },
    )

    randomize_base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="BASE_LINK"),
            "com_range": {"x": (-0.02, 0.02), "y": (-0.02, 0.02), "z": (-0.01, 0.01)},
        },
    )

    # add_link_mass = EventTerm(
    #     func=mdp.randomize_rigid_body_mass,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=["F.*", "R.*"]),
    #         "mass_distribution_params": (-0.1, 0.1),
    #         "recompute_inertia": True,
    #         "operation": "add",
    #     },
    # )

    # add_joint_friction = EventTerm(
    #     func=mdp.randomize_joint_parameters,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
    #         "operation": "add",
    #         "friction_distribution_params": (-0.2, 0.2),
    #     },
    # )

    # Reset events
    # base_external_force_torque = EventTerm(
    #     func=mdp.apply_external_force_torque,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names="BASE_LINK"),
    #         "force_range": (-2.0, 2.0),
    #         "torque_range": (-1.0, 1.0),
    #     },
    # )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.1, 0.1),
                "y": (-0.1, 0.1),
                "z": (-0.1, 0.1),
                "roll": (-0.1, 0.1),
                "pitch": (-0.1, 0.1),
                "yaw": (-0.1, 0.1),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )

    # Interval events
    # push_robot = EventTerm(
    #     func=mdp.push_by_setting_velocity,
    #     mode="interval",
    #     interval_range_s=(10.0, 15.0),
    #     params={"velocity_range": {"x": (-0.4, 0.4), "y": (-0.4, 0.4)}},
    # )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="BASE_LINK"), "threshold": BASE_CONTACT_TARGET_THRESHOLD},
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    # base_contact_curriculum = CurrTerm(
    #     func=mdp.modify_term_cfg,
    #     params={
    #         "address": "terminations.base_contact.params.threshold",
    #         "modify_fn": mdp.base_contact_threshold_decay,
    #         "modify_params": {
    #             "initial_threshold": BASE_CONTACT_INIT_THRESHOLD,
    #             "target_threshold": BASE_CONTACT_TARGET_THRESHOLD,
    #             "flat_steps": BASE_CONTACT_FLAT_STEPS,
    #             "decay_steps": BASE_CONTACT_DECAY_STEPS,
    #         },
    #     }
    # )
    terrain_levels = CurrTerm(
        func=mdp.terrain_levels_velocityError,
        params={
            "error_threshold_up": CURRICULUM_ERROR_THRESHOLD_UP,
            "error_threshold_down": CURRICULUM_ERROR_THRESHOLD_DOWN,
        }
    )
    # stay_flat_penalty_weight_levels = CurrTerm(
    #     func=mdp.flat_reward_anneal_terrainLevels,
    #     params={
    #         "target_term_name": "stay_flat",
    #         "scale_levels": 4,
    #         "min_factor": 0.1,
    #         "min_factor_terrainLevel": 4,
    #     },
    # )
    # abd_deviate_penalty_weight_levels = CurrTerm(
    #     func=mdp.joint_deviation_penalty_levels,
    #     params={
    #         "target_term_name": "abd_pos_deviate",
    #         "scale_levels": ABD_POS_DEVIATE_SCALE_LEVELS,
    #         "min_factor": ABD_POS_DEVIATE_MIN_FACTOR,
    #         "min_factor_terrainLevel": ABD_POS_DEVIATE_MIN_FACTOR_TERRAIN_LEVEL,
    #     },
    # )
    # leg_deviate_penalty_weight_levels = CurrTerm(
    #     func=mdp.joint_deviation_penalty_levels,
    #     params={
    #         "target_term_name": "leg_pos_deviate",
    #         "scale_levels": LEG_POS_DEVIATE_SCALE_LEVELS,
    #         "min_factor": LEG_POS_DEVIATE_MIN_FACTOR,
    #         "min_factor_terrainLevel": LEG_POS_DEVIATE_MIN_FACTOR_TERRAIN_LEVEL,
    #     },
    # )


##
# Environment configuration
##


@configclass
class BlindLocomotionCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""
    # Scene settings
    scene: wheelDog_RL_sceneCfg = wheelDog_RL_sceneCfg(num_envs=4096, env_spacing=8)
    # State settings
    commands: CommandsCfg = CommandsCfg()
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    # recorder: RecorderCfg = RecorderCfg()

    # Post initialization overrides.
    def __post_init__(self):
        """Post initialization."""
        # Simulation and episode settings
        # Note that control period is sim.dt*decimation
        self.sim.dt = 0.005
        self.decimation = 4
        self.episode_length_s = 15.0
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        # Update sensor update periods
        # Tick priviledged sensors based on the smallest update period (physics update period)
        # if self.scene.height_scanner is not None:
        #     self.scene.height_scanner.update_period = self.sim.dt
        # if self.scene.contact_forces is not None:
        #     self.scene.contact_forces.update_period = self.sim.dt

        # Check terrain generator existence and enable terrain curriculum
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False
