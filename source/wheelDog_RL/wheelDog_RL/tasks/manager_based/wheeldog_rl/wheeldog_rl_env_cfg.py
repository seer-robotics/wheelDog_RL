# Library imports. 
import math
import torch

# Isaac Lab imports
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

# Import scene definition. 
from wheelDog_RL.tasks.manager_based.wheeldog_rl.sceneCfg import wheelDog_RL_sceneCfg

# Local mdp module inherited from Isaac.
from wheelDog_RL.tasks.manager_based.wheeldog_rl import mdp

# Import custom modules.
from .commandSpaceCfg import CommandsCfg, CrippledCommandsCfg
from .actionSpaceCfg import ActionsCfg, CrippledActionsCfg
from .observationSpaceCfg import ObservationsCfg
from .rewardCfg import RewardsCfg, CrippledRewardsCfg

# Import settings. 
from wheelDog_RL.tasks.manager_based.wheeldog_rl.settings import \
    CPU_POOL_BUCKET_SIZE, \
    CURRICULUM_ERROR_THRESHOLD_UP, \
    CURRICULUM_ERROR_THRESHOLD_DOWN, \
    CRIPPLE_PENALTY_TARGET_WEIGHT, \
    CRIPPLE_PENALTY_FLAT_STEPS, \
    CRIPPLE_PENALTY_DECAY_STEPS


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

    add_link_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["F.*", "R.*"]),
            "mass_distribution_params": (-0.1, 0.1),
            "recompute_inertia": True,
            "operation": "add",
        },
    )

    add_joint_friction = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "operation": "add",
            "friction_distribution_params": (-0.2, 0.2),
        },
    )

    # Reset events
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="BASE_LINK"),
            "force_range": (-20.0, 20.0),
            "torque_range": (-10.0, 10.0),
        },
    )

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
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(8.0, 10.0),
        params={"velocity_range": {"x": (-1.2, 1.2), "y": (-0.8, 0.8)}},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    fallen = DoneTerm(
        func=mdp.terminate_fallen,
        params={
        "sensor_cfg": SceneEntityCfg("contact_forces", body_names="BASE_LINK"),"threshold": 1.0
        },
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    # command_stages = CurrTerm(
    #     func=mdp.command_staged_curriculum,
    # )
    terrain_levels = CurrTerm(
        func=mdp.terrain_levels_velocityError,
        params={
            "error_threshold_up": CURRICULUM_ERROR_THRESHOLD_UP,
            "error_threshold_down": CURRICULUM_ERROR_THRESHOLD_DOWN,
        }
    )


@configclass
class CrippledCurriculumCfg:
    """Curriculum terms for the MDP."""

    cripplePosePenalty_curriculum = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "rewards.pretend_cripple_penalty.weight",
            "modify_fn": mdp.reward_weight_anneal,
            "modify_params": {
                "target_threshold": CRIPPLE_PENALTY_TARGET_WEIGHT,
                "flat_steps": CRIPPLE_PENALTY_FLAT_STEPS,
                "decay_steps": CRIPPLE_PENALTY_DECAY_STEPS,
            },
        }
    )


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

        # Check terrain generator existence and enable terrain curriculum
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False


@configclass
class CrippleLocomotionCfg(ManagerBasedRLEnvCfg):
    """
    Configuration for the cripple locomotion velocity-tracking environment.
    """
    # Scene settings
    scene: wheelDog_RL_sceneCfg = wheelDog_RL_sceneCfg(num_envs=4096, env_spacing=8)
    # State settings
    commands: CommandsCfg = CrippledCommandsCfg()
    actions: ActionsCfg = CrippledActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    # MDP settings
    rewards: RewardsCfg = CrippledRewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    # curriculum: CurriculumCfg = CrippledCurriculumCfg()
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

        # Override commands.
        self.commands.base_velocity.rel_standing_envs = 1.0

        # Override events.
        self.events.push_robot = None
        self.events.base_external_force_torque = None
        self.events.add_base_mass.params["mass_distribution_params"] = (0.85, 1.15)

        # Change terrain to flat.
        self.scene.terrain.terrain_type = "plane"

        # No terrain curriculum.
        self.scene.terrain.terrain_generator = None

        # No height scans for cripple dog.
        self.scene.fl_leg_ray = None
        self.scene.fr_leg_ray = None
        self.scene.rl_leg_ray = None
        self.scene.rr_leg_ray = None
        self.scene.height_scanner = None
        self.observations.critic.fl_foot_normals = None
        self.observations.critic.fr_foot_normals = None
        self.observations.critic.rl_foot_normals = None
        self.observations.critic.rr_foot_normals = None
        self.observations.critic.fl_foot_scan = None
        self.observations.critic.fr_foot_scan = None
        self.observations.critic.rl_foot_scan = None
        self.observations.critic.rr_foot_scan = None
        self.observations.critic.base_height_scan = None

        # Check terrain generator existence and enable terrain curriculum
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False
