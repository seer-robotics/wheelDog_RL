# Library imports. 
import math

# Isaac Lab imports
from isaaclab.envs import mdp
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import RecorderTermCfg as  RecTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveGaussianNoiseCfg as Gnoise

# Import scene definition. 
from wheelDog_RL.tasks.manager_based.wheeldog_rl.sceneCfg import wheelDog_RL_sceneCfg

# Import custom modules. 
from wheelDog_RL.tasks.manager_based.wheeldog_rl import customObservations
from wheelDog_RL.tasks.manager_based.wheeldog_rl import customCurriculum
from wheelDog_RL.tasks.manager_based.wheeldog_rl import customRewards

# Import settings. 
from wheelDog_RL.tasks.manager_based.wheeldog_rl.settings import OBS_HISTORY_LEN, CPU_POOL_BUCKET_SIZE


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    # https://github.com/isaac-sim/IsaacLab/discussions/2620
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(6.0, 10.0),
        rel_standing_envs=0.01,
        rel_heading_envs=0.99,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.6, 1.6), lin_vel_y=(-0.4, 0.4), ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi)
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # Uses position action for non-wheel joints.
    # Uses velocity action for wheel joints.
    # Leaves on-hardware torque-velocity handling for abstraction. 
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            ".*_ABAD_JOINT",
            ".*_HIP_JOINT",
            ".*_KNEE_JOINT",
        ],
        scale=1.0,
        use_default_offset=True,
    )
    wheel_vel = mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=[".*_FOOT_JOINT"],
        scale=1.0,
        use_default_offset=True,
        clip={".*_FOOT_JOINT": (-160.0, 160.0)}
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        ##
        # observation terms (order preserved)
        # Implement observation history on a per-term basis.
        ##

        # Base states history
        # Model these base states with an EKF on the physical robot. 
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel, 
            noise=Gnoise(mean=0, std=0.20),
            history_length=3,
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel, 
            noise=Gnoise(mean=0, std=0.08),
            history_length=3,
        )

        # IMU sensor history. 
        imu_ang_vel = ObsTerm(
            func=mdp.imu_ang_vel,
            noise=Gnoise(mean=0, std=0.035),
            params={"asset_cfg": SceneEntityCfg(name="base_IMU")},
            history_length=3,
        )
        imu_projected_gravity = ObsTerm(
            func=mdp.imu_projected_gravity,
            noise=Gnoise(mean=0, std=0.06),
            params={"asset_cfg": SceneEntityCfg(name="base_IMU")},
            history_length=3,
        )

        # Commands. 
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, 
            params={"command_name": "base_velocity"},
            history_length=0,
        )

        # Joint states history. 
        # Excludes the wheel positions from the joint positions history. 
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel, 
            noise=Gnoise(mean=0, std=0.03),
            params={
                "asset_cfg": SceneEntityCfg("robot", 
                    joint_names=[".*_ABAD_JOINT", ".*_HIP_JOINT", ".*_KNEE_JOINT"]),
                },
            history_length=OBS_HISTORY_LEN,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel, 
            noise=Gnoise(mean=0, std=0.08),
            history_length=OBS_HISTORY_LEN,
        )

        # Action history. 
        velocity_actions = ObsTerm(
            func=mdp.last_action,
            history_length=3,
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group."""
        # Base states history.
        base_pos_z = ObsTerm(func=mdp.base_pos_z)
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            history_length=3,
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            history_length=3,
        )
        projected_gravity = ObsTerm(func=mdp.projected_gravity)

        # Commands.
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, 
            params={"command_name": "base_velocity"},
        )

        # Joint states history. 
        # Excludes the wheel positions from the joint positions history. 
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg("robot", 
                    joint_names=[".*_ABAD_JOINT", ".*_HIP_JOINT", ".*_KNEE_JOINT"]),
                },
            history_length=OBS_HISTORY_LEN,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            history_length=OBS_HISTORY_LEN,
        )

        # Priviledged info.
        base_height_scan = ObsTerm(
            func=mdp.height_scan,
            params={
                "sensor_cfg": SceneEntityCfg("height_scanner"),
                "offset": 0.5,
            },
            clip=(-1.5, 1.5),
            history_length=2,
        )
        feet_contacts = ObsTerm(
            # Feet binary contact states.
            func=customObservations.contact_states,
            params={
                "threshold": 1,
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces",
                    body_names=[".*_FOOT_LINK"],
                )
            },
            history_length=3,
        )
        fl_foot_forces = ObsTerm(
            # Front left foot normal and tangential contact forces.
            func=customObservations.contact_forces,
            params={"sensor_cfg": SceneEntityCfg("fl_foot_contacts")},
            history_length=3,
        )
        rl_foot_forces = ObsTerm(
            # Rear left foot normal and tangential contact forces.
            func=customObservations.contact_forces,
            params={"sensor_cfg": SceneEntityCfg("rl_foot_contacts")},
            history_length=3,
        )
        fr_foot_forces = ObsTerm(
            # Front right foot normal and tangential contact forces.
            func=customObservations.contact_forces,
            params={"sensor_cfg": SceneEntityCfg("fr_foot_contacts")},
            history_length=3,
        )
        rr_foot_forces = ObsTerm(
            # Rear right foot normal and tangential contact forces.
            func=customObservations.contact_forces,
            params={"sensor_cfg": SceneEntityCfg("rr_foot_contacts")},
            history_length=3,
        )
        fl_foot_normals = ObsTerm(
            # Terrain normals around front left foot.
            func=customObservations.terrain_normals,
            params={"sensor_cfg": SceneEntityCfg("fl_leg_ray")},
            history_length=2,
        )
        rl_foot_normals = ObsTerm(
            # Terrain normals around rear left foot.
            func=customObservations.terrain_normals,
            params={"sensor_cfg": SceneEntityCfg("rl_leg_ray")},
            history_length=2,
        )
        fr_foot_normals = ObsTerm(
            # Terrain normals around front right foot.
            func=customObservations.terrain_normals,
            params={"sensor_cfg": SceneEntityCfg("fr_leg_ray")},
            history_length=2,
        )
        rr_foot_normals = ObsTerm(
            # Terrain normals around rear right foot.
            func=customObservations.terrain_normals,
            params={"sensor_cfg": SceneEntityCfg("rr_leg_ray")},
            history_length=2,
        )
        
        # Dynamics randomization terms
        contact_friction = ObsTerm(
            func=customObservations.contact_friction,
            params={
                "link_names": [
                    "FBL_FOOT_LINK",
                    "FAR_FOOT_LINK",
                    "RBL_FOOT_LINK",
                    "RAR_FOOT_LINK",
                ]
            },
        )
        
        # # Optional: wheel-specific slip velocity or contact quality
        # wheel_slip = ObsTerm(
        #     func=mdp.joint_vel_rel,  # but only for wheel joints, compared against base velocity
        #     params={
        #         "asset_cfg": SceneEntityCfg("robot",
        #             joint_names=[".*_WHEEL_JOINT"]),  # your wheel joint regex
        #     },
        #     history_length=3,
        # )  # Alternatively, implement a custom mdp.wheel_slip_velocity term
        
        def __post_init__(self):
            # No noise for priviledged information.
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


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
            "static_friction_range": (0.7, 1.3),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (0.0, 0.1),
            "num_buckets": CPU_POOL_BUCKET_SIZE,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="BASE_LINK"),
            "mass_distribution_params": (-1.0, 2.5),
            "recompute_inertia": True,
            "operation": "add",
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
            "force_range": (-2.0, 2.0),
            "torque_range": (-1.0, 1.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.25, 0.25),
            "velocity_range": (0.0, 0.0),
        },
    )

    # Interval events
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.4, 0.4), "y": (-0.4, 0.4)}},
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- rewards
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=1.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=0.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    
    # -- penalties
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-1.0e-2)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0e-5)
    feet_air_time = RewTerm(
        # Note here that feet air time is penalized instead of rewarded, as we are training a wheeled robot. The idea is to get it to keep its wheels on the ground. 
        func=customRewards.feet_air_time,
        weight=-0.125,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*FOOT_LINK"),
            "command_name": "base_velocity",
            "threshold": 0.5,
        },
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_HIP_LINK"]), "threshold": 1.0},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="BASE_LINK"), "threshold": 1.0},
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(
        func=customCurriculum.terrain_levels_velocityError,
        params={
            "error_threshold_up": 0.5,
            "error_threshold_down": 2.0,
        }
        )


##
# Environment configuration
##


@configclass
class BlindLocomotionCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""
    # Scene settings
    scene: wheelDog_RL_sceneCfg = wheelDog_RL_sceneCfg(num_envs=8192, env_spacing=8)
    # State settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
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
        self.episode_length_s = 32.0
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        # Update sensor update periods
        # Tick priviledged sensors based on the smallest update period (physics update period)
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        # Check terrain generator existence and enable terrain curriculum
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False
