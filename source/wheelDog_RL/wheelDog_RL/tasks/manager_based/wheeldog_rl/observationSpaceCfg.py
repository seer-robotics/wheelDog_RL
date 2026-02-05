# Isaac Lab imports
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils.noise import AdditiveGaussianNoiseCfg as Gnoise

# Local mdp module inherited from Isaac.
from wheelDog_RL.tasks.manager_based.wheeldog_rl import mdp

# Import settings. 
from wheelDog_RL.tasks.manager_based.wheeldog_rl.settings import STATE_HISTORY, SHORT_HISTORY, BASE_HEIGHT_THRESHOLD

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

        # Base states.
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel, 
            noise=Gnoise(mean=0, std=0.20),
            history_length=STATE_HISTORY,
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel, 
            noise=Gnoise(mean=0, std=0.08),
            history_length=STATE_HISTORY,
        )

        # IMU sensor. 
        imu_ang_vel = ObsTerm(
            func=mdp.imu_ang_vel,
            noise=Gnoise(mean=0, std=0.035),
            params={"asset_cfg": SceneEntityCfg(name="base_IMU")},
            history_length=STATE_HISTORY,
        )
        imu_projected_gravity = ObsTerm(
            func=mdp.imu_projected_gravity,
            noise=Gnoise(mean=0, std=0.06),
            params={"asset_cfg": SceneEntityCfg(name="base_IMU")},
            history_length=STATE_HISTORY,
        )

        # Commands. 
        commands_history = ObsTerm(
            func=mdp.generated_commands, 
            params={"command_name": "base_velocity"},
            history_length=SHORT_HISTORY,
        )

        # Actions. 
        joint_actions = ObsTerm(
            func=mdp.last_action,
            history_length=STATE_HISTORY,
        )

        # Joint states history. 
        # Excludes the wheel positions from the joint positions history. 
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel, 
            noise=Gnoise(mean=0, std=0.03),
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", 
                    joint_names=[
                        "FBL_ABAD_JOINT",
                        "FAR_ABAD_JOINT",
                        "RBL_ABAD_JOINT",
                        "RAR_ABAD_JOINT",
                        "FBL_HIP_JOINT",
                        "FAR_HIP_JOINT",
                        "RBL_HIP_JOINT",
                        "RAR_HIP_JOINT",
                        "FBL_KNEE_JOINT",
                        "FAR_KNEE_JOINT",
                        "RBL_KNEE_JOINT",
                        "RAR_KNEE_JOINT",
                    ],
                    preserve_order=True,
                ),
            },
            history_length=STATE_HISTORY,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel, 
            noise=Gnoise(mean=0, std=0.08),
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", 
                    joint_names=[
                        "FBL_ABAD_JOINT",
                        "FAR_ABAD_JOINT",
                        "RBL_ABAD_JOINT",
                        "RAR_ABAD_JOINT",
                        "FBL_HIP_JOINT",
                        "FAR_HIP_JOINT",
                        "RBL_HIP_JOINT",
                        "RAR_HIP_JOINT",
                        "FBL_KNEE_JOINT",
                        "FAR_KNEE_JOINT",
                        "RBL_KNEE_JOINT",
                        "RAR_KNEE_JOINT",
                        "FBL_FOOT_JOINT",
                        "FAR_FOOT_JOINT",
                        "RBL_FOOT_JOINT",
                        "RAR_FOOT_JOINT",
                    ],
                    preserve_order=True,
                ),
            },
            history_length=STATE_HISTORY,
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group."""
        # Base states.
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
        )
        projected_gravity = ObsTerm(func=mdp.projected_gravity)

        # Commands.
        commands_history = ObsTerm(
            func=mdp.generated_commands, 
            params={"command_name": "base_velocity"},
        )

        # Action. 
        joint_actions = ObsTerm(
            func=mdp.last_action,
        )

        # Joint states history. 
        # Excludes the wheel positions from the joint positions history. 
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", 
                    joint_names=[
                        "FBL_ABAD_JOINT",
                        "FAR_ABAD_JOINT",
                        "RBL_ABAD_JOINT",
                        "RAR_ABAD_JOINT",
                        "FBL_HIP_JOINT",
                        "FAR_HIP_JOINT",
                        "RBL_HIP_JOINT",
                        "RAR_HIP_JOINT",
                        "FBL_KNEE_JOINT",
                        "FAR_KNEE_JOINT",
                        "RBL_KNEE_JOINT",
                        "RAR_KNEE_JOINT",
                    ],
                    preserve_order=True,
                ),
            },
            history_length=SHORT_HISTORY,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", 
                    joint_names=[
                        "FBL_ABAD_JOINT",
                        "FAR_ABAD_JOINT",
                        "RBL_ABAD_JOINT",
                        "RAR_ABAD_JOINT",
                        "FBL_HIP_JOINT",
                        "FAR_HIP_JOINT",
                        "RBL_HIP_JOINT",
                        "RAR_HIP_JOINT",
                        "FBL_KNEE_JOINT",
                        "FAR_KNEE_JOINT",
                        "RBL_KNEE_JOINT",
                        "RAR_KNEE_JOINT",
                        "FBL_FOOT_JOINT",
                        "FAR_FOOT_JOINT",
                        "RBL_FOOT_JOINT",
                        "RAR_FOOT_JOINT",
                    ],
                    preserve_order=True,
                ),
            },
            history_length=SHORT_HISTORY,
        )

        # Contact terms.
        feet_contacts = ObsTerm(
            # Feet binary contact states.
            func=mdp.contact_states,
            params={
                "threshold": 1,
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces",
                    body_names=[
                        "FBL_FOOT_LINK",
                        "FAR_FOOT_LINK",
                        "RBL_FOOT_LINK",
                        "RAR_FOOT_LINK",
                    ],
                    preserve_order=True,
                )
            },
            history_length=SHORT_HISTORY,
        )
        feet_forces = ObsTerm(
            # Feet normal contact states.
            func=mdp.normal_forces,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces",
                    body_names=[
                        "FBL_FOOT_LINK",
                        "FAR_FOOT_LINK",
                        "RBL_FOOT_LINK",
                        "RAR_FOOT_LINK",
                    ],
                    preserve_order=True,
                )
            },
            history_length=SHORT_HISTORY,
        )
        # fl_foot_forces = ObsTerm(
        #     # Front left foot normal and tangential contact forces.
        #     func=customObservations.normal_forces,
        #     params={"sensor_cfg": SceneEntityCfg("fl_foot_contacts")},
        #     history_length=BASE_STATES_HISTORY,
        # )
        # fr_foot_forces = ObsTerm(
        #     # Front right foot normal and tangential contact forces.
        #     func=customObservations.normal_forces,
        #     params={"sensor_cfg": SceneEntityCfg("fr_foot_contacts")},
        #     history_length=BASE_STATES_HISTORY,
        # )
        # rl_foot_forces = ObsTerm(
        #     # Rear left foot normal and tangential contact forces.
        #     func=customObservations.normal_forces,
        #     params={"sensor_cfg": SceneEntityCfg("rl_foot_contacts")},
        #     history_length=BASE_STATES_HISTORY,
        # )
        # rr_foot_forces = ObsTerm(
        #     # Rear right foot normal and tangential contact forces.
        #     func=customObservations.normal_forces,
        #     params={"sensor_cfg": SceneEntityCfg("rr_foot_contacts")},
        #     history_length=BASE_STATES_HISTORY,
        # )

        # Height scan terms.
        fl_foot_normals = ObsTerm(
            # Terrain normals around front left foot.
            func=mdp.terrain_normals,
            params={"sensor_cfg": SceneEntityCfg("fl_leg_ray")},
        )
        fr_foot_normals = ObsTerm(
            # Terrain normals around front right foot.
            func=mdp.terrain_normals,
            params={"sensor_cfg": SceneEntityCfg("fr_leg_ray")},
        )
        rl_foot_normals = ObsTerm(
            # Terrain normals around rear left foot.
            func=mdp.terrain_normals,
            params={"sensor_cfg": SceneEntityCfg("rl_leg_ray")},
        )
        rr_foot_normals = ObsTerm(
            # Terrain normals around rear right foot.
            func=mdp.terrain_normals,
            params={"sensor_cfg": SceneEntityCfg("rr_leg_ray")},
        )
        fl_foot_scan = ObsTerm(
            # Height scan around front left foot.
            func=mdp.height_scan,
            params={
                "sensor_cfg": SceneEntityCfg("fl_leg_ray"),
                "offset": 0.1,
            },
            clip=(-1.5, 1.5),
        )
        fr_foot_scan = ObsTerm(
            # Height scan around front right foot.
            func=mdp.height_scan,
            params={
                "sensor_cfg": SceneEntityCfg("fr_leg_ray"),
                "offset": 0.1,
            },
            clip=(-1.5, 1.5),
        )
        rl_foot_scan = ObsTerm(
            # Height scan around rear left foot.
            func=mdp.height_scan,
            params={
                "sensor_cfg": SceneEntityCfg("rl_leg_ray"),
                "offset": 0.1,
            },
            clip=(-1.5, 1.5),
        )
        rr_foot_scan = ObsTerm(
            # Height scan around rear right foot.
            func=mdp.height_scan,
            params={
                "sensor_cfg": SceneEntityCfg("rr_leg_ray"),
                "offset": 0.1,
            },
            clip=(-1.5, 1.5),
        )
        base_height_scan = ObsTerm(
            func=mdp.height_scan,
            params={
                "sensor_cfg": SceneEntityCfg("height_scanner"),
                "offset": BASE_HEIGHT_THRESHOLD,
            },
            clip=(-1.5, 1.5),
        )
        
        # Dynamics randomization observation terms.
        contact_friction = ObsTerm(
            func=mdp.contact_friction,
            params={
                "link_names": [
                    "FBL_FOOT_LINK",
                    "FAR_FOOT_LINK",
                    "RBL_FOOT_LINK",
                    "RAR_FOOT_LINK",
                ]
            },
        )
        
        def __post_init__(self):
            # No noise for priviledged information.
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()