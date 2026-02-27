# Library imports. 
import math

# Isaac Lab imports
from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg

# Local mdp module inherited from Isaac.
from wheelDog_RL.tasks.manager_based.wheeldog_rl import mdp

# Import settings.
from wheelDog_RL.tasks.manager_based.wheeldog_rl.settings import CRIPPLE_TARGET


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- rewards
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.16)}
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.16)}
    )
    good_stance = RewTerm(
        func=mdp.default_joint_pos,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", 
                joint_names=[
                    ".*_ABAD_JOINT",
                ],
                preserve_order=True,
            ),
            "std": math.sqrt(0.16),
        }
    )
    
    # -- penalties
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.3)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-5e-1)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-07)
    dof_torque_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-2.5e-5)
    # dof_energy_l1 = RewTerm(func=mdp.joint_energy_l1, weight=-2.5e-8)
    # dof_energy_legs = RewTerm(
    #     func=mdp.joint_energy_l1,
    #     weight=-1.0e-5,
    #     params={
    #         "asset_cfg": SceneEntityCfg(
    #             "robot", 
    #             joint_names=[
    #                 ".*_ABAD_JOINT",
    #                 ".*_HIP_JOINT",
    #                 ".*_KNEE_JOINT",
    #             ],
    #             preserve_order=True,
    #         ),
    #     }
    # )
    # dof_energy_wheels = RewTerm(
    #     func=mdp.joint_energy_l1,
    #     weight=-1.0e-6,
    #     params={
    #         "asset_cfg": SceneEntityCfg(
    #             "robot", 
    #             joint_names=[".*_FOOT_JOINT"],
    #             preserve_order=True,
    #         ),
    #     }
    # )
    leg_action_rate_l2 = RewTerm(
        func=mdp.actionTerm_rate_l2,
        weight=-1e-2,
        params={
            "term_names": [
                "abdomen_joint_pos",
                "hip_joint_pos",
                "knee_joint_pos",
            ]
        }
    )
    wheel_action_rate_l2 = RewTerm(
        func=mdp. actionTerm_rate_l2,
        weight=-0.5e-2,
        params={
            "term_names": [
                "wheel_joint_vel",
            ]
        }
    )
    dof_pos_deviate = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.3,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", 
                joint_names=[
                    ".*_ABAD_JOINT",
                    ".*_HIP_JOINT",
                    ".*_KNEE_JOINT",
                ],
                preserve_order=True,
            ),
        }
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=[
                    "BASE_LINK",
                    ".*_HIP_LINK",
                    ".*_KNEE_LINK",
                ]
            ),
            "threshold": 1.0
        },
    )
    parallel_to_terrain =RewTerm(
        func=mdp.terrain_orientation,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("height_scanner"),
        },
    )
    zero_drift_l2 = RewTerm(
        func=mdp.zero_drift_l2,
        weight=-10.0,
        params={
            "zero_cmd_threshold": 0.1,
        }
    )
    # kinematic_slip_l2 = RewTerm(
    #     func=mdp.kinematic_slip,
    #     weight=-1e-3,
    #     params={
    #         "asset_cfg": SceneEntityCfg(
    #             "robot", 
    #             joint_names=[
    #                 "FBL_FOOT_JOINT",
    #                 "FAR_FOOT_JOINT",
    #                 "RBL_FOOT_JOINT",
    #                 "RAR_FOOT_JOINT",
    #             ],
    #             preserve_order=True,
    #         ),
    #         "wheel_y_positions": [0.2, -0.2, 0.24, -0.24],
    #         "wheel_radius": 0.08,
    #     }
    # )
    # base_height_threshold = RewTerm(
    #     func=mdp.base_height_threshold,
    #     weight=-2.0,
    #     params={
    #         "height_threshold": BASE_HEIGHT_THRESHOLD,
    #         "asset_cfg": SceneEntityCfg("robot"),
    #         "sensor_cfg": SceneEntityCfg("height_scanner"),
    #     }
    # )


@configclass
class CrippledRewardsCfg:
    """Reward terms for the MDP."""

    # -- rewards
    pretend_cripple_reward = RewTerm(
        func=mdp.joint_pos_target_reward_l2,
        weight=1.0,
        params={
            "target": CRIPPLE_TARGET,
            "std": math.sqrt(0.36),
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "FBL_ABAD_JOINT",
                    "FBL_HIP_JOINT",
                    "FBL_KNEE_JOINT",
                    "FBL_FOOT_JOINT",
                    "FAR_ABAD_JOINT",
                    "FAR_HIP_JOINT",
                    "FAR_KNEE_JOINT",
                    "FAR_FOOT_JOINT",
                ],
                preserve_order=True,
            ),
        },
    )
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.16)}
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.16)}
    )
    
    # -- penalties
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.3)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-5e-1)
    abd_pos_deviate = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", 
                joint_names=[
                    "RBL_ABAD_JOINT",
                    "RAR_ABAD_JOINT",
                ],
                preserve_order=True,
            ),
        }
    )
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-07)
    dof_torque_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-2.5e-5)
    leg_action_rate_l2 = RewTerm(
        func=mdp.actionTerm_rate_l2,
        weight=-1e-2,
        params={
            "term_names": [
                "abdomen_joint_pos",
                "front_hip_joint_pos",
                "rear_hip_joint_pos",
                "front_knee_joint_pos",
                "rear_knee_joint_pos",
                "front_wheel_joint_pos",
            ]
        }
    )
    wheel_action_rate_l2 = RewTerm(
        func=mdp. actionTerm_rate_l2,
        weight=-0.5e-2,
        params={
            "term_names": [
                "rear_wheel_joint_vel",
            ]
        }
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=[
                    "BASE_LINK",
                    ".*_HIP_LINK",
                    ".*_KNEE_LINK",
                    "FBL_FOOT_LINK",
                    "FAR_FOOT_LINK",
                ]
            ),
            "threshold": 1.0
        },
    )
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    base_height = RewTerm(
        func=mdp.base_height_l2,
        weight=-1.0,
        params={"target_height": 0.35},
    )
    support_deviation = RewTerm(
        func=mdp.rear_feet_com_alignment,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", 
                body_names=[
                    "RBL_FOOT_LINK",
                    "RAR_FOOT_LINK",
                ],
                preserve_order=True,
            ),
        }
    )
    pretend_cripple_penalty = RewTerm(
        func=mdp.joint_pos_target_penalty_l2,
        weight=-1.0,
        params={
            "target": CRIPPLE_TARGET,
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "FBL_ABAD_JOINT",
                    "FBL_HIP_JOINT",
                    "FBL_KNEE_JOINT",
                    "FBL_FOOT_JOINT",
                    "FAR_ABAD_JOINT",
                    "FAR_HIP_JOINT",
                    "FAR_KNEE_JOINT",
                    "FAR_FOOT_JOINT",
                ],
                preserve_order=True,
            ),
        },
    )
