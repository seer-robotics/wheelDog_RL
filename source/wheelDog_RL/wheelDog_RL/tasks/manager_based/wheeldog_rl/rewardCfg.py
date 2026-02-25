# Library imports. 
import math

# Isaac Lab imports
from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg

# Local mdp module inherited from Isaac.
from wheelDog_RL.tasks.manager_based.wheeldog_rl import mdp

# Import settings.
from wheelDog_RL.tasks.manager_based.wheeldog_rl.settings import BASE_HEIGHT_THRESHOLD, WHEEL_RATE_INIT_WEIGHT


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
        weight=1.5,
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
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.4)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-5e-1)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-07)
    dof_torque_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-2.5e-5)
    # dof_energy_l1 = RewTerm(func=mdp.joint_energy_l1, weight=-1.0e-5)
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
        weight=-0.2,
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
    # leg_pos_deviate = RewTerm(
    #     func=mdp.joint_deviation_l1,
    #     weight=-0.2,
    #     params={
    #         "asset_cfg": SceneEntityCfg(
    #             "robot", 
    #             joint_names=[
    #                 ".*_HIP_JOINT",
    #                 ".*_KNEE_JOINT",
    #             ],
    #             preserve_order=True,
    #         ),
    #     }
    # )
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
    # base_height_threshold = RewTerm(
    #     func=mdp.base_height_threshold,
    #     weight=-2.0,
    #     params={
    #         "height_threshold": BASE_HEIGHT_THRESHOLD,
    #         "asset_cfg": SceneEntityCfg("robot"),
    #         "sensor_cfg": SceneEntityCfg("height_scanner"),
    #     }
    # )
