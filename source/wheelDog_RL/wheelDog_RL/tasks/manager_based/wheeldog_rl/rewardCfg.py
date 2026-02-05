# Library imports. 
import math

# Isaac Lab imports
from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg

# Local mdp module inherited from Isaac.
from wheelDog_RL.tasks.manager_based.wheeldog_rl import mdp

# Import settings.
from wheelDog_RL.tasks.manager_based.wheeldog_rl.settings import BASE_HEIGHT_THRESHOLD

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- rewards
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=1.2,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    stay_alive = RewTerm(mdp.is_alive, weight=1.0)
    feet_ground_time = RewTerm(
        # Reward keeping the feet on the ground.
        func=mdp.feet_ground_time,
        weight=0.5,
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
            ),
            "command_name": "base_velocity",
            "threshold": 0.04,
        },
    )
    
    # -- penalties
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.4)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-1e-2)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1e-1)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-07)
    dof_pos_deviate = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.3,
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
        }
    )
    stay_flat =RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=[
                    ".*_HIP_LINK",
                    ".*_KNEE_LINK",
                ]
            ),
            "threshold": 1.0
        },
    )
    base_height_threshold = RewTerm(
        func=mdp.base_height_threshold_l2,
        weight=-1.0,
        params={
            "height_threshold": BASE_HEIGHT_THRESHOLD,
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
        }
    )