# Isaac Lab imports
from isaaclab.utils import configclass

# Local mdp module inherited from Isaac.
from wheelDog_RL.tasks.manager_based.wheeldog_rl import mdp

# Import robot asset configuration. 
from wheelDog_RL.assets.configs.xg_wheel_su import XG_WHEEL_SU_CFG

# Import settings. 
from wheelDog_RL.tasks.manager_based.wheeldog_rl.settings import LEG_ACTION_SCALE, WHEEL_ACTION_SCALE, ABD_ACTION_SCALE


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # Uses position action for non-wheel joints.
    # Uses velocity action for wheel joints.
    # Leaves on-hardware torque-velocity handling for abstraction. 
    abdomen_joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "FBL_ABAD_JOINT",
            "FAR_ABAD_JOINT",
            "RBL_ABAD_JOINT",
            "RAR_ABAD_JOINT",
        ],
        scale=ABD_ACTION_SCALE,
        preserve_order=True,
        use_default_offset=True,
    )
    hip_joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "FBL_HIP_JOINT",
            "FAR_HIP_JOINT",
            "RBL_HIP_JOINT",
            "RAR_HIP_JOINT",
        ],
        scale=LEG_ACTION_SCALE,
        preserve_order=True,
        use_default_offset=True,
    )
    knee_joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "FBL_KNEE_JOINT",
            "FAR_KNEE_JOINT",
            "RBL_KNEE_JOINT",
            "RAR_KNEE_JOINT",
        ],
        scale=LEG_ACTION_SCALE,
        preserve_order=True,
        use_default_offset=True,
    )
    wheel_joint_vel = mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=[
            "FBL_FOOT_JOINT",
            "FAR_FOOT_JOINT",
            "RBL_FOOT_JOINT",
            "RAR_FOOT_JOINT",
        ],
        scale=WHEEL_ACTION_SCALE,
        preserve_order=True,
        use_default_offset=True,
        clip={
            "FBL_FOOT_JOINT": (-160.0, 160.0),
            "FAR_FOOT_JOINT": (-160.0, 160.0),
            "RBL_FOOT_JOINT": (-160.0, 160.0),
            "RAR_FOOT_JOINT": (-160.0, 160.0),
        },
    )
