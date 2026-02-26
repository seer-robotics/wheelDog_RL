# Isaac Lab imports
from isaaclab.utils import configclass

# Local mdp module inherited from Isaac.
from wheelDog_RL.tasks.manager_based.wheeldog_rl import mdp

# Import robot asset configuration. 
from wheelDog_RL.assets.configs.xg_wheel_su import XG_WHEEL_SU_CFG

# Import settings. 
from wheelDog_RL.tasks.manager_based.wheeldog_rl.settings import \
    ABD_OFFSET, \
    ABD_ACTION_SCALE, \
    HIP_OFFSET, \
    HIP_ACTION_SCALE, \
    KNEE_OFFSET, \
    KNEE_ACTION_SCALE, \
    WHEEL_ACTION_SCALE


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
        use_default_offset=False,
        offset=ABD_OFFSET,
        scale=ABD_ACTION_SCALE,
        preserve_order=True,
        clip={".*": (-0.49, 0.49)},
    )
    hip_joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "FBL_HIP_JOINT",
            "FAR_HIP_JOINT",
            "RBL_HIP_JOINT",
            "RAR_HIP_JOINT",
        ],
        use_default_offset=False,
        offset=HIP_OFFSET,
        scale=HIP_ACTION_SCALE,
        preserve_order=True,
        clip={".*": (-1.15, 2.97)},
    )
    knee_joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "FBL_KNEE_JOINT",
            "FAR_KNEE_JOINT",
            "RBL_KNEE_JOINT",
            "RAR_KNEE_JOINT",
        ],
        use_default_offset=False,
        offset=KNEE_OFFSET,
        scale=KNEE_ACTION_SCALE,
        preserve_order=True,
        clip={".*": (-2.72, -0.60)},
    )
    # leg_joint_pos = mdp.JointPositionToLimitsActionCfg(
    #     asset_name="robot",
    #     joint_names=[
    #         "FBL_ABAD_JOINT",
    #         "FAR_ABAD_JOINT",
    #         "RBL_ABAD_JOINT",
    #         "RAR_ABAD_JOINT",
    #         "FBL_HIP_JOINT",
    #         "FAR_HIP_JOINT",
    #         "RBL_HIP_JOINT",
    #         "RAR_HIP_JOINT",
    #         "FBL_KNEE_JOINT",
    #         "FAR_KNEE_JOINT",
    #         "RBL_KNEE_JOINT",
    #         "RAR_KNEE_JOINT",
    #     ],
    #     scale=1.0,
    #     preserve_order=True,
    # )
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
        clip={".*": (-160.0, 160.0)},
    )


@configclass
class CrippledActionsCfg:
    """Action specifications for the MDP."""
    abdomen_joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "FBL_ABAD_JOINT",
            "FAR_ABAD_JOINT",
            "RBL_ABAD_JOINT",
            "RAR_ABAD_JOINT",
        ],
        use_default_offset=False,
        offset=0.0,
        scale=0.4,
        preserve_order=True,
        clip={".*": (-0.49, 0.49)},
    )
    front_hip_joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "FBL_HIP_JOINT",
            "FAR_HIP_JOINT",
        ],
        use_default_offset=False,
        offset=1.0,
        scale=0.6,
        preserve_order=True,
        clip={".*": (-1.15, 2.97)},
    )
    rear_hip_joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "RBL_HIP_JOINT",
            "RAR_HIP_JOINT",
        ],
        use_default_offset=False,
        offset=0.3,
        scale=0.6,
        preserve_order=True,
        clip={".*": (-1.15, 2.97)},
    )
    front_knee_joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "FBL_KNEE_JOINT",
            "FAR_KNEE_JOINT",
        ],
        use_default_offset=False,
        offset=-2.0,
        scale=0.6,
        preserve_order=True,
        clip={".*": (-2.72, -0.60)},
    )
    rear_knee_joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "RBL_KNEE_JOINT",
            "RAR_KNEE_JOINT",
        ],
        use_default_offset=False,
        offset=-1.2,
        scale=0.5,
        preserve_order=True,
        clip={".*": (-2.72, -0.60)},
    )
    front_wheel_joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "FBL_FOOT_JOINT",
            "FAR_FOOT_JOINT",
        ],
        use_default_offset=False,
        offset=0.0,
        scale=1.0,
        preserve_order=True,
    )
    rear_wheel_joint_vel = mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=[
            "RBL_FOOT_JOINT",
            "RAR_FOOT_JOINT",
        ],
        scale=WHEEL_ACTION_SCALE,
        preserve_order=True,
        use_default_offset=True,
        clip={".*": (-160.0, 160.0)},
    )
