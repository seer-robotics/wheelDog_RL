# Isaac Lab imports
from isaaclab.utils import configclass

# Local mdp module inherited from Isaac.
from wheelDog_RL.tasks.manager_based.wheeldog_rl import mdp

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # Uses position action for non-wheel joints.
    # Uses velocity action for wheel joints.
    # Leaves on-hardware torque-velocity handling for abstraction. 
    abdomen_joint_pos = mdp.RelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "FBL_ABAD_JOINT",
            "FAR_ABAD_JOINT",
            "RBL_ABAD_JOINT",
            "RAR_ABAD_JOINT",
        ],
        scale=0.25,
        clip={".*": (-0.49, 0.49)},
        preserve_order=True,
        use_default_offset=True,
    )
    hip_joint_pos = mdp.RelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "FBL_HIP_JOINT",
            "FAR_HIP_JOINT",
            "RBL_HIP_JOINT",
            "RAR_HIP_JOINT",
        ],
        scale=0.25,
        clip={".*": (-1.15, 2.97)},
        preserve_order=True,
        use_default_offset=True,
    )
    knee_joint_pos = mdp.RelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "FBL_KNEE_JOINT",
            "FAR_KNEE_JOINT",
            "RBL_KNEE_JOINT",
            "RAR_KNEE_JOINT",
        ],
        scale=0.25,
        clip={".*": (-2.72, -0.60)},
        preserve_order=True,
        use_default_offset=True,
    )
    # joint_pos = mdp.JointPositionToLimitsActionCfg(
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
        scale=5.0,
        preserve_order=True,
        use_default_offset=True,
        clip={
            "FBL_FOOT_JOINT": (-160.0, 160.0),
            "FAR_FOOT_JOINT": (-160.0, 160.0),
            "RBL_FOOT_JOINT": (-160.0, 160.0),
            "RAR_FOOT_JOINT": (-160.0, 160.0),
        },
    )
