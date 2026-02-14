# Library imports.
from pathlib import Path
import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.actuators import DCMotorCfg, ImplicitActuatorCfg

XG_WHEEL_SU_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        # Path(__file__).resolve().parents[6] is the folder that contains the entire project folder.
        usd_path=str(
            Path(__file__).resolve().parents[6] / \
            "USD_Files" / "xg_wheel_su_USD" / "xg_wheel_su.usd"
        ),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, 
            solver_position_iteration_count=4, 
            solver_velocity_iteration_count=0
        ),
    ),
    # Initial States on entity spawn. 
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.45),
        joint_pos={
            ".*_ABAD_JOINT": 0.00,
            ".*_HIP_JOINT": 0.8,
            ".*_KNEE_JOINT": -1.5,
            ".*_FOOT_JOINT": 0.00,
        },  # Default joint positions
        joint_vel={".*": 0.0},
    ),
    # Safety factor for joint limits
    soft_joint_pos_limit_factor=0.95,  
    actuators={
        "legs": DCMotorCfg(
            joint_names_expr=[".*_ABAD_JOINT", ".*_HIP_JOINT", ".*_KNEE_JOINT"],
            effort_limit=33.5,
            saturation_effort=33.5,
            velocity_limit=28.0,
            stiffness=30.0,
            damping=0.5,
            friction=0.0,
        ),
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=[".*_FOOT_JOINT"],
            effort_limit_sim=80.0,
            velocity_limit_sim=60.0,
            stiffness=0.0,
            damping=1.5,
        )
    },
)
