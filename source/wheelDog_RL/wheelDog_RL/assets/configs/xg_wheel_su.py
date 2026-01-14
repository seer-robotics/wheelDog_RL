import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.actuators import ActuatorNetMLPCfg, DCMotorCfg, ImplicitActuatorCfg

YOUR_ROBOT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        # usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/YourRobot/your_robot.usd",  # Or local path if not on Nucleus
        usd_path="~/workspace/Projects/USD_Files/xg_wheel_su_USD/xg_wheel_su.usd",
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
        pos=(0.0, 0.0, 1.0),  # Initial position (e.g., above ground)
        joint_pos={
            ".*_ABAD_JOINT": 0.00,
            ".*_HIP_JOINT": 0.40,
            ".*_KNEE_JOINT": -0.80,
            ".*_FOOT_JOINT": 0.00,
        },  # Default joint positions
        joint_vel={".*": 0.0},
    ),
    # Safety factor for joint limits
    soft_joint_pos_limit_factor=0.95,  
    # All joints use Implicit Actuator model definition. 
    # Thus, actuator behavior is handled by the simulation engine. 
    # Performs continuous-time ideal PD integration. 
    # We use this given that we trust the OEM supplied joint control API. 
    actuators={
        
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*_ABAD_JOINT", ".*_HIP_JOINT", ".*_KNEE_JOINT"],
            effort_limit=30,
            velocity_limit=28,
            stiffness=0.0,
            damping=1e5 # Empirical range (1e4, 1e6)
        ),
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=[".*_FOOT_JOINT"],
            effort_limit=10.5,
            velocity_limit=165,
            stiffness=0.0,
            damping=1e6 # Empirical range (1e4, 1e6)
        ),
    },
)
