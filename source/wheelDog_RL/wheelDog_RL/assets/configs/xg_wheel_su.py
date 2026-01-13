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
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5),  # Initial position (e.g., above ground)
        joint_pos={
            ".*_ABAD_JOINT": 0.00,
            ".*_HIP_JOINT": 0.40,
            ".*_KNEE_JOINT": -0.80,
            ".*_FOOT_JOINT": 0.00,
        },  # Default joint positions
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.95,  # Safety factor for joint limits
    actuators={
        "legs_and_wheels": ArticulationCfg.ActuatorCfg(  # Customize groups as per your robot
            joint_names_expr=[".*_HIP_JOINT", ".*_THIGH_JOINT", ".*_CALF_JOINT", ".*_WHEEL_JOINT"],  # Regex for joint names
            effort_limit=20.0,  # Torque limit (adjust based on hardware)
            velocity_limit=100.0,  # Speed limit
            stiffness=80.0,  # PD controller stiffness
            damping=0.5,  # PD controller damping
        ),
    },
)
