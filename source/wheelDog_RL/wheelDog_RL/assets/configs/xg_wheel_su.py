from omni.isaac.lab.assets import ArticulationCfg as ArticCfg
from omni.isaac.lab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from omni.isaac.lab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from omni.isaac.lab.utils.assets import ISAACLAB_NUCLEUS_DIR

YOUR_ROBOT_CFG = ArticCfg(
    spawn=UsdFileCfg(
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/YourRobot/your_robot.usd",  # Or local path if not on Nucleus
        rigid_props=RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=100.0,
        ),
        activate_contact_sensors=True,
    ),
    init_state=ArticCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5),  # Initial position (e.g., above ground)
        rot=(1.0, 0.0, 0.0, 0.0),  # Quaternion orientation
        joint_pos={},  # Default joint positions if needed
    ),
    soft_joint_pos_limit_factor=0.95,  # Safety factor for joint limits
    actuators={
        "legs_and_wheels": ArticCfg.ActuatorCfg(  # Customize groups as per your robot
            joint_names_expr=[".*_HIP_JOINT", ".*_THIGH_JOINT", ".*_CALF_JOINT", ".*_WHEEL_JOINT"],  # Regex for joint names
            effort_limit=20.0,  # Torque limit (adjust based on hardware)
            velocity_limit=100.0,  # Speed limit
            stiffness=80.0,  # PD controller stiffness
            damping=0.5,  # PD controller damping
        ),
    },
)