# Library imports. 
import math

# Isaac Lab imports
from isaaclab.utils import configclass

# Local mdp module inherited from Isaac.
from wheelDog_RL.tasks.manager_based.wheeldog_rl import mdp

# Import settings. 
# from wheelDog_RL.tasks.manager_based.wheeldog_rl.settings import \
#     CMD_CURRICULUM_INIT_MIN_RANGES

@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    # https://github.com/isaac-sim/IsaacLab/discussions/2620
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(6.0, 10.0),
        rel_standing_envs=0.01,
        rel_heading_envs=0.99,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.3, 0.3),
            lin_vel_y=(-0.0, 0.0),
            ang_vel_z=(-0.08, 0.08),
            heading=(-math.pi, math.pi)
        ),
    )