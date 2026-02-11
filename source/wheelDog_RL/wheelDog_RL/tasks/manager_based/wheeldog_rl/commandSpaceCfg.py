# Library imports. 
import math

# Isaac Lab imports
from isaaclab.utils import configclass

# Local mdp module inherited from Isaac.
from wheelDog_RL.tasks.manager_based.wheeldog_rl import mdp

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
            lin_vel_x=(-1.2, 1.2), lin_vel_y=(-0.05, 0.05), ang_vel_z=(-0.8, 0.8), heading=(-math.pi, math.pi)
        ),
    )