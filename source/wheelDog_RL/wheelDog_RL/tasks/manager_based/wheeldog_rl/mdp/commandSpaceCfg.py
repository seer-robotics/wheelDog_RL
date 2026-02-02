# Library imports. 
import math

# Isaac Lab imports
from isaaclab.envs import mdp as isaac_mdp
from isaaclab.utils import configclass

@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    # https://github.com/isaac-sim/IsaacLab/discussions/2620
    base_velocity = isaac_mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(6.0, 10.0),
        rel_standing_envs=0.01,
        rel_heading_envs=0.99,
        heading_command=False,
        debug_vis=True,
        ranges=isaac_mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.2, 1.2), lin_vel_y=(-0.4, 0.4), ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi)
        ),
    )