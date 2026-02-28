"""Script for a controlled simulation of an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Controlled simulation of an RL agent from RSL-RL.")
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Wheeldog-Rl-v0-play", help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--real-time", action="store_true", default=True, help="Run in real-time, if possible.")

# Onnx file path.
parser.add_argument("--onnx-path", type=str, required=True)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)

# parse the arguments
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# Setup WebRTC streaming. 
args_cli.headless = True
args_cli.livestream = 2
args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Library imports.
from isaaclab_tasks.utils import parse_env_cfg
import onnxruntime as ort
import gymnasium as gym
import numpy as np
import torch

# Import the module to register the gym environment. 
import wheelDog_RL.tasks  # noqa: F401


def main():
    """Random actions agent with Isaac Lab environment."""
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs
    )

    # Load ONNX model
    providers = ['CUDAExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
    session = ort.InferenceSession(args_cli.onnx_path, providers=providers)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # Load the environment.
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs
    )
    env = gym.make(args_cli.task, cfg=env_cfg)

    # reset environment
    env.reset()
    actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
    obs, rew, terminated, truncated, info = env.step(actions)

    # simulate environment
    while simulation_app.is_running():
        # run everything under inference mode
        with torch.inference_mode():
            # Hardcoded command for testing
            # commands = np.array([0.6, 0.0, 0.0], dtype=np.float32)

            # Insert command into policy observations
            policyObs = obs["policy"]
            # policyObs[..., :3] = torch.from_numpy(commands).to(policyObs.device, dtype=policyObs.dtype)

            # Run policy inference
            input_data = policyObs.cpu().numpy() if isinstance(policyObs, torch.Tensor) else policyObs
            onnxActions = session.run([output_name], {input_name: input_data})[0]
            actions = onnxActions

            # Convert back to torch if needed
            if isinstance(actions, np.ndarray):
                actions = torch.from_numpy(actions).to(env.unwrapped.device)

            # Apply actions
            obs, rew, terminated, truncated, info = env.step(actions)

    # close the simulator
    env.close()

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
