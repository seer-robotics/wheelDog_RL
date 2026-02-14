# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##


gym.register(
    id="Wheeldog-Rl-v0",
    entry_point="wheelDog_RL.tasks.manager_based.wheeldog_rl.envEntry:WheelDog_BlindLocomotionEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.robotCfg:WheelDog_BlindLocomotionEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)

gym.register(
    id="Wheeldog-Rl-v0-play",
    entry_point="wheelDog_RL.tasks.manager_based.wheeldog_rl.envEntry:WheelDog_BlindLocomotionEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.robotCfg:WheelDog_BlindLocomotionEnvPlayCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)

gym.register(
    id="Crippledog-Rl-v0",
    entry_point="wheelDog_RL.tasks.manager_based.wheeldog_rl.envEntry:CrippleDog_BlindLocomotionEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.robotCfg:CrippleDog_BlindLocomotionEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg_crippleDog",
    },
)
