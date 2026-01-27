#!/bin/bash

# Make this file executable:
# chmod a+x scripts/rsl_rl/trainBlindWheelDog.sh

# Run the train.py script with arguments
python scripts/rsl_rl/train.py \
    --task Wheeldog-Rl-v0 \
    --headless \
    --livestream 1 \
    --enable_cameras

# /home/renda/miniforge3/envs/bin_isaaclab/lib/python3.11/site-packages/rsl_rl/utils/utils.py:245: UserWarning: The observation configuration dictionary 'obs_groups' must contain the 'policy' key. As an observation group with the name 'policy' was found, this is assumed to be the observation set. Consider adding the 'policy' key to the 'obs_groups' dictionary for clarity. This behavior will be removed in a future version.
#   warnings.warn(
# /home/renda/miniforge3/envs/bin_isaaclab/lib/python3.11/site-packages/rsl_rl/utils/utils.py:283: UserWarning: The observation configuration dictionary 'obs_groups' must contain the 'critic' key. As an observation group with the name 'critic' was found, this is assumed to be the observation set. Consider adding the 'critic' key to the 'obs_groups' dictionary for clarity. This behavior will be removed in a future version.
