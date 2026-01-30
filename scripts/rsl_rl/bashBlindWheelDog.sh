#!/bin/bash

# Make this file executable:
# chmod a+x scripts/rsl_rl/bashBlindWheelDog.sh

# Run the train.py script with arguments
# python scripts/rsl_rl/train.py \
    # --num_envs 4096 \
    # --task Wheeldog-Rl-v0 \
    # --headless \
    # --livestream 2 \
    # --enable_cameras

HYDRA_FULL_ERROR=1 python scripts/rsl_rl/train.py \
    --num_envs 4096 \
    --task Wheeldog-Rl-v0 \
    --headless \
    --livestream 0
