#!/bin/bash

# Make this file executable:
# chmod a+x scripts/rsl_rl/bashPlayBlindWheelDog.sh

# Run the train.py script with arguments
python scripts/rsl_rl/play.py \
    --num_envs 32 \
    --task Wheeldog-Rl-v0-play \
    --load_run 2026-02-25_18-11-38 \
    --headless \
    --livestream 2 \
    --enable_cameras

# python scripts/rsl_rl/play.py \
#     --num_envs 8 \
#     --task Wheeldog-Rl-v0-play \
#     --headless \
#     --livestream 2 \
#     --enable_cameras
