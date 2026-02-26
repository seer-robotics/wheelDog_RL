#!/bin/bash

python scripts/rsl_rl/play.py \
    --num_envs 8 \
    --task Crippledog-Rl-v0-play \
    --headless \
    --livestream 2 \
    --enable_cameras
