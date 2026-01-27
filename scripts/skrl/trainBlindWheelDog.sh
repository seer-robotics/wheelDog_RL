#!/bin/bash

# Make this file executable:
# chmod a+x scripts/skrl/trainBlindWheelDog.sh

# Run the train.py script with arguments
python scripts/skrl/train.py \
    --task Wheeldog-Rl-v0 \
    --headless \
    --livestream 1 \
    --enable_cameras
