#!/bin/bash

# Override path to have onnxruntime use conda's libraries.
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Run the controlledSim.py script with arguments
python scripts/controlledSim.py \
    --task Wheeldog-Rl-v0-play \
    --onnx-path logs/rsl_rl/wheelDog_Blind_Managed/2026-02-28_11-30-50/exported/policy.onnx \
    --headless \
    --livestream 2 \
    --enable_cameras
