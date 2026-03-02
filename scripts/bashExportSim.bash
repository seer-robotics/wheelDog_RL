#!/bin/bash

# Override path to have onnxruntime use conda's libraries.
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Run the exportSim.py script with arguments
python scripts/exportSim.py \
    --task Wheeldog-Rl-v0-play \
    --onnx-path logs/rsl_rl/wheelDog_Blind_Managed/2026-03-02_10-20-54/exported/policy.onnx \
    --headless \
    --livestream 2 \
    --enable_cameras
