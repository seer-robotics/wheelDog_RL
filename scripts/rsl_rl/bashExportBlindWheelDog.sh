#!/usr/bin/env bash

# Script to export the trained blind locomotion policy for the wheeled quadruped using Isaac Lab + rsl_rl

# Usage examples:
# bash script/rsl_rl/bashExportBlindWheelDog.sh --num_envs 64
# bash script/rsl_rl/bashExportBlindWheelDog.sh --load_run 2025-02-12_17-34-11 --checkpoint model_500.pt
# bash script/rsl_rl/bashExportBlindWheelDog.sh

set -euo pipefail

# ──────────────────────────────────────────────────────────────────────────────
#  Default values
# ──────────────────────────────────────────────────────────────────────────────

TASK_NAME="Wheeldog-Rl-v0-play"
NUM_ENVS=64
LOAD_RUN=""
CHECKPOINT=""

# ──────────────────────────────────────────────────────────────────────────────
#  Parse arguments
# ──────────────────────────────────────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
    case $1 in
        --task)
            TASK_NAME="$2"
            shift 2
            ;;
        --task=*)
            TASK_NAME="${1#*=}"
            shift
            ;;
        --num_envs)
            NUM_ENVS="$2"
            shift 2
            ;;
        --num_envs=*)
            NUM_ENVS="${1#*=}"
            shift
            ;;
        --load_run)
            LOAD_RUN="$2"
            shift 2
            ;;
        --load_run=*)
            LOAD_RUN="${1#*=}"
            shift
            ;;
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --checkpoint=*)
            CHECKPOINT="${1#*=}"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --task <name>              (required) Task name, e.g. WheeledQuadruped"
            echo "  --num_envs <int>           Number of environments (default: 32)"
            echo "  --load_run <directory>     Specific training run folder name"
            echo "                             If omitted, uses the latest run"
            echo "  --checkpoint <file.pt>     Specific checkpoint file"
            echo "                             If omitted, uses the latest checkpoint"
            echo "  -h, --help                 Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            echo "Use --help for usage information." >&2
            exit 1
            ;;
    esac
done

# ──────────────────────────────────────────────────────────────────────────────
#  Validation
# ──────────────────────────────────────────────────────────────────────────────

if [[ -z "${TASK_NAME}" ]]; then
    echo "Error: --task is required" >&2
    echo "Example: $0 --task WheeledQuadrupedRough" >&2
    exit 1
fi

if ! [[ "${NUM_ENVS}" =~ ^[0-9]+$ ]] || [[ "${NUM_ENVS}" -lt 1 ]]; then
    echo "Error: --num_envs must be a positive integer" >&2
    exit 1
fi

# ──────────────────────────────────────────────────────────────────────────────
#  Build command
# ──────────────────────────────────────────────────────────────────────────────

CMD="python scripts/rsl_rl/play.py --headless"
CMD="${CMD} --task ${TASK_NAME}"
CMD="${CMD} --num_envs ${NUM_ENVS}"

if [[ -n "${LOAD_RUN}" ]]; then
    CMD="${CMD} --load_run ${LOAD_RUN}"
fi

if [[ -n "${CHECKPOINT}" ]]; then
    CMD="${CMD} --checkpoint ${CHECKPOINT}"
fi

# ──────────────────────────────────────────────────────────────────────────────
#  Execute
# ──────────────────────────────────────────────────────────────────────────────

echo "Executing: ${CMD}"
echo ""

${CMD}
