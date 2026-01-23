#!/usr/bin/env bash
set -euo pipefail

echo "Starting EXPO SageMaker train.sh on $(hostname) at $(date)"

# ---- Defaults (override via SageMaker Estimator environment) ----
: "${EXPO_SCRIPT:=expo_transport.sh}"     # e.g. expo_square.sh, expo_toolhang.sh, ...
: "${SEED:=$(date +%Y%m%d%H%M%S)}"       # timestamp seed
: "${USER:=unknown_user}"                # your scripts use $USER for /data/user_data/$USER/...
: "${CUDA_VISIBLE_DEVICES:=0}"

# Extra args to forward to your EXPO bash script (must be "--k=v" form)
: "${EXPO_ARGS:=}"                       # e.g. "--use_success_buffer=True --offline_steps=100000"

# Common runtime vars (can be overridden per job)
: "${MUJOCO_GL:=egl}"
: "${XLA_PYTHON_CLIENT_PREALLOCATE:=false}"

export MUJOCO_GL
export XLA_PYTHON_CLIENT_PREALLOCATE
export CUDA_VISIBLE_DEVICES
export USER

# If WANDB_API_KEY is missing, don't hard-crash: run offline.
if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "WANDB_API_KEY not set -> running W&B in offline/disabled mode."
  export WANDB_MODE=offline
  export WANDB_DISABLED=true
fi

REPO_ROOT="/opt/ml/code"
EXPO_DIR="${REPO_ROOT}/scripts/expo"
SCRIPT_PATH="${EXPO_DIR}/${EXPO_SCRIPT}"

if [[ ! -f "${SCRIPT_PATH}" ]]; then
  echo "ERROR: EXPO script not found: ${SCRIPT_PATH}"
  echo "Available scripts:"
  ls -1 "${EXPO_DIR}" || true
  exit 2
fi

# Ensure dirs expected by your scripts exist
mkdir -p "/data/user_data/${USER}/expo"
chmod -R 777 "/data/user_data/${USER}" || true

echo "Repo root: ${REPO_ROOT}"
echo "EXPO script: ${SCRIPT_PATH}"
echo "SEED: ${SEED}"
echo "EXPO_ARGS: ${EXPO_ARGS}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
nvidia-smi -L || true

# Your scripts do: cd ../.. to reach repo root.
cd "${EXPO_DIR}"

SEED_ARG="--seed=${SEED}"

# shellcheck disable=SC2086
exec /bin/bash "./${EXPO_SCRIPT}" ${SEED_ARG} ${EXPO_ARGS}
