#!/bin/bash
# Debug script for Push-T training (run without SLURM)
# Usage: bash run_pusht_debug.sh [--seed=0] [--other_param=value]

echo "Working directory: $(pwd)"
echo "Running on node: $(hostname)"

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONWARNINGS="ignore::DeprecationWarning"

source ~/miniconda/etc/profile.d/conda.sh && conda activate expo
cd /home/mananaga/EXPO/

export C_INCLUDE_PATH=$CONDA_PREFIX/include:$C_INCLUDE_PATH
export LIBRARY_PATH=$CONDA_PREFIX/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export FLAX_USE_ORBAX_CHECKPOINTING=0
XLA_PYTHON_CLIENT_PREALLOCATE=false

# Default parameters (can be overridden via command line arguments)
# Using OGPO pusht hyperparameters where applicable
seed=0
run_name="expo_pusht_debug_${seed}"
utd_ratio=1  # From OGPO: utd_warmup=1, utd_online=1
start_training=5000  # From EXPO square default
max_steps=2000000  # From OGPO: online_steps=2000000
pretrain_steps=500000  # From OGPO: offline_steps=500000
horizon=4  # From OGPO: horizon_length=4
batch_size=256  # Default batch size
eval_episodes=50  # More than OGPO's 2 (with 64 parallel envs), but reasonable for sequential
offline_ratio=0.5
project_name="EXPO_paper"
use_success_buffer=False  # From OGPO: use_success_buffer=True
success_buffer_batch_size=256  # Default value
clip_bc=True  # From OGPO: clip_bc=True

# Parse command line arguments to override defaults
for arg in "$@"; do
  case $arg in
    --*=*)
      key="${arg%%=*}"      # part before '='
      value="${arg#*=}"     # part after '='
      key="${key#--}"       # strip leading '--'
      eval "$key=\"$value\"" # set variable dynamically
      ;;
    *)
      echo "Unknown argument: $arg"
      ;;
  esac
done

echo "Starting EXPO Push-T training job on $(hostname) at $(date)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Using GPU: $(nvidia-smi -L)"
echo "Seed: $seed"
echo "Run name: $run_name"

# Run the training
echo "Running training with parameters:"
echo "  env_name: pusht-keypoints-v0"
echo "  seed: $seed"
echo "  run_name: $run_name"
echo "  utd_ratio: $utd_ratio"
echo "  start_training: $start_training"
echo "  max_steps: $max_steps"
echo "  pretrain_steps: $pretrain_steps"
echo "  horizon: $horizon"
echo "  batch_size: $batch_size"
echo "  eval_episodes: $eval_episodes"
echo "  offline_ratio: $offline_ratio"
echo "  project_name: $project_name"
echo "  use_success_buffer: $use_success_buffer"
echo "  clip_bc: $clip_bc"

python train_pusht.py \
    --env_name=pusht-keypoints-v0 \
    --seed=$seed \
    --run_name=$run_name \
    --utd_ratio=$utd_ratio \
    --start_training=$start_training \
    --max_steps=$max_steps \
    --pretrain_steps=$pretrain_steps \
    --horizon=$horizon \
    --batch_size=$batch_size \
    --eval_episodes=$eval_episodes \
    --offline_ratio=$offline_ratio \
    --config=configs/expo_config.py \
    --config.backup_entropy=False \
    --config.hidden_dims="(512, 512, 512, 512)" \
    --config.discount=0.95 \
    --config.tau=0.05 \
    --config.N=8 \
    --config.n_edit_samples=8 \
    --config.edit_action_scale=0.05 \
    --project_name=$project_name \
    --use_success_buffer=$use_success_buffer \
    --success_buffer_batch_size=$success_buffer_batch_size \
    --clip_bc=$clip_bc

echo "Job completed at $(date)"

