#!/bin/bash
#SBATCH --partition=h200
#SBATCH --job-name=expo_kitchen_complete
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=100:00:00
#SBATCH --output=/scratch/sws0/user/ssaxena/logs/expo_kitchen_complete_%j.out
#SBATCH --error=/scratch/sws0/user/ssaxena/logs/expo_kitchen_complete_%j.out


echo "Working directory: $(pwd)"
echo "Job ID: $SLURM_JOB_ID"
echo "Job submitted from: $SLURM_SUBMIT_DIR"
echo "Running on node: $SLURMD_NODENAME"

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1  # Disable Python output buffering for real-time logs
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ssaxena/.mujoco/mujoco210/bin:/home/ssaxena/.mujoco/mjpro150/bin:/usr/lib/nvidia

# Set cache directories to scratch to avoid disk quota issues
export HF_HOME=/scratch/sws0/user/ssaxena/.cache/huggingface
export TRANSFORMERS_CACHE=/scratch/sws0/user/ssaxena/.cache/huggingface
export HF_DATASETS_CACHE=/scratch/sws0/user/ssaxena/.cache/huggingface
mkdir -p $HF_HOME

export MPLCONFIGDIR=/scratch/sws0/user/ssaxena/.cache/matplotlib
mkdir -p $MPLCONFIGDIR

export PIP_CACHE_DIR=/scratch/sws0/user/ssaxena/.cache/pip
export UV_CACHE_DIR=/scratch/sws0/user/ssaxena/.cache/uv
mkdir -p $PIP_CACHE_DIR $UV_CACHE_DIR

# Activate conda environment
source /scratch/sws0/user/ssaxena/miniforge3/etc/profile.d/conda.sh && conda activate expo

# Add conda environment's lib path for CuDNN
export LD_LIBRARY_PATH=/scratch/sws0/user/ssaxena/miniforge3/envs/expo/lib/python3.9/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH

# Default parameters (can be overridden via command line arguments)
seed=0
env_name="D4RL/kitchen/complete-v2"
run_name="expo_kitchen_complete_${seed}"
utd_ratio=20
start_training=5000
max_steps=3000000
pretrain_steps=500000
project_name="EXPO_franka_kitchen"
horizon=4
batch_size=256
eval_episodes=100
log_interval=5000
eval_interval=20000
offline_eval_interval=20000
offline_ratio=0.5
clip_bc=True
use_success_buffer=False

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

echo "Starting EXPO training job on $(hostname) at $(date)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Using GPU: $(nvidia-smi -L)"
echo "Seed: $seed"
echo "Run name: $run_name"

# Create necessary directories
mkdir -p /scratch/sws0/user/ssaxena/logs

# Change to project directory
cd /scratch/sws0/user/ssaxena/EXPO

# Run the training
echo "Running training with parameters:"
echo "  env_name: $env_name"
echo "  seed: $seed"
echo "  run_name: $run_name"
echo "  utd_ratio: $utd_ratio"
echo "  start_training: $start_training"
echo "  max_steps: $max_steps"
echo "  pretrain_steps: $pretrain_steps"
echo "  project_name: $project_name"
echo "  horizon: $horizon"

python train_franka_kitchen.py \
    --env_name=$env_name \
    --seed=$seed \
    --run_name=$run_name \
    --utd_ratio=$utd_ratio \
    --start_training=$start_training \
    --max_steps=$max_steps \
    --pretrain_steps=$pretrain_steps \
    --batch_size=$batch_size \
    --eval_episodes=$eval_episodes \
    --log_interval=$log_interval \
    --eval_interval=$eval_interval \
    --offline_eval_interval=$offline_eval_interval \
    --horizon=$horizon \
    --offline_ratio=$offline_ratio \
    --clip_bc=$clip_bc \
    --use_success_buffer=$use_success_buffer \
    --config=configs/expo_config.py \
    --config.backup_entropy=False \
    --config.hidden_dims="(512, 512, 512, 512)" \
    --config.N=8 \
    --config.n_edit_samples=8 \
    --config.edit_action_scale=0.05 \
    --config.discount=0.99 \
    --project_name=$project_name \
    --checkpoint_model=True

echo "Job completed at $(date)"
