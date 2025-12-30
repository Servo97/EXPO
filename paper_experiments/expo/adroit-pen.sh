#!/bin/bash
#SBATCH --partition=general
#SBATCH --job-name=ogpo_training
#SBATCH --gres=gpu:1
#SBATCH --constraint=VRAM_48GB
#SBATCH --cpus-per-task=20
#SBATCH --mem=40G
#SBATCH --time=48:00:00
#SBATCH --output=/home/mananaga/logs/%j/.out
#SBATCH --error=/home/mananaga/logs/%j/.out
mkdir -p logs

echo "Working directory: $(pwd)"
echo "Job ID: $SLURM_JOB_ID"
echo "Job submitted from: $SLURM_SUBMIT_DIR"
echo "Running on node: $SLURMD_NODENAME"

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTHONWARNINGS="ignore::DeprecationWarning"

# Activate conda environment (all other env vars set automatically via activation script)
source ~/miniconda/etc/profile.d/conda.sh && conda activate expo-adroit
cd /home/mananaga/EXPO/

# Optional: Disable Orbax checkpointing if needed
# export FLAX_USE_ORBAX_CHECKPOINTING=0

# Default parameters (can be overridden via command line arguments)
seed=0
run_name="expo_pen_debug_${seed}"
utd_ratio=20
start_training=0
max_steps=2000000
expo=True
project_name="EXPO_paper"

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

echo "Starting EXPO Adroit Pen training job on $(hostname) at $(date)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Using GPU: $(nvidia-smi -L)"
echo "Seed: $seed"
echo "Run name: $run_name"

# Run the training
echo "Running training with parameters:"
echo "  env_name: pen-binary-v0"
echo "  seed: $seed"
echo "  run_name: $run_name"
echo "  utd_ratio: $utd_ratio"
echo "  start_training: $start_training"
echo "  max_steps: $max_steps"
echo "  expo: $expo"
echo "  project_name: $project_name"

python train_finetuning.py \
    --env_name=pen-binary-v0 \
    --seed=$seed \
    --run_name=$run_name \
    --utd_ratio=$utd_ratio \
    --start_training=$start_training \
    --max_steps=$max_steps \
    --expo=$expo \
    --config=configs/expo_config.py \
    --config.backup_entropy=False \
    --config.hidden_dims="(256, 256, 256)" \
    --config.N=8 \
    --config.n_edit_samples=8 \
    --config.edit_action_scale=0.7 \
    --config.actor_drop=0.1 \
    --project_name=$project_name

echo "Job completed at $(date)"

