#!/bin/bash
#SBATCH --partition=preempt
#SBATCH --job-name=expo_pusht_sparse_slow
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
# Note: CUDA_VISIBLE_DEVICES is managed by SLURM via --gres=gpu:1
# Do not override it manually, or it may point to an unallocated GPU
export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONWARNINGS="ignore::DeprecationWarning"

source ~/miniconda/etc/profile.d/conda.sh
conda activate expo

cd /home/mananaga/EXPO/

# Ensure conda libraries are in the path, but system CUDA takes priority
export C_INCLUDE_PATH=$CONDA_PREFIX/include:$C_INCLUDE_PATH
export LIBRARY_PATH=$CONDA_PREFIX/lib:$LIBRARY_PATH
# Put system CUDA paths at the FRONT of LD_LIBRARY_PATH for GPU access
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/nvidia:/usr/lib64/nvidia:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export FLAX_USE_ORBAX_CHECKPOINTING=0

# Point JAX to system CUDA installation (conda env doesn't have CUDA libs)
# Check for CUDA in common locations
if [ -d "/usr/local/cuda" ]; then
    export XLA_FLAGS="--xla_gpu_cuda_data_dir=/usr/local/cuda"
    echo "Using CUDA from: /usr/local/cuda"
elif [ -d "/usr/local/cuda-11" ]; then
    export XLA_FLAGS="--xla_gpu_cuda_data_dir=/usr/local/cuda-11"
    echo "Using CUDA from: /usr/local/cuda-11"
elif [ -d "/usr/local/cuda-12" ]; then
    export XLA_FLAGS="--xla_gpu_cuda_data_dir=/usr/local/cuda-12"
    echo "Using CUDA from: /usr/local/cuda-12"
else
    echo "WARNING: Could not find system CUDA installation"
fi

# Default parameters (can be overridden via command line arguments)
# Using OGPO pusht hyperparameters where applicable
seed=0
rew_fn="sparse_slow"
run_name="expo_pusht_${rew_fn}_${seed}"
output_dir="/data/user_data/mananaga/expo/logs_sparse_slow"
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

echo "Starting EXPO Push-T training job (SPARSE_SLOW reward) on $(hostname) at $(date)"
echo "SLURM allocated GPUs: $CUDA_VISIBLE_DEVICES"
echo ""
echo "GPU Status:"
nvidia-smi -L
nvidia-smi -q -d COMPUTE | grep -A 2 "Compute Mode" || echo "Could not query compute mode"
echo ""
echo "CUDA Environment:"
echo "  CONDA_PREFIX: $CONDA_PREFIX"
echo "  LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "  XLA_FLAGS: $XLA_FLAGS"
which nvcc || echo "nvcc not found in PATH"
echo ""
echo "Checking CUDA libraries in conda env:"
ls -la $CONDA_PREFIX/lib/libcuda* 2>/dev/null || echo "No libcuda* in $CONDA_PREFIX/lib"
ls -la $CONDA_PREFIX/lib/libcudart* 2>/dev/null || echo "No libcudart* in $CONDA_PREFIX/lib"
echo ""
echo "Python/JAX CUDA test:"
python -c "import jax; print(f'JAX version: {jax.__version__}'); print(f'JAX devices: {jax.devices()}'); print(f'JAX local devices: {jax.local_devices()}')" 2>&1
echo ""
echo "Seed: $seed"
echo "Reward function: $rew_fn"
echo "Run name: $run_name"

# Run the training
echo "Running training with parameters:"
echo "  env_name: pusht-keypoints-v0"
echo "  seed: $seed"
echo "  rew_fn: $rew_fn"
echo "  run_name: $run_name"
echo "  output_dir: $output_dir"
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
    --rew_fn=$rew_fn \
    --run_name=$run_name \
    --output_dir=$output_dir \
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
