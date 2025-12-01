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
export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONWARNINGS="ignore::DeprecationWarning"

source ~/miniconda/etc/profile.d/conda.sh && conda activate expo
cd /home/mananaga/EXPO/

export C_INCLUDE_PATH=$CONDA_PREFIX/include:$C_INCLUDE_PATH
export LIBRARY_PATH=$CONDA_PREFIX/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
XLA_PYTHON_CLIENT_PREALLOCATE=false

python train_finetuning.py --env_name=hammer-binary-v0 \
                                --seed=$1 \
                                --run_name=expo_hammer_$1 \
                                --utd_ratio=20 \
                                --start_training 5000 \
                                --max_steps 2000000 \
                                --expo=True \
                                --config=configs/expo_config.py \
                                --config.backup_entropy=False \
                                --config.hidden_dims="(512, 512, 512, 512)" \
                                --config.num_min_qs=1 \
                                --config.N=8 \
                                --config.n_edit_samples=8 \
                                --config.edit_action_scale=0.05 \
                                --project_name=EXPO_paper

