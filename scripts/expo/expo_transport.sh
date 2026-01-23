# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Activate your conda environment (uncomment if needed)
# source ~/miniconda3/etc/profile.d/conda.sh && conda activate /data/user_data/ssaxena2/tdmpc2-jax
source ~/miniconda3/etc/profile.d/conda.sh && conda activate expo

# Default parameters (can be overridden via command line arguments)
seed=0
run_name="expo_transport_${seed}"
utd_ratio=20
start_training=20000
max_steps=6000000
pretrain_steps=1000000
dataset_dir="/home/nagababa/.robomimic/transport/mh/"
project_name="EXPO_paper"
horizon=8
batch_size=256
eval_episodes=64
log_interval=10000
eval_interval=25000
offline_eval_interval=50000
offline_ratio=0.0
clip_bc=False
use_success_buffer=False

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

echo "Starting QPPO training job on $(hostname) at $(date)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Using GPU: $(nvidia-smi -L)"

cd ../..

python train_robo.py \
    --env_name=transport \
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
    --config.hidden_dims="(512, 512, 512, 512, 512)" \
    --config.N=8 \
    --config.n_edit_samples=8 \
    --config.edit_action_scale=0.1 \
    --config.discount=0.999 \
    --project_name=$project_name \
    --dataset_dir=$dataset_dir

echo "Job completed at $(date)"
