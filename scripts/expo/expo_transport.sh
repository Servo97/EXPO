# Set environment variables
export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Activate your conda environment (uncomment if needed)
# source ~/miniconda3/etc/profile.d/conda.sh && conda activate /data/user_data/ssaxena2/tdmpc2-jax
source ~/miniconda3/etc/profile.d/conda.sh && conda activate ogpo

# Set up variables
device=0
seeds=$1
log=True
env_name=transport-mh-low_dim
run_name="transport_flow"
wandb_name="transport_flow"
horizon_length=8
bc_coeff=1
pg_coeff=1
utd_warmup=1
utd_online=1
discount=0.999
SAVE_DIR="/data/user_data/$USER/ogpo"
restore_actor_path=None
restore_critic_path=None
ep_resume=0
offline_steps=1000000
calql_steps=0
q_warmup_steps=0
online_steps=5500000
best_of_n=8
use_constant_scheduler_for_bc=False
num_qs=10
q_agg=mean
subsample_bon=True
offline_ratio=0.0
clip_bc=False
use_success_buffer=True
plot_q_vs_mc=False
error_correct_sde_to_ode=True
use_denoiser=True

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

export CUDA_VISIBLE_DEVICES=$device

echo "Starting QPPO training job on $(hostname) at $(date)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Using GPU: $(nvidia-smi -L)"

cd ../..

# Run the training
seeds="${seeds//,/ }"
for seed in $seeds
do
    echo "Running with seed: $seed"
    python main_ogpo.py \
    --agent=agents/reinflow_calql.py \
    --project=OGPO_transport_ablations \
    --run_group=transport_flow \
    --run_name=$run_name \
    --wandb_name=$wandb_name \
    --log=$log \
    --restore_actor_path=$restore_actor_path \
    --restore_critic_path=$restore_critic_path \
    --ep_resume=$ep_resume \
    --offline_steps=$offline_steps \
    --calql_steps=$calql_steps \
    --q_warmup_steps=$q_warmup_steps \
    --online_steps=$online_steps \
    --clip_bc=$clip_bc \
    --use_success_buffer=$use_success_buffer \
    --plot_q_vs_mc=$plot_q_vs_mc \
    --best_of_n=$best_of_n \
    --agent.q_agg=$q_agg \
    --agent.subsample_bon=$subsample_bon \
    --agent.num_qs=$num_qs \
    --seed=$seed \
    --discount=$discount \
    --utd_warmup=$utd_warmup \
    --utd_online=$utd_online \
    --env_name=$env_name \
    --sparse=False \
    --horizon_length=$horizon_length \
    --agent.flow_steps=10 \
    --n_eval_envs=64 \
    --eval_episodes=2 \
    --log_interval=10000 \
    --eval_interval=50000 \
    --eval_interval_bc=200000 \
    --save_interval=2000000 \
    --save_dir=$SAVE_DIR \
    --agent.clip_epsilon=0.01 \
    --agent.entropy_coeff=0.0 \
    --agent.min_noise_std=0.01 \
    --agent.max_noise_std=0.01 \
    --agent.use_constant_noise=True \
    --agent.constant_noise_std=0.01 \
    --agent.ppo_batch_size=256 \
    --agent.grpo_num_samples=32 \
    --offline_ratio=$offline_ratio \
    --agent.use_bc_regularization=True \
    --agent.use_constant_scheduler_for_bc=$use_constant_scheduler_for_bc \
    --agent.bc_coeff=$bc_coeff \
    --agent.pg_coeff=$pg_coeff \
    --start_training=40000 \
    --agent.actor_scheduler=cosine \
    --agent.critic_scheduler=constant \
    --agent.value_hidden_dims=512,512,512,512,512 \
    --agent.error_correct_sde_to_ode=$error_correct_sde_to_ode \
    --agent.use_denoiser=$use_denoiser \
    --agent.policy_type=flow
done

echo "Job completed at $(date)"
