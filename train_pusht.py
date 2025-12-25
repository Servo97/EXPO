"""Training script for Push-T environment in EXPO"""
#! /usr/bin/env python
import os
import pickle
import warnings

# Suppress all deprecation warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import numpy as np
import tqdm
from absl import app, flags
import jax

try:
    from flax.training import checkpoints
except:
    print("Not loading checkpointing functionality.")
from ml_collections import config_flags

import wandb
from expo.agents import EXPOLearner
from expo.data import ReplayBuffer
from expo.evaluation import evaluate

FLAGS = flags.FLAGS

flags.DEFINE_string("project_name", "EXPO_paper", "wandb project name.")
flags.DEFINE_string("run_name", None, "wandb run name. If None, wandb will auto-generate a name.")
flags.DEFINE_string("env_name", "pusht-keypoints-v0", "Push-T environment name.")
flags.DEFINE_float("offline_ratio", 0.5, "Offline ratio.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("eval_episodes", 50, "Number of episodes used for evaluation.")
flags.DEFINE_integer("log_interval", 1000, "Logging interval.")
flags.DEFINE_integer("eval_interval", 20000, "Eval interval.")
flags.DEFINE_integer("offline_eval_interval", 20000, "Eval interval.")
flags.DEFINE_integer("batch_size", 256, "Mini batch size.")
flags.DEFINE_integer("max_steps", int(2e6), "Number of training steps.")
flags.DEFINE_integer("start_training", int(1e4), "Number of training steps to start training.")
flags.DEFINE_string("output_dir", "/data/user_data/mananaga/expo/logs", "Directory for saving logs and checkpoints.")
flags.DEFINE_integer("pretrain_steps", 500000, "Number of offline updates.")
flags.DEFINE_boolean("tqdm", True, "Use tqdm progress bar.")
flags.DEFINE_boolean("save_video", False, "Save videos during evaluation.")
flags.DEFINE_boolean("checkpoint_model", False, "Save agent checkpoint on evaluation.")
flags.DEFINE_boolean("checkpoint_buffer", False, "Save agent replay buffer on evaluation.")
flags.DEFINE_integer("utd_ratio", 1, "Update to data ratio.")
flags.DEFINE_integer("horizon", 4, "Action chunking horizon.")

config_flags.DEFINE_config_file(
    "config",
    "configs/expo_config.py",
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)

flags.DEFINE_bool('clip_bc', True, "Clip BC to 50%")
flags.DEFINE_integer('success_buffer_batch_size', 256, "batch size of the success buffer.")
flags.DEFINE_bool('use_success_buffer', True, "whether to use the success buffer in the bc loss")


def combine(one_dict, other_dict):
    combined = {}

    for k, v in one_dict.items():
        if isinstance(v, dict):
            combined[k] = combine(v, other_dict[k])
        else:
            # Handle edge cases where one batch is empty
            if v.shape[0] == 0:
                combined[k] = other_dict[k]
            elif other_dict[k].shape[0] == 0:
                combined[k] = v
            else:
                # Interleave the two batches
                tmp = np.empty(
                    (v.shape[0] + other_dict[k].shape[0], *v.shape[1:]), dtype=v.dtype
                )
                tmp[0::2] = v
                tmp[1::2] = other_dict[k]
                combined[k] = tmp

    return combined


def create_success_buffer_batch(replay_buffer, batch_size, seq_len, discount):
    """Sample a batch from successful trajectories only."""
    success_batch = replay_buffer.sample_sequence(
        batch_size=batch_size,
        sequence_length=seq_len,
        discount=discount,
        success_only=True,
    )
    return success_batch


class PushTD4RLDataset:
    """Dataset wrapper for Push-T that provides D4RL-like interface."""
    
    def __init__(self, env_name='pusht-keypoints-v0', num_data=0):
        """
        Args:
            env_name: Name of the Push-T environment
            num_data: Number of data points to use (0 = use all)
        """
        # Import from EXPO's pusht environment
        from expo.envs.pusht_utils import get_dataset
        
        # Load dataset
        print(f"Loading Push-T dataset for {env_name}...")
        dataset = get_dataset(env_name)
        
        # Limit dataset size if requested
        if num_data > 0 and num_data < len(dataset['observations']):
            for key in dataset.keys():
                dataset[key] = dataset[key][:num_data]
        
        self.dataset_dict = dataset
        self._size = len(dataset['observations'])
        print(f"Loaded Push-T dataset with {self._size} transitions")
        
    def sample(self, batch_size):
        """Sample a batch of transitions."""
        indices = np.random.randint(0, self._size, size=batch_size)
        
        batch = {
            'observations': self.dataset_dict['observations'][indices],
            'actions': self.dataset_dict['actions'][indices],
            'rewards': self.dataset_dict['rewards'][indices],
            'next_observations': self.dataset_dict['next_observations'][indices],
            'masks': self.dataset_dict['masks'][indices],
            'terminals': self.dataset_dict['terminals'][indices],
        }
        return batch
    
    def sample_sequence(self, batch_size, sequence_length, discount):
        """Sample sequences for action chunking."""
        # Find valid starting points (not too close to episode end)
        valid_starts = []
        
        # Track episode boundaries
        episode_starts = [0]
        for i in range(1, self._size):
            if self.dataset_dict['terminals'][i-1]:
                episode_starts.append(i)
        episode_starts.append(self._size)
        
        # Find valid starting positions
        for ep_start, ep_end in zip(episode_starts[:-1], episode_starts[1:]):
            ep_len = ep_end - ep_start
            if ep_len >= sequence_length:
                valid_starts.extend(range(ep_start, ep_end - sequence_length + 1))
        
        valid_starts = np.array(valid_starts)
        
        # Sample starting indices
        start_indices = valid_starts[np.random.randint(0, len(valid_starts), size=batch_size)]
        
        # Build sequences
        observations = []
        actions = []
        rewards = []
        masks = []
        next_observations = []
        valid = []
        
        for start_idx in start_indices:
            seq_obs = []
            seq_act = []
            seq_rew = []
            seq_mask = []
            seq_valid = []
            
            for offset in range(sequence_length):
                idx = start_idx + offset
                seq_obs.append(self.dataset_dict['observations'][idx])
                seq_act.append(self.dataset_dict['actions'][idx])
                seq_rew.append(self.dataset_dict['rewards'][idx])
                seq_mask.append(self.dataset_dict['masks'][idx])
                seq_valid.append(1.0)  # All valid within sequence
            
            observations.append(seq_obs)
            actions.append(seq_act)
            rewards.append(seq_rew)
            masks.append(seq_mask)
            valid.append(seq_valid)
            next_observations.append(self.dataset_dict['next_observations'][start_idx + sequence_length - 1])
        
        batch = {
            'observations': np.array(observations)[:, 0],  # Take first obs
            'actions': np.array(actions),
            'rewards': np.array(rewards),
            'masks': np.array(masks),
            'valid': np.array(valid),
            'next_observations': np.array(next_observations),
        }
        
        return batch


def main(_):
    assert FLAGS.offline_ratio >= 0.0 and FLAGS.offline_ratio <= 1.0

    wandb.init(project=FLAGS.project_name, name=FLAGS.run_name)
    wandb.config.update(FLAGS)

    exp_prefix = f"s{FLAGS.seed}_{FLAGS.pretrain_steps}pretrain"
    if hasattr(FLAGS.config, "critic_layer_norm") and FLAGS.config.critic_layer_norm:
        exp_prefix += "_LN"

    log_dir = os.path.join(FLAGS.output_dir, exp_prefix)

    if FLAGS.checkpoint_model:
        chkpt_dir = os.path.join(log_dir, "checkpoints")
        os.makedirs(chkpt_dir, exist_ok=True)

    if FLAGS.checkpoint_buffer:
        buffer_dir = os.path.join(log_dir, "buffers")
        os.makedirs(buffer_dir, exist_ok=True)

    # Create Push-T environment
    from expo.envs.pusht_utils import make_env
    
    env = make_env(FLAGS.env_name, seed=FLAGS.seed)
    eval_env = make_env(FLAGS.env_name, seed=FLAGS.seed + 1000)
    
    # Load dataset
    ds = PushTD4RLDataset(env_name=FLAGS.env_name)
    
    # Get example data
    example_observation = ds.dataset_dict['observations'][0]
    example_action = ds.dataset_dict['actions'][0]
    
    max_traj_len = 300  # Standard for Push-T

    kwargs = dict(FLAGS.config)
    model_cls = kwargs.pop("model_cls")
    kwargs["horizon"] = FLAGS.horizon
    agent = globals()[model_cls].create(
        FLAGS.seed, example_observation, example_action, **kwargs
    )

    replay_buffer = ReplayBuffer(
        example_observation, example_action, FLAGS.max_steps
    )
    replay_buffer.seed(FLAGS.seed)

    actual_pretrain_steps = 0  # Track actual steps completed (in case of early BC clipping)
    for i in tqdm.tqdm(range(0, FLAGS.pretrain_steps), smoothing=0.1, disable=not FLAGS.tqdm):
        if FLAGS.horizon > 1:
            offline_batch = ds.sample_sequence(FLAGS.batch_size * FLAGS.utd_ratio, sequence_length=FLAGS.horizon, discount=FLAGS.config.discount)
            offline_batch = jax.tree_map(lambda x: x.reshape((FLAGS.utd_ratio, FLAGS.batch_size) + x.shape[1:]), offline_batch)
        else:
            offline_batch = ds.sample(FLAGS.batch_size * FLAGS.utd_ratio)
        
        batch = {}
        for k, v in offline_batch.items():
            batch[k] = v

        if FLAGS.horizon > 1:
            # Flatten actions: (utd_ratio, batch_size, horizon, action_dim) -> (utd_ratio * batch_size, horizon * action_dim)
            batch['actions'] = batch['actions'].reshape(batch['actions'].shape[0] * batch['actions'].shape[1], -1)
            # Take n-step reward and mask: (utd_ratio, batch_size, horizon) -> (utd_ratio * batch_size,)
            batch['rewards'] = batch['rewards'][:, :, -1].reshape(-1)
            batch['masks'] = batch['masks'][:, :, -1].reshape(-1)
            # Flatten observations and next_observations: (utd_ratio, batch_size, ...) -> (utd_ratio * batch_size, ...)
            batch['observations'] = batch['observations'].reshape(-1, *batch['observations'].shape[2:])
            batch['next_observations'] = batch['next_observations'].reshape(-1, *batch['next_observations'].shape[2:])

        agent, update_info = agent.update_offline(batch, FLAGS.utd_ratio, False, False)

        if i % FLAGS.log_interval == 0:
            for k, v in update_info.items():
                wandb.log({f"offline-training/{k}": v}, step=i)

        if i % FLAGS.offline_eval_interval == 0:
            eval_info = evaluate(agent, eval_env, num_episodes=FLAGS.eval_episodes)

            for k, v in eval_info.items():
                wandb.log({f"offline-evaluation/{k}": v}, step=i)
                wandb.log({f"evaluation/{k}": v}, step=i)
            
            print(eval_info)
            print(f"Offline evaluation return: {eval_info.get('return', 0.0)}")
            
            if FLAGS.clip_bc and eval_info.get("return", 0.0) >= 0.45:
                if FLAGS.checkpoint_model:
                    try:
                        checkpoints.save_checkpoint(
                            chkpt_dir, agent, step=i, keep=20, overwrite=True
                        )
                    except Exception as e:
                        print(f"Could not save model checkpoint during BC clipping: {e}")

                print("breaking due to bc clipping (offline pretraining)")
                actual_pretrain_steps = i + 1
                break
    else:
        actual_pretrain_steps = FLAGS.pretrain_steps

    print(f"Completed {actual_pretrain_steps} pretraining steps", flush=True)

    observation, done = env.reset(), False
    log_returns = 0
    action_queue = []
    action_dim = example_action.shape[-1]
    trajectory_buffer = []

    # Print configuration
    if FLAGS.use_success_buffer:
        print("=" * 60, flush=True)
        print(f"SUCCESS BUFFER ENABLED", flush=True)
        print(f"  - Success buffer batch size: {FLAGS.success_buffer_batch_size}", flush=True)
        print(f"  - Horizon: {FLAGS.horizon}", flush=True)
        print(f"  - Min required successful transitions: {FLAGS.success_buffer_batch_size * FLAGS.horizon}", flush=True)
        print("=" * 60, flush=True)

    for i in tqdm.tqdm(
        range(0, FLAGS.max_steps + 1), smoothing=0.1, disable=not FLAGS.tqdm
    ):
        if i < FLAGS.start_training:
            action = np.random.uniform(-1, 1, size=(action_dim,))
        else:
            if len(action_queue) == 0:
                if FLAGS.horizon > 1:
                    # Sample chunk of actions
                    action_chunk, agent = agent.sample_actions(observation)
                    action_chunk = np.array(action_chunk).reshape(-1, action_dim)
                    for act in action_chunk:
                        action_queue.append(act)
                else:
                    action, agent = agent.sample_actions(observation)
                    action_queue.append(action)
            
            action = action_queue.pop(0)

        next_observation, reward, done, info = env.step(action)

        if not done or "TimeLimit.truncated" in info:
            mask = 1.0
        else:
            mask = 0.0

        transition = dict(
            observations=observation,
            actions=action,
            rewards=reward,
            masks=mask,
            dones=done,
            next_observations=next_observation,
        )
        trajectory_buffer.append(transition)
        log_returns += reward
        observation = next_observation

        if done:
            # Mark all transitions in trajectory with success
            is_success = float(info.get('success', 0.0))
            score = is_success if is_success != 0 else 1e-9
            for traj_transition in trajectory_buffer:
                traj_transition['is_success'] = is_success
                traj_transition['score'] = score
                replay_buffer.insert(traj_transition)
            
            # Print success buffer status every successful episode
            if is_success > 0:
                num_successful = replay_buffer._traj_success_mask[:replay_buffer._size].sum()
                print(f"[Step {i}] SUCCESS! Total successful transitions in buffer: {num_successful}/{replay_buffer._size}", flush=True)
            
            trajectory_buffer = []
            observation, done = env.reset(), False
            action_queue = []

            wandb.log({f"training/episode_return": log_returns}, step=i + actual_pretrain_steps)
            wandb.log({f"training/success": is_success}, step=i + actual_pretrain_steps)
            log_returns = 0

        if i >= FLAGS.start_training:
            # Check if buffer has enough data for sequence sampling
            if FLAGS.horizon > 1:
                min_required_size = max(
                    int(FLAGS.batch_size * FLAGS.utd_ratio * (1 - FLAGS.offline_ratio)) * FLAGS.horizon,
                    FLAGS.horizon
                )
            else:
                min_required_size = int(FLAGS.batch_size * FLAGS.utd_ratio * (1 - FLAGS.offline_ratio))

            if replay_buffer._size >= min_required_size:
                # Proceed with sampling and update
                if FLAGS.horizon > 1:
                    online_batch = replay_buffer.sample_sequence(
                        int(FLAGS.batch_size * FLAGS.utd_ratio * (1 - FLAGS.offline_ratio)),
                        sequence_length=FLAGS.horizon,
                        discount=FLAGS.config.discount
                    )
                else:
                    online_batch = replay_buffer.sample(
                        int(FLAGS.batch_size * FLAGS.utd_ratio * (1 - FLAGS.offline_ratio))
                    )

                if FLAGS.horizon > 1:
                    offline_batch = ds.sample_sequence(
                        int(FLAGS.batch_size * FLAGS.utd_ratio * FLAGS.offline_ratio),
                        sequence_length=FLAGS.horizon,
                        discount=FLAGS.config.discount
                    )
                else:
                    offline_batch = ds.sample(
                        int(FLAGS.batch_size * FLAGS.utd_ratio * FLAGS.offline_ratio)
                    )

                batch = combine(offline_batch, online_batch)

                if FLAGS.horizon > 1:
                     # Flatten actions: (B, T, D) -> (B, T*D)
                     batch['actions'] = batch['actions'].reshape(batch['actions'].shape[0], -1)
                     # Take n-step reward and mask: (B, T) -> (B,)
                     batch['rewards'] = batch['rewards'][:, -1]
                     batch['masks'] = batch['masks'][:, -1]

                agent, update_info = agent.update(batch, FLAGS.utd_ratio)
            else:
                # Skip update - buffer still filling up
                if i % FLAGS.log_interval == 0:
                    print(f"[Step {i}] Skipping update: buffer size {replay_buffer._size} < {min_required_size} required", flush=True)
                update_info = {}

            # Success buffer BC regularization
            if FLAGS.use_success_buffer and FLAGS.horizon > 1:
                num_successful = replay_buffer._traj_success_mask[:replay_buffer._size].sum()
                min_required = FLAGS.success_buffer_batch_size * FLAGS.horizon
                
                if i % FLAGS.log_interval == 0:
                    print(f"[Step {i}] Success buffer check: {num_successful} successful / {min_required} required", flush=True)
                
                if num_successful >= min_required:
                    try:
                        success_batch = create_success_buffer_batch(
                            replay_buffer,
                            FLAGS.success_buffer_batch_size,
                            FLAGS.horizon,
                            FLAGS.config.discount
                        )
                        # Flatten actions for BC
                        success_batch_flat = dict(success_batch)
                        success_batch_flat['actions'] = success_batch['actions'].reshape(
                            success_batch['actions'].shape[0], -1
                        )
                        success_batch_flat['observations'] = success_batch['observations']
                        
                        agent, bc_info = agent.update_actor_bc(success_batch_flat)
                        update_info.update({f"success_bc/{k}": v for k, v in bc_info.items()})
                        
                        if i % FLAGS.log_interval == 0:
                            print(f"[Step {i}] ✓ Applied success buffer BC update, loss: {bc_info.get('bc_loss', 'N/A'):.4f}", flush=True)
                    except ValueError as e:
                        if i % FLAGS.log_interval == 0:
                            print(f"[Step {i}] ✗ Failed to sample from success buffer: {e}", flush=True)
                        pass

            if i % FLAGS.log_interval == 0:
                for k, v in update_info.items():
                    wandb.log({f"training/{k}": v}, step=i + actual_pretrain_steps)

        if i % FLAGS.eval_interval == 0:
            eval_info = evaluate(
                agent,
                eval_env,
                num_episodes=FLAGS.eval_episodes,
                save_video=FLAGS.save_video,
            )

            for k, v in eval_info.items():
                wandb.log({f"evaluation/{k}": v}, step=i + actual_pretrain_steps)

            if FLAGS.checkpoint_model:
                try:
                    checkpoints.save_checkpoint(
                        chkpt_dir, agent, step=i, keep=20, overwrite=True
                    )
                except:
                    print("Could not save model checkpoint.")

            if FLAGS.checkpoint_buffer:
                try:
                    with open(os.path.join(buffer_dir, f"buffer"), "wb") as f:
                        pickle.dump(replay_buffer, f, pickle.HIGHEST_PROTOCOL)
                except:
                    print("Could not save agent buffer.")


if __name__ == "__main__":
    app.run(main)

