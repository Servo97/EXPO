"""Training script for EXPO on Franka Kitchen tasks using Minari datasets"""
#! /usr/bin/env python
import os
import pickle
import warnings

# Suppress all deprecation warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import gymnasium as gym
import minari
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
from expo.agents import SACLearner
from expo.data import ReplayBuffer
from expo.data import RoboReplayBuffer
from expo.data.dataset import Dataset
from expo.evaluation import evaluate, evaluate_robo

FLAGS = flags.FLAGS

flags.DEFINE_string("project_name", "EXPO_franka_kitchen", "wandb project name.")
flags.DEFINE_string("run_name", None, "wandb run name. If None, wandb will auto-generate a name.")
flags.DEFINE_string("env_name", "D4RL/kitchen/partial-v2", "Kitchen dataset name (e.g., D4RL/kitchen/partial-v2).")
flags.DEFINE_float("offline_ratio", 0.5, "Offline ratio.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("eval_episodes", 100, "Number of episodes used for evaluation.")
flags.DEFINE_integer("log_interval", 1000, "Logging interval.")
flags.DEFINE_integer("eval_interval", 10000, "Eval interval.")
flags.DEFINE_integer("offline_eval_interval", 10000, "Eval interval.")
flags.DEFINE_integer("batch_size", 256, "Mini batch size.")
flags.DEFINE_integer("max_steps", int(3e6), "Number of training steps.")
flags.DEFINE_integer(
    "start_training", int(1e4), "Number of training steps to start training."
)
flags.DEFINE_string("output_dir", "data/user_data/ssaxena2/expo/logs", "Directory for saving logs and checkpoints.")
flags.DEFINE_integer("pretrain_steps", 1000000, "Number of offline updates.")
flags.DEFINE_boolean("tqdm", True, "Use tqdm progress bar.")
flags.DEFINE_boolean("save_video", False, "Save videos during evaluation.")
flags.DEFINE_boolean("checkpoint_model", False, "Save agent checkpoint on evaluation.")
flags.DEFINE_boolean(
    "checkpoint_buffer", False, "Save agent replay buffer on evaluation."
)
flags.DEFINE_integer("utd_ratio", 1, "Update to data ratio.")
flags.DEFINE_boolean(
    "pretrain_edit", False, "Whether to pretrain edit policy."
)
flags.DEFINE_boolean(
    "pretrain_q", False, "Whether to pretrain Q-function."
)
flags.DEFINE_integer("horizon", 4, "Action chunking horizon.")

config_flags.DEFINE_config_file(
    "config",
    "configs/sac_config.py",
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)

flags.DEFINE_bool('clip_bc', True, "Clip BC to 50%")
flags.DEFINE_integer('success_buffer_batch_size', 256, "batch size of the success buffer.")
flags.DEFINE_bool('use_success_buffer', False, "whether to use the success buffer in the bc loss")


def flatten_obs_dict(obs_dict):
    """
    Flattens a dictionary observation into a single numpy array.
    This is for the live environment wrapper.
    """
    goal_keys = sorted(obs_dict['desired_goal'].keys())

    desired_goal_parts = [np.atleast_1d(obs_dict['desired_goal'][key]) for key in goal_keys]
    desired_goal = np.concatenate(desired_goal_parts, axis=-1)

    achieved_goal_parts = [np.atleast_1d(obs_dict['achieved_goal'][key]) for key in goal_keys]
    achieved_goal = np.concatenate(achieved_goal_parts, axis=-1)

    return np.concatenate([obs_dict['observation'], desired_goal, achieved_goal], axis=-1)


class FrankaKitchenWrapper(gym.Wrapper):
    """
    A wrapper for the FrankaKitchen-v1 environment that flattens the
    dictionary-based observation into a single vector for the RL agent.
    Also converts gymnasium API to gym API for compatibility with EXPO.
    """
    def __init__(self, env):
        super().__init__(env)
        sample_obs, _ = env.reset()
        flattened_sample = flatten_obs_dict(sample_obs)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=flattened_sample.shape, dtype=np.float32
        )
        self._return_info = False

    def reset(self, **kwargs):
        # Check if caller expects the new API (observation, info) or old API (observation)
        obs_dict, info = self.env.reset(**kwargs)
        info['tasks_completed'] = 0
        info['success'] = 0.0
        flattened_obs = flatten_obs_dict(obs_dict)

        # For compatibility: return just observation (gym API)
        # EXPO's evaluation and training code will handle unpacking if needed
        if self._return_info:
            return flattened_obs, info
        return flattened_obs

    def step(self, action):
        obs_dict, reward, terminated, truncated, info = self.env.step(action)
        reward = reward - 7
        if reward == -3:
            reward = 0
        info['reward'] = reward
        info['tasks_completed'] = len(info.get('episode_task_completions', []))
        info['success'] = float(terminated)
        done = terminated or truncated
        return flatten_obs_dict(obs_dict), reward, done, info


def make_env(env_name: str, seed=None):
    """Creates the FrankaKitchen-v1 environment and applies the flattening wrapper."""
    dataset = minari.load_dataset(env_name, download=True)
    env = dataset.recover_environment(terminate_on_tasks_completed=True, max_episode_steps=600)
    env = FrankaKitchenWrapper(env)
    if seed is not None:
        env.reset(seed=seed)
    return env


def get_dataset(env_name: str):
    """
    Loads and processes the Franka Kitchen Minari dataset, handling its complex
    observation structure and potential data corruption issues.
    """
    dataset = minari.load_dataset(env_name, download=True)
    total_steps = dataset.total_steps
    print(f"Loading Minari dataset '{env_name}' with {total_steps} timesteps...")

    # Robustly find the first valid episode for shape inference.
    first_valid_episode = None
    for episode in dataset:
        if len(episode.actions) > 0 and len(episode.observations['observation']) > 0:
            first_valid_episode = episode
            break
    if first_valid_episode is None:
        raise ValueError("Dataset contains no valid episodes.")

    achieved_goal_t0 = {
        key: val[0] for key, val in first_valid_episode.observations['achieved_goal'].items()
    }
    desired_goal_t0 = {
        key: val[0] for key, val in first_valid_episode.observations['desired_goal'].items()
    }

    sample_obs_dict = {
        'observation': first_valid_episode.observations['observation'][0],
        'achieved_goal': achieved_goal_t0,
        'desired_goal': desired_goal_t0,
    }

    sample_flat_obs = flatten_obs_dict(sample_obs_dict)
    sample_action = first_valid_episode.actions[0]

    all_obs = np.zeros((total_steps, sample_flat_obs.shape[0]), dtype=np.float32)
    all_actions = np.zeros((total_steps, sample_action.shape[0]), dtype=np.float32)
    all_rewards = np.zeros(total_steps, dtype=np.float32)
    all_terminals = np.zeros(total_steps, dtype=np.float32)

    current_idx = 0
    for episode in tqdm.tqdm(dataset, desc=f"Processing '{env_name}'"):
        if len(episode.actions) == 0 or len(episode.observations['observation']) == 0:
            continue

        num_steps = min(len(episode.actions), len(episode.observations['observation']))
        idx_slice = slice(current_idx, current_idx + num_steps)

        # Vectorized flattening of the structured observation arrays.
        obs_part = episode.observations['observation'][:num_steps]

        # Handle the case where goal observations are a dict of arrays
        goal_keys = sorted(episode.observations['desired_goal'].keys())

        desired_goal_parts = [episode.observations['desired_goal'][key][:num_steps] for key in goal_keys]
        desired_goal_flat = np.concatenate(desired_goal_parts, axis=1)

        achieved_goal_parts = [episode.observations['achieved_goal'][key][:num_steps] for key in goal_keys]
        achieved_goal_flat = np.concatenate(achieved_goal_parts, axis=1)

        episode_obs_flat = np.concatenate([obs_part, desired_goal_flat, achieved_goal_flat], axis=1)

        # Populate the pre-allocated arrays.
        all_obs[idx_slice] = episode_obs_flat.astype(np.float32)
        all_actions[idx_slice] = episode.actions[:num_steps].astype(np.float32)
        all_rewards[idx_slice] = episode.rewards[:num_steps].astype(np.float32)
        dones = np.logical_or(episode.terminations, episode.truncations)
        all_terminals[idx_slice] = dones[:num_steps].astype(np.float32)
        current_idx += num_steps

    # Trim arrays if corrupted data was skipped.
    if current_idx < total_steps:
        all_obs = all_obs[:current_idx]
        all_actions = all_actions[:current_idx]
        all_rewards = all_rewards[:current_idx]
        all_terminals = all_terminals[:current_idx]

    # Standard method for creating next_observations.
    next_observations = np.roll(all_obs, -1, axis=0)
    episode_ends = np.where(all_terminals == 1.0)[0]
    for end_idx in episode_ends:
        if end_idx < len(all_terminals) - 1:
            next_observations[end_idx] = all_obs[end_idx]

    masks = 1.0 - all_terminals
    dones = all_terminals.astype(bool)

    dataset_dict = {
        'observations': all_obs,
        'actions': all_actions,
        'rewards': all_rewards,
        'terminals': all_terminals,
        'masks': masks,
        'next_observations': next_observations,
        'dones': dones,
    }

    return Dataset(dataset_dict)


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

    # Create Franka Kitchen environment using Minari
    env = make_env(FLAGS.env_name, seed=FLAGS.seed)
    eval_env = make_env(FLAGS.env_name, seed=FLAGS.seed + 42)

    # Load Minari dataset
    ds = get_dataset(FLAGS.env_name)

    # Get example observation and action
    example_observation = ds.dataset_dict['observations'][0]
    example_action = ds.dataset_dict['actions'][0]

    kwargs = dict(FLAGS.config)
    model_cls = kwargs.pop("model_cls")
    kwargs["horizon"] = FLAGS.horizon
    agent = globals()[model_cls].create(
        FLAGS.seed, example_observation, example_action, **kwargs
    )

    replay_buffer = RoboReplayBuffer(
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

        agent, update_info = agent.update_offline(batch, FLAGS.utd_ratio, FLAGS.pretrain_q, FLAGS.pretrain_edit)

        if i % FLAGS.log_interval == 0:
            for k, v in update_info.items():
                wandb.log({f"offline-training/{k}": v}, step=i)

        if i % FLAGS.offline_eval_interval == 0:
            eval_info = evaluate_robo(agent, eval_env, num_episodes=FLAGS.eval_episodes, max_traj_len=600)

            for k, v in eval_info.items():
                wandb.log({f"offline-evaluation/{k}": v}, step=i)
                wandb.log({f"evaluation/{k}": v}, step=i)
            print(eval_info)
            print(f"Offline evaluation return: {eval_info.get('return', 0.0)}")
            if FLAGS.clip_bc and eval_info.get("return", 0.0) >= 2.0:  # Kitchen success threshold
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
        # Loop completed without break (no BC clipping)
        actual_pretrain_steps = FLAGS.pretrain_steps

    print(f"Completed {actual_pretrain_steps} pretraining steps", flush=True)

    observation, done = env.reset(), False
    action_queue = []
    action_dim = env.action_space.shape[0]
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
            action = env.action_space.sample()
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

        if not done:
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
            if is_success > 0 and hasattr(replay_buffer, '_traj_success_mask'):
                num_successful = replay_buffer._traj_success_mask[:replay_buffer._size].sum()
                print(f"[Step {i}] SUCCESS! Total successful transitions in buffer: {num_successful}/{replay_buffer._size}", flush=True)

            trajectory_buffer = []
            observation, done = env.reset(), False
            action_queue = []

            if 'episode' in info:
                for k, v in info["episode"].items():
                    wandb.log({f"training/{k}": v}, step=i + actual_pretrain_steps)

        if i >= FLAGS.start_training:
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

            # Success buffer BC regularization
            if FLAGS.use_success_buffer and FLAGS.horizon > 1 and hasattr(replay_buffer, '_traj_success_mask'):
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
                        # Not enough successful trajectories yet
                        if i % FLAGS.log_interval == 0:
                            print(f"[Step {i}] ✗ Failed to sample from success buffer: {e}", flush=True)
                        pass

            if i % FLAGS.log_interval == 0:
                for k, v in update_info.items():
                    wandb.log({f"training/{k}": v}, step=i + actual_pretrain_steps)

        if i % FLAGS.eval_interval == 0:
            eval_info = evaluate_robo(
                agent,
                eval_env,
                num_episodes=FLAGS.eval_episodes,
                max_traj_len=600,
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
