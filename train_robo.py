"""Starter code from the RLPD repository https://github.com/ikostrikov/rlpd"""
#! /usr/bin/env python
import os
import pickle
import glob
import warnings

# Suppress all deprecation warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import d4rl
import d4rl.gym_mujoco
import d4rl.locomotion
import dmcgym
import gym
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
from expo.data.d4rl_datasets import D4RLDataset

import mimicgen
from robomimic.utils.dataset import SequenceDataset
from expo.data.robomimic_datasets import (
    process_robomimic_dataset, get_mimicgen_env, get_robomimic_env, RoboD4RLDataset, 
    ENV_TO_HORIZON_MAP, MIMICGEN_ENV_TO_HORIZON_MAP, OBS_KEYS
)
import cloudpickle as pickle

try:
    from expo.data.binary_datasets import BinaryDataset
except:
    print("not importing binary dataset")
from expo.evaluation import evaluate, evaluate_diffusion, evaluate_robo
from expo.wrappers import wrap_gym

FLAGS = flags.FLAGS

flags.DEFINE_string("project_name", "expo", "wandb project name.")
flags.DEFINE_string("env_name", "halfcheetah-expert-v2", "D4rl dataset name.")
flags.DEFINE_float("offline_ratio", 0.0, "Offline ratio.")
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
flags.DEFINE_integer(
    "num_data", 0, "Number of training steps to start training."
)
flags.DEFINE_string("dataset_dir", "halfcheetah-expert-v2", "D4rl dataset name.")
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
    "binary_include_bc", True, "Whether to include BC data in the binary datasets."
)
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



def combine(one_dict, other_dict):
    combined = {}

    for k, v in one_dict.items():
        if isinstance(v, dict):
            combined[k] = combine(v, other_dict[k])
        else:
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

    wandb.init(project=FLAGS.project_name)
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


    if FLAGS.env_name in ENV_TO_HORIZON_MAP:

        # Handle custom dataset directory path
        if FLAGS.dataset_dir != '' and FLAGS.dataset_dir != 'mh' and FLAGS.dataset_dir != 'ph' and os.path.isdir(FLAGS.dataset_dir):
            # Custom directory path provided - find hdf5 file
            hdf5_files = glob.glob(os.path.join(FLAGS.dataset_dir, 'low_dim_*.hdf5'))
            if not hdf5_files:
                raise FileNotFoundError(f"No low_dim_*.hdf5 file found in {FLAGS.dataset_dir}")
            dataset_path = hdf5_files[0]  # Use first matching hdf5 file
            seq_dataset = SequenceDataset(hdf5_path=dataset_path,
                                        obs_keys=OBS_KEYS,
                                        dataset_keys=("actions", "rewards", "dones"),
                                        hdf5_cache_mode="all",
                                        load_next_obs=True)
            dataset = process_robomimic_dataset(seq_dataset)
        elif FLAGS.dataset_dir != '' and FLAGS.dataset_dir != 'mh' and FLAGS.dataset_dir != 'ph' and os.path.isfile(FLAGS.dataset_dir):
            # Custom pickle file path provided
            with open(FLAGS.dataset_dir, 'rb') as handle:
                dataset = pickle.load(handle)
            dataset['rewards'] = dataset['rewards'].squeeze()
            dataset['terminals'] = dataset['terminals'].squeeze()
            dataset_path = f'./robomimic/datasets/{FLAGS.env_name}/ph/low_dim_v141.hdf5'  # Default for env creation
        elif FLAGS.dataset_dir == 'ph':
            dataset_path = f'./robomimic/datasets/{FLAGS.env_name}/ph/low_dim_v141.hdf5'
            seq_dataset = SequenceDataset(hdf5_path=dataset_path,
                                        obs_keys=OBS_KEYS,
                                        dataset_keys=("actions", "rewards", "dones"),
                                        hdf5_cache_mode="all",
                                        load_next_obs=True)
            dataset = process_robomimic_dataset(seq_dataset)
        else:
            # Default to mh
            dataset_path = f'./robomimic/datasets/{FLAGS.env_name}/mh/low_dim_v141.hdf5'
            seq_dataset = SequenceDataset(hdf5_path=dataset_path,
                                        obs_keys=OBS_KEYS,
                                        dataset_keys=("actions", "rewards", "dones"),
                                        hdf5_cache_mode="all",
                                        load_next_obs=True)
            dataset = process_robomimic_dataset(seq_dataset)
        ds = RoboD4RLDataset(env=None, num_data=FLAGS.num_data, custom_dataset=dataset)


        example_observation = ds.dataset_dict['observations'][0][np.newaxis]
        example_action = ds.dataset_dict['actions'][0][np.newaxis]
        env = get_robomimic_env(dataset_path, example_action, FLAGS.env_name)
        eval_env = get_robomimic_env(dataset_path, example_action, FLAGS.env_name)
        max_traj_len = ENV_TO_HORIZON_MAP[FLAGS.env_name]

    else:

        dataset_path = f'./mimicgen/datasets/source/{FLAGS.env_name}.hdf5'
        if FLAGS.dataset_dir != '' and FLAGS.dataset_dir != 'mh' and FLAGS.dataset_dir != 'ph':
            dataset_dir = FLAGS.dataset_dir
        else:
            dataset_dir = f'./mimicgen/datasets/{FLAGS.env_name}/dataset.pkl'
        with open(dataset_dir, 'rb') as handle:
            dataset = pickle.load(handle)
        
        dataset['rewards'] = dataset['rewards'].squeeze()
        dataset['terminals'] = dataset['terminals'].squeeze()


        ds = RoboD4RLDataset(env=None, custom_dataset=dataset, num_data=FLAGS.num_data)
        example_observation = ds.dataset_dict['observations'][0][np.newaxis]
        example_action = ds.dataset_dict['actions'][0][np.newaxis]


        env = get_mimicgen_env(dataset_path, example_action, FLAGS.env_name)
        eval_env = get_mimicgen_env(dataset_path, example_action, FLAGS.env_name)
        max_traj_len = MIMICGEN_ENV_TO_HORIZON_MAP[FLAGS.env_name]


    kwargs = dict(FLAGS.config)
    model_cls = kwargs.pop("model_cls")
    kwargs["horizon"] = FLAGS.horizon
    agent = globals()[model_cls].create(
        FLAGS.seed, example_observation.squeeze(), example_action.squeeze(), **kwargs
    )

    replay_buffer = RoboReplayBuffer(
        example_observation.squeeze(), example_action.squeeze(), FLAGS.max_steps
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
            if "antmaze" in FLAGS.env_name and k == "rewards":
                batch[k] -= 1

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
            eval_info = evaluate_robo(agent, eval_env, max_traj_len=max_traj_len, num_episodes=FLAGS.eval_episodes)


            for k, v in eval_info.items():
                wandb.log({f"offline-evaluation/{k}": v}, step=i)
                wandb.log({f"evaluation/{k}": v}, step=i)
            # wandb.log({}, commit=True)  # Force flush
            print(eval_info)
            print(f"Offline evaluation success rate: {eval_info.get('return', 0.0)}")
            if FLAGS.clip_bc and eval_info.get("return", 0.0) >= 0.45:
                if FLAGS.checkpoint_model:
                    try:
                        checkpoints.save_checkpoint(
                            chkpt_dir, agent, step=i, keep=20, overwrite=True
                        )
                    except Exception as e:
                        print(f"Could not save model checkpoint during BC clipping: {e}")

                print("breaking due to bc clipping (offline pretraining)")
                actual_pretrain_steps = i + 1  # +1 because loop index starts at 0
                break
    else:
        # Loop completed without break (no BC clipping)
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
            action = np.random.uniform(-1, 1, size=(example_action.shape[1], ))
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

            for k, v in info["episode"].items():
                decode = {"r": "return", "l": "length"}
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
                max_traj_len=max_traj_len, 
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
