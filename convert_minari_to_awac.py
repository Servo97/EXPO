#!/usr/bin/env python3
"""
Convert Minari D4RL hammer dataset to AWAC format for EXPO.

This script:
1. Loads the Minari D4RL/hammer/expert-v2 dataset
2. Converts it to the AWAC format expected by EXPO
3. Saves hammer2_sparse.npy and hammer_bc_sparse4.npy files
"""

import numpy as np
import minari
from tqdm import tqdm

def convert_minari_to_awac_expert_format(dataset_id, output_path, num_expert_trajectories=25):
    """
    Convert Minari dataset to AWAC expert format (hammer2_sparse.npy).

    AWAC expert format is a numpy array of trajectories, where each trajectory is a dict with:
    - observations: list of dicts with 'state_observation' key
    - actions: list of actions
    - rewards: numpy array of rewards (has one extra entry)
    - next_observations: list of dicts with 'state_observation' key
    - terminals: numpy array of terminals
    - agent_infos: empty list
    - env_infos: empty list
    """
    print(f"Loading Minari dataset '{dataset_id}'...")

    # Download and load dataset
    minari.download_dataset(dataset_id)
    dataset = minari.load_dataset(dataset_id)

    print(f"Total episodes in dataset: {len(dataset)}")
    print(f"Converting first {num_expert_trajectories} episodes to AWAC expert format...")

    awac_trajectories = []

    for i, episode in enumerate(tqdm(dataset, desc="Converting episodes")):
        if i >= num_expert_trajectories:
            break

        num_steps = len(episode.actions)
        if num_steps == 0:
            continue

        # Get observations (excluding the last one which is post-terminal)
        obs = episode.observations[:-1]  # Shape: (num_steps, obs_dim)
        next_obs = episode.observations[1:]  # Shape: (num_steps, obs_dim)
        actions = episode.actions  # Shape: (num_steps, action_dim)
        rewards = episode.rewards  # Shape: (num_steps,)

        # Get terminals
        dones = np.logical_or(episode.terminations, episode.truncations)
        terminals = dones.astype(np.float64)  # Shape: (num_steps,)

        # Convert DENSE rewards to SPARSE/BINARY rewards (like AWAC format)
        # SPARSE reward: -1 for all steps except potentially last step
        # This matches the format in door2_sparse.npy where all rewards are -1
        rewards = np.full(num_steps, -1.0, dtype=np.float64)

        # Convert to AWAC format: observations and next_observations as list of dicts
        obs_list = [{'state_observation': obs[t]} for t in range(num_steps)]
        next_obs_list = [{'state_observation': next_obs[t]} for t in range(num_steps)]
        actions_list = [actions[t] for t in range(num_steps)]

        # AWAC format has one extra reward entry (see comment in binary_datasets.py line 34)
        # We'll just duplicate the last reward
        rewards_with_extra = np.append(rewards, rewards[-1])

        # Add both 'observation' and 'state_observation' keys to match original format
        obs_list_full = [{'observation': obs[t], 'state_observation': obs[t]} for t in range(num_steps)]
        next_obs_list_full = [{'observation': next_obs[t], 'state_observation': next_obs[t]} for t in range(num_steps)]

        trajectory = {
            'observations': obs_list_full,
            'actions': actions_list,
            'rewards': rewards_with_extra,
            'next_observations': next_obs_list_full,
            'terminals': terminals,
            'agent_infos': [],
            'env_infos': []
        }

        awac_trajectories.append(trajectory)

    # Convert to numpy array
    awac_array = np.array(awac_trajectories, dtype=object)

    print(f"\nSaving to {output_path}")
    print(f"  Total trajectories: {len(awac_array)}")
    np.save(output_path, awac_array)
    print(f"  Successfully saved!")

    return awac_array


def convert_minari_to_awac_bc_format(dataset_id, output_path, num_bc_trajectories=500):
    """
    Convert Minari dataset to AWAC BC format (hammer_bc_sparse4.npy).

    AWAC BC format is a Python list of dicts, where each dict represents a trajectory:
    - observations: numpy array (T, obs_dim)
    - actions: numpy array (T, action_dim)
    - rewards: numpy array (T, 1)
    - next_observations: numpy array (T, obs_dim)
    - terminals: numpy array (T, 1) bool
    - agent_infos: empty list
    - env_infos: empty list
    """
    print(f"\nLoading Minari dataset '{dataset_id}' for BC format...")

    # Download and load dataset
    minari.download_dataset(dataset_id)
    dataset = minari.load_dataset(dataset_id)

    print(f"Converting {min(num_bc_trajectories, len(dataset))} episodes to AWAC BC format...")

    awac_bc_list = []

    for i, episode in enumerate(tqdm(dataset, desc="Converting BC episodes")):
        if i >= num_bc_trajectories:
            break

        num_steps = len(episode.actions)
        if num_steps == 0:
            continue

        # Get observations (excluding the last one which is post-terminal)
        obs = episode.observations[:-1]  # Shape: (num_steps, obs_dim)
        next_obs = episode.observations[1:]  # Shape: (num_steps, obs_dim)
        actions = episode.actions  # Shape: (num_steps, action_dim)
        rewards = episode.rewards  # Shape: (num_steps,)

        # Get terminals
        dones = np.logical_or(episode.terminations, episode.truncations)
        terminals = dones.astype(bool)  # Shape: (num_steps,)

        # Convert DENSE rewards to SPARSE/BINARY rewards (like AWAC format)
        # SPARSE reward: -1 for all steps (matching door_bc_sparse4.npy format)
        rewards = np.full(num_steps, -1.0, dtype=np.float64)

        # BC format: everything as numpy arrays with proper shapes
        trajectory = {
            'observations': obs.astype(np.float64),  # (T, obs_dim)
            'actions': actions.astype(np.float32),  # (T, action_dim)
            'rewards': rewards.reshape(-1, 1).astype(np.int64),  # (T, 1)
            'next_observations': next_obs.astype(np.float64),  # (T, obs_dim)
            'terminals': terminals.reshape(-1, 1),  # (T, 1)
            'agent_infos': [],
            'env_infos': []
        }

        awac_bc_list.append(trajectory)

    print(f"\nSaving to {output_path}")
    print(f"  Total trajectories: {len(awac_bc_list)}")
    np.save(output_path, awac_bc_list)
    print(f"  Successfully saved!")

    return awac_bc_list


if __name__ == "__main__":
    import os

    dataset_id = "D4RL/hammer/expert-v2"
    output_dir = os.path.expanduser("~/.datasets/awac-data/")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Convert to expert format (hammer2_sparse.npy)
    expert_output = os.path.join(output_dir, "hammer2_sparse.npy")
    convert_minari_to_awac_expert_format(dataset_id, expert_output, num_expert_trajectories=25)

    # Convert to BC format (hammer_bc_sparse4.npy)
    bc_output = os.path.join(output_dir, "hammer_bc_sparse4.npy")
    convert_minari_to_awac_bc_format(dataset_id, bc_output, num_bc_trajectories=500)

    print("\n" + "="*60)
    print("Conversion complete!")
    print(f"  Expert data: {expert_output}")
    print(f"  BC data: {bc_output}")
    print("="*60)
