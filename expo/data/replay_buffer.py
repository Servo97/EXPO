import collections
from typing import Dict, Optional, Union

import gym
import gym.spaces
import jax
import numpy as np
from flax.core import frozen_dict

from expo.data.dataset import Dataset, DatasetDict


def _init_replay_dict(
    obs_space: gym.Space, capacity: int
) -> Union[np.ndarray, DatasetDict]:
    if isinstance(obs_space, gym.spaces.Box):
        return np.empty((capacity, *obs_space.shape), dtype=obs_space.dtype)
    elif isinstance(obs_space, gym.spaces.Dict):
        data_dict = {}
        for k, v in obs_space.spaces.items():
            data_dict[k] = _init_replay_dict(v, capacity)
        return data_dict
    else:
        raise TypeError()


def _insert_recursively(
    dataset_dict: DatasetDict, data_dict: DatasetDict, insert_index: int
):
    if isinstance(dataset_dict, np.ndarray):
        dataset_dict[insert_index] = data_dict
    elif isinstance(dataset_dict, dict):
        assert dataset_dict.keys() == data_dict.keys()
        for k in dataset_dict.keys():
            _insert_recursively(dataset_dict[k], data_dict[k], insert_index)
    else:
        raise TypeError()


def _init_robo_replay_dict(
    example_observation: gym.Space, capacity: int
) -> Union[np.ndarray, DatasetDict]:
    # if isinstance(obs_space, gym.spaces.Box):
    return np.empty((capacity, *example_observation.shape), dtype=example_observation.dtype)


class RoboReplayBuffer(Dataset):
    def __init__(
        self,
        example_observation,
        example_action,
        capacity: int,
    ):

        observation_data = _init_robo_replay_dict(example_observation, capacity)
        next_observation_data = _init_robo_replay_dict(example_observation, capacity)
        dataset_dict = dict(
            observations=observation_data,
            next_observations=next_observation_data,
            actions=np.empty((capacity, *example_action.shape), dtype=example_action.dtype),
            rewards=np.empty((capacity,), dtype=np.float32),
            masks=np.empty((capacity,), dtype=np.float32),
            dones=np.empty((capacity,), dtype=bool),
            is_success=np.zeros((capacity,), dtype=np.float32),
            score=np.zeros((capacity,), dtype=np.float32),
        )

        super().__init__(dataset_dict)

        self._size = 0
        self._capacity = capacity
        self._insert_index = 0
        self._traj_success_mask = np.zeros(capacity, dtype=np.bool_)

    def __len__(self) -> int:
        return self._size

    def insert(self, data_dict: DatasetDict):
        # Handle is_success and score fields
        data_dict = data_dict.copy()
        if 'is_success' not in data_dict:
            data_dict['is_success'] = 0.0
        if 'score' not in data_dict:
            data_dict['score'] = 1.0 if float(data_dict.get('is_success', 0.0)) > 0 else 1e-9
        
        _insert_recursively(self.dataset_dict, data_dict, self._insert_index)
        self._traj_success_mask[self._insert_index] = (float(data_dict['is_success']) > 0)

        self._insert_index = (self._insert_index + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)
    

    def insert_dataset(self, dataset: Dict[str, np.ndarray]):
        """Insert an entire offline dataset at once."""
        dataset_size = len(dataset['observations'])
        
        if dataset_size <= self._capacity:
            # Dataset fits entirely in buffer
            for key in self.dataset_dict.keys():
                if key in dataset:
                    self.dataset_dict[key][:dataset_size] = dataset[key]
            
            self._size = dataset_size
            self._insert_index = min(self._size + 1, self._capacity)  # Start overwriting from beginning
        else:
            # Dataset is larger than buffer capacity - take random subset
            indices = np.random.choice(dataset_size, self._capacity, replace=False)
            
            for key in self.dataset_dict.keys():
                if key in dataset:
                    self.dataset_dict[key][:] = dataset[key][indices]
            
            self._size = self._capacity
            self._insert_index = 0

    def sample_sequence(self, batch_size, sequence_length, discount, ret_next_act=False, ret_mc=False, filter_fn=None, success_only=None, by_score=False):
        # Determine valid start indices to avoid crossing the insert_index (head of circular buffer)
        valid_indices = []
        
        # Range 1: [0, insert_index - sequence_length]
        # This is valid if insert_index >= sequence_length
        # Note: indices are inclusive for start, so we go up to insert_index - sequence_length
        if self._insert_index >= sequence_length:
            valid_indices.append(np.arange(self._insert_index - sequence_length + 1))
            
        # Range 2: [insert_index, capacity - sequence_length]
        # Only if buffer is full (size == capacity)
        # We can sample starting from insert_index up to capacity - sequence_length
        if self._size == self._capacity:
            if self._capacity - self._insert_index >= sequence_length:
                valid_indices.append(np.arange(self._insert_index, self._capacity - sequence_length + 1))
                
        if not valid_indices:
             # Buffer too small
             raise ValueError("Replay buffer too small for requested sequence length.")
        
        valid_indices = np.concatenate(valid_indices)
        
        # Apply success filter if requested
        if success_only is not None:
            success_mask = self._traj_success_mask[valid_indices]
            if success_only:
                valid_indices = valid_indices[success_mask]
            else:
                valid_indices = valid_indices[~success_mask]
            
            if len(valid_indices) == 0:
                raise ValueError("No transitions found matching success filter")
        
        # Sample from valid indices
        idxs = np.random.choice(valid_indices, size=batch_size)
        
        # 2) Build a (B, T) index matrix and gather everything at once
        T = sequence_length
        offs = np.arange(T, dtype=np.int64)[None, :]          # (1, T)
        seq_idxs = idxs[:, None].astype(np.int64) + offs      # (B, T)

        # Helper to get data
        def get_data(key):
            if key in self.dataset_dict:
                return self.dataset_dict[key][seq_idxs]
            return None

        obs_seq         = get_data('observations')      # (B, T, obs_dim)
        next_obs_seq    = get_data('next_observations') # (B, T, obs_dim)
        actions_seq     = get_data('actions')           # (B, T, act_dim)
        rewards_seq     = get_data('rewards')           # (B, T)
        masks_seq       = get_data('masks')             # (B, T)
        terminals_seq   = 1.0 - masks_seq # (B, T)

        # 3) Running mask/terminal over time
        masks_prefix     = np.minimum.accumulate(masks_seq, axis=1)
        terminals_prefix = np.maximum.accumulate(terminals_seq, axis=1)

        # 4) Valid: 1 at i==0; for i>0 it's 1 - terminals_prefix at i-1
        valid = np.ones_like(masks_seq, dtype=np.float32)
        valid[:, 1:] = 1.0 - terminals_prefix[:, :-1]

        # 5) Prefix discounted return
        rdtype = rewards_seq.dtype
        disc_pows = (discount ** np.arange(T)).astype(rdtype, copy=False)  # (T,)
        rewards_prefix = np.cumsum(rewards_seq * disc_pows[None, :], axis=1)

        # 6) "observations" = first frame only
        first_obs = self.dataset_dict['observations'][idxs]  # (B, obs_dim)
        last_obs = next_obs_seq[:, -1, ...]

        result = dict(
            observations=first_obs.copy(),
            actions=actions_seq,
            masks=masks_prefix,
            rewards=rewards_prefix,
            terminals=terminals_prefix,
            valid=valid,
            next_observations=last_obs,
        )
        
        return frozen_dict.freeze(result)


    def get_iterator(self, queue_size: int = 2, sample_args: dict = {}):
        # See https://flax.readthedocs.io/en/latest/_modules/flax/jax_utils.html#prefetch_to_device
        # queue_size = 2 should be ok for one GPU.

        queue = collections.deque()

        def enqueue(n):
            for _ in range(n):
                data = self.sample(**sample_args)
                queue.append(jax.device_put(data))

        enqueue(queue_size)
        while queue:
            yield queue.popleft()
            enqueue(1)


class ReplayBuffer(Dataset):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        capacity: int,
        next_observation_space: Optional[gym.Space] = None,
    ):
        if next_observation_space is None:
            next_observation_space = observation_space

        observation_data = _init_replay_dict(observation_space, capacity)
        next_observation_data = _init_replay_dict(next_observation_space, capacity)
        dataset_dict = dict(
            observations=observation_data,
            next_observations=next_observation_data,
            actions=np.empty((capacity, *action_space.shape), dtype=action_space.dtype),
            rewards=np.empty((capacity,), dtype=np.float32),
            masks=np.empty((capacity,), dtype=np.float32),
            dones=np.empty((capacity,), dtype=bool),
        )

        super().__init__(dataset_dict)

        self._size = 0
        self._capacity = capacity
        self._insert_index = 0

    def __len__(self) -> int:
        return self._size

    def insert(self, data_dict: DatasetDict):
        _insert_recursively(self.dataset_dict, data_dict, self._insert_index)

        self._insert_index = (self._insert_index + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)

    def insert_dataset(self, dataset: Dict[str, np.ndarray]):
        dataset_size = len(dataset['observations'])
        
        if dataset_size <= self._capacity:
            # Dataset fits entirely in buffer
            for key in self.dataset_dict.keys():
                if key in dataset:
                    self.dataset_dict[key][:dataset_size] = dataset[key]
            
            self._size = dataset_size
            self._insert_index = min(self._size + 1, self._capacity)  # Start overwriting from beginning
        else:
            # Dataset is larger than buffer capacity - take random subset
            indices = np.random.choice(dataset_size, self._capacity, replace=False)
            
            for key in self.dataset_dict.keys():
                if key in dataset:
                    self.dataset_dict[key][:] = dataset[key][indices]
            
            self._size = self._capacity
            self._insert_index = 0

    def get_iterator(self, queue_size: int = 2, sample_args: dict = {}):
        # See https://flax.readthedocs.io/en/latest/_modules/flax/jax_utils.html#prefetch_to_device
        # queue_size = 2 should be ok for one GPU.

        queue = collections.deque()

        def enqueue(n):
            for _ in range(n):
                data = self.sample(**sample_args)
                queue.append(jax.device_put(data))

        enqueue(queue_size)
        while queue:
            yield queue.popleft()
            enqueue(1)
