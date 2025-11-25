from functools import partial
from random import sample
from typing import Dict, Iterable, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from flax.core import frozen_dict
from gym.utils import seeding

from expo.types import DataType

DatasetDict = Dict[str, DataType]


def _check_lengths(dataset_dict: DatasetDict, dataset_len: Optional[int] = None) -> int:
    for v in dataset_dict.values():
        if isinstance(v, dict):
            dataset_len = dataset_len or _check_lengths(v, dataset_len)
        elif isinstance(v, np.ndarray):
            item_len = len(v)
            dataset_len = dataset_len or item_len
            assert dataset_len == item_len, "Inconsistent item lengths in the dataset."
        else:
            raise TypeError("Unsupported type.")
    return dataset_len


def _subselect(dataset_dict: DatasetDict, index: np.ndarray) -> DatasetDict:
    new_dataset_dict = {}
    for k, v in dataset_dict.items():
        if isinstance(v, dict):
            new_v = _subselect(v, index)
        elif isinstance(v, np.ndarray):
            new_v = v[index]
        else:
            raise TypeError("Unsupported type.")
        new_dataset_dict[k] = new_v
    return new_dataset_dict


def _sample(
    dataset_dict: Union[np.ndarray, DatasetDict], indx: np.ndarray
) -> DatasetDict:
    if isinstance(dataset_dict, np.ndarray):
        return dataset_dict[indx]
    elif isinstance(dataset_dict, dict):
        batch = {}
        for k, v in dataset_dict.items():
            batch[k] = _sample(v, indx)
    else:
        raise TypeError("Unsupported type.")
    return batch


class Dataset(object):
    def __init__(self, dataset_dict: DatasetDict, seed: Optional[int] = None):
        self.dataset_dict = dataset_dict
        self.dataset_len = _check_lengths(dataset_dict)

        # Seeding similar to OpenAI Gym:
        # https://github.com/openai/gym/blob/master/gym/spaces/space.py#L46
        self._np_random = None
        self._seed = None
        if seed is not None:
            self.seed(seed)

    @property
    def np_random(self) -> np.random.RandomState:
        if self._np_random is None:
            self.seed()
        return self._np_random

    def seed(self, seed: Optional[int] = None) -> list:
        self._np_random, self._seed = seeding.np_random(seed)
        return [self._seed]

    def __len__(self) -> int:
        return self.dataset_len

    def sample(
        self,
        batch_size: int,
        keys: Optional[Iterable[str]] = None,
        indx: Optional[np.ndarray] = None,
    ) -> frozen_dict.FrozenDict:
        if indx is None:
            if hasattr(self.np_random, "integers"):
                indx = self.np_random.integers(len(self), size=batch_size)
            else:
                indx = self.np_random.randint(len(self), size=batch_size)

        batch = dict()

        if keys is None:
            keys = self.dataset_dict.keys()

        for k in keys:
            if isinstance(self.dataset_dict[k], dict):
                batch[k] = _sample(self.dataset_dict[k], indx)
            else:
                batch[k] = self.dataset_dict[k][indx]

        return frozen_dict.freeze(batch)

    def sample_jax(self, batch_size: int, keys: Optional[Iterable[str]] = None):
        if not hasattr(self, "rng"):
            self.rng = jax.random.PRNGKey(self._seed or 42)

            if keys is None:
                keys = self.dataset_dict.keys()

            jax_dataset_dict = {k: self.dataset_dict[k] for k in keys}
            jax_dataset_dict = jax.device_put(jax_dataset_dict)

            @jax.jit
            def _sample_jax(rng):
                key, rng = jax.random.split(rng)
                indx = jax.random.randint(
                    key, (batch_size,), minval=0, maxval=len(self)
                )
                return rng, jax.tree_map(
                    lambda d: jnp.take(d, indx, axis=0), jax_dataset_dict
                )

            self._sample_jax = _sample_jax

        self.rng, sample = self._sample_jax(self.rng)
        return sample

    def split(self, ratio: float) -> Tuple["Dataset", "Dataset"]:
        assert 0 < ratio and ratio < 1
        train_index = np.index_exp[: int(self.dataset_len * ratio)]
        test_index = np.index_exp[int(self.dataset_len * ratio) :]

        index = np.arange(len(self), dtype=np.int32)
        self.np_random.shuffle(index)
        train_index = index[: int(self.dataset_len * ratio)]
        test_index = index[int(self.dataset_len * ratio) :]

        train_dataset_dict = _subselect(self.dataset_dict, train_index)
        test_dataset_dict = _subselect(self.dataset_dict, test_index)
        return Dataset(train_dataset_dict), Dataset(test_dataset_dict)

    def _trajectory_boundaries_and_returns(self) -> Tuple[list, list, list]:
        episode_starts = [0]
        episode_ends = []

        episode_return = 0
        episode_returns = []

        for i in range(len(self)):
            episode_return += self.dataset_dict["rewards"][i]

            if self.dataset_dict["dones"][i]:
                episode_returns.append(episode_return)
                episode_ends.append(i + 1)
                if i + 1 < len(self):
                    episode_starts.append(i + 1)
                episode_return = 0.0

        return episode_starts, episode_ends, episode_returns

    def filter(
        self, take_top: Optional[float] = None, threshold: Optional[float] = None
    ):
        assert (take_top is None and threshold is not None) or (
            take_top is not None and threshold is None
        )

        (
            episode_starts,
            episode_ends,
            episode_returns,
        ) = self._trajectory_boundaries_and_returns()

        if take_top is not None:
            threshold = np.percentile(episode_returns, 100 - take_top)

        bool_indx = np.full((len(self),), False, dtype=bool)

        for i in range(len(episode_returns)):
            if episode_returns[i] >= threshold:
                bool_indx[episode_starts[i] : episode_ends[i]] = True

        self.dataset_dict = _subselect(self.dataset_dict, bool_indx)

        self.dataset_len = _check_lengths(self.dataset_dict)

    def normalize_returns(self, scaling: float = 1000):
        (_, _, episode_returns) = self._trajectory_boundaries_and_returns()
        self.dataset_dict["rewards"] *= scaling

    def sample_sequence(self, batch_size, sequence_length, discount, ret_next_act=False, ret_mc=False, filter_fn=None, success_only=None, by_score=False):
        # 1) Sample start indices
        #    Valid starts are [0, size - sequence_length]
        #    But we must also respect terminal boundaries if we want strict trajectories?
        #    Actually, the original code just samples indices and then constructs sequences.
        #    Let's look at how they do it.
        #    They rely on the fact that they can just grab [t, t+T].
        #    If a terminal happens in the middle, they handle it via masks.
        
        # Filter valid indices (start of sequences)
        # We need indices i such that i + sequence_length <= size
        # AND we probably don't want to cross episodes if we can avoid it, 
        # OR we handle it with masks. 
        # The reference implementation just samples indices.
        
        # Let's assume we just sample valid start indices.
        
        # For now, let's implement a simplified version that matches the reference logic
        # Reference logic:
        # valid_indices = np.arange(self.size - sequence_length + 1)
        # idxs = np.random.choice(valid_indices, size=batch_size)
        
        # We need to be careful about `self.dataset_len` vs `self.size`. 
        # In this class it is `self.dataset_len`.
        
        max_start = self.dataset_len - sequence_length
        if max_start < 0:
            raise ValueError("Dataset is too small for the requested sequence length.")
            
        # We can just use random integers
        if hasattr(self.np_random, "integers"):
            idxs = self.np_random.integers(0, max_start + 1, size=batch_size)
        else:
            idxs = self.np_random.randint(0, max_start + 1, size=batch_size)

        # 2) Build a (B, T) index matrix and gather everything at once
        T = sequence_length
        offs = np.arange(T, dtype=np.int64)[None, :]          # (1, T)
        seq_idxs = idxs[:, None].astype(np.int64) + offs      # (B, T)

        # We need to access self.dataset_dict directly
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
        terminals_seq   = 1.0 - masks_seq # (B, T) # Assuming masks are 1.0 - terminals

        # 3) Running mask/terminal over time
        #    (note: np.minimum/maximum are ufuncs; .accumulate supports axis)
        masks_prefix     = np.minimum.accumulate(masks_seq, axis=1)
        terminals_prefix = np.maximum.accumulate(terminals_seq, axis=1)

        # 4) Valid: 1 at i==0; for i>0 it's 1 - terminals_prefix at i-1
        valid = np.ones_like(masks_seq, dtype=np.float32)
        valid[:, 1:] = 1.0 - terminals_prefix[:, :-1]

        # 5) Prefix discounted return (same semantics as your loop)
        #    No stop-at-terminal masking (matches original).
        rdtype = rewards_seq.dtype
        disc_pows = (discount ** np.arange(T)).astype(rdtype, copy=False)  # (T,)
        rewards_prefix = np.cumsum(rewards_seq * disc_pows[None, :], axis=1)

        # 6) "observations" = first frame only (keep your original API)
        first_obs = self.dataset_dict['observations'][idxs]  # (B, obs_dim)
        last_obs = next_obs_seq[:, -1, ...]

        # TODO: Augmentation if needed (skipped for now as it wasn't in the original Dataset class explicitly)

        result = dict(
            observations=first_obs.copy(),          # matches your return
            actions=actions_seq,
            masks=masks_prefix,
            rewards=rewards_prefix,
            terminals=terminals_prefix,
            valid=valid,
            next_observations=last_obs,
        )
        
        return frozen_dict.freeze(result)

