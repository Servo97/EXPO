from typing import Dict
import warnings

# Suppress deprecation warnings from gym
warnings.filterwarnings('ignore', category=DeprecationWarning)

import gym
import numpy as np
import multiprocessing as mp
import time
import copy
import jax.numpy as jnp
import jax
import flax
from flax.core import frozen_dict
from gym.utils import seeding
from collections import defaultdict

from expo.wrappers.wandb_video import WANDBVideo


def evaluate(
    agent, env: gym.Env, num_episodes: int, save_video: bool = False
) -> Dict[str, float]:
    if save_video:
        env = WANDBVideo(env, name="eval_video", max_videos=1)
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=num_episodes)

    for i in range(num_episodes):
        observation, done = env.reset(), False
        while not done:
            action = agent.eval_actions(observation)
            observation, _, done, _ = env.step(action)
    return {"return": np.mean(env.return_queue), "length": np.mean(env.length_queue)}


def evaluate_diffusion(
    agent, env: gym.Env, num_episodes: int, save_video: bool = False
) -> Dict[str, float]:
    if save_video:
        env = WANDBVideo(env, name="eval_video", max_videos=1)
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=num_episodes)

    for i in range(num_episodes):
        observation, done = env.reset(), False
        while not done:
            action, agent = agent.eval_actions(observation)
            observation, _, done, _ = env.step(action)
    return {"return": np.mean(env.return_queue), "length": np.mean(env.length_queue)}, agent


def evaluate_diffusion_steps(
    agent, env: gym.Env, num_steps: int, save_video: bool = False, return_trajs=False,
) -> Dict[str, float]:
    if save_video:
        env = WANDBVideo(env, name="eval_video", max_videos=1)
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=1000)

    all_trajs = []

    steps = 0


    while steps < num_steps:
        observation, done = env.reset(), False
        observations = []
        dones = []
        next_observations = []
        actions = []
        rewards = []

        while not done and steps < num_steps:
            action, agent = agent.eval_actions(observation)
            next_observation, r, done, _ = env.step(action)

            

            observations += [observation]
            rewards += [r]
            actions += [action]
            dones += [done]
            next_observations += [next_observation]

            observation = next_observation

            steps += 1




        
        traj = {
                'observations': np.array(observations, dtype=np.float32),
                'actions': np.array(actions, dtype=np.float32),
                'rewards': np.array(rewards, dtype=np.float32),
                'next_observations': np.array(next_observations, dtype=np.float32),
                'dones': np.array(dones, dtype=np.float32),
            }
        
        all_trajs += [traj]


    if return_trajs:
        return all_trajs, {"return": np.mean(env.return_queue), "length": np.mean(env.length_queue)}, agent


    return {"return": np.mean(env.return_queue), "length": np.mean(env.length_queue)}, agent



class SamplerPolicy(object):
    def __init__(self, agent):
        self.agent = agent

    def update_params(self, params):
        return self

    def __call__(self, observations, deterministic=False, add_noise=0.0, **kwargs):
        actions = self.agent.eval_actions(observations)

        if len(actions) == 2:
            actions, agent = actions
            self.agent = agent


        assert jnp.all(jnp.isfinite(actions))
        return jax.device_get(actions)




def evaluate_robo(agent, env: gym.Env, num_episodes: int, max_traj_len: int, save_video: bool = False, return_trajs=False): 
    eval_sampler = QueueTrajSampler(env, max_traj_len)
    sampler_policy = SamplerPolicy(agent)
    trajs = eval_sampler.sample(
                    sampler_policy,
                    num_episodes
                )

    if return_trajs:
        return trajs, {"return": np.mean([np.sum(t['rewards']) for t in trajs]), "length": np.mean([len(t['rewards']) for t in trajs])}
    return {"return": np.mean([np.sum(t['rewards']) for t in trajs]), "length": np.mean([len(t['rewards']) for t in trajs])}





class TrajSamplerProc:
    def __init__(self, process_id, seed, env, filter, n_trajs, terminal_queue, max_traj_length=1000, rlpd=False):
        self.process_id = process_id
        self.seed = seed
        self.env = env
        self.filter = filter
        self.n_trajs = n_trajs
        self.terminal_queue = terminal_queue
        self.max_traj_length = max_traj_length
        self.rlpd = rlpd
        
        # Queues for communication
        self.send_queue = mp.Queue()
        self.recv_queue = mp.Queue()
    
    def start(self):
        # Suppress deprecation warnings in worker process
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        
        env = copy.deepcopy(self.env)

        
        trajs = []
        for traj_idx in range(self.n_trajs):
            np.random.seed(self.seed[traj_idx])
            # Initialize trajectory storage
            observations = []
            actions = []
            rewards = []
            next_observations = []
            dones = []
            
            # Reset environment
            observation = env.reset()
            
            # Handle observation indexing
            
            # Run trajectory
            # Run trajectory
            action_queue = []
            for step in range(self.max_traj_length):
                if len(action_queue) == 0:
                    # Send observation to main process to get action
                    self.send_queue.put((self.process_id, traj_idx, observation))
                    
                    # Wait for action from main process
                    action_chunk = self.recv_queue.get()
                    
                    # Handle action chunking
                    # Try to get action dimension from env
                    if hasattr(env, 'action_space'):
                         if isinstance(env.action_space, np.ndarray):
                             act_dim = env.action_space.shape[-1]
                         elif hasattr(env.action_space, 'shape'):
                             act_dim = env.action_space.shape[-1]
                         else:
                             # Fallback or error?
                             # Assuming flat action if no shape
                             act_dim = len(action_chunk) 
                    else:
                         # Fallback if no action_space
                         # If we don't know act_dim, we can't easily reshape if it's flat.
                         # But usually we know it.
                         # For now assume it is available as it is a gym wrapper.
                         act_dim = len(action_chunk)

                    if action_chunk.shape[0] > act_dim:
                         action_chunk = action_chunk.reshape(-1, act_dim)
                         for a in action_chunk:
                             action_queue.append(a)
                    else:
                         action_queue.append(action_chunk)

                action = action_queue.pop(0)
                
                # Step environment
                next_observation, reward, done, info = env.step(action)
                
                # Handle RLPD specific logic
                if self.rlpd and "TimeLimit.truncated" in info:
                    done = True
                
                # Store step data
                observations.append(observation)
                actions.append(action)
                rewards.append(reward)
                dones.append(done)
                next_observations.append(next_observation)
                
                # Update observation
                observation = next_observation
                
                # Check if trajectory is done
                if done:
                    break
            
            # Filter based on reward if needed
            if self.filter and not np.sum(rewards) > 0:
                continue
            
            # Create trajectory dictionary
            traj = {
                'observations': np.array(observations, dtype=np.float32),
                'actions': np.array(actions, dtype=np.float32),
                'rewards': np.array(rewards, dtype=np.float32),
                'next_observations': np.array(next_observations, dtype=np.float32),
                'dones': np.array(dones, dtype=np.float32),
            }
            
            trajs.append(traj)
        
        # Send all trajectories back to main process
        self.terminal_queue.put((self.process_id, trajs))
        
        return


class QueueTrajSampler(object):
    def __init__(self, env, max_traj_length=1000, rlpd=False, seed=1):
        self.max_traj_length = max_traj_length
        self._env = env
        self.seed = seed * 10000
        self.rlpd = rlpd
    
    def sample(self, policy, n_trajs, replay_buffer=None, deterministic=False, obs_index=False, 
               add_noise=0, filter=False, n_workers=None):
        """
        Sample trajectories in parallel using multiple processes with queue-based communication.
        
        Args:
            policy: Policy to sample trajectories from
            n_trajs: Number of trajectories to sample
            replay_buffer: Optional replay buffer to add samples to
            deterministic: Whether to sample deterministically from policy
            obs_index: Whether to index observation with 'observation' key
            add_noise: Amount of noise to add to policy actions
            filter: Whether to filter out trajectories with non-positive returns
            n_workers: Number of worker processes to use (defaults to CPU count)
        """
        if n_workers is None:
            n_workers = min(mp.cpu_count() // 2, n_trajs)
        else:
            n_workers = min(n_workers, n_trajs, mp.cpu_count())
        

        print("n w: ", n_workers)


        # Calculate trajectories per worker
        trajs_per_worker = [n_trajs // n_workers + (1 if i < n_trajs % n_workers else 0) for i in range(n_workers)]

        n_traj = [[-1]]

        for i in range(len(trajs_per_worker)):
            n_traj += [list(range(n_traj[-1][-1] + 1, n_traj[-1][-1] + 1 + trajs_per_worker[i]))]
        
        n_traj = n_traj[1:] # mapping from process id and process traj idx to trajectory id
        
        # Create communication queues
        terminal_queue = mp.Queue()
        
        # Create sampler processes
        sampler_procs = []
        for i in range(n_workers):
            seeds = list(range(self.seed + sum(trajs_per_worker[:i]), self.seed + sum(trajs_per_worker[:i+1])))
            sampler_procs.append(
                TrajSamplerProc(
                    process_id=i,
                    seed=seeds,
                    env=self._env,
                    filter=filter,
                    n_trajs=trajs_per_worker[i],
                    terminal_queue=terminal_queue,
                    max_traj_length=self.max_traj_length,
                    rlpd=self.rlpd
                )
            )
        
        # Create mapping from process_id to queues
        send_queues = {i: proc.recv_queue for i, proc in enumerate(sampler_procs)}
        get_queues = {i: proc.send_queue for i, proc in enumerate(sampler_procs)}
        
        # Start processes
        processes = {i: mp.Process(target=proc.start) for i, proc in enumerate(sampler_procs)}
        for _, p in processes.items():
            p.start()
        
        # Run sampling
        start_time = time.time()
        all_trajs = []
        
        # Continue until all processes are done
        while len(processes) > 0:
            # Check for completed processes
            while not terminal_queue.empty():
                proc_id, trajs = terminal_queue.get()
                all_trajs.extend(trajs)
                
                # Clean up process
                processes[proc_id].join()
                processes.pop(proc_id)
                send_queues.pop(proc_id)
                get_queues.pop(proc_id)
            
            # Process observations from workers and send back actions
            observations = []
            process_ids = []
            traj_indices = []
            
            # Collect observations from all queues
            for proc_id, queue in get_queues.items():
                if not queue.empty():
                    proc_id, traj_idx, obs = queue.get()
                    process_ids.append(proc_id)
                    traj_indices.append(traj_idx)
                    observations.append(obs)
            
            # If we have observations, process them and send actions
            if observations:
                # Convert to batch and get actions
                obs_batch = np.array(observations)


                actions = []


                for i in range(obs_batch.shape[0]):
                    actions += [policy(obs_batch[i], deterministic=deterministic, add_noise=add_noise, multihead_idx=n_traj[process_ids[i]][traj_indices[i]])]


                # actions = policy(obs_batch, deterministic=deterministic, add_noise=add_noise)
                
                # Send actions back to respective processes
                for i, proc_id in enumerate(process_ids):
                    send_queues[proc_id].put(actions[i])
            
            # Small sleep to prevent CPU spinning
            if not observations:
                time.sleep(0.001)
        
        print(f"Sampling completed in {time.time() - start_time:.2f} seconds")


        all_trajs = all_trajs[:n_trajs]

        
        # Add to replay buffer if provided
        if replay_buffer is not None:
            for traj in all_trajs:
                for i in range(len(traj['observations'])):
                    replay_buffer.add_sample(
                        traj['observations'][i],
                        traj['actions'][i],
                        traj['rewards'][i],
                        traj['next_observations'][i],
                        traj['dones'][i]
                    )
        
        return all_trajs
    
    @property
    def env(self):
        return self._env


