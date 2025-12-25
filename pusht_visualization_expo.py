"""
Visualization script for EXPO Push-T agent.
Loads EXPO checkpoints and plots trajectory visualizations.

NOTE: This script uses the old Gym API (pre-0.26) where:
    - env.reset() returns only observation (not obs, info)
    - env.step() returns (obs, reward, done, info) (not obs, reward, terminated, truncated, info)

USAGE EXAMPLES:
    
    # Use default checkpoint directory and settings (uses cache if available)
    python pusht_visualization_expo.py
    
    # Specify custom checkpoint directory
    python pusht_visualization_expo.py --checkpoint_dir /path/to/checkpoints
    
    # Load specific checkpoint step
    python pusht_visualization_expo.py --checkpoint_step 1000000
    
    # Force regenerate rollouts (ignore cache)
    python pusht_visualization_expo.py --no_cache
    
    # Collect more rollouts for better statistics
    python pusht_visualization_expo.py --num_rollouts 100
    
    # Custom output file
    python pusht_visualization_expo.py --output my_visualization.png
    
    # Full example with all options
    python pusht_visualization_expo.py \
        --checkpoint_dir /data/user_data/mananaga/expo/logs/s42_500000pretrain_LN/checkpoints \
        --checkpoint_step 500000 \
        --num_rollouts 50 \
        --seed 42 \
        --output expo_500k.png

DEFAULT VALUES:
    checkpoint_dir: /data/user_data/mananaga/expo/logs/s0_500000pretrain_LN/checkpoints
    checkpoint_step: 40000 (latest available)
    cache_path: expo_rollouts_data.npz
    num_rollouts: 50
    seed: 42
    output: pusht_expo_visualization.png
"""

import os
import sys
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from ml_collections import ConfigDict

# Add EXPO to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from expo.agents import EXPOLearner
from expo.envs.pusht_utils import make_env, get_dataset

# Import Flax checkpoints for loading
try:
    from flax.training import checkpoints
except ImportError:
    print("Warning: Could not import flax.training.checkpoints")
    checkpoints = None


def load_expo_checkpoint(checkpoint_dir, step=None):
    """
    Load EXPO checkpoint from directory.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        step: Specific step to load (if None, loads latest)
    
    Returns:
        agent: Loaded EXPO agent
    """
    if checkpoints is None:
        raise RuntimeError("flax.training.checkpoints not available")
    
    print(f"Loading checkpoint from {checkpoint_dir}", flush=True)
    
    # First, we need to create a dummy agent to get the structure
    # Load a sample observation and action from the dataset
    dataset = get_dataset('pusht-keypoints-v0')
    example_obs = dataset['observations'][0]
    example_action = dataset['actions'][0]
    
    # Create config (matching the actual training config from run_pusht_debug.sh)
    config = ConfigDict({
        'actor_lr': 3e-4,
        'critic_lr': 3e-4,
        'temp_lr': 3e-4,
        'hidden_dims': (512, 512, 512, 512),  # From training script
        'discount': 0.95,  # From training script
        'tau': 0.05,  # From training script
        'num_qs': 10,
        'num_min_qs': 2,
        'critic_layer_norm': True,
        'N': 8,  # From training script
        'n_edit_samples': 8,  # From training script
        'entropy_scale': 1.0,
        'edit_action_scale': 0.05,  # From training script
        'actor_drop': 0.0,
        'd_actor_drop': 0.0,
        'batch_split': 1,
        'T': 10,
        'horizon': 4,  # Action chunking horizon
        'backup_entropy': False,  # From training script
    })
    
    # Create dummy agent
    print("Creating EXPO agent structure...", flush=True)
    agent = EXPOLearner.create(
        seed=42,
        observation_space=example_obs,
        action_space=example_action,
        **config
    )
    
    # Load checkpoint
    if step is not None:
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_{step}')
        if not os.path.exists(checkpoint_path):
            print(f"Warning: Checkpoint at step {step} not found, loading latest")
            step = None
    
    if step is None:
        # Find latest checkpoint
        restored_agent = checkpoints.restore_checkpoint(
            checkpoint_dir, 
            target=agent,
            orbax_checkpointer=False
        )
    else:
        restored_agent = checkpoints.restore_checkpoint(
            checkpoint_dir,
            target=agent,
            step=step,
            orbax_checkpointer=False
        )
    
    print(f"Successfully loaded EXPO checkpoint", flush=True)
    return restored_agent


def collect_expo_rollouts(agent, env, num_successes=50, seed=42, max_steps=300, agent_name="EXPO"):
    """
    Collect successful rollouts from EXPO agent.
    
    Args:
        agent: EXPO agent
        env: Push-T environment
        num_successes: Number of successful rollouts to collect
        seed: Random seed (used for agent's RNG only; env uses fixed seed from creation)
        max_steps: Maximum steps per episode
        agent_name: Name for logging
    
    Returns:
        List of successful rollout dictionaries
    """
    successful_rollouts = []
    
    pbar_limit = 1000  # Safety limit
    success_count = 0
    total_attempts = 0
    
    print(f"Collecting {num_successes} successful rollouts for {agent_name}...", flush=True)
    
    rng = jax.random.PRNGKey(seed)
    
    for i in range(pbar_limit):
        if success_count >= num_successes:
            break
        
        total_attempts += 1
        
        # Reset env (old Gym API - returns only observation)
        obs = env.reset()
        obs = np.array(obs)
        
        episode_obs = [obs]
        episode_actions = []
        episode_next_obs = []
        
        done = False
        truncated = False
        step = 0
        
        action_queue = []
        action_dim = env.action_space.shape[0]
        
        # Update agent's RNG
        agent = agent.replace(rng=rng)
        
        while not (done or truncated) and step < max_steps:
            # Sample action if queue is empty
            if len(action_queue) == 0:
                # Sample action chunk from agent
                action_chunk, agent = agent.sample_actions(obs)
                action_chunk = np.array(action_chunk)
                
                # Reshape and queue
                if action_chunk.ndim == 1:
                    # Single action or flattened chunk
                    if action_chunk.shape[0] > action_dim:
                        # It's a flattened chunk, reshape
                        action_chunk = action_chunk.reshape(-1, action_dim)
                        for a in action_chunk:
                            action_queue.append(a)
                    else:
                        # Single action
                        action_queue.append(action_chunk)
                elif action_chunk.ndim == 2:
                    # Already shaped as (horizon, action_dim)
                    for a in action_chunk:
                        action_queue.append(a)
                else:
                    # Just take the first action
                    action_queue.append(action_chunk.flatten()[:action_dim])
            
            # Pop action
            action_np = action_queue.pop(0)
            
            # Clip action
            action_np = np.clip(action_np, -1, 1)
            
            # Step env
            next_obs, reward, done, info = env.step(action_np)
            truncated = False  # Old gym API doesn't separate truncated from done
            
            episode_obs.append(obs)
            episode_actions.append(action_np)
            episode_next_obs.append(next_obs)
            
            obs = next_obs
            step += 1
            
            # Update RNG for next iteration
            rng = agent.rng
        
        # Check success
        is_success = info.get('success', 0.0) > 0.99 or (reward == 0.0 and done)
        
        if is_success:
            success_count += 1
            print(f"Success {success_count}/{num_successes} (success rate: {success_count/total_attempts:.1%})", flush=True)
            successful_rollouts.append({
                'obs': np.array(episode_obs),
                'actions': np.array(episode_actions),
                'next_obs': np.array(episode_next_obs)
            })
        else:
            if i % 20 == 0:
                print(f"Episode {i} failed. Success count: {success_count}/{total_attempts} ({success_count/max(total_attempts,1):.1%})", flush=True)
    
    print(f"\n{agent_name} final success rate: {success_count}/{total_attempts} = {success_count/max(total_attempts,1):.1%}\n", flush=True)
    return successful_rollouts


def save_rollouts(expo_rollouts, save_path='expo_rollouts_data.npz'):
    """Save rollout data to npz file for faster replotting."""
    print(f"\nSaving rollout data to {save_path}...", flush=True)
    
    data = {
        'expo_rollouts': expo_rollouts,
    }
    
    np.savez_compressed(save_path, **data)
    print(f"Saved rollout data to {save_path}", flush=True)


def load_rollouts(load_path='expo_rollouts_data.npz'):
    """Load rollout data from npz file."""
    print(f"\nLoading rollout data from {load_path}...", flush=True)
    
    if not os.path.exists(load_path):
        print(f"Warning: {load_path} not found. Will generate new rollouts.", flush=True)
        return None
    
    data = np.load(load_path, allow_pickle=True)
    expo_rollouts = data['expo_rollouts'].tolist()
    
    print(f"Loaded rollouts: EXPO={len(expo_rollouts)}", flush=True)
    
    return expo_rollouts


def create_gradient_colormap(hex_color, name='custom'):
    """Create a colormap from dark to light version of a hex color."""
    # Convert hex to RGB (0-1 range)
    hex_color = hex_color.lstrip('#')
    r, g, b = tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))
    
    # Create gradient from dark to light
    colors = [
        (r * 0.5, g * 0.5, b * 0.5),  # Dark version
        (r, g, b),                      # Original color
        (0.5 + r * 1.1, 0.5 + g * 1.1, 0.5 + b * 1.1),  # Light version
    ]
    
    n_bins = 256
    cmap = LinearSegmentedColormap.from_list(name, colors, N=n_bins)
    return cmap


def plot_expo_rollouts(expo_rollouts, env, viz_seed=42, output_path='pusht_expo_visualization.png'):
    """
    Plot EXPO rollouts on Push-T environment.
    
    Args:
        expo_rollouts: List of rollout dictionaries
        env: Push-T environment
        viz_seed: Seed for environment rendering
        output_path: Where to save the plot
    """
    print(f"Plotting results to {output_path}...", flush=True)
    
    # Set up publication-quality plotting parameters
    plt.rcParams.update({
        'font.family': 'monospace',
        'font.monospace': ['Courier New', 'DejaVu Sans Mono', 'Courier'],
        'font.size': 16,
        'axes.labelsize': 18,
        'axes.titlesize': 20,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        'figure.titlesize': 22,
        'axes.linewidth': 1.5,
        'grid.linewidth': 0.8,
        'lines.linewidth': 2.0,
    })
    
    # Create custom gradient colormap for EXPO (blue-ish)
    expo_cmap = create_gradient_colormap('3366CC', 'expo_gradient')
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=150)
    
    # Get background image from initial state
    obs = env.reset()
    bg_img = env.render()  # Should be (512, 512, 3)
    
    def unnormalize_action(x):
        """Unnormalize actions from [-1, 1] to [0, 512]"""
        return (x + 1) / 2 * 512
    
    # Set background
    ax.imshow(bg_img, extent=[0, 512, 512, 0], alpha=0.8)
    ax.set_xlim(0, 512)
    ax.set_ylim(512, 0)
    
    # Remove all decorations for clean look
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title('EXPO Push-T Trajectories', fontsize=20, pad=10)
    
    # Remove all borders/spines
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # Plot all rollouts
    for rollout_idx, rollout in enumerate(expo_rollouts):
        obs = rollout['obs']
        actions = rollout['actions']  # Executed actions (puck positions)
        
        # Get executed actions (first action in each chunk if action chunking)
        # actions shape: [T, action_dim] where action_dim might be horizon * 2
        if actions.shape[1] > 2:
            # Action chunking - take first action
            executed_actions = actions[:, :2]
        else:
            executed_actions = actions
        
        # Clip actions to valid range [-1, 1] before unnormalizing
        executed_actions = np.clip(executed_actions, -1.0, 1.0)
        executed_actions_unnorm = unnormalize_action(executed_actions)
        
        # Get number of timesteps for this trajectory
        T = len(executed_actions_unnorm)
        
        # Create color array based on time progression
        time_indices = np.arange(T)
        colors = expo_cmap(time_indices / max(T - 1, 1))
        
        # Plot connecting lines with color gradient
        if T > 1:
            points = executed_actions_unnorm.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            
            alpha_line = 0.15 if len(expo_rollouts) > 10 else 0.5
            lc = LineCollection(segments, colors=colors[:-1], linewidth=1.5,
                               alpha=alpha_line, zorder=5)
            ax.add_collection(lc)
        
        # Plot executed action trajectory with time-based color gradient
        alpha_ext = 0.3 if len(expo_rollouts) > 10 else 0.7
        ax.scatter(executed_actions_unnorm[:, 0], executed_actions_unnorm[:, 1],
                  c=colors, s=25, alpha=alpha_ext, edgecolors='none',
                  zorder=10)
    
    plt.tight_layout(pad=0)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved plot to {output_path}", flush=True)
    plt.close()


def main(checkpoint_dir=None, checkpoint_step=None, use_cache=True, cache_path='expo_rollouts_data.npz', 
         num_rollouts=50, viz_seed=42, output_path='pusht_expo_visualization.png'):
    """
    Main function to generate or load rollouts and create visualization.
    
    Args:
        checkpoint_dir: Directory containing EXPO checkpoints
        checkpoint_step: Specific checkpoint step to load (None = latest)
        use_cache: If True, try to load from cache. If False, regenerate rollouts.
        cache_path: Path to the npz file for caching rollouts.
        num_rollouts: Number of successful rollouts to collect
        viz_seed: Seed for environment
        output_path: Where to save the visualization
    """
    
    env_name = 'pusht-keypoints-v0'
    
    # Try to load from cache first
    if use_cache:
        cached_data = load_rollouts(cache_path)
        if cached_data is not None:
            expo_rollouts = cached_data
            
            # Still need to create env for plotting
            print(f"Creating environment {env_name} for plotting...", flush=True)
            env = make_env(env_name, seed=viz_seed)
            env.reset()  # Initialize environment
            
            # Plot and exit
            plot_expo_rollouts(expo_rollouts, env, viz_seed=viz_seed, output_path=output_path)
            return
    
    # If not using cache or cache not found, generate rollouts
    if checkpoint_dir is None:
        raise ValueError("checkpoint_dir must be provided when generating new rollouts")
    
    print("\n" + "="*80)
    print("GENERATING NEW ROLLOUTS FOR EXPO")
    print("="*80 + "\n")
    
    # Load EXPO agent
    print("Loading EXPO Agent...", flush=True)
    agent = load_expo_checkpoint(checkpoint_dir, step=checkpoint_step)
    
    # Create env
    print(f"Creating environment {env_name}...", flush=True)
    env = make_env(env_name, seed=viz_seed)
    
    # Collect EXPO Rollouts
    expo_rollouts = collect_expo_rollouts(
        agent, env, 
        num_successes=num_rollouts, 
        seed=viz_seed, 
        agent_name="EXPO"
    )
    
    # Save rollouts to cache for future use
    save_rollouts(expo_rollouts, save_path=cache_path)
    
    # Plot
    plot_expo_rollouts(expo_rollouts, env, viz_seed=viz_seed, output_path=output_path)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize EXPO Push-T trajectories')
    parser.add_argument('--checkpoint_dir', type=str, 
                        default='/data/user_data/mananaga/expo/logs/s0_500000pretrain_LN/checkpoints',
                        help='Directory containing EXPO checkpoints (default: logs/s0_500000pretrain_LN/checkpoints)')
    parser.add_argument('--checkpoint_step', type=int, default=40000,
                        help='Specific checkpoint step to load (default: 40000, the latest available)')
    parser.add_argument('--no_cache', action='store_true',
                        help='Do not use cached rollouts (regenerate from scratch)')
    parser.add_argument('--cache_path', type=str, default='expo_rollouts_data.npz',
                        help='Path to cache file (default: expo_rollouts_data.npz)')
    parser.add_argument('--num_rollouts', type=int, default=50,
                        help='Number of successful rollouts to collect (default: 50)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for environment (default: 42)')
    parser.add_argument('--output', type=str, default='pusht_expo_visualization.png',
                        help='Output path for visualization (default: pusht_expo_visualization.png)')
    
    args = parser.parse_args()
    
    # By default use cache unless --no_cache is specified
    use_cache = not args.no_cache
    
    main(
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_step=args.checkpoint_step,
        use_cache=use_cache,
        cache_path=args.cache_path,
        num_rollouts=args.num_rollouts,
        viz_seed=args.seed,
        output_path=args.output
    )

