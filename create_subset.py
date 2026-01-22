#!/usr/bin/env python3
"""
Create a small subset from the existing MineRL extracted dataset.

This script copies a specified number of episodes from the main dataset
to a new subset directory for fast testing.
"""

import os
import shutil
from pathlib import Path
import argparse
import random


def create_subset(
    source_dir: str,
    output_dir: str,
    num_episodes: int = 10,
    seed: int = 42
):
    """
    Create a subset of episodes from the source dataset.
    
    Args:
        source_dir: Path to the main MineRL extracted dataset
        output_dir: Path where subset will be created
        num_episodes: Number of episodes to include in subset
        seed: Random seed for episode selection
    """
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    if not source_path.exists():
        raise ValueError(f"Source directory does not exist: {source_dir}")
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all episode directories grouped by environment
    # Look for directories containing frames.npy
    episodes_by_env = {}
    
    # Check for nested structure (environment/episode)
    for frames_file in source_path.rglob("frames.npy"):
        episode_dir = frames_file.parent
        
        # Determine environment (parent directory of episode)
        # Structure: source/env/episode/frames.npy
        env_dir = episode_dir.parent
        env_name = env_dir.name
        
        # Skip if this is the source directory itself
        if env_dir == source_path:
            continue
        
        if env_name not in episodes_by_env:
            episodes_by_env[env_name] = []
        
        if episode_dir not in episodes_by_env[env_name]:
            episodes_by_env[env_name].append(episode_dir)
    
    # Also check for direct episode directories (if no nested structure)
    if len(episodes_by_env) == 0:
        for ep_dir in source_path.iterdir():
            if ep_dir.is_dir():
                frames_npy = ep_dir / "frames.npy"
                if frames_npy.exists():
                    if "direct" not in episodes_by_env:
                        episodes_by_env["direct"] = []
                    if ep_dir not in episodes_by_env["direct"]:
                        episodes_by_env["direct"].append(ep_dir)
    
    if len(episodes_by_env) == 0:
        raise ValueError(f"No episode directories found in {source_dir}")
    
    # Print summary
    total_episodes = sum(len(eps) for eps in episodes_by_env.values())
    print(f"Found {total_episodes} total episodes across {len(episodes_by_env)} environments:")
    for env_name, eps in episodes_by_env.items():
        print(f"  {env_name}: {len(eps)} episodes")
    
    # Ensure we have at least one episode from each environment
    # Then randomly select remaining episodes
    random.seed(seed)
    selected_episodes = []
    
    # First, select at least one episode from each environment
    for env_name, eps in episodes_by_env.items():
        if len(eps) > 0:
            selected = random.choice(eps)
            selected_episodes.append(selected)
            eps.remove(selected)
    
    print(f"Selected 1 episode from each of {len(episodes_by_env)} environments")
    
    # Calculate remaining episodes needed
    remaining_needed = num_episodes - len(selected_episodes)
    
    if remaining_needed > 0:
        # Collect all remaining episodes
        all_remaining = []
        for eps in episodes_by_env.values():
            all_remaining.extend(eps)
        
        if len(all_remaining) > 0:
            # Randomly select remaining episodes
            num_to_select = min(remaining_needed, len(all_remaining))
            additional = random.sample(all_remaining, num_to_select)
            selected_episodes.extend(additional)
            print(f"Selected {len(additional)} additional episodes randomly")
    
    print(f"Total selected: {len(selected_episodes)} episodes")
    
    # Copy selected episodes
    copied = 0
    envs_included = set()
    
    for i, ep_dir in enumerate(selected_episodes):
        try:
            # Determine relative path from source
            rel_path = ep_dir.relative_to(source_path)
            
            # Track which environments are included
            # Structure: env/episode or just episode
            parts = rel_path.parts
            if len(parts) >= 2:
                envs_included.add(parts[0])
            elif len(parts) == 1:
                envs_included.add("direct")
            
            # Create destination path
            dest_dir = output_path / rel_path
            dest_dir.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy entire episode directory
            if dest_dir.exists():
                shutil.rmtree(dest_dir)
            shutil.copytree(ep_dir, dest_dir)
            
            copied += 1
            if (i + 1) % 5 == 0:
                print(f"  Copied {i + 1}/{len(selected_episodes)} episodes...")
                
        except Exception as e:
            print(f"  Warning: Failed to copy {ep_dir}: {e}")
    
    print(f"\n✓ Successfully created subset with {copied} episodes")
    print(f"  Environments included: {sorted(envs_included)}")
    print(f"  Source: {source_dir}")
    print(f"  Output: {output_dir}")
    
    return copied


def main():
    parser = argparse.ArgumentParser(description="Create a small subset from MineRL dataset")
    parser.add_argument(
        "--source-dir",
        type=str,
        default="data/mineRL_extracted",
        help="Path to main MineRL extracted dataset"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/mineRL_subset",
        help="Path where subset will be created"
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=10,
        help="Number of episodes to include in subset"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for episode selection"
    )
    
    args = parser.parse_args()
    
    create_subset(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        num_episodes=args.num_episodes,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
