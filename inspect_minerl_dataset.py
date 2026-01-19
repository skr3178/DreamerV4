#!/usr/bin/env python3
"""
Inspect MineRL Dataset Dimensions
Works with downloaded MineRL zip files directly.
"""

import os
import sys
import zipfile
import json
from pathlib import Path
import numpy as np
from PIL import Image
import io

def inspect_npz_file(npz_data):
    """Inspect contents of an npz file."""
    info = {}
    for key in npz_data.keys():
        arr = npz_data[key]
        info[key] = {
            "shape": arr.shape,
            "dtype": str(arr.dtype),
            "min": float(arr.min()) if arr.size > 0 else None,
            "max": float(arr.max()) if arr.size > 0 else None,
        }
        if arr.size < 100:
            info[key]["sample"] = arr.tolist()
    return info

def inspect_zip_file(zip_path: Path, num_episodes: int = 3):
    """Inspect a MineRL zip file."""
    print("=" * 70)
    print(f"Inspecting {zip_path.name}")
    print("=" * 70)
    
    if not zip_path.exists():
        print(f"File not found: {zip_path}")
        return None
    
    size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"File size: {size_mb:.1f} MB\n")
    
    with zipfile.ZipFile(zip_path, 'r') as zf:
        file_list = zf.namelist()
        
        # Find episode directories
        episode_dirs = {}
        for f in file_list:
            parts = f.split('/')
            if len(parts) >= 2:
                ep_dir = '/'.join(parts[:2])  # Get environment/episode
                if ep_dir not in episode_dirs:
                    episode_dirs[ep_dir] = []
                episode_dirs[ep_dir].append(f)
        
        print(f"Found {len(episode_dirs)} episode directories")
        
        # Inspect first few episodes
        obs_shapes = []
        action_info = {}
        reward_info = {}
        
        for i, (ep_dir, files) in enumerate(list(episode_dirs.items())[:num_episodes]):
            print(f"\n--- Episode {i+1}: {ep_dir.split('/')[-1]} ---")
            
            # Find rendered.npz (contains observations, actions, rewards)
            npz_files = [f for f in files if f.endswith('rendered.npz')]
            metadata_files = [f for f in files if f.endswith('metadata.json')]
            
            if npz_files:
                try:
                    npz_file = npz_files[0]
                    with zf.open(npz_file) as f:
                        npz_data = np.load(io.BytesIO(f.read()))
                        
                        print(f"  NPZ file: {npz_file}")
                        print(f"  Keys in NPZ: {list(npz_data.keys())}")
                        
                        # Inspect each array
                        for key in npz_data.keys():
                            arr = npz_data[key]
                            print(f"\n  {key}:")
                            print(f"    Shape: {arr.shape}")
                            print(f"    Dtype: {arr.dtype}")
                            
                            if 'pov' in key.lower() or 'image' in key.lower() or 'obs' in key.lower():
                                obs_shapes.append(arr.shape)
                                print(f"    Range: [{arr.min()}, {arr.max()}]")
                                if arr.ndim >= 3:
                                    print(f"    Image dimensions: {arr.shape[-3:] if arr.ndim >= 3 else 'N/A'}")
                            
                            elif 'action' in key.lower():
                                action_info[key] = {
                                    "shape": arr.shape,
                                    "dtype": arr.dtype
                                }
                                if arr.dtype in [np.int8, np.int16, np.int32, np.int64]:
                                    unique_vals = np.unique(arr)
                                    print(f"    Unique values: {len(unique_vals)}")
                                    print(f"    Range: [{unique_vals.min()}, {unique_vals.max()}]")
                                    if len(unique_vals) < 50:
                                        print(f"    Values: {unique_vals.tolist()}")
                                else:
                                    print(f"    Range: [{arr.min():.3f}, {arr.max():.3f}]")
                            
                            elif 'reward' in key.lower():
                                reward_info[key] = {
                                    "shape": arr.shape,
                                    "dtype": arr.dtype,
                                    "min": float(arr.min()),
                                    "max": float(arr.max()),
                                    "mean": float(arr.mean())
                                }
                                print(f"    Range: [{arr.min():.3f}, {arr.max():.3f}], Mean: {arr.mean():.3f}")
                            
                            # Show sample if small
                            if arr.size < 50:
                                print(f"    Sample: {arr.flatten()[:10]}")
                
                except Exception as e:
                    print(f"  Error reading NPZ: {e}")
            
            if metadata_files:
                try:
                    with zf.open(metadata_files[0]) as f:
                        metadata = json.load(f)
                        print(f"\n  Metadata keys: {list(metadata.keys())}")
                        if 'action_space' in metadata:
                            print(f"  Action space: {metadata['action_space']}")
                        if 'observation_space' in metadata:
                            print(f"  Observation space: {metadata['observation_space']}")
                except Exception as e:
                    print(f"  Error reading metadata: {e}")
        
        # Summary
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        
        if obs_shapes:
            print(f"\nObservation Space:")
            unique_shapes = list(set(obs_shapes))
            for shape in unique_shapes:
                print(f"  - Shape: {shape}")
        
        if action_info:
            print(f"\nAction Space:")
            for key, info in action_info.items():
                print(f"  - {key}: shape={info['shape']}, dtype={info['dtype']}")
        
        if reward_info:
            print(f"\nReward Space:")
            for key, info in reward_info.items():
                print(f"  - {key}: shape={info['shape']}, range=[{info['min']:.3f}, {info['max']:.3f}], mean={info['mean']:.3f}")
        
        return {
            "observation_shapes": unique_shapes if obs_shapes else [],
            "action_info": action_info,
            "reward_info": reward_info
        }

def main():
    dataset_dir = Path("/media/skr/storage/dreamerv4/data/mineRL")
    
    print("=" * 70)
    print("MineRL Dataset Inspector")
    print("=" * 70)
    print(f"\nDataset directory: {dataset_dir}")
    
    # Find zip files
    zip_files = list(dataset_dir.glob("*.zip"))
    
    if not zip_files:
        print(f"\nNo zip files found in {dataset_dir}")
        print("Please download MineRL datasets first.")
        return
    
    print(f"\nFound {len(zip_files)} zip file(s):")
    for zf in zip_files:
        size_mb = zf.stat().st_size / (1024 * 1024)
        print(f"  - {zf.name} ({size_mb:.1f} MB)")
    
    # Inspect each zip file
    results = {}
    for zip_file in zip_files:
        if zip_file.stat().st_size > 0:
            result = inspect_zip_file(zip_file, num_episodes=3)
            results[zip_file.name] = result
            print("\n")
    
    # Final summary
    print("=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print("\nBased on DreamerV4 paper and MineRL documentation:")
    print("- Action space: 23 keyboard binary + 121 mouse categorical")
    print("  Total: ~144 discrete actions (or combined into single categorical)")
    print("- Observation space: 64x64x3 RGB images (POV)")
    print("- Reward: Scalar per timestep")
    print("\nNote: MINERL_DATA_ROOT should be set to the dataset directory")
    print(f"  export MINERL_DATA_ROOT={dataset_dir}")

if __name__ == "__main__":
    main()
