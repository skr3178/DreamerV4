#!/usr/bin/env python3
"""
Analysis script for DreamerV4 mixed-small dataset files.
Analyzes the structure and relevance of .pt and .png files.
"""

import torch
from tensordict import TensorDict
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None  # Disable decompression bomb check for large images

def analyze_pt_file(pt_path):
    """Analyze the structure of a .pt TensorDict file."""
    print(f"\n{'='*70}")
    print(f"ANALYZING: {pt_path}")
    print(f"{'='*70}")
    
    data = torch.load(pt_path, map_location='cpu', weights_only=False)
    
    print(f"\nType: {type(data)}")
    print(f"Total timesteps: {len(data)}")
    
    # First timestep structure
    first_item = data[0]
    print(f"\nKeys in each timestep: {list(first_item.keys())}")
    
    print(f"\n{'='*70}")
    print("FIELD DETAILS")
    print(f"{'='*70}")
    for key in first_item.keys():
        val = first_item[key]
        if hasattr(val, 'shape'):
            print(f"\n{key}:")
            print(f"  Shape: {val.shape}")
            print(f"  Dtype: {val.dtype}")
            if val.dtype in [torch.float32, torch.float64]:
                valid_vals = val[~torch.isnan(val)] if torch.isnan(val).any() else val
                if len(valid_vals) > 0:
                    print(f"  Min: {valid_vals.min().item():.4f}")
                    print(f"  Max: {valid_vals.max().item():.4f}")
                    print(f"  Mean: {valid_vals.mean().item():.4f}")
            elif val.dtype == torch.bool:
                print(f"  Value: {val.item()}")
            if val.numel() <= 10:
                print(f"  Sample values: {val.tolist()}")
    
    # Episode analysis
    episodes = []
    for i in range(len(data)):
        if 'episode' in data[i]:
            ep = data[i]['episode']
            if ep.numel() == 1:
                episodes.append(ep.item())
    
    unique_episodes = set(episodes)
    print(f"\n{'='*70}")
    print("EPISODE STATISTICS")
    print(f"{'='*70}")
    print(f"Total timesteps: {len(data)}")
    print(f"Total unique episodes: {len(unique_episodes)}")
    if len(unique_episodes) > 0:
        print(f"Average timesteps per episode: {len(data) / len(unique_episodes):.2f}")
    
    return data

def analyze_png_file(png_path):
    """Analyze the structure of a .png file (concatenated frames)."""
    print(f"\n{'='*70}")
    print(f"ANALYZING: {png_path}")
    print(f"{'='*70}")
    
    img = Image.open(png_path)
    img_array = np.array(img)
    
    print(f"\nImage Properties:")
    print(f"  Size: {img.size} (width x height)")
    print(f"  Mode: {img.mode}")
    print(f"  Format: {img.format}")
    print(f"  Array shape: {img_array.shape}")
    print(f"  Array dtype: {img_array.dtype}")
    
    height = img_array.shape[0]
    width = img_array.shape[1] if len(img_array.shape) >= 2 else 1
    
    print(f"\nFrame Structure Analysis:")
    print(f"  Dimensions: {width} x {height}")
    
    # Common frame sizes in RL datasets
    frame_sizes = [64, 84, 96, 128, 224]
    print(f"\n  Possible frame configurations:")
    for frame_size in frame_sizes:
        if width % frame_size == 0:
            num_frames = width // frame_size
            print(f"    - {num_frames} frames of {frame_size}x{frame_size}")
    
    # Sample statistics
    if width > 1000:
        sample = img_array[:, :min(500, width), :] if len(img_array.shape) == 3 else img_array[:, :min(500, width)]
        print(f"\n  Sample pixel range: [{sample.min()}, {sample.max()}]")
        print(f"  Sample mean: {sample.mean():.2f}")
    
    return img_array

def main():
    base_path = "data/NH_dataset/mixed-small/acrobot-swingup"
    
    print("="*70)
    print("DREAMERV4 MIXED-SMALL DATASET ANALYSIS")
    print("Task: acrobot-swingup")
    print("="*70)
    
    # Analyze .pt file
    pt_data = analyze_pt_file(f"{base_path}.pt")
    
    # Analyze PNG files
    for i in range(3):
        png_path = f"{base_path}-{i}.png"
        try:
            analyze_png_file(png_path)
        except Exception as e:
            print(f"Error analyzing {png_path}: {e}")
    
    print(f"\n{'='*70}")
    print("FILE TYPE RELEVANCE SUMMARY")
    print(f"{'='*70}")
    print("""
.pt FILE (TensorDict):
  - Contains: Structured trajectory data with observations, actions, rewards, etc.
  - Structure: Each timestep has:
    * 'obs': Observation vector (state representation)
    * 'action': Action vector
    * 'reward': Scalar reward value
    * 'terminated': Boolean termination flag
    * 'episode': Episode identifier
  - Usage: Primary data source for training world models and policies
  - Format: PyTorch TensorDict (efficient batch operations)
  
.png FILES (Concatenated Frames):
  - Contains: Visual observations concatenated horizontally
  - Structure: Very wide images (e.g., 897792 x 224) containing many frames
  - Purpose: Visual representation of the environment states
  - Usage: Can be used for visualization or converted to individual frames
  - Format: RGB images, uint8, concatenated frames side-by-side
  
RELATIONSHIP:
  - The .pt file contains the structured, numerical data (observations as vectors)
  - The .png files contain the visual/raw pixel data
  - They represent the same trajectories but in different formats:
    * .pt: Processed, structured data ready for model training
    * .png: Raw visual data for visualization or image-based models
  - The numbered PNG files (-0, -1, -2) likely represent different splits
    or different portions of the dataset
    """)

if __name__ == "__main__":
    main()
