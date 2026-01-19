#!/usr/bin/env python3
"""
Extract POV frames from MineRL dataset video files.

Extracts frames from .mp4 files in MineRL zip archives and saves them
as numpy arrays or images for training.
"""

import os
import sys
import zipfile
import json
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
from PIL import Image
import io
import cv2

try:
    import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("Warning: tqdm not installed, progress bars disabled")


def extract_frames_from_video(video_bytes: bytes, target_fps: int = 20, target_size: tuple = (64, 64)) -> np.ndarray:
    """
    Extract frames from video bytes.
    
    Args:
        video_bytes: Raw video file bytes
        target_fps: Target frames per second (downsample if needed)
        target_size: Target frame size (height, width)
    
    Returns:
        Frames as numpy array (T, H, W, C) in RGB format, values in [0, 255]
    """
    # Write video bytes to temporary file-like object
    video_buffer = io.BytesIO(video_bytes)
    
    # Use OpenCV to read video
    # We need to save to a temporary file for OpenCV
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
        tmp_file.write(video_bytes)
        tmp_path = tmp_file.name
    
    try:
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame skip to achieve target_fps
        frame_skip = max(1, int(fps / target_fps))
        
        frames = []
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Only keep frames at target FPS
            if frame_idx % frame_skip == 0:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Resize to target size
                if frame_rgb.shape[:2] != target_size:
                    frame_rgb = cv2.resize(frame_rgb, (target_size[1], target_size[0]), 
                                         interpolation=cv2.INTER_LINEAR)
                
                frames.append(frame_rgb)
            
            frame_idx += 1
        
        cap.release()
        
        if len(frames) == 0:
            return np.array([])
        
        # Stack frames: (T, H, W, C)
        frames_array = np.stack(frames, axis=0).astype(np.uint8)
        return frames_array
    
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def extract_episode_data(zip_path: Path, episode_path: str, output_dir: Path, 
                        target_size: tuple = (64, 64), target_fps: int = 20):
    """
    Extract frames, actions, and rewards from a single episode.
    
    Args:
        zip_path: Path to the MineRL zip file
        episode_path: Path to episode directory within zip
        output_dir: Directory to save extracted data
        target_size: Target frame size (height, width)
        target_fps: Target frames per second
    
    Returns:
        Dictionary with extraction results
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'r') as zf:
        # Find files
        episode_files = [f for f in zf.namelist() if f.startswith(episode_path)]
        
        video_file = None
        npz_file = None
        metadata_file = None
        
        for f in episode_files:
            if f.endswith('.mp4') and 'recording' in f:
                video_file = f
            elif f.endswith('rendered.npz'):
                npz_file = f
            elif f.endswith('metadata.json'):
                metadata_file = f
        
        results = {
            "episode": episode_path,
            "frames_extracted": 0,
            "actions_shape": None,
            "rewards_shape": None,
            "success": False
        }
        
        # Extract frames from video
        if video_file:
            try:
                with zf.open(video_file) as f:
                    video_bytes = f.read()
                    frames = extract_frames_from_video(video_bytes, target_fps, target_size)
                    
                    if frames.size > 0:
                        # Save frames as numpy array
                        frames_file = output_dir / "frames.npy"
                        np.save(frames_file, frames)
                        results["frames_extracted"] = frames.shape[0]
                        results["frames_shape"] = frames.shape
            except Exception as e:
                print(f"  Warning: Could not extract frames: {e}")
        
        # Extract actions and rewards from NPZ
        if npz_file:
            try:
                with zf.open(npz_file) as f:
                    npz_data = np.load(io.BytesIO(f.read()))
                    
                    # Extract actions
                    action_keys = [k for k in npz_data.keys() if k.startswith('action$')]
                    if action_keys:
                        actions_dict = {k: npz_data[k] for k in action_keys}
                        
                        # Save actions
                        actions_file = output_dir / "actions.npz"
                        np.savez(actions_file, **actions_dict)
                        results["actions_shape"] = {k: v.shape for k, v in actions_dict.items()}
                    
                    # Extract rewards
                    if 'reward' in npz_data:
                        rewards = npz_data['reward']
                        rewards_file = output_dir / "rewards.npy"
                        np.save(rewards_file, rewards)
                        results["rewards_shape"] = rewards.shape
                    
                    # Save all observations
                    obs_keys = [k for k in npz_data.keys() if k.startswith('observation$')]
                    if obs_keys:
                        obs_dict = {k: npz_data[k] for k in obs_keys}
                        obs_file = output_dir / "observations.npz"
                        np.savez(obs_file, **obs_dict)
            
            except Exception as e:
                print(f"  Warning: Could not extract NPZ data: {e}")
        
        # Extract metadata
        if metadata_file:
            try:
                with zf.open(metadata_file) as f:
                    metadata = json.load(f)
                    
                    # Calculate FPS from metadata
                    duration_ms = metadata.get('duration_ms', 0)
                    true_video_frame_count = metadata.get('true_video_frame_count', 0)
                    if duration_ms > 0 and true_video_frame_count > 0:
                        duration_sec = duration_ms / 1000.0
                        calculated_fps = true_video_frame_count / duration_sec
                        metadata['calculated_fps'] = round(calculated_fps, 2)
                    
                    metadata_file_out = output_dir / "metadata.json"
                    with open(metadata_file_out, 'w') as mf:
                        json.dump(metadata, mf, indent=2)
            except Exception as e:
                print(f"  Warning: Could not extract metadata: {e}")
        
        results["success"] = results["frames_extracted"] > 0
        
        return results


def process_zip_file(zip_path: Path, output_base_dir: Path, 
                    target_size: tuple = (64, 64), target_fps: int = 20,
                    max_episodes: Optional[int] = None):
    """
    Process all episodes in a MineRL zip file.
    
    Args:
        zip_path: Path to MineRL zip file
        output_base_dir: Base directory for output
        target_size: Target frame size (height, width)
        target_fps: Target frames per second
        max_episodes: Maximum number of episodes to process (None = all)
    """
    env_name = zip_path.stem
    output_dir = output_base_dir / env_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print(f"Processing {env_name}")
    print("=" * 70)
    print(f"Output directory: {output_dir}")
    print(f"Target frame size: {target_size}")
    print(f"Target FPS: {target_fps}\n")
    
    with zipfile.ZipFile(zip_path, 'r') as zf:
        # Find all episode directories
        file_list = zf.namelist()
        episode_dirs = set()
        
        for f in file_list:
            parts = f.split('/')
            if len(parts) >= 2:
                ep_dir = '/'.join(parts[:2])  # environment/episode
                if ep_dir != env_name:  # Skip root directory
                    episode_dirs.add(ep_dir)
        
        episode_dirs = sorted(list(episode_dirs))
        
        if max_episodes:
            episode_dirs = episode_dirs[:max_episodes]
        
        print(f"Found {len(episode_dirs)} episodes")
        print(f"Processing {len(episode_dirs)} episodes...\n")
        
        successful = 0
        failed = 0
        
        iterator = tqdm.tqdm(episode_dirs) if HAS_TQDM else episode_dirs
        
        for ep_dir in iterator:
            ep_name = ep_dir.split('/')[-1]
            ep_output_dir = output_dir / ep_name
            
            try:
                result = extract_episode_data(
                    zip_path, ep_dir, ep_output_dir, 
                    target_size=target_size, target_fps=target_fps
                )
                
                if result["success"]:
                    successful += 1
                else:
                    failed += 1
                    if not HAS_TQDM:
                        print(f"  ✗ {ep_name}: Failed")
            
            except Exception as e:
                failed += 1
                if not HAS_TQDM:
                    print(f"  ✗ {ep_name}: Error - {e}")
        
        print(f"\nCompleted: {successful} successful, {failed} failed")
        return successful, failed


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract POV frames from MineRL dataset")
    parser.add_argument("--input-dir", type=str, 
                       default="/media/skr/storage/dreamerv4/data/mineRL",
                       help="Directory containing MineRL zip files")
    parser.add_argument("--output-dir", type=str,
                       default="/media/skr/storage/dreamerv4/data/mineRL_extracted",
                       help="Output directory for extracted frames")
    parser.add_argument("--target-size", type=int, nargs=2, default=[64, 64],
                       help="Target frame size (height width)")
    parser.add_argument("--target-fps", type=int, default=20,
                       help="Target frames per second")
    parser.add_argument("--max-episodes", type=int, default=None,
                       help="Maximum episodes per environment (None = all)")
    parser.add_argument("--environments", type=str, nargs="+", default=None,
                       help="Specific environments to process (default: all)")
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    # Find zip files
    zip_files = list(input_dir.glob("*.zip"))
    
    if not zip_files:
        print(f"No zip files found in {input_dir}")
        return
    
    # Filter by environment if specified
    if args.environments:
        zip_files = [zf for zf in zip_files if any(env in zf.name for env in args.environments)]
    
    print(f"Found {len(zip_files)} zip file(s) to process")
    
    total_successful = 0
    total_failed = 0
    
    for zip_file in zip_files:
        successful, failed = process_zip_file(
            zip_file, output_dir,
            target_size=tuple(args.target_size),
            target_fps=args.target_fps,
            max_episodes=args.max_episodes
        )
        total_successful += successful
        total_failed += failed
        print()
    
    print("=" * 70)
    print("EXTRACTION SUMMARY")
    print("=" * 70)
    print(f"Total successful: {total_successful}")
    print(f"Total failed: {total_failed}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
