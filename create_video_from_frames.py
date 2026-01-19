#!/usr/bin/env python3
"""
Create video file from extracted MineRL frames.

Converts frames.npy numpy arrays back into .mp4 video files for visualization.
Uses imageio for better codec support.
"""

import argparse
from pathlib import Path
import numpy as np

try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False
    import cv2

def get_fps_from_metadata(frames_path: Path) -> float:
    """Get FPS from metadata.json if available."""
    metadata_path = frames_path.parent / "metadata.json"
    if metadata_path.exists():
        try:
            import json
            with open(metadata_path) as f:
                metadata = json.load(f)
            
            # Try calculated_fps first (added by extraction script)
            if 'calculated_fps' in metadata:
                return metadata['calculated_fps']
            
            # Calculate from duration and frame count
            duration_ms = metadata.get('duration_ms', 0)
            true_video_frame_count = metadata.get('true_video_frame_count', 0)
            if duration_ms > 0 and true_video_frame_count > 0:
                duration_sec = duration_ms / 1000.0
                return true_video_frame_count / duration_sec
        except Exception as e:
            print(f"  Warning: Could not read FPS from metadata: {e}")
    
    return None

def create_video_with_imageio(frames_path: Path, output_path: Path, fps: float = None):
    """Create video using imageio (better codec support)."""
    print(f"Loading frames from: {frames_path}")
    frames = np.load(frames_path)
    
    print(f"Frames shape: {frames.shape}")
    print(f"Number of frames: {frames.shape[0]}")
    
    # Get FPS from metadata if not provided
    if fps is None:
        fps_from_metadata = get_fps_from_metadata(frames_path)
        if fps_from_metadata:
            fps = fps_from_metadata
            print(f"Using FPS from metadata: {fps:.2f}")
        else:
            fps = 20.0
            print(f"Using default FPS: {fps}")
    else:
        print(f"Using specified FPS: {fps}")
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating video: {output_path}")
    print(f"Resolution: {frames.shape[2]}x{frames.shape[1]}, FPS: {fps:.2f}")
    
    # Use imageio to write video with H.264 codec
    writer = imageio.get_writer(
        str(output_path),
        fps=fps,
        codec='libx264',
        quality=8,
        pixelformat='yuv420p'  # Ensures compatibility
    )
    
    for i, frame in enumerate(frames):
        writer.append_data(frame)
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(frames)} frames")
    
    writer.close()
    print(f"✓ Video created: {output_path}")
    print(f"  Total frames: {len(frames)}")
    print(f"  Duration: {len(frames) / fps:.2f} seconds")

def create_video_with_opencv(frames_path: Path, output_path: Path, fps: float = None):
    """Create video using OpenCV (fallback)."""
    print(f"Loading frames from: {frames_path}")
    frames = np.load(frames_path)
    
    print(f"Frames shape: {frames.shape}")
    print(f"Number of frames: {frames.shape[0]}")
    
    # Get FPS from metadata if not provided
    if fps is None:
        fps_from_metadata = get_fps_from_metadata(frames_path)
        if fps_from_metadata:
            fps = fps_from_metadata
            print(f"Using FPS from metadata: {fps:.2f}")
        else:
            fps = 20.0
            print(f"Using default FPS: {fps}")
    else:
        print(f"Using specified FPS: {fps}")
    
    # Get video dimensions
    height, width = frames.shape[1], frames.shape[2]
    
    # If output is .mp4, try to use .avi instead for better compatibility
    if output_path.suffix == '.mp4':
        output_path = output_path.with_suffix('.avi')
        print(f"Note: Using .avi format for better codec support")
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    if not out.isOpened():
        raise RuntimeError("Could not open video writer. Try installing imageio: pip install imageio imageio-ffmpeg")
    
    print(f"Creating video: {output_path}")
    print(f"Resolution: {width}x{height}, FPS: {fps}")
    
    for i, frame in enumerate(frames):
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
        
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(frames)} frames")
    
    out.release()
    print(f"✓ Video created: {output_path}")
    print(f"  Total frames: {len(frames)}")
    print(f"  Duration: {len(frames) / fps:.2f} seconds")

def main():
    parser = argparse.ArgumentParser(description="Create video from extracted MineRL frames")
    parser.add_argument("--frames", type=str, required=True,
                       help="Path to frames.npy file")
    parser.add_argument("--output", type=str, required=True,
                       help="Output video file path (.mp4 or .avi)")
    parser.add_argument("--fps", type=float, default=None,
                       help="Frames per second (default: auto-detect from metadata, or 20)")
    parser.add_argument("--format", type=str, choices=['mp4', 'avi'], default='mp4',
                       help="Output format (default: mp4)")
    
    args = parser.parse_args()
    
    frames_path = Path(args.frames)
    output_path = Path(args.output)
    
    if not frames_path.exists():
        print(f"Error: Frames file not found: {frames_path}")
        return
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Set extension based on format
    if args.format == 'avi':
        output_path = output_path.with_suffix('.avi')
    elif args.format == 'mp4':
        output_path = output_path.with_suffix('.mp4')
    
    # Use imageio if available (better codec support)
    if HAS_IMAGEIO:
        try:
            create_video_with_imageio(frames_path, output_path, args.fps)
        except Exception as e:
            print(f"Error with imageio: {e}")
            print("Falling back to OpenCV...")
            create_video_with_opencv(frames_path, output_path, args.fps)
    else:
        print("imageio not available, using OpenCV (may have codec issues)")
        print("For better results, install: pip install imageio imageio-ffmpeg")
        create_video_with_opencv(frames_path, output_path, args.fps)

if __name__ == "__main__":
    main()
