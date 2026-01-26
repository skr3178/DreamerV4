#!/usr/bin/env python3
"""
Generate comparison videos from Phase 1 trained model.

This script:
1. Loads a Phase 1 checkpoint (tokenizer, dynamics)
2. Takes input frames from the dataset
3. Generates:
   - Tokenizer reconstruction videos (encode-decode)
   - Dynamics prediction videos (rollout)
4. Creates side-by-side comparison videos with original ground truth
"""

import torch
import yaml
from pathlib import Path
import numpy as np
import argparse
import sys
from tqdm import tqdm
from typing import Dict

# Try to import imageio for better video codec support
try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False
    try:
        import cv2
    except ImportError:
        print("Warning: Neither imageio nor cv2 available. Install one for video saving.")
        HAS_IMAGEIO = False
        cv2 = None

# Add flush helper
def print_flush(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()
    sys.stderr.flush()

from dreamer.models import CausalTokenizer, DynamicsModel
from dreamer.data import create_dataloader
from eval_phase1 import load_phase1_checkpoint


def normalize_for_display(img: torch.Tensor) -> torch.Tensor:
    """Normalize image tensor to [0, 1] for display."""
    if img.min() < 0:
        img = (img + 1.0) / 2.0
    return img.clamp(0, 1)


def decode_latents_to_frames(tokenizer, latents):
    """
    Decode latents back to video frames.
    
    Args:
        tokenizer: CausalTokenizer model
        latents: (B, T, num_latent, latent_dim) tensor
    
    Returns:
        frames: (B, T, C, H, W) tensor in [0, 1] range
    """
    B, T, num_latent, latent_dim = latents.shape
    
    all_frames = []
    
    for t in range(T):
        # Get latents for this timestep
        latent_t = latents[:, t]  # (B, num_latent, latent_dim)
        
        # Decode using tokenizer
        try:
            decoded_frame = tokenizer.decode(latent_t)  # (B, C, H, W)
            all_frames.append(decoded_frame)
        except Exception as e:
            print_flush(f"Warning: Decode failed at timestep {t}: {e}")
            # Fallback: create dummy frame
            decoded_frame = torch.zeros(B, 3, 64, 64, device=latents.device)
            all_frames.append(decoded_frame)
    
    frames = torch.stack(all_frames, dim=1)  # (B, T, C, H, W)
    
    # Normalize to [0, 1] if needed
    frames = normalize_for_display(frames)
    
    return frames


def generate_tokenizer_reconstruction(
    tokenizer: CausalTokenizer,
    frames: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Generate reconstruction from tokenizer (encode-decode).
    
    Uses the reconstructed patches directly from the encode output,
    which is what the tokenizer was trained to produce.
    
    Args:
        tokenizer: Tokenizer model
        frames: (B, T, C, H, W) input frames
        device: Device
    
    Returns:
        reconstructed: (B, T, C, H, W) reconstructed frames
    """
    tokenizer.eval()
    
    # Reshape for tokenizer if needed
    if frames.dim() == 5 and frames.shape[2] != tokenizer.in_channels:
        frames_reshaped = frames.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)
    else:
        frames_reshaped = frames
    
    with torch.no_grad():
        # Encode - this returns reconstructed patches directly
        output = tokenizer.encode(frames_reshaped, mask_ratio=0.0)
        
        # Use reconstructed patches from encode output (not decode method)
        # This is what the tokenizer was trained to produce
        reconstructed_patches = output["reconstructed"]  # (B, T, num_patches, patch_dim)
        
        # Unpatchify to get frames
        B, T, num_patches, patch_dim = reconstructed_patches.shape
        all_frames = []
        
        for t in range(T):
            # Get reconstructed patches for this timestep
            patches_t = reconstructed_patches[:, t]  # (B, num_patches, patch_dim)
            
            # Unpatchify to get frame
            frame = tokenizer.decode_patches(patches_t)  # (B, C, H, W)
            all_frames.append(frame)
        
        reconstructed = torch.stack(all_frames, dim=1)  # (B, T, C, H, W)
        
        # Normalize to [0, 1] for display
        reconstructed = normalize_for_display(reconstructed)
    
    return reconstructed


def generate_dynamics_rollout(
    tokenizer: CausalTokenizer,
    dynamics: DynamicsModel,
    initial_frames: torch.Tensor,
    actions: torch.Tensor,
    device: torch.device,
    rollout_steps: int = 16,
    num_denoising_steps: int = 4,
) -> torch.Tensor:
    """
    Generate video rollout using dynamics model with proper shortcut forcing.
    
    Uses the dynamics.generate() method which properly handles shortcut forcing
    with iterative denoising for better long-term stability.
    
    Args:
        tokenizer: Tokenizer model
        dynamics: Dynamics model
        initial_frames: (B, T, C, H, W) initial frames
        actions: (B, T) actions for rollout
        device: Device
        rollout_steps: Number of steps to rollout
        num_denoising_steps: Number of denoising steps (K) for shortcut forcing
    
    Returns:
        generated_frames: (B, rollout_steps, C, H, W) generated frames
    """
    tokenizer.eval()
    dynamics.eval()
    
    # Reshape frames for tokenizer
    if initial_frames.dim() == 5 and initial_frames.shape[2] != tokenizer.in_channels:
        frames_reshaped = initial_frames.permute(0, 2, 1, 3, 4)
    else:
        frames_reshaped = initial_frames
    
    B = initial_frames.shape[0]
    
    with torch.no_grad():
        # Encode initial frames
        tokenizer_output = tokenizer.encode(frames_reshaped, mask_ratio=0.0)
        initial_latents = tokenizer_output["latents"]  # (B, T, num_latent, latent_dim)
        
        # Use last frame's latent as starting point
        starting_latent = initial_latents[:, -1]  # (B, num_latent, latent_dim)
        
        # Prepare actions for rollout (pad if needed)
        if actions.shape[1] < rollout_steps:
            # Repeat last action to fill rollout
            last_action = actions[:, -1:]  # (B, 1)
            actions_rollout = torch.cat([actions, last_action.repeat(1, rollout_steps - actions.shape[1])], dim=1)
        else:
            actions_rollout = actions[:, :rollout_steps]  # (B, rollout_steps)
        
        # Use dynamics.generate() method which properly handles shortcut forcing
        # This uses iterative denoising for better stability
        generated_latents = dynamics.generate(
            initial_latents=starting_latent,
            actions=actions_rollout,
            num_steps=num_denoising_steps,  # K=4 denoising steps
            discrete_actions=True,
        )  # (B, rollout_steps, num_latent, latent_dim)
        
        # Decode all latents to frames
        generated_frames = decode_latents_to_frames(tokenizer, generated_latents)
    
    return generated_frames


def create_side_by_side_video(
    original: torch.Tensor,
    generated: torch.Tensor,
    output_path: Path,
    fps: int = 10,
    labels: tuple = ("Original", "Generated"),
):
    """
    Create side-by-side comparison video.
    
    Args:
        original: (T, C, H, W) or (B, T, C, H, W) original frames
        generated: (T, C, H, W) or (B, T, C, H, W) generated frames
        output_path: Path to save video
        fps: Frames per second
        labels: Tuple of (left_label, right_label)
    """
    # Handle batch dimension
    if original.dim() == 5:
        original = original[0]  # (T, C, H, W)
    if generated.dim() == 5:
        generated = generated[0]  # (T, C, H, W)
    
    T, C, H, W = original.shape
    
    # Ensure same number of frames
    min_T = min(original.shape[0], generated.shape[0])
    original = original[:min_T]
    generated = generated[:min_T]
    
    # Convert to numpy and normalize
    original_np = normalize_for_display(original).permute(0, 2, 3, 1).cpu().numpy()  # (T, H, W, C)
    generated_np = normalize_for_display(generated).permute(0, 2, 3, 1).cpu().numpy()  # (T, H, W, C)
    
    # Create side-by-side frames
    side_by_side = np.concatenate([original_np, generated_np], axis=2)  # (T, H, 2*W, C)
    
    # Convert to uint8
    frames_uint8 = (side_by_side * 255).astype(np.uint8)
    
    # Save video
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if HAS_IMAGEIO:
        try:
            writer = imageio.get_writer(
                str(output_path),
                fps=fps,
                codec='libx264',
                quality=8,
                pixelformat='yuv420p'
            )
            for frame in frames_uint8:
                writer.append_data(frame)
            writer.close()
            print_flush(f"✓ Saved comparison video to {output_path}")
            return
        except Exception as e:
            print_flush(f"Warning: imageio failed: {e}, trying OpenCV...")
    
    # Fallback to OpenCV
    if cv2 is not None:
        frames_bgr = frames_uint8[..., ::-1]  # RGB -> BGR
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (2*W, H))
        if out.isOpened():
            for frame in frames_bgr:
                out.write(frame)
            out.release()
            print_flush(f"✓ Saved comparison video to {output_path}")
        else:
            print_flush(f"Error: Could not open video writer for {output_path}")
    else:
        print_flush(f"Error: No video library available. Install imageio or opencv-python")


def save_video(frames: torch.Tensor, output_path: Path, fps: int = 10):
    """Save frames as video file."""
    # Handle batch dimension
    if frames.dim() == 5:
        frames = frames[0]  # (T, C, H, W)
    
    T, C, H, W = frames.shape
    
    # Normalize and convert to numpy
    frames_np = normalize_for_display(frames).permute(0, 2, 3, 1).cpu().numpy()  # (T, H, W, C)
    frames_uint8 = (frames_np * 255).astype(np.uint8)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if HAS_IMAGEIO:
        try:
            writer = imageio.get_writer(
                str(output_path),
                fps=fps,
                codec='libx264',
                quality=8,
                pixelformat='yuv420p'
            )
            for frame in frames_uint8:
                writer.append_data(frame)
            writer.close()
            print_flush(f"✓ Saved video to {output_path}")
            return
        except Exception as e:
            print_flush(f"Warning: imageio failed: {e}")
    
    if cv2 is not None:
        frames_bgr = frames_uint8[..., ::-1]  # RGB -> BGR
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (W, H))
        if out.isOpened():
            for frame in frames_bgr:
                out.write(frame)
            out.release()
            print_flush(f"✓ Saved video to {output_path}")
        else:
            print_flush(f"Error: Could not open video writer")


def main():
    parser = argparse.ArgumentParser(description="Generate comparison videos from Phase 1 checkpoint")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/minerl.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to Phase 1 checkpoint (default: latest in checkpoints/dreamerv4_minerl/)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="videos/phase1_comparisons",
        help="Output directory for videos",
    )
    parser.add_argument(
        "--num_videos",
        type=int,
        default=5,
        help="Number of videos to generate",
    )
    parser.add_argument(
        "--rollout_steps",
        type=int,
        default=16,
        help="Number of steps for dynamics rollout",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Frames per second for output videos",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (overrides config)",
    )
    args = parser.parse_args()
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        requested_device = config["experiment"]["device"]
        if requested_device == "cuda" and not torch.cuda.is_available():
            print_flush("⚠️  Warning: CUDA requested but not available. Falling back to CPU.")
            device = torch.device("cpu")
        else:
            device = torch.device(requested_device)
    
    print_flush(f"Using device: {device}")
    
    # Find checkpoint
    if args.checkpoint is None:
        checkpoint_dir = Path(config["experiment"]["checkpoint_dir"]) / config["experiment"]["name"]
        checkpoints = sorted(checkpoint_dir.glob("phase1_*.pt"), key=lambda x: x.stat().st_mtime, reverse=True)
        if not checkpoints:
            print_flush(f"Error: No checkpoints found in {checkpoint_dir}")
            sys.exit(1)
        checkpoint_path = checkpoints[0]
        print_flush(f"Using latest checkpoint: {checkpoint_path}")
    else:
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            print_flush(f"Error: Checkpoint not found at {checkpoint_path}")
            sys.exit(1)
    
    # Load checkpoint
    print_flush(f"Loading checkpoint from {checkpoint_path}")
    tokenizer, dynamics, ckpt = load_phase1_checkpoint(checkpoint_path, config, device)
    tokenizer.eval()
    dynamics.eval()
    
    print_flush(f"Loaded checkpoint from step {ckpt.get('global_step', 'unknown')}")
    print_flush(f"Epoch: {ckpt.get('epoch', 'unknown')}")
    
    # Create dataloader
    print_flush("\nCreating data loader...")
    from torch.utils.data import DataLoader
    from dreamer.data.minerl_dataset import MineRLDataset
    
    dataset = MineRLDataset(
        data_path=config["data"]["path"],
        sequence_length=config["data"]["sequence_length"],
        image_size=(config["data"]["image_height"], config["data"]["image_width"]),
        split="train",
        max_episodes=config["data"].get("max_episodes", None),
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print_flush(f"\nGenerating {args.num_videos} comparison videos...")
    print_flush(f"Output directory: {output_dir}")
    
    data_iter = iter(dataloader)
    
    for video_idx in range(args.num_videos):
        print_flush(f"\n--- Generating Video {video_idx + 1}/{args.num_videos} ---")
        
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)
        
        frames = batch["frames"].to(device)  # (B, T, C, H, W)
        actions = batch["actions"].to(device)  # (B, T)
        
        # Handle action format
        if actions.dim() == 3 and actions.shape[-1] == 1:
            actions = actions.squeeze(-1)
        
        print_flush(f"  Input frames shape: {frames.shape}")
        print_flush(f"  Actions shape: {actions.shape}")
        
        # 1. Tokenizer reconstruction
        print_flush(f"  Generating tokenizer reconstruction...")
        with torch.no_grad():
            reconstructed = generate_tokenizer_reconstruction(tokenizer, frames, device)
        
        # Save original and reconstruction comparison
        recon_comparison_path = output_dir / f"reconstruction_{video_idx + 1}.mp4"
        create_side_by_side_video(
            frames,
            reconstructed,
            recon_comparison_path,
            fps=args.fps,
            labels=("Original", "Reconstructed"),
        )
        
        # 2. Dynamics rollout
        print_flush(f"  Generating dynamics rollout ({args.rollout_steps} steps)...")
        with torch.no_grad():
            rollout_frames = generate_dynamics_rollout(
                tokenizer,
                dynamics,
                frames,
                actions,
                device,
                rollout_steps=args.rollout_steps,
            )
        
        # Create comparison: original initial frames + generated rollout
        # Use first few frames as context, then show rollout
        context_frames = frames[:, :min(4, frames.shape[1])]  # First 4 frames
        full_generated = torch.cat([context_frames, rollout_frames], dim=1)  # (B, context+rollout, C, H, W)
        
        # Pad original to same length for comparison
        if full_generated.shape[1] > frames.shape[1]:
            # Repeat last frame to match length
            last_frame = frames[:, -1:].repeat(1, full_generated.shape[1] - frames.shape[1], 1, 1, 1)
            original_padded = torch.cat([frames, last_frame], dim=1)
        else:
            original_padded = frames[:, :full_generated.shape[1]]
        
        rollout_comparison_path = output_dir / f"rollout_{video_idx + 1}.mp4"
        create_side_by_side_video(
            original_padded,
            full_generated,
            rollout_comparison_path,
            fps=args.fps,
            labels=("Original", "Rollout"),
        )
        
        # Also save individual videos
        original_path = output_dir / f"original_{video_idx + 1}.mp4"
        save_video(frames, original_path, fps=args.fps)
        
        reconstructed_path = output_dir / f"reconstructed_only_{video_idx + 1}.mp4"
        save_video(reconstructed, reconstructed_path, fps=args.fps)
        
        rollout_path = output_dir / f"rollout_only_{video_idx + 1}.mp4"
        save_video(full_generated, rollout_path, fps=args.fps)
    
    print_flush(f"\n✓ Generated {args.num_videos} comparison videos in {output_dir}/")
    print_flush(f"  - reconstruction_*.mp4: Original vs Tokenizer reconstruction")
    print_flush(f"  - rollout_*.mp4: Original vs Dynamics rollout")
    print_flush(f"  - original_*.mp4: Original videos")
    print_flush(f"  - reconstructed_only_*.mp4: Tokenizer reconstructions")
    print_flush(f"  - rollout_only_*.mp4: Dynamics rollouts")


if __name__ == "__main__":
    main()
