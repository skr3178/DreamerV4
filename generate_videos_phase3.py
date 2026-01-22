#!/usr/bin/env python3
"""
Generate videos from Phase 3 trained model.

This script:
1. Loads Phase 2 checkpoint for tokenizer and dynamics (world model)
2. Loads Phase 3 checkpoint for trained heads (policy, value, reward)
3. Takes input frames from the dataset
4. Uses the dynamics model to predict future frames
5. Decodes latents back to video frames
6. Saves as video file
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
    import cv2

# Add flush helper
def print_flush(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()
    sys.stderr.flush()

from dreamer.models import CausalTokenizer, DynamicsModel
from dreamer.models import PolicyHead, ValueHead, RewardHead
from dreamer.data import create_dataloader
from dreamer.utils import set_seed, load_phase2_world_model as _load_phase2_world_model, load_phase3_heads as _load_phase3_heads
from eval_phase2 import load_config, create_tokenizer, create_dynamics_model, create_heads


def load_phase2_world_model(checkpoint_path: str, config: Dict, device: torch.device):
    """Load Phase 2 checkpoint for tokenizer and dynamics only (not heads)."""
    return _load_phase2_world_model(
        checkpoint_path, config, device,
        create_tokenizer_fn=create_tokenizer,
        create_dynamics_fn=create_dynamics_model,
    )


def load_phase3_heads(checkpoint_path: str, config: Dict, device: torch.device):
    """Load Phase 3 checkpoint for heads (policy, value, reward)."""
    return _load_phase3_heads(
        checkpoint_path, config, device,
        create_heads_fn=create_heads,
    )


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
        except Exception as e:
            print_flush(f"Warning: Decode failed: {e}, using fallback")
            # Fallback: create dummy frame
            decoded_frame = torch.zeros(B, 3, 64, 64, device=latents.device)
        
        all_frames.append(decoded_frame)
    
    frames = torch.stack(all_frames, dim=1)  # (B, T, C, H, W)
    
    # Normalize to [0, 1] if needed
    if frames.min() < 0:
        frames = (frames + 1) / 2  # [-1, 1] -> [0, 1]
    frames = frames.clamp(0, 1)
    
    return frames


def generate_rollout_video(
    tokenizer: CausalTokenizer,
    dynamics: DynamicsModel,
    heads: Dict,
    initial_frames: torch.Tensor,
    actions: torch.Tensor,
    device: torch.device,
    rollout_steps: int = 16,
) -> torch.Tensor:
    """
    Generate a video rollout using dynamics model.
    
    Args:
        tokenizer: Tokenizer model
        dynamics: Dynamics model
        heads: Dictionary of heads (policy, value, reward)
        initial_frames: (B, T, C, H, W) initial frames
        actions: (B, T) actions for rollout
        device: Device
        rollout_steps: Number of steps to roll out
    
    Returns:
        generated_frames: (B, rollout_steps, C, H, W) generated frames
    """
    tokenizer.eval()
    dynamics.eval()
    
    # Reshape frames for tokenizer: (B, C, T, H, W)
    if initial_frames.dim() == 5 and initial_frames.shape[2] != tokenizer.in_channels:
        frames_reshaped = initial_frames.permute(0, 2, 1, 3, 4)
    else:
        frames_reshaped = initial_frames
    
    B = initial_frames.shape[0]
    
    with torch.inference_mode():
        # Encode initial frames to latents
        tokenizer_output = tokenizer.encode(frames_reshaped, mask_ratio=0.0)
        initial_latents = tokenizer_output["latents"]  # (B, T, num_latent, latent_dim)
        
        # Start with initial latents
        current_latents = initial_latents  # (B, T, num_latent, latent_dim)
        
        generated_frames_list = []
        
        # Rollout for specified steps
        for step in range(rollout_steps):
            # Get action for this step (use last action or zero if not available)
            if step < actions.shape[1]:
                action = actions[:, step:step+1]  # (B, 1)
            else:
                # Use policy head to predict action
                # Flatten latents for head
                batch_size, time_steps, num_latent, latent_dim = current_latents.shape
                latents_flat = current_latents.reshape(batch_size, time_steps, -1)
                
                # Get last timestep's latent and reshape for policy head
                last_latent = latents_flat[:, -1]  # (B, input_dim) - squeeze time dimension
                
                # Predict action using policy head
                policy_output = heads["policy"](last_latent)
                if isinstance(policy_output, dict):
                    logits = policy_output.get("logits", policy_output.get("action_logits"))
                else:
                    logits = policy_output
                
                # Sample action - handle different logit shapes
                if logits.dim() == 4:
                    # MTP output: (B, 1, L+1, num_actions) -> take first prediction
                    logits = logits[:, 0, 0]  # (B, num_actions)
                elif logits.dim() == 3:
                    # MTP output: (B, L+1, num_actions) -> take first prediction
                    logits = logits[:, 0]  # (B, num_actions)
                
                # Now logits should be 2D (B, num_actions)
                # argmax gives (B,) -> unsqueeze to (B, 1)
                action = torch.argmax(logits, dim=-1).unsqueeze(-1)  # (B, 1)
            
            # Use dynamics to predict next latent
            last_latent_step = current_latents[:, -1:]  # (B, 1, num_latent, latent_dim)
            
            # Prepare action: dynamics expects (B, T) for discrete actions
            if action.dim() == 2:
                action_for_dynamics = action  # (B, 1)
            else:
                action_for_dynamics = action.unsqueeze(1)  # (B, 1)
            
            # Use dynamics forward pass
            dynamics_output = dynamics(
                latents=last_latent_step,
                actions=action_for_dynamics,
                discrete_actions=True,
            )
            
            # Get predicted next latent
            predicted_latent = dynamics_output["predicted_latents"][:, -1:]  # (B, 1, num_latent, latent_dim)
            
            # Append to sequence
            current_latents = torch.cat([current_latents, predicted_latent], dim=1)
            
            # Decode predicted latent to frame
            decoded_frame = decode_latents_to_frames(tokenizer, predicted_latent)  # (B, 1, C, H, W)
            generated_frames_list.append(decoded_frame[:, 0])  # (B, C, H, W)
        
        # Stack all generated frames
        generated_frames = torch.stack(generated_frames_list, dim=1)  # (B, rollout_steps, C, H, W)
        
        return generated_frames


def save_video(frames: torch.Tensor, output_path: str, fps: int = 10, codec: str = 'libx264'):
    """
    Save frames as video file with proper codec.
    
    Args:
        frames: (T, H, W, C) or (B, T, C, H, W) numpy array in [0, 1] range
        output_path: Path to save video
        fps: Frames per second
        codec: Video codec ('libx264' for H.264, 'libx265' for H.265, 'mpeg4' for MPEG-4)
    """
    # Handle different input shapes
    if frames.dim() == 5:
        # (B, T, C, H, W) -> take first batch, convert to (T, H, W, C)
        frames = frames[0].permute(0, 2, 3, 1).cpu().numpy()
    elif frames.dim() == 4 and frames.shape[1] == 3:
        # (T, C, H, W) -> (T, H, W, C)
        frames = frames.permute(0, 2, 3, 1).cpu().numpy()
    
    T, H, W, C = frames.shape
    
    # Convert to uint8 and ensure RGB format
    frames_uint8 = (frames * 255).astype(np.uint8)
    
    # Use imageio if available (better codec support)
    if HAS_IMAGEIO:
        try:
            writer = imageio.get_writer(
                output_path,
                fps=fps,
                codec=codec,
                quality=8,
                pixelformat='yuv420p'
            )
            for frame in frames_uint8:
                writer.append_data(frame)
            writer.close()
            print_flush(f"✓ Saved video to {output_path} (codec: {codec})")
            return
        except Exception as e:
            print_flush(f"  Warning: imageio failed: {e}, trying OpenCV...")
    
    # Fallback to OpenCV
    if C == 3:
        frames_bgr = frames_uint8[..., ::-1]  # RGB -> BGR
    else:
        frames_bgr = frames_uint8
    
    # Try different codecs
    codecs_to_try = [
        ('H264', cv2.VideoWriter_fourcc(*'H264')),
        ('XVID', cv2.VideoWriter_fourcc(*'XVID')),
        ('mp4v', cv2.VideoWriter_fourcc(*'mp4v')),
    ]
    
    video_written = False
    for codec_name, fourcc in codecs_to_try:
        try:
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (W, H))
            if out.isOpened():
                for frame in frames_bgr:
                    out.write(frame)
                out.release()
                video_written = True
                print_flush(f"✓ Saved video to {output_path} (codec: {codec_name})")
                break
        except Exception as e:
            print_flush(f"  Warning: Codec {codec_name} failed: {e}, trying next...")
            continue
    
    if not video_written:
        # Final fallback: Save as image sequence
        print_flush(f"  Warning: All codecs failed, saving as image sequence...")
        output_dir = Path(output_path).parent
        output_stem = Path(output_path).stem
        for i, frame in enumerate(frames_uint8):
            frame_path = output_dir / f"{output_stem}_frame_{i:04d}.png"
            cv2.imwrite(str(frame_path), frame)
        print_flush(f"✓ Saved {len(frames_uint8)} frames as PNG images in {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate videos from Phase 3 model")
    parser.add_argument(
        "--phase2-checkpoint",
        type=str,
        required=True,
        help="Path to Phase 2 checkpoint (for tokenizer and dynamics)"
    )
    parser.add_argument(
        "--phase3-checkpoint",
        type=str,
        required=True,
        help="Path to Phase 3 checkpoint (for heads)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/minerl_subset.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to data (overrides config)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="videos/phase3",
        help="Directory to save videos"
    )
    parser.add_argument(
        "--num-videos",
        type=int,
        default=3,
        help="Number of videos to generate"
    )
    parser.add_argument(
        "--rollout-steps",
        type=int,
        default=16,
        help="Number of steps to roll out"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use"
    )
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Setup device
    device = torch.device(args.device or config["experiment"]["device"])
    set_seed(config["experiment"]["seed"])
    
    print_flush(f"Device: {device}")
    print_flush(f"Phase 2 checkpoint: {args.phase2_checkpoint}")
    print_flush(f"Phase 3 checkpoint: {args.phase3_checkpoint}")
    
    # Load Phase 2 checkpoint (world model)
    tokenizer, dynamics, phase2_checkpoint = load_phase2_world_model(
        args.phase2_checkpoint, config, device
    )
    
    # Load Phase 3 checkpoint (heads)
    heads, phase3_checkpoint = load_phase3_heads(
        args.phase3_checkpoint, config, device
    )
    
    # Create data loader
    data_path = args.data_path or config["data"]["path"]
    print_flush(f"\nLoading data from: {data_path}")
    
    from dreamer.data.minerl_dataset import MineRLDataset
    from torch.utils.data import DataLoader
    
    eval_dataset = MineRLDataset(
        data_path=data_path,
        sequence_length=config["data"]["sequence_length"],
        image_size=(config["data"]["image_height"], config["data"]["image_width"]),
        split="train",
        max_episodes=config["data"].get("max_episodes", None),
    )
    
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=1,  # One video at a time
        shuffle=True,  # Shuffle to get different samples
        num_workers=0,
        pin_memory=True,
    )
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print_flush(f"\nGenerating {args.num_videos} videos...")
    print_flush(f"Output directory: {output_dir}")
    
    # Generate videos
    data_iter = iter(eval_loader)
    
    for video_idx in range(args.num_videos):
        print_flush(f"\n--- Generating Video {video_idx + 1}/{args.num_videos} ---")
        
        try:
            # Get a batch
            batch = next(data_iter)
        except StopIteration:
            # Restart iterator if we run out
            data_iter = iter(eval_loader)
            batch = next(data_iter)
        
        frames = batch["frames"].to(device)  # (B, T, C, H, W)
        actions = batch["actions"].to(device)  # (B, T, ...)
        
        # Handle action format
        if actions.dim() == 3 and actions.shape[-1] == 1:
            actions = actions.squeeze(-1)
        
        print_flush(f"  Input frames shape: {frames.shape}")
        print_flush(f"  Actions shape: {actions.shape}")
        
        # First, show reconstruction quality (encode and decode original frames)
        print_flush(f"  Testing reconstruction quality...")
        with torch.inference_mode():
            # Reshape for tokenizer
            if frames.dim() == 5 and frames.shape[2] != tokenizer.in_channels:
                frames_reshaped = frames.permute(0, 2, 1, 3, 4)
            else:
                frames_reshaped = frames
            
            # Encode and decode
            tokenizer_output = tokenizer.encode(frames_reshaped, mask_ratio=0.0)
            latents = tokenizer_output["latents"]  # (B, T, num_latent, latent_dim)
            reconstructed_frames = decode_latents_to_frames(tokenizer, latents)
        
        # Save reconstruction video
        recon_video_path = output_dir / f"reconstruction_{video_idx + 1}.mp4"
        save_video(reconstructed_frames, str(recon_video_path), fps=10)
        
        # Save original for comparison
        original_video_path = output_dir / f"original_{video_idx + 1}.mp4"
        save_video(frames, str(original_video_path), fps=10)
        
        # Generate rollout
        print_flush(f"  Generating {args.rollout_steps} step rollout...")
        try:
            generated_frames = generate_rollout_video(
                tokenizer=tokenizer,
                dynamics=dynamics,
                heads=heads,
                initial_frames=frames,
                actions=actions,
                device=device,
                rollout_steps=args.rollout_steps,
            )
            
            print_flush(f"  Generated frames shape: {generated_frames.shape}")
            
            # Save rollout video
            rollout_video_path = output_dir / f"rollout_{video_idx + 1}.mp4"
            save_video(generated_frames, str(rollout_video_path), fps=10)
        except Exception as e:
            print_flush(f"  Warning: Rollout generation failed: {e}")
            import traceback
            traceback.print_exc()
    
    print_flush(f"\n✓ Generated {args.num_videos} videos in {output_dir}")


if __name__ == "__main__":
    main()
