#!/usr/bin/env python3
"""
Generate reconstruction comparison images from Phase 1 checkpoint.
Creates grid images showing original vs reconstructed frames.
"""

import torch
import yaml
from pathlib import Path
import numpy as np
import argparse
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Dict

# Add flush helper
def print_flush(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()
    sys.stderr.flush()

from dreamer.models import CausalTokenizer, DynamicsModel
from dreamer.data import create_dataloader
from eval_phase1 import load_phase1_checkpoint
from generate_videos_phase1 import generate_tokenizer_reconstruction, normalize_for_display


def create_reconstruction_grid(
    original_frames: torch.Tensor,
    reconstructed_frames: torch.Tensor,
    num_frames: int = 8,
    save_path: Path = None,
    title: str = "Original vs Reconstructed",
):
    """
    Create a grid image showing original vs reconstructed frames.
    
    Args:
        original_frames: (B, T, C, H, W) or (T, C, H, W) tensor
        reconstructed_frames: (B, T, C, H, W) or (T, C, H, W) tensor
        num_frames: Number of frames to display
        save_path: Path to save the image
        title: Title for the image
    """
    # Handle batch dimension
    if original_frames.dim() == 5:
        original_frames = original_frames[0]  # (T, C, H, W)
    if reconstructed_frames.dim() == 5:
        reconstructed_frames = reconstructed_frames[0]  # (T, C, H, W)
    
    T = original_frames.shape[0]
    num_frames = min(num_frames, T)
    
    # Select frames evenly spaced
    frame_indices = np.linspace(0, T - 1, num_frames, dtype=int)
    
    # Normalize frames to [0, 1]
    original_norm = normalize_for_display(original_frames)
    reconstructed_norm = normalize_for_display(reconstructed_frames)
    
    # Convert to numpy (T, H, W, C)
    original_np = original_norm.permute(0, 2, 3, 1).cpu().numpy()
    reconstructed_np = reconstructed_norm.permute(0, 2, 3, 1).cpu().numpy()
    
    # Create figure with two rows: original on top, reconstructed below
    fig, axes = plt.subplots(2, num_frames, figsize=(2 * num_frames, 4))
    if num_frames == 1:
        axes = axes.reshape(2, 1)
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    for i, t_idx in enumerate(frame_indices):
        # Original frame (top row)
        axes[0, i].imshow(original_np[t_idx])
        axes[0, i].set_title(f"Frame {t_idx}", fontsize=10)
        axes[0, i].axis('off')
        
        # Reconstructed frame (bottom row)
        axes[1, i].imshow(reconstructed_np[t_idx])
        axes[1, i].axis('off')
    
    # Add row labels
    fig.text(0.02, 0.75, 'Original', rotation=90, fontsize=12, fontweight='bold', va='center')
    fig.text(0.02, 0.25, 'Reconstructed', rotation=90, fontsize=12, fontweight='bold', va='center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print_flush(f"✓ Saved reconstruction grid to {save_path}")
    
    plt.close()


def create_side_by_side_grid(
    original_frames: torch.Tensor,
    reconstructed_frames: torch.Tensor,
    num_frames: int = 8,
    save_path: Path = None,
    title: str = "Original vs Reconstructed",
):
    """
    Create a grid image showing original and reconstructed frames side by side.
    
    Args:
        original_frames: (B, T, C, H, W) or (T, C, H, W) tensor
        reconstructed_frames: (B, T, C, H, W) or (T, C, H, W) tensor
        num_frames: Number of frames to display
        save_path: Path to save the image
        title: Title for the image
    """
    # Handle batch dimension
    if original_frames.dim() == 5:
        original_frames = original_frames[0]  # (T, C, H, W)
    if reconstructed_frames.dim() == 5:
        reconstructed_frames = reconstructed_frames[0]  # (T, C, H, W)
    
    T = original_frames.shape[0]
    num_frames = min(num_frames, T)
    
    # Select frames evenly spaced
    frame_indices = np.linspace(0, T - 1, num_frames, dtype=int)
    
    # Normalize frames to [0, 1]
    original_norm = normalize_for_display(original_frames)
    reconstructed_norm = normalize_for_display(reconstructed_frames)
    
    # Convert to numpy (T, H, W, C)
    original_np = original_norm.permute(0, 2, 3, 1).cpu().numpy()
    reconstructed_np = reconstructed_norm.permute(0, 2, 3, 1).cpu().numpy()
    
    # Create figure with side-by-side pairs
    fig, axes = plt.subplots(num_frames, 2, figsize=(4, 2 * num_frames))
    if num_frames == 1:
        axes = axes.reshape(1, 2)
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    for i, t_idx in enumerate(frame_indices):
        # Original frame (left)
        axes[i, 0].imshow(original_np[t_idx])
        if i == 0:
            axes[i, 0].set_title("Original", fontsize=12, fontweight='bold')
        axes[i, 0].set_ylabel(f"Frame {t_idx}", fontsize=10)
        axes[i, 0].axis('off')
        
        # Reconstructed frame (right)
        axes[i, 1].imshow(reconstructed_np[t_idx])
        if i == 0:
            axes[i, 1].set_title("Reconstructed", fontsize=12, fontweight='bold')
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print_flush(f"✓ Saved side-by-side grid to {save_path}")
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate reconstruction comparison images from Phase 1 checkpoint")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/minerl_subset.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to Phase 1 checkpoint",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="reconstruction_images",
        help="Output directory for images",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=5,
        help="Number of images to generate",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=8,
        help="Number of frames to show per image",
    )
    parser.add_argument(
        "--layout",
        type=str,
        default="stacked",
        choices=["stacked", "side_by_side"],
        help="Layout: 'stacked' (original top, reconstructed bottom) or 'side_by_side' (pairs)",
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
    
    # Load checkpoint
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print_flush(f"Error: Checkpoint not found at {checkpoint_path}")
        sys.exit(1)
    
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
    
    print_flush(f"\nGenerating {args.num_images} reconstruction images...")
    print_flush(f"Output directory: {output_dir}")
    print_flush(f"Layout: {args.layout}")
    
    data_iter = iter(dataloader)
    
    for img_idx in range(args.num_images):
        print_flush(f"\n--- Generating Image {img_idx + 1}/{args.num_images} ---")
        
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)
        
        frames = batch["frames"].to(device)  # (B, T, C, H, W)
        
        print_flush(f"  Input frames shape: {frames.shape}")
        
        # Generate reconstruction
        print_flush(f"  Generating tokenizer reconstruction...")
        with torch.no_grad():
            reconstructed = generate_tokenizer_reconstruction(tokenizer, frames, device)
        
        # Create grid image
        if args.layout == "stacked":
            save_path = output_dir / f"reconstruction_stacked_{img_idx + 1}.png"
            create_reconstruction_grid(
                frames,
                reconstructed,
                num_frames=args.num_frames,
                save_path=save_path,
                title=f"Reconstruction Comparison {img_idx + 1} (Epoch {ckpt.get('epoch', 'unknown')})",
            )
        else:  # side_by_side
            save_path = output_dir / f"reconstruction_sidebyside_{img_idx + 1}.png"
            create_side_by_side_grid(
                frames,
                reconstructed,
                num_frames=args.num_frames,
                save_path=save_path,
                title=f"Reconstruction Comparison {img_idx + 1} (Epoch {ckpt.get('epoch', 'unknown')})",
            )
    
    print_flush(f"\n✓ Generated {args.num_images} reconstruction images in {output_dir}/")
    print_flush(f"  - reconstruction_*.png: Grid comparison images")


if __name__ == "__main__":
    main()
