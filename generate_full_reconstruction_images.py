#!/usr/bin/env python3
"""
Generate full reconstruction comparison images showing:
1. Original frames
2. Reconstructed frames (from patches)
3. Decoder images (from CNN decoder)
"""

import torch
import yaml
from pathlib import Path
import numpy as np
import argparse
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add flush helper
def print_flush(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()
    sys.stderr.flush()

from dreamer.models import CausalTokenizer, DynamicsModel
from dreamer.data import create_dataloader
from eval_phase1 import load_phase1_checkpoint
from generate_videos_phase1 import normalize_for_display


def create_full_reconstruction_grid(
    original_frames: torch.Tensor,
    reconstructed_frames: torch.Tensor,
    decoder_frames: torch.Tensor,
    num_frames: int = 8,
    save_path: Path = None,
    title: str = "Original vs Reconstructed vs Decoder",
):
    """
    Create a grid image showing original, reconstructed (patches), and decoder (CNN) frames.
    
    Args:
        original_frames: (B, T, C, H, W) or (T, C, H, W) tensor
        reconstructed_frames: (B, T, C, H, W) or (T, C, H, W) tensor - from patches
        decoder_frames: (B, T, C, H, W) or (T, C, H, W) tensor - from CNN decoder
        num_frames: Number of frames to display
        save_path: Path to save the image
        title: Title for the image
    """
    # Handle batch dimension
    if original_frames.dim() == 5:
        original_frames = original_frames[0]  # (T, C, H, W)
    if reconstructed_frames.dim() == 5:
        reconstructed_frames = reconstructed_frames[0]  # (T, C, H, W)
    if decoder_frames.dim() == 5:
        decoder_frames = decoder_frames[0]  # (T, C, H, W)
    
    T = original_frames.shape[0]
    num_frames = min(num_frames, T)
    
    # Select frames evenly spaced
    frame_indices = np.linspace(0, T - 1, num_frames, dtype=int)
    
    # Normalize frames to [0, 1]
    original_norm = normalize_for_display(original_frames)
    reconstructed_norm = normalize_for_display(reconstructed_frames)
    decoder_norm = normalize_for_display(decoder_frames)
    
    # Convert to numpy (T, H, W, C)
    original_np = original_norm.permute(0, 2, 3, 1).cpu().numpy()
    reconstructed_np = reconstructed_norm.permute(0, 2, 3, 1).cpu().numpy()
    decoder_np = decoder_norm.permute(0, 2, 3, 1).cpu().numpy()
    
    # Create figure with three rows: original, reconstructed, decoder
    fig, axes = plt.subplots(3, num_frames, figsize=(2 * num_frames, 6))
    if num_frames == 1:
        axes = axes.reshape(3, 1)
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    for i, t_idx in enumerate(frame_indices):
        # Original frame (top row)
        axes[0, i].imshow(original_np[t_idx])
        if i == 0:
            axes[0, i].set_ylabel("Original", fontsize=12, fontweight='bold')
        axes[0, i].set_title(f"Frame {t_idx}", fontsize=10)
        axes[0, i].axis('off')
        
        # Reconstructed frame (middle row) - from patches
        axes[1, i].imshow(reconstructed_np[t_idx])
        if i == 0:
            axes[1, i].set_ylabel("Reconstructed\n(Patches)", fontsize=12, fontweight='bold')
        axes[1, i].axis('off')
        
        # Decoder frame (bottom row) - from CNN decoder
        axes[2, i].imshow(decoder_np[t_idx])
        if i == 0:
            axes[2, i].set_ylabel("Decoder\n(CNN)", fontsize=12, fontweight='bold')
        axes[2, i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print_flush(f"✓ Saved full reconstruction grid to {save_path}")
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate full reconstruction comparison images (Original, Reconstructed, Decoder)")
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
        default="reconstruction_images_full",
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
    
    print_flush(f"\nGenerating {args.num_images} full reconstruction images...")
    print_flush(f"Output directory: {output_dir}")
    
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
        
        # Reshape for tokenizer if needed
        if frames.dim() == 5 and frames.shape[2] != tokenizer.in_channels:
            frames_reshaped = frames.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)
        else:
            frames_reshaped = frames
        
        # Generate all three types of outputs
        print_flush(f"  Generating tokenizer outputs...")
        with torch.no_grad():
            # Forward through tokenizer to get all outputs
            output = tokenizer(frames_reshaped, mask_ratio=0.0)
            
            # Get reconstructed patches and decode them
            reconstructed_patches = output["reconstructed"]  # (B, T, num_patches, patch_dim)
            B, T, num_patches, patch_dim = reconstructed_patches.shape
            
            # Decode patches to frames
            reconstructed_frames_list = []
            for t in range(T):
                patches_t = reconstructed_patches[:, t]  # (B, num_patches, patch_dim)
                frame = tokenizer.decode_patches(patches_t)  # (B, C, H, W)
                reconstructed_frames_list.append(frame)
            reconstructed_frames = torch.stack(reconstructed_frames_list, dim=1)  # (B, T, C, H, W)
            
            # Get decoder images (from CNN decoder)
            decoder_frames = output.get("decoder_images")  # (B, T, C, H, W)
            if decoder_frames is None:
                print_flush("  Warning: decoder_images not found in output, generating from latents...")
                # Fallback: decode latents through CNN decoder
                latents = output["latents"]  # (B, T, num_latent, latent_dim)
                decoder_frames_list = []
                for t in range(T):
                    latent_t = latents[:, t]  # (B, num_latent, latent_dim)
                    decoded_t = tokenizer.decode(latent_t)  # (B, C, H, W)
                    decoder_frames_list.append(decoded_t)
                decoder_frames = torch.stack(decoder_frames_list, dim=1)  # (B, T, C, H, W)
            
            # Original frames (permute back if needed)
            if frames_reshaped.shape[1] == tokenizer.in_channels:
                original_frames = frames_reshaped.permute(0, 2, 1, 3, 4)  # (B, T, C, H, W)
            else:
                original_frames = frames
        
        # Create grid image
        save_path = output_dir / f"reconstructions_epoch{ckpt.get('epoch', 'unknown')}_{img_idx + 1}.png"
        create_full_reconstruction_grid(
            original_frames,
            reconstructed_frames,
            decoder_frames,
            num_frames=args.num_frames,
            save_path=save_path,
            title=f"Reconstruction Comparison {img_idx + 1} (Epoch {ckpt.get('epoch', 'unknown')})",
        )
    
    print_flush(f"\n✓ Generated {args.num_images} full reconstruction images in {output_dir}/")
    print_flush(f"  - reconstructions_epoch*.png: Full comparison (Original, Reconstructed, Decoder)")


if __name__ == "__main__":
    main()
