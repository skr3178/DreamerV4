#!/usr/bin/env python3
"""Evaluate Phase 1 trained model."""

import torch
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
import argparse
import sys

from dreamer.models import CausalTokenizer, DynamicsModel
from dreamer.data import create_dataloader


def load_phase1_checkpoint(checkpoint_path, config, device):
    """Load trained Phase 1 models."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Create tokenizer
    tokenizer = CausalTokenizer(
        image_height=config["data"]["image_height"],
        image_width=config["data"]["image_width"],
        in_channels=config["data"]["in_channels"],
        patch_size=config["tokenizer"]["patch_size"],
        embed_dim=config["tokenizer"]["embed_dim"],
        latent_dim=config["tokenizer"]["latent_dim"],
        num_latent_tokens=config["tokenizer"]["num_latent_tokens"],
        depth=config["tokenizer"]["depth"],
        num_heads=config["tokenizer"]["num_heads"],
        dropout=config["tokenizer"]["dropout"],
        num_registers=config["tokenizer"]["num_registers"],
        mask_ratio=config["tokenizer"]["mask_ratio"],
    ).to(device)

    # Create dynamics
    dynamics = DynamicsModel(
        latent_dim=config["tokenizer"]["latent_dim"],
        num_latent_tokens=config["tokenizer"]["num_latent_tokens"],
        embed_dim=config["tokenizer"]["embed_dim"],
        depth=config["tokenizer"]["depth"],
        num_heads=config["tokenizer"]["num_heads"],
        dropout=config["tokenizer"]["dropout"],
        num_discrete_actions=config["dynamics"]["num_discrete_actions"],
        num_registers=config["dynamics"]["num_registers"],
        max_shortcut_steps=config["dynamics"]["max_shortcut_steps"],
    ).to(device)

    tokenizer.load_state_dict(checkpoint["tokenizer_state_dict"])
    dynamics.load_state_dict(checkpoint["dynamics_state_dict"])

    return tokenizer, dynamics, checkpoint


def evaluate_reconstruction(tokenizer, dataloader, device, num_batches=5):
    """Evaluate reconstruction quality on patches."""
    tokenizer.eval()
    
    total_mse = 0.0
    total_samples = 0
    
    print("Evaluating reconstruction quality...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= num_batches:
                break
                
            frames = batch["frames"].to(device)  # (B, T, C, H, W)
            
            # Reshape for tokenizer: (B, C, T, H, W)
            if frames.dim() == 5 and frames.shape[2] != tokenizer.in_channels:
                frames_reshaped = frames.permute(0, 2, 1, 3, 4)
            else:
                frames_reshaped = frames
            
            # Forward pass with no masking for evaluation
            output = tokenizer(frames_reshaped, mask_ratio=0.0)
            
            # Get reconstructed patches and original patches
            reconstructed_patches = output["reconstructed"]  # (B, T, num_patches, patch_dim)
            original_patches = output.get("original_patches")
            
            if original_patches is None:
                print(f"  Batch {batch_idx + 1}: Warning - original_patches not in output, skipping")
                continue
            
            # Compute MSE on patches (this is the standard evaluation metric)
            mse = ((original_patches - reconstructed_patches) ** 2).mean().item()
            total_mse += mse * frames.shape[0] * frames.shape[1]
            total_samples += frames.shape[0] * frames.shape[1]
            
            print(f"  Batch {batch_idx + 1}/{num_batches}: Patch MSE = {mse:.6f}")
    
    avg_mse = total_mse / total_samples if total_samples > 0 else 0.0
    return avg_mse


def visualize_reconstruction(tokenizer, batch, device, save_path="reconstruction.png"):
    """Visualize original vs reconstructed frames."""
    tokenizer.eval()
    
    frames = batch["frames"].to(device)  # (B, T, C, H, W)
    
    # Reshape for tokenizer: (B, C, T, H, W)
    if frames.dim() == 5 and frames.shape[2] != tokenizer.in_channels:
        frames_reshaped = frames.permute(0, 2, 1, 3, 4)
    else:
        frames_reshaped = frames
    
    with torch.no_grad():
        output = tokenizer(frames_reshaped, mask_ratio=0.0)
        
        # Get original and reconstructed patches
        original_patches = output.get("original_patches")
        reconstructed_patches = output["reconstructed"]
        
        if original_patches is None:
            print("Cannot visualize: original_patches not available in output")
            return
        
        # Unpatchify to get full frames
        # Check if tokenizer has patch_embed with unpatchify method
        if hasattr(tokenizer, "patch_embed") and hasattr(tokenizer.patch_embed, "unpatchify"):
            # Unpatchify original and reconstructed patches
            B, T = original_patches.shape[:2]
            original_frames = []
            reconstructed_frames = []
            for t in range(T):
                orig_patches_t = original_patches[:, t]  # (B, num_patches, patch_dim)
                recon_patches_t = reconstructed_patches[:, t]  # (B, num_patches, patch_dim)
                
                orig_frame = tokenizer.patch_embed.unpatchify(orig_patches_t)  # (B, C, H, W)
                recon_frame = tokenizer.patch_embed.unpatchify(recon_patches_t)  # (B, C, H, W)
                
                original_frames.append(orig_frame)
                reconstructed_frames.append(recon_frame)
            
            original_frames = torch.stack(original_frames, dim=2)  # (B, C, T, H, W)
            reconstructed_frames = torch.stack(reconstructed_frames, dim=2)  # (B, C, T, H, W)
        else:
            # Fallback: use original frames directly (frames_reshaped is already in (B, C, T, H, W))
            original_frames = frames_reshaped
            print("Warning: Cannot unpatchify patches, using original frames for visualization")
            # For reconstructed, we'll show the original frames as placeholder
            reconstructed_frames = frames_reshaped
        
        # Select first batch item and first timestep
        # original_frames is (B, C, T, H, W), so [0, :, 0] gives (C, H, W)
        original_frame = original_frames[0, :, 0].cpu().permute(1, 2, 0).numpy()  # (H, W, C)
        reconstructed_frame = reconstructed_frames[0, :, 0].cpu().permute(1, 2, 0).numpy()  # (H, W, C)
        
        # Normalize to [0, 1] for display
        # Frames might be in [-1, 1] or [0, 1]
        if original_frame.min() < 0:
            original_frame = (original_frame * 0.5 + 0.5).clip(0, 1)
        else:
            original_frame = original_frame.clip(0, 1)
            
        if reconstructed_frame.min() < 0:
            reconstructed_frame = (reconstructed_frame * 0.5 + 0.5).clip(0, 1)
        else:
            reconstructed_frame = reconstructed_frame.clip(0, 1)
        
        # Plot
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(original_frame)
        plt.title("Original")
        plt.axis("off")
        
        plt.subplot(1, 2, 2)
        plt.imshow(reconstructed_frame)
        plt.title("Reconstructed")
        plt.axis("off")
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved visualization to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Phase 1 trained model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/minerl_subset.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/media/skr/storage/dreamerv4/checkpoints/subset/dreamerv4_minerl_subset/phase1_final.pt",
        help="Path to Phase 1 checkpoint",
    )
    parser.add_argument(
        "--num_batches",
        type=int,
        default=5,
        help="Number of batches to evaluate",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualization of reconstructions",
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
            print("⚠️  Warning: CUDA requested but not available. Falling back to CPU.")
            device = torch.device("cpu")
        else:
            device = torch.device(requested_device)
    
    print(f"Using device: {device}")
    
    # Load checkpoint
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        sys.exit(1)
    
    print(f"Loading checkpoint from {checkpoint_path}")
    tokenizer, dynamics, ckpt = load_phase1_checkpoint(
        checkpoint_path, config, device
    )
    tokenizer.eval()
    dynamics.eval()
    
    print(f"Loaded checkpoint from step {ckpt.get('global_step', 'unknown')}")
    print(f"Epoch: {ckpt.get('epoch', 'unknown')}")
    
    # Create dataloader
    print("\nCreating data loader...")
    dataloader = create_dataloader(
        data_path=config["data"]["path"],
        batch_size=4,
        sequence_length=config["data"]["sequence_length"],
        image_size=(config["data"]["image_height"], config["data"]["image_width"]),
        num_workers=0,
        split="train",
        max_episodes=5,
    )
    
    # Evaluate reconstruction
    print("\n" + "=" * 80)
    print("Reconstruction Evaluation")
    print("=" * 80)
    mse = evaluate_reconstruction(tokenizer, dataloader, device, num_batches=args.num_batches)
    print(f"\nAverage Reconstruction MSE (on patches): {mse:.6f}")
    
    # Visualize if requested
    if args.visualize:
        print("\nGenerating visualization...")
        batch = next(iter(dataloader))
        visualize_reconstruction(tokenizer, batch, device)
    
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
