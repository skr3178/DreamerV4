#!/usr/bin/env python3
"""
Generate and Compare Images from Trained Checkpoint

This script loads a trained checkpoint and generates images from the dynamics model,
then compares them with the original MNIST images.
"""

import sys
from pathlib import Path

# Add parent directory to path to import dreamer modules
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
import numpy as np
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import models
from dreamer.models import CausalTokenizer, DynamicsModel
from dataset import create_mnist_dataloader


def load_checkpoint(
    checkpoint_path: str,
    device: torch.device,
    latent_dim: int = 16,
    num_latent_tokens: int = 8,
    embed_dim: int = 128,
    depth: int = 4,
    num_heads: int = 4,
    max_shortcut_steps: int = 4,
):
    """Load models from checkpoint."""
    checkpoint_file = Path(checkpoint_path)
    
    if not checkpoint_file.exists():
        print(f"Error: Checkpoint file not found: {checkpoint_path}")
        
        # Check if checkpoints directory exists
        checkpoints_dir = Path("checkpoints")
        if checkpoints_dir.exists():
            available = list(checkpoints_dir.glob("*.pt"))
            if available:
                print(f"\nAvailable checkpoints in {checkpoints_dir}:")
                for ckpt in sorted(available):
                    print(f"  - {ckpt}")
                print(f"\nTry using one of these checkpoints, e.g.:")
                print(f"  python generate_images.py --checkpoint {available[0]}")
            else:
                print(f"\nNo checkpoint files found in {checkpoints_dir}/")
        else:
            print(f"\nCheckpoints directory does not exist.")
            print("Please run training first:")
            print("  python test_shortcut.py")
        
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading checkpoint from {checkpoint_path}...")
    
    # Use weights_only=False for now (contains model state dicts, not just weights)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Get config from checkpoint if available
    if "config" in checkpoint:
        config = checkpoint["config"]
        latent_dim = config.get("latent_dim", latent_dim)
        num_latent_tokens = config.get("num_latent_tokens", num_latent_tokens)
        embed_dim = config.get("embed_dim", embed_dim)
        depth = config.get("depth", depth)
        num_heads = config.get("num_heads", num_heads)
        max_shortcut_steps = config.get("max_shortcut_steps", max_shortcut_steps)
    
    # Create models
    tokenizer = CausalTokenizer(
        image_height=28,
        image_width=28,
        in_channels=1,
        patch_size=4,
        embed_dim=embed_dim,
        latent_dim=latent_dim,
        num_latent_tokens=num_latent_tokens,
        depth=depth,
        num_heads=num_heads,
        dropout=0.0,
        num_registers=2,
        mask_ratio=0.0,
    ).to(device)
    
    dynamics = DynamicsModel(
        latent_dim=latent_dim,
        num_latent_tokens=num_latent_tokens,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        dropout=0.0,
        num_discrete_actions=10,
        num_registers=2,
        max_shortcut_steps=max_shortcut_steps,
    ).to(device)
    
    # Load state dicts
    tokenizer.load_state_dict(checkpoint["tokenizer_state_dict"])
    dynamics.load_state_dict(checkpoint["dynamics_state_dict"])
    
    tokenizer.eval()
    dynamics.eval()
    
    print(f"✓ Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    
    return tokenizer, dynamics


def generate_next_frame(
    tokenizer: CausalTokenizer,
    dynamics: DynamicsModel,
    frames: torch.Tensor,
    actions: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Generate the next frame given previous frames.
    
    Args:
        frames: (B, T, C, H, W) input frames
        actions: (B, T) actions
    
    Returns:
        generated_frame: (B, C, H, W) generated next frame
    """
    with torch.no_grad():
        # Encode frames to latents
        tokenizer_output = tokenizer.encode(frames, mask_ratio=0.0)
        latents = tokenizer_output["latents"]  # (B, T, num_latent, latent_dim)
        
        # Use dynamics to predict next latent
        # For generation, we'll use the last frame as context and predict the next
        # In this simplified version, we'll use the dynamics model's generate method
        # or manually predict the next latent
        
        # Get last frame's latent
        last_latent = latents[:, -1:]  # (B, 1, num_latent, latent_dim)
        last_action = actions[:, -1:]  # (B, 1)
        
        # Use dynamics forward to predict next latent
        # The dynamics model predicts latents for all timesteps
        # We want to predict what the next frame would be given the sequence
        output = dynamics(
            latents=latents,
            actions=actions,
            discrete_actions=True,
            add_noise_to_latents=False,  # Don't add noise for generation
        )
        
        # Get predicted latent for the last timestep (this is the prediction)
        # predicted_latents shape: (B, T, num_latent, latent_dim)
        predicted_latent = output["predicted_latents"][:, -1]  # (B, num_latent, latent_dim)
        
        # Decode predicted latent to image
        # The tokenizer's decode method expects (B, num_latent, latent_dim)
        generated_frame = tokenizer.decode(predicted_latent)  # (B, C, H, W)
        
        # The model appears to predict inverted values, so invert here
        # Invert in [-1, 1] space: -x
        generated_frame = -generated_frame
        
        return generated_frame


def compare_images(
    tokenizer: CausalTokenizer,
    dynamics: DynamicsModel,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    num_samples: int = 10,
    output_dir: str = "generated_images",
):
    """Generate images and compare with originals."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print(f"\nGenerating {num_samples} image comparisons...")
    
    all_originals = []
    all_generated = []
    all_errors = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Generating")):
            if batch_idx >= num_samples:
                break
            
            frames = batch["frames"].to(device)  # (B, T, C, H, W)
            labels = batch["labels"].to(device)  # (B, T)
            actions = labels.long()
            
            # Get the last frame as target (what we want to predict)
            target_frame = frames[:, -1]  # (B, C, H, W)
            
            # Generate next frame
            generated_frame = generate_next_frame(
                tokenizer, dynamics, frames, actions, device
            )
            
            # Denormalize images (from [-1, 1] to [0, 1])
            # Note: generated_frame is already inverted in generate_next_frame
            target_frame_denorm = (target_frame + 1.0) / 2.0
            generated_frame_denorm = (generated_frame + 1.0) / 2.0
            generated_frame_denorm = torch.clamp(generated_frame_denorm, 0, 1)
            
            # Compute error
            error = torch.abs(target_frame_denorm - generated_frame_denorm)
            
            # Store for visualization
            all_originals.append(target_frame_denorm.cpu())
            all_generated.append(generated_frame_denorm.cpu())
            all_errors.append(error.cpu())
            
            # Save individual comparison
            comparison = torch.cat([
                target_frame_denorm.cpu(),
                generated_frame_denorm.cpu(),
                error.cpu(),
            ], dim=0)  # (3*B, C, H, W)
            
            save_image(
                comparison,
                output_path / f"comparison_batch_{batch_idx}.png",
                nrow=frames.shape[0],
                padding=2,
            )
    
    # Create grid comparison
    if all_originals:
        print("\nCreating grid comparison...")
        
        # Stack all images
        originals_grid = torch.cat(all_originals, dim=0)  # (N, C, H, W)
        generated_grid = torch.cat(all_generated, dim=0)
        errors_grid = torch.cat(all_errors, dim=0)
        
        # Create side-by-side comparison
        comparison_grid = torch.cat([
            originals_grid,
            generated_grid,
            errors_grid,
        ], dim=0)  # (3*N, C, H, W)
        
        # Save grid
        grid_image = make_grid(
            comparison_grid,
            nrow=num_samples,
            padding=2,
            normalize=False,
        )
        save_image(grid_image, output_path / "grid_comparison.png")
        
        # Create matplotlib figure for better visualization
        fig, axes = plt.subplots(3, num_samples, figsize=(2*num_samples, 6))
        if num_samples == 1:
            axes = axes.reshape(3, 1)
        
        for i in range(num_samples):
            # Original
            axes[0, i].imshow(originals_grid[i].permute(1, 2, 0).squeeze(), cmap='gray')
            axes[0, i].set_title(f"Original {i+1}")
            axes[0, i].axis('off')
            
            # Generated
            axes[1, i].imshow(generated_grid[i].permute(1, 2, 0).squeeze(), cmap='gray')
            axes[1, i].set_title(f"Generated {i+1}")
            axes[1, i].axis('off')
            
            # Error
            axes[2, i].imshow(errors_grid[i].permute(1, 2, 0).squeeze(), cmap='hot')
            axes[2, i].set_title(f"Error {i+1}")
            axes[2, i].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path / "comparison_visualization.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved comparison images to {output_path}/")
        
        # Compute statistics
        mse = F.mse_loss(generated_grid, originals_grid)
        mae = F.l1_loss(generated_grid, originals_grid)
        
        print(f"\nReconstruction Statistics:")
        print(f"  MSE: {mse.item():.6f}")
        print(f"  MAE: {mae.item():.6f}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate and compare images from checkpoint")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint file (default: auto-detect latest checkpoint)"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Number of samples to generate"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="generated_images",
        help="Output directory for generated images"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use"
    )
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Auto-detect checkpoint if not provided
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        checkpoints_dir = Path("checkpoints")
        if checkpoints_dir.exists():
            # Look for final checkpoint first, then latest epoch checkpoint
            final_ckpt = checkpoints_dir / "final_checkpoint.pt"
            if final_ckpt.exists():
                checkpoint_path = str(final_ckpt)
            else:
                # Find latest epoch checkpoint
                epoch_ckpts = sorted(checkpoints_dir.glob("checkpoint_epoch_*.pt"))
                if epoch_ckpts:
                    checkpoint_path = str(epoch_ckpts[-1])
                    print(f"Auto-detected latest checkpoint: {checkpoint_path}")
                else:
                    print("Error: No checkpoints found. Please run training first:")
                    print("  python test_shortcut.py")
                    return
        else:
            print("Error: Checkpoints directory does not exist. Please run training first:")
            print("  python test_shortcut.py")
            return
    
    # Load checkpoint
    tokenizer, dynamics = load_checkpoint(checkpoint_path, device)
    
    # Create dataloader
    test_loader = create_mnist_dataloader(
        batch_size=1,  # Process one at a time for clearer visualization
        sequence_length=8,
        train=False,
        num_workers=2,
    )
    
    # Generate and compare
    compare_images(
        tokenizer=tokenizer,
        dynamics=dynamics,
        dataloader=test_loader,
        device=device,
        num_samples=args.num_samples,
        output_dir=args.output_dir,
    )
    
    print("\n✓ Image generation complete!")


if __name__ == "__main__":
    main()
