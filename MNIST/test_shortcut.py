#!/usr/bin/env python3
"""
Test Shortcut Model on MNIST Dataset

This script tests the shortcut forcing loss and dynamics model on MNIST sequences.
It references the core shortcut model code without modifying it.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path to import dreamer modules
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm

# Import shortcut model code (we reference, don't modify)
from dreamer.models import CausalTokenizer, DynamicsModel
from dreamer.losses import ShortcutForcingLoss
from dreamer.losses.shortcut_loss import BootstrapTargetComputer
from dataset import create_mnist_dataloader


def create_models(
    latent_dim: int = 16,
    num_latent_tokens: int = 8,
    embed_dim: int = 128,
    depth: int = 4,
    num_heads: int = 4,
    max_shortcut_steps: int = 4,
    device: str = "cuda",
):
    """Create tokenizer and dynamics models for MNIST."""
    
    # Tokenizer: encodes images to latents
    tokenizer = CausalTokenizer(
        image_height=28,
        image_width=28,
        in_channels=1,  # MNIST is grayscale
        patch_size=4,  # Smaller patches for 28x28 images
        embed_dim=embed_dim,
        latent_dim=latent_dim,
        num_latent_tokens=num_latent_tokens,
        depth=depth,
        num_heads=num_heads,
        dropout=0.0,
        num_registers=2,
        mask_ratio=0.0,  # No masking for dynamics training
    ).to(device)
    
    # Dynamics model: predicts future latents
    # For MNIST, we don't have actions, so we'll use a dummy action space
    # In practice, you might want to condition on digit transitions or other features
    dynamics = DynamicsModel(
        latent_dim=latent_dim,
        num_latent_tokens=num_latent_tokens,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        dropout=0.0,
        num_discrete_actions=10,  # 10 digit classes as "actions" (dummy)
        num_registers=2,
        max_shortcut_steps=max_shortcut_steps,
    ).to(device)
    
    return tokenizer, dynamics


def train_step(
    tokenizer: CausalTokenizer,
    dynamics: DynamicsModel,
    batch: dict,
    loss_fn: ShortcutForcingLoss,
    bootstrap_computer: BootstrapTargetComputer,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    use_bootstrap: bool = True,
) -> dict:
    """Single training step."""
    optimizer.zero_grad()
    
    # Get frames: (B, T, C, H, W)
    frames = batch["frames"].to(device)
    batch_size, seq_len = frames.shape[0], frames.shape[1]
    
    # Reshape for tokenizer: (B, T, C, H, W) - tokenizer expects this format
    # The encode method handles both (B, C, T, H, W) and (B, T, C, H, W)
    
    # Encode frames to latents (no gradient for tokenizer in this test)
    with torch.no_grad():
        tokenizer_output = tokenizer.encode(frames, mask_ratio=0.0)
        latents = tokenizer_output["latents"]  # (B, T, num_latent, latent_dim)
    
    # Create dummy actions (using digit labels as "actions" for conditioning)
    # In a real scenario, you might want to use actual digit transitions
    labels = batch["labels"].to(device)  # (B, T)
    actions = labels.long()  # Use digit labels as discrete actions
    
    # Forward pass through dynamics
    output = dynamics(
        latents=latents,
        actions=actions,
        discrete_actions=True,
    )
    
    # Compute bootstrap targets if needed
    bootstrap_targets = None
    if use_bootstrap:
        # Get indices where d is not minimum
        d_other_mask = ~output["d_is_min"]
        if d_other_mask.any():
            # For bootstrap, we need to compute targets for samples with d > d_min
            # The bootstrap target is computed using two half-steps
            # We'll compute it for the last timestep (target frame)
            
            # Get noisy latents (dynamics already added noise internally, but we need to recreate)
            # We'll use the same signal_level and step_size from output
            noisy_latents = dynamics.add_noise_with_context_corruption(
                latents,
                output["signal_level"],
            )
            
            # Extract last timestep for prediction
            target_latents = latents[:, -1:]  # (B, 1, num_latent, latent_dim)
            noisy_target = noisy_latents[:, -1:]  # (B, 1, num_latent, latent_dim)
            target_actions = actions[:, -1:]  # (B, 1)
            
            # Compute bootstrap targets for samples with d > d_min
            batch_indices = d_other_mask.nonzero(as_tuple=True)[0]
            if len(batch_indices) > 0:
                bootstrap_list = []
                for idx in batch_indices:
                    bootstrap_target = bootstrap_computer.compute_bootstrap_target(
                        dynamics,
                        noisy_target[idx:idx+1],
                        target_actions[idx:idx+1],
                        output["signal_level"][idx:idx+1],
                        output["step_size"][idx:idx+1],
                        discrete_actions=True,
                    )
                    bootstrap_list.append(bootstrap_target)
                
                if bootstrap_list:
                    # Stack bootstrap targets
                    bootstrap_batch = torch.cat(bootstrap_list, dim=0)  # (num_d_other, 1, num_latent, latent_dim)
                    
                    # Create full bootstrap tensor matching predicted_latents shape
                    bootstrap_full = output["target_latents"].clone()
                    # Only replace targets for samples with d > d_min, and only for last timestep
                    bootstrap_full[batch_indices, -1:] = bootstrap_batch
                    bootstrap_targets = bootstrap_full
    
    # Compute loss
    loss_dict = loss_fn(
        predicted_latents=output["predicted_latents"],
        target_latents=output["target_latents"],
        signal_level=output["signal_level"],
        step_size=output["step_size"],
        d_is_min=output["d_is_min"],
        bootstrap_targets=bootstrap_targets,
    )
    
    loss = loss_dict["loss"]
    
    # Backward pass
    loss.backward()
    torch.nn.utils.clip_grad_norm_(dynamics.parameters(), max_norm=1.0)
    optimizer.step()
    
    return {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in loss_dict.items()}


def evaluate(
    tokenizer: CausalTokenizer,
    dynamics: DynamicsModel,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict:
    """Evaluate the model."""
    tokenizer.eval()
    dynamics.eval()
    
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            frames = batch["frames"].to(device)
            labels = batch["labels"].to(device)
            
            # Encode to latents (tokenizer handles both formats)
            tokenizer_output = tokenizer.encode(frames, mask_ratio=0.0)
            latents = tokenizer_output["latents"]
            
            # Forward pass
            actions = labels.long()
            output = dynamics(
                latents=latents,
                actions=actions,
                discrete_actions=True,
            )
            
            # Simple MSE loss for evaluation
            mse = torch.nn.functional.mse_loss(
                output["predicted_latents"],
                output["target_latents"],
            )
            
            total_loss += mse.item()
            num_batches += 1
    
    return {"mse_loss": total_loss / num_batches if num_batches > 0 else 0.0}


def main():
    """Main training loop."""
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Hyperparameters
    batch_size = 32
    sequence_length = 8
    num_epochs = 15  # Increased for longer training
    learning_rate = 1e-4
    
    # Create models
    print("Creating models...")
    tokenizer, dynamics = create_models(
        latent_dim=16,
        num_latent_tokens=8,
        embed_dim=128,
        depth=4,
        num_heads=4,
        max_shortcut_steps=4,
        device=device,
    )
    
    # Freeze tokenizer (pretrained or fixed for this test)
    for param in tokenizer.parameters():
        param.requires_grad = False
    tokenizer.eval()
    
    # Create loss function and bootstrap computer
    loss_fn = ShortcutForcingLoss()
    bootstrap_computer = BootstrapTargetComputer()
    
    # Create optimizer (only for dynamics model)
    optimizer = AdamW(dynamics.parameters(), lr=learning_rate)
    
    # Create dataloaders
    print("Creating dataloaders...")
    train_loader = create_mnist_dataloader(
        batch_size=batch_size,
        sequence_length=sequence_length,
        train=True,
        num_workers=2,
    )
    
    test_loader = create_mnist_dataloader(
        batch_size=batch_size,
        sequence_length=sequence_length,
        train=False,
        num_workers=2,
    )
    
    print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")
    
    # Create checkpoint directory
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Training loop
    print("\nStarting training...")
    for epoch in range(num_epochs):
        dynamics.train()
        
        epoch_losses = []
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch in progress_bar:
            loss_dict = train_step(
                tokenizer=tokenizer,
                dynamics=dynamics,
                batch=batch,
                loss_fn=loss_fn,
                bootstrap_computer=bootstrap_computer,
                optimizer=optimizer,
                device=device,
                use_bootstrap=True,
            )
            
            epoch_losses.append(loss_dict["loss"])
            
            # Update progress bar
            progress_bar.set_postfix({
                "loss": f"{loss_dict['loss']:.4f}",
                "d_min": f"{loss_dict['d_min_loss']:.4f}",
                "d_other": f"{loss_dict['d_other_loss']:.4f}",
            })
        
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"\nEpoch {epoch+1} - Average Loss: {avg_loss:.4f}")
        
        # Evaluate on test set
        if (epoch + 1) % 2 == 0:
            eval_metrics = evaluate(tokenizer, dynamics, test_loader, device)
            print(f"Test MSE Loss: {eval_metrics['mse_loss']:.4f}")
        
        # Save checkpoint
        checkpoint = {
            "epoch": epoch + 1,
            "tokenizer_state_dict": tokenizer.state_dict(),
            "dynamics_state_dict": dynamics.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": {
                "latent_dim": 16,
                "num_latent_tokens": 8,
                "embed_dim": 128,
                "depth": 4,
                "num_heads": 4,
                "max_shortcut_steps": 4,
            }
        }
        torch.save(checkpoint, checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pt")
        print(f"Saved checkpoint to {checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pt'}")
    
    # Save final checkpoint
    final_checkpoint = {
        "epoch": num_epochs,
        "tokenizer_state_dict": tokenizer.state_dict(),
        "dynamics_state_dict": dynamics.state_dict(),
        "config": {
            "latent_dim": 16,
            "num_latent_tokens": 8,
            "embed_dim": 128,
            "depth": 4,
            "num_heads": 4,
            "max_shortcut_steps": 4,
        }
    }
    torch.save(final_checkpoint, checkpoint_dir / "final_checkpoint.pt")
    print(f"\nSaved final checkpoint to {checkpoint_dir / 'final_checkpoint.pt'}")
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
