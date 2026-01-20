"""
DreamerV4 Phase 1: World Model Pretraining

Trains the tokenizer and dynamics model on video data.

Phase 1 consists of:
1. Tokenizer training: Masked autoencoding with MSE + LPIPS loss (Eq. 5)
2. Dynamics training: Shortcut forcing objective (Eq. 7)
"""

import os
import argparse
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import yaml

from dreamer.models import CausalTokenizer, DynamicsModel
from dreamer.losses import TokenizerLoss, ShortcutForcingLoss
from dreamer.data import MineRLDataset, create_dataloader
from dreamer.utils import set_seed, count_parameters, freeze_module


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_tokenizer(config: Dict) -> CausalTokenizer:
    """Create tokenizer model from config."""
    return CausalTokenizer(
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
    )


def create_dynamics_model(config: Dict) -> DynamicsModel:
    """Create dynamics model from config."""
    return DynamicsModel(
        latent_dim=config["tokenizer"]["latent_dim"],
        num_latent_tokens=config["tokenizer"]["num_latent_tokens"],
        embed_dim=config["tokenizer"]["embed_dim"],
        depth=config["tokenizer"]["depth"],
        num_heads=config["tokenizer"]["num_heads"],
        dropout=config["tokenizer"]["dropout"],
        num_discrete_actions=config["dynamics"]["num_discrete_actions"],
        num_registers=config["dynamics"]["num_registers"],
        max_shortcut_steps=config["dynamics"]["max_shortcut_steps"],
    )


def train_tokenizer_step(
    tokenizer: CausalTokenizer,
    batch: Dict[str, torch.Tensor],
    loss_fn: TokenizerLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_grad_norm: float = 1.0,
) -> Dict[str, float]:
    """Single training step for tokenizer."""
    optimizer.zero_grad()
    
    # Move data to device
    frames = batch["frames"].to(device)  # (B, T, C, H, W)
    
    # Reshape for tokenizer: (B, C, T, H, W)
    if frames.dim() == 5 and frames.shape[2] != tokenizer.in_channels:
        frames = frames.permute(0, 2, 1, 3, 4)
    
    # Forward pass
    output = tokenizer(frames, mask_ratio=tokenizer.mask_ratio)
    
    # Compute loss
    loss_dict = loss_fn(
        predicted_patches=output["reconstructed"],
        target_patches=output["original_patches"],
        mask=output.get("mask"),
    )
    
    loss = loss_dict["loss"]
    
    # Backward pass
    loss.backward()
    
    # Gradient clipping
    if max_grad_norm > 0:
        torch.nn.utils.clip_grad_norm_(tokenizer.parameters(), max_grad_norm)
    
    optimizer.step()
    
    return {k: v.item() for k, v in loss_dict.items()}


def train_dynamics_step(
    tokenizer: CausalTokenizer,
    dynamics: DynamicsModel,
    batch: Dict[str, torch.Tensor],
    loss_fn: ShortcutForcingLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_grad_norm: float = 1.0,
) -> Dict[str, float]:
    """Single training step for dynamics model."""
    optimizer.zero_grad()
    
    # Move data to device
    frames = batch["frames"].to(device)
    actions = batch["actions"].to(device)
    
    # Reshape frames for tokenizer
    if frames.dim() == 5 and frames.shape[2] != tokenizer.in_channels:
        frames = frames.permute(0, 2, 1, 3, 4)
    
    # Get latents from tokenizer (no gradient)
    with torch.no_grad():
        tokenizer_output = tokenizer.encode(frames, mask_ratio=0.0)
        latents = tokenizer_output["latents"]  # (B, T, num_latent, latent_dim)
    
    # Handle action format
    discrete_actions = actions.dim() == 2 or (actions.dim() == 3 and actions.shape[-1] == 1)
    if discrete_actions and actions.dim() == 3:
        actions = actions.squeeze(-1)
    
    # Forward pass through dynamics
    output = dynamics(
        latents=latents,
        actions=actions,
        discrete_actions=discrete_actions,
    )
    
    # Compute loss
    loss_dict = loss_fn(
        predicted_latents=output["predicted_latents"],
        target_latents=output["target_latents"],
        signal_level=output["signal_level"],
        step_size=output["step_size"],
        d_is_min=output["d_is_min"],
    )
    
    loss = loss_dict["loss"]
    
    # Backward pass
    loss.backward()
    
    # Gradient clipping
    if max_grad_norm > 0:
        torch.nn.utils.clip_grad_norm_(dynamics.parameters(), max_grad_norm)
    
    optimizer.step()
    
    return {k: v.item() for k, v in loss_dict.items()}


def train_phase1(config: Dict):
    """
    Train Phase 1: World Model Pretraining.
    
    1. Train tokenizer with MAE + LPIPS
    2. Train dynamics with shortcut forcing
    """
    # Setup
    device = torch.device(config["experiment"]["device"])
    set_seed(config["experiment"]["seed"])
    
    # Create directories
    log_dir = Path(config["experiment"]["log_dir"]) / config["experiment"]["name"]
    ckpt_dir = Path(config["experiment"]["checkpoint_dir"]) / config["experiment"]["name"]
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    # Create models
    print("Creating models...")
    tokenizer = create_tokenizer(config).to(device)
    dynamics = create_dynamics_model(config).to(device)
    
    print(f"Tokenizer parameters: {count_parameters(tokenizer):,}")
    print(f"Dynamics parameters: {count_parameters(dynamics):,}")
    
    # Create data loader
    print("Creating data loader...")
    train_loader = create_dataloader(
        data_path=config["data"]["path"],
        batch_size=config["data"]["batch_size"],
        sequence_length=config["data"]["sequence_length"],
        image_size=(config["data"]["image_height"], config["data"]["image_width"]),
        num_workers=config["data"]["num_workers"],
        split="train",
        max_episodes=config["data"].get("max_episodes", None),
    )
    
    # Create losses
    tokenizer_loss_fn = TokenizerLoss(
        lpips_weight=config["training"]["phase1"]["lpips_weight"],
        use_lpips=True,
    )
    dynamics_loss_fn = ShortcutForcingLoss()
    
    # Create optimizers
    phase1_config = config["training"]["phase1"]
    
    tokenizer_optimizer = AdamW(
        tokenizer.parameters(),
        lr=phase1_config["learning_rate"],
        weight_decay=phase1_config["weight_decay"],
        betas=tuple(config["optimizer"]["betas"]),
        eps=config["optimizer"]["eps"],
    )
    
    dynamics_optimizer = AdamW(
        dynamics.parameters(),
        lr=phase1_config["learning_rate"],
        weight_decay=phase1_config["weight_decay"],
        betas=tuple(config["optimizer"]["betas"]),
        eps=config["optimizer"]["eps"],
    )
    
    # Create schedulers
    total_steps = len(train_loader) * phase1_config["epochs"]
    tokenizer_scheduler = CosineAnnealingLR(
        tokenizer_optimizer,
        T_max=total_steps,
        eta_min=config["scheduler"]["min_lr"],
    )
    dynamics_scheduler = CosineAnnealingLR(
        dynamics_optimizer,
        T_max=total_steps,
        eta_min=config["scheduler"]["min_lr"],
    )
    
    # Tensorboard
    writer = SummaryWriter(log_dir)
    
    # Training loop
    print("Starting Phase 1 training...")
    global_step = 0
    
    for epoch in range(phase1_config["epochs"]):
        tokenizer.train()
        dynamics.train()
        
        epoch_losses = {"tokenizer": [], "dynamics": []}
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{phase1_config['epochs']}")
        
        for batch in pbar:
            # Train tokenizer
            tok_loss = train_tokenizer_step(
                tokenizer=tokenizer,
                batch=batch,
                loss_fn=tokenizer_loss_fn,
                optimizer=tokenizer_optimizer,
                device=device,
                max_grad_norm=phase1_config["max_grad_norm"],
            )
            epoch_losses["tokenizer"].append(tok_loss["loss"])
            tokenizer_scheduler.step()
            
            # Train dynamics
            dyn_loss = train_dynamics_step(
                tokenizer=tokenizer,
                dynamics=dynamics,
                batch=batch,
                loss_fn=dynamics_loss_fn,
                optimizer=dynamics_optimizer,
                device=device,
                max_grad_norm=phase1_config["max_grad_norm"],
            )
            epoch_losses["dynamics"].append(dyn_loss["loss"])
            dynamics_scheduler.step()
            
            # Update progress bar
            pbar.set_postfix({
                "tok_loss": f"{tok_loss['loss']:.4f}",
                "dyn_loss": f"{dyn_loss['loss']:.4f}",
            })
            
            # Log to tensorboard
            if global_step % config["logging"]["log_every"] == 0:
                writer.add_scalar("loss/tokenizer", tok_loss["loss"], global_step)
                writer.add_scalar("loss/dynamics", dyn_loss["loss"], global_step)
                writer.add_scalar("lr/tokenizer", tokenizer_scheduler.get_last_lr()[0], global_step)
            
            global_step += 1
        
        # Epoch summary
        avg_tok_loss = sum(epoch_losses["tokenizer"]) / len(epoch_losses["tokenizer"])
        avg_dyn_loss = sum(epoch_losses["dynamics"]) / len(epoch_losses["dynamics"])
        
        print(f"Epoch {epoch+1}: Tokenizer Loss = {avg_tok_loss:.4f}, Dynamics Loss = {avg_dyn_loss:.4f}")
        
        writer.add_scalar("epoch/tokenizer_loss", avg_tok_loss, epoch)
        writer.add_scalar("epoch/dynamics_loss", avg_dyn_loss, epoch)
        
        # Save checkpoint
        if (epoch + 1) % phase1_config["save_every"] == 0:
            checkpoint = {
                "epoch": epoch,
                "tokenizer_state_dict": tokenizer.state_dict(),
                "dynamics_state_dict": dynamics.state_dict(),
                "tokenizer_optimizer": tokenizer_optimizer.state_dict(),
                "dynamics_optimizer": dynamics_optimizer.state_dict(),
                "config": config,
            }
            torch.save(checkpoint, ckpt_dir / f"phase1_epoch_{epoch+1}.pt")
            print(f"Saved checkpoint to {ckpt_dir / f'phase1_epoch_{epoch+1}.pt'}")
    
    # Save final checkpoint
    final_checkpoint = {
        "epoch": phase1_config["epochs"],
        "tokenizer_state_dict": tokenizer.state_dict(),
        "dynamics_state_dict": dynamics.state_dict(),
        "config": config,
    }
    torch.save(final_checkpoint, ckpt_dir / "phase1_final.pt")
    print(f"Saved final checkpoint to {ckpt_dir / 'phase1_final.pt'}")
    
    writer.close()
    print("Phase 1 training complete!")


def main():
    parser = argparse.ArgumentParser(description="DreamerV4 Phase 1 Training")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/minerl.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (overrides config)",
    )
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override device if specified
    if args.device:
        config["experiment"]["device"] = args.device
    
    # Run training
    train_phase1(config)


if __name__ == "__main__":
    main()
