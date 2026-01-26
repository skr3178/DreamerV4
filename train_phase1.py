"""
DreamerV4 Phase 1: World Model Pretraining

Trains the tokenizer and dynamics model on video data.

Phase 1 consists of:
1. Tokenizer training: Masked autoencoding with MSE + LPIPS loss (Eq. 5)
2. Dynamics training: Shortcut forcing objective (Eq. 7)
"""

import os
import sys
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

# Helper function for flushed printing (ensures output appears in log files immediately)
def print_flush(*args, **kwargs):
    """Print and immediately flush stdout and stderr."""
    print(*args, **kwargs)
    sys.stdout.flush()
    sys.stderr.flush()

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

    # Reconstruct images from patches for LPIPS
    B, T = output["reconstructed"].shape[:2]
    reconstructed_images = torch.stack([
        tokenizer.decode_patches(output["reconstructed"][:, t])
        for t in range(T)
    ], dim=1)  # (B, T, C, H, W)

    # Target images (permute back to B, T, C, H, W)
    target_images = frames.permute(0, 2, 1, 3, 4) if frames.shape[1] == tokenizer.in_channels else frames

    # Compute loss with images for LPIPS
    loss_dict = loss_fn(
        predicted_patches=output["reconstructed"],
        target_patches=output["original_patches"],
        mask=output.get("mask"),
        predicted_images=reconstructed_images,
        target_images=target_images,
        decoder_images=output.get("decoder_images"),
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


def train_phase1(config: Dict, checkpoint_path: Optional[str] = None):
    """
    Train Phase 1: World Model Pretraining.
    
    1. Train tokenizer with MAE + LPIPS
    2. Train dynamics with shortcut forcing
    
    Args:
        config: Training configuration
        checkpoint_path: Optional path to checkpoint to resume from
    """
    # Setup
    requested_device = config["experiment"]["device"]
    # Check CUDA availability and fallback to CPU if needed
    if requested_device == "cuda" and not torch.cuda.is_available():
        print_flush("⚠️  Warning: CUDA requested but not available. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(requested_device)
    
    print_flush(f"Using device: {device}")
    if device.type == "cuda":
        print_flush(f"  GPU: {torch.cuda.get_device_name(0)}")
        print_flush(f"  CUDA version: {torch.version.cuda}")
        # Clear any leftover GPU memory
        torch.cuda.empty_cache()
        memory_allocated = torch.cuda.memory_allocated(0) / 1024**2
        memory_reserved = torch.cuda.memory_reserved(0) / 1024**2
        if memory_allocated > 0 or memory_reserved > 0:
            print_flush(f"  Warning: GPU memory already in use: {memory_allocated:.2f} MB allocated, {memory_reserved:.2f} MB reserved")
            print_flush(f"  Clearing cache...")
            torch.cuda.empty_cache()
    
    set_seed(config["experiment"]["seed"])
    
    # Create directories
    log_dir = Path(config["experiment"]["log_dir"]) / config["experiment"]["name"]
    ckpt_dir = Path(config["experiment"]["checkpoint_dir"]) / config["experiment"]["name"]
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    # Create models
    print_flush("Creating models...")
    tokenizer = create_tokenizer(config).to(device)
    dynamics = create_dynamics_model(config).to(device)
    
    # Load checkpoint if provided
    start_epoch = 0
    global_step = 0
    checkpoint = None
    if checkpoint_path:
        checkpoint_file = Path(checkpoint_path)
        if checkpoint_file.exists():
            print_flush(f"Loading checkpoint from {checkpoint_file}")
            checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=False)
            tokenizer.load_state_dict(checkpoint["tokenizer_state_dict"])
            dynamics.load_state_dict(checkpoint["dynamics_state_dict"])
            start_epoch = checkpoint.get("epoch", 0)
            global_step = checkpoint.get("global_step", 0)
            print_flush(f"Resumed from epoch {start_epoch}, step {global_step}")
        else:
            print_flush(f"Warning: Checkpoint not found at {checkpoint_file}, starting from scratch")
    
    # Verify models are on correct device
    tokenizer_device = next(tokenizer.parameters()).device
    dynamics_device = next(dynamics.parameters()).device
    print_flush(f"Tokenizer device: {tokenizer_device}")
    print_flush(f"Dynamics device: {dynamics_device}")
    
    print_flush(f"Tokenizer parameters: {count_parameters(tokenizer):,}")
    print_flush(f"Dynamics parameters: {count_parameters(dynamics):,}")
    
    # Create data loader
    print_flush("Creating data loader...")
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
    use_lpips = config["training"]["phase1"].get("use_lpips", True)
    tokenizer_loss_fn = TokenizerLoss(
        lpips_weight=config["training"]["phase1"]["lpips_weight"],
        use_lpips=use_lpips,
    )
    # Move LPIPS to device if available
    if hasattr(tokenizer_loss_fn, 'lpips_fn') and tokenizer_loss_fn.lpips_fn is not None:
        tokenizer_loss_fn.lpips_fn = tokenizer_loss_fn.lpips_fn.to(device)
        print_flush("LPIPS model moved to device")
    elif not use_lpips:
        print_flush("LPIPS disabled for faster training")
    
    dynamics_loss_fn = ShortcutForcingLoss()
    
    # Print training configuration
    print_flush(f"\nTraining Configuration:")
    print_flush(f"  Batch size: {config['data']['batch_size']}")
    print_flush(f"  Num workers: {config['data']['num_workers']}")
    print_flush(f"  Sequence length: {config['data']['sequence_length']}")
    print_flush(f"  LPIPS enabled: {use_lpips}")
    print_flush(f"  Device: {device}")
    if device.type == "cuda":
        print_flush(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print_flush()
    
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
    
    # Load optimizer states if resuming
    if checkpoint is not None:
        if "tokenizer_optimizer" in checkpoint:
            tokenizer_optimizer.load_state_dict(checkpoint["tokenizer_optimizer"])
            print_flush("Loaded tokenizer optimizer state")
        if "dynamics_optimizer" in checkpoint:
            dynamics_optimizer.load_state_dict(checkpoint["dynamics_optimizer"])
            print_flush("Loaded dynamics optimizer state")
    
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
    
    # Fast-forward schedulers if resuming
    if checkpoint is not None and global_step > 0:
        for _ in range(global_step):
            tokenizer_scheduler.step()
            dynamics_scheduler.step()
        print_flush(f"Fast-forwarded schedulers to step {global_step}")
    
    # Tensorboard
    writer = SummaryWriter(log_dir)
    
    # Training loop
    print_flush("Starting Phase 1 training...")
    print_flush(f"Total batches per epoch: {len(train_loader)}")
    max_steps = phase1_config.get("max_steps", None)
    if max_steps:
        print_flush(f"Max steps: {max_steps} (will stop early if reached)")
    print_flush(f"Logging every {config['logging']['log_every']} steps")
    if checkpoint is not None:
        print_flush(f"Resuming from epoch {start_epoch + 1}/{phase1_config['epochs']}, step {global_step}\n")
    else:
        print_flush()
    
    for epoch in range(start_epoch, phase1_config["epochs"]):
        tokenizer.train()
        dynamics.train()
        
        # Track all loss components
        epoch_losses = {
            "tokenizer": {
                "total": [],
                "mse": [],
                "lpips": [],
                "mse_norm": [],
                "lpips_norm": [],
            },
            "dynamics": {
                "total": [],
                "d_min": [],
                "d_other": [],
                "mean_weight": [],
                "mean_signal": [],
                "mean_step": [],
            }
        }
        
        # Calculate starting batch index if resuming mid-epoch
        start_batch_idx = 0
        if checkpoint is not None and epoch == start_epoch:
            # If resuming in the same epoch, calculate which batch to start from
            # global_step tells us how many batches we've processed
            batches_per_epoch = len(train_loader)
            if global_step > 0:
                # Calculate which batch in this epoch corresponds to global_step
                # global_step = (epoch * batches_per_epoch) + batch_idx
                batch_idx_in_epoch = global_step % batches_per_epoch
                start_batch_idx = batch_idx_in_epoch + 1  # Continue from next batch
                if start_batch_idx >= batches_per_epoch:
                    # If we've passed the end of this epoch, move to next epoch
                    start_batch_idx = 0
                    continue
                print_flush(f"Resuming from batch {start_batch_idx} in epoch {epoch+1} (was at step {global_step})")
        
        # Configure tqdm to flush output immediately
        pbar = tqdm(
            train_loader, 
            desc=f"Epoch {epoch+1}/{phase1_config['epochs']}",
            file=sys.stdout,  # Explicitly use stdout
            mininterval=1.0,  # Update at least every second
        )
        
        # Set progress bar to correct starting position if resuming
        # Note: tqdm will auto-increment, so we need to account for that
        if start_batch_idx > 0:
            # We'll manually update the display during skip
            pass
        
        for batch_idx, batch in enumerate(pbar):
            # Skip batches if resuming mid-epoch
            if batch_idx < start_batch_idx:
                # Update global_step to keep it in sync, but don't train
                global_step += 1
                # Manually set progress bar position (tqdm auto-increments, so we correct it)
                pbar.n = batch_idx + 1
                pbar.refresh()
                continue
            # Train tokenizer
            tok_loss = train_tokenizer_step(
                tokenizer=tokenizer,
                batch=batch,
                loss_fn=tokenizer_loss_fn,
                optimizer=tokenizer_optimizer,
                device=device,
                max_grad_norm=phase1_config["max_grad_norm"],
            )
            
            # Track tokenizer losses
            epoch_losses["tokenizer"]["total"].append(tok_loss["loss"])
            epoch_losses["tokenizer"]["mse"].append(tok_loss.get("mse_loss", 0.0))
            epoch_losses["tokenizer"]["lpips"].append(tok_loss.get("lpips_loss", 0.0))
            epoch_losses["tokenizer"]["mse_norm"].append(tok_loss.get("mse_loss_normalized", 0.0))
            epoch_losses["tokenizer"]["lpips_norm"].append(tok_loss.get("lpips_loss_normalized", 0.0))
            
            tokenizer_scheduler.step()
            tok_lr = tokenizer_scheduler.get_last_lr()[0]
            
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
            
            # Track dynamics losses
            epoch_losses["dynamics"]["total"].append(dyn_loss["loss"])
            epoch_losses["dynamics"]["d_min"].append(dyn_loss.get("d_min_loss", 0.0))
            epoch_losses["dynamics"]["d_other"].append(dyn_loss.get("d_other_loss", 0.0))
            epoch_losses["dynamics"]["mean_weight"].append(dyn_loss.get("mean_weight", 0.0))
            epoch_losses["dynamics"]["mean_signal"].append(dyn_loss.get("mean_signal_level", 0.0))
            epoch_losses["dynamics"]["mean_step"].append(dyn_loss.get("mean_step_size", 0.0))
            
            dynamics_scheduler.step()
            dyn_lr = dynamics_scheduler.get_last_lr()[0]
            
            # Update progress bar with detailed metrics
            pbar.set_postfix({
                "tok": f"{tok_loss['loss']:.4f}",
                "tok_mse": f"{tok_loss.get('mse_loss', 0.0):.4f}",
                "tok_lpips": f"{tok_loss.get('lpips_loss', 0.0):.4f}",
                "dyn": f"{dyn_loss['loss']:.4f}",
                "d_min": f"{dyn_loss.get('d_min_loss', 0.0):.4f}",
                "d_other": f"{dyn_loss.get('d_other_loss', 0.0):.4f}",
                "lr_tok": f"{tok_lr:.2e}",
                "lr_dyn": f"{dyn_lr:.2e}",
            })
            # Flush progress bar output
            pbar.refresh()
            sys.stdout.flush()
            sys.stderr.flush()
            
            # Log to tensorboard
            if global_step % config["logging"]["log_every"] == 0:
                # Tokenizer losses
                writer.add_scalar("loss/tokenizer/total", tok_loss["loss"], global_step)
                writer.add_scalar("loss/tokenizer/mse", tok_loss.get("mse_loss", 0.0), global_step)
                writer.add_scalar("loss/tokenizer/lpips", tok_loss.get("lpips_loss", 0.0), global_step)
                writer.add_scalar("loss/tokenizer/mse_normalized", tok_loss.get("mse_loss_normalized", 0.0), global_step)
                writer.add_scalar("loss/tokenizer/lpips_normalized", tok_loss.get("lpips_loss_normalized", 0.0), global_step)
                
                # Dynamics losses
                writer.add_scalar("loss/dynamics/total", dyn_loss["loss"], global_step)
                writer.add_scalar("loss/dynamics/d_min", dyn_loss.get("d_min_loss", 0.0), global_step)
                writer.add_scalar("loss/dynamics/d_other", dyn_loss.get("d_other_loss", 0.0), global_step)
                writer.add_scalar("loss/dynamics/mean_weight", dyn_loss.get("mean_weight", 0.0), global_step)
                writer.add_scalar("loss/dynamics/mean_signal_level", dyn_loss.get("mean_signal_level", 0.0), global_step)
                writer.add_scalar("loss/dynamics/mean_step_size", dyn_loss.get("mean_step_size", 0.0), global_step)
                
                # Learning rates
                writer.add_scalar("lr/tokenizer", tok_lr, global_step)
                writer.add_scalar("lr/dynamics", dyn_lr, global_step)
                
                # Flush after logging
                writer.flush()
                sys.stdout.flush()
                sys.stderr.flush()
            
            global_step += 1

            # Save mid-epoch checkpoint (at halfway point of each epoch)
            num_batches = len(train_loader)
            if batch_idx == num_batches // 2:
                mid_checkpoint = {
                    "epoch": epoch,
                    "batch_idx": batch_idx,
                    "global_step": global_step,
                    "tokenizer_state_dict": tokenizer.state_dict(),
                    "dynamics_state_dict": dynamics.state_dict(),
                    "tokenizer_optimizer": tokenizer_optimizer.state_dict(),
                    "dynamics_optimizer": dynamics_optimizer.state_dict(),
                    "config": config,
                }
                mid_path = ckpt_dir / f"phase1_epoch{epoch+1}_mid.pt"
                torch.save(mid_checkpoint, mid_path)
                print_flush(f"\nSaved mid-epoch checkpoint to {mid_path}")

            # Save checkpoint periodically during training (for max_steps scenarios)
            save_every_steps = phase1_config.get("save_every_steps", None)
            if save_every_steps and global_step % save_every_steps == 0:
                checkpoint = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "tokenizer_state_dict": tokenizer.state_dict(),
                    "dynamics_state_dict": dynamics.state_dict(),
                    "tokenizer_optimizer": tokenizer_optimizer.state_dict(),
                    "dynamics_optimizer": dynamics_optimizer.state_dict(),
                    "config": config,
                }
                step_checkpoint_path = ckpt_dir / f"phase1_step_{global_step}.pt"
                torch.save(checkpoint, step_checkpoint_path)
                print_flush(f"Saved checkpoint at step {global_step} to {step_checkpoint_path}")
            
            # Check if we've reached max_steps
            if max_steps and global_step >= max_steps:
                print_flush(f"\nReached max_steps ({max_steps}), stopping training early...")
                # Save final checkpoint before stopping
                final_checkpoint = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "tokenizer_state_dict": tokenizer.state_dict(),
                    "dynamics_state_dict": dynamics.state_dict(),
                    "tokenizer_optimizer": tokenizer_optimizer.state_dict(),
                    "dynamics_optimizer": dynamics_optimizer.state_dict(),
                    "config": config,
                }
                final_step_path = ckpt_dir / f"phase1_step_{global_step}_final.pt"
                torch.save(final_checkpoint, final_step_path)
                print_flush(f"Saved final checkpoint at step {global_step} to {final_step_path}")
                break
        
        # Check if we should break out of epoch loop
        if max_steps and global_step >= max_steps:
            break
        
        # Compute epoch averages
        def avg(lst):
            return sum(lst) / len(lst) if lst else 0.0
        
        avg_tok_total = avg(epoch_losses["tokenizer"]["total"])
        avg_tok_mse = avg(epoch_losses["tokenizer"]["mse"])
        avg_tok_lpips = avg(epoch_losses["tokenizer"]["lpips"])
        avg_tok_mse_norm = avg(epoch_losses["tokenizer"]["mse_norm"])
        avg_tok_lpips_norm = avg(epoch_losses["tokenizer"]["lpips_norm"])
        
        avg_dyn_total = avg(epoch_losses["dynamics"]["total"])
        avg_dyn_d_min = avg(epoch_losses["dynamics"]["d_min"])
        avg_dyn_d_other = avg(epoch_losses["dynamics"]["d_other"])
        avg_dyn_weight = avg(epoch_losses["dynamics"]["mean_weight"])
        avg_dyn_signal = avg(epoch_losses["dynamics"]["mean_signal"])
        avg_dyn_step = avg(epoch_losses["dynamics"]["mean_step"])
        
        # Detailed epoch summary
        print_flush(f"\n{'='*80}")
        print_flush(f"Epoch {epoch+1}/{phase1_config['epochs']} Summary")
        print_flush(f"{'='*80}")
        print_flush(f"Tokenizer Losses:")
        print_flush(f"  Total:        {avg_tok_total:.6f}")
        print_flush(f"  MSE:          {avg_tok_mse:.6f}")
        print_flush(f"  LPIPS:        {avg_tok_lpips:.6f}")
        print_flush(f"  MSE (norm):   {avg_tok_mse_norm:.6f}")
        print_flush(f"  LPIPS (norm): {avg_tok_lpips_norm:.6f}")
        print_flush(f"  Learning Rate: {tok_lr:.2e}")
        print_flush(f"\nDynamics Losses:")
        print_flush(f"  Total:        {avg_dyn_total:.6f}")
        print_flush(f"  d_min:        {avg_dyn_d_min:.6f}")
        print_flush(f"  d_other:      {avg_dyn_d_other:.6f}")
        print_flush(f"  Mean Weight:  {avg_dyn_weight:.6f}")
        print_flush(f"  Mean Signal:  {avg_dyn_signal:.6f}")
        print_flush(f"  Mean Step:    {avg_dyn_step:.6f}")
        print_flush(f"  Learning Rate: {dyn_lr:.2e}")
        print_flush(f"{'='*80}\n")
        
        # Log epoch averages to tensorboard
        writer.add_scalar("epoch/tokenizer/total", avg_tok_total, epoch)
        writer.add_scalar("epoch/tokenizer/mse", avg_tok_mse, epoch)
        writer.add_scalar("epoch/tokenizer/lpips", avg_tok_lpips, epoch)
        writer.add_scalar("epoch/tokenizer/mse_normalized", avg_tok_mse_norm, epoch)
        writer.add_scalar("epoch/tokenizer/lpips_normalized", avg_tok_lpips_norm, epoch)
        writer.add_scalar("epoch/dynamics/total", avg_dyn_total, epoch)
        writer.add_scalar("epoch/dynamics/d_min", avg_dyn_d_min, epoch)
        writer.add_scalar("epoch/dynamics/d_other", avg_dyn_d_other, epoch)
        writer.add_scalar("epoch/dynamics/mean_weight", avg_dyn_weight, epoch)
        writer.add_scalar("epoch/dynamics/mean_signal_level", avg_dyn_signal, epoch)
        writer.add_scalar("epoch/dynamics/mean_step_size", avg_dyn_step, epoch)
        writer.add_scalar("epoch/lr/tokenizer", tok_lr, epoch)
        writer.add_scalar("epoch/lr/dynamics", dyn_lr, epoch)
        
        # Save checkpoint at end of every epoch
        checkpoint = {
            "epoch": epoch,
            "global_step": global_step,
            "tokenizer_state_dict": tokenizer.state_dict(),
            "dynamics_state_dict": dynamics.state_dict(),
            "tokenizer_optimizer": tokenizer_optimizer.state_dict(),
            "dynamics_optimizer": dynamics_optimizer.state_dict(),
            "config": config,
        }
        epoch_checkpoint_path = ckpt_dir / f"phase1_epoch_{epoch+1}.pt"
        torch.save(checkpoint, epoch_checkpoint_path)
        print_flush(f"Saved end-of-epoch checkpoint to {epoch_checkpoint_path}")
    
    # Save final checkpoint
    final_checkpoint = {
        "epoch": epoch if max_steps and global_step >= max_steps else phase1_config["epochs"],
        "global_step": global_step,
        "tokenizer_state_dict": tokenizer.state_dict(),
        "dynamics_state_dict": dynamics.state_dict(),
        "tokenizer_optimizer": tokenizer_optimizer.state_dict(),
        "dynamics_optimizer": dynamics_optimizer.state_dict(),
        "config": config,
    }
    torch.save(final_checkpoint, ckpt_dir / "phase1_final.pt")
    print_flush(f"Saved final checkpoint (step {global_step}) to {ckpt_dir / 'phase1_final.pt'}")
    
    writer.close()
    print_flush("Phase 1 training complete!")


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
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override device if specified
    if args.device:
        config["experiment"]["device"] = args.device
    
    # Run training
    train_phase1(config, checkpoint_path=args.checkpoint)


if __name__ == "__main__":
    main()
