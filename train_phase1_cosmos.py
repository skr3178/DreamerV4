"""
DreamerV4 Phase 1 with Pretrained Cosmos Tokenizer

Trains ONLY the dynamics model using frozen NVIDIA Cosmos CV8x8x8 tokenizer.
This replaces custom tokenizer training with a pretrained video encoder.

Key differences from train_phase1.py:
- Cosmos tokenizer is frozen (no tokenizer training)
- No TokenizerLoss or LPIPS computation
- Supports gradient accumulation for memory efficiency
- Uses bfloat16 for Cosmos encoder
- ~40% fewer training steps due to stable features
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
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import yaml


def print_flush(*args, **kwargs):
    """Print and immediately flush stdout and stderr."""
    print(*args, **kwargs)
    sys.stdout.flush()
    sys.stderr.flush()


from dreamer.models import DynamicsModel
from dreamer.models.cosmos_tokenizer_wrapper import CosmosTokenizerWrapper, create_cosmos_tokenizer
from dreamer.losses import ShortcutForcingLoss
from dreamer.data import MineRLDataset, create_dataloader
from dreamer.utils import set_seed, count_parameters


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_cosmos_tokenizer_from_config(config: Dict, device: str) -> CosmosTokenizerWrapper:
    """Create Cosmos tokenizer wrapper from config."""
    cosmos_config = config.get("cosmos_tokenizer", {})

    return create_cosmos_tokenizer(
        checkpoint_path=cosmos_config.get("checkpoint_path", "cosmos_tokenizer/CV8x8x8"),
        pool_tokens=cosmos_config.get("pool_tokens", 16),
        input_resolution=cosmos_config.get("input_resolution", 256),
        device=device,
        dtype=cosmos_config.get("dtype", "bfloat16"),
    )


def create_dynamics_model_for_cosmos(config: Dict) -> DynamicsModel:
    """Create dynamics model configured for Cosmos tokenizer output."""
    # Cosmos outputs 16 tokens x 16 dim after pooling
    tokenizer_config = config.get("tokenizer", {})
    dynamics_config = config.get("dynamics", {})

    return DynamicsModel(
        latent_dim=tokenizer_config.get("latent_dim", 16),
        num_latent_tokens=tokenizer_config.get("num_latent_tokens", 16),
        embed_dim=dynamics_config.get("embed_dim", 256),
        depth=dynamics_config.get("num_layers", 12),
        num_heads=dynamics_config.get("num_heads", 8),
        dropout=tokenizer_config.get("dropout", 0.0),
        num_discrete_actions=dynamics_config.get("num_discrete_actions", 144),
        num_registers=dynamics_config.get("num_registers", 4),
        max_shortcut_steps=dynamics_config.get("max_shortcut_steps", 6),
    )


def train_dynamics_step(
    cosmos_tokenizer: CosmosTokenizerWrapper,
    dynamics: DynamicsModel,
    batch: Dict[str, torch.Tensor],
    loss_fn: ShortcutForcingLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[GradScaler] = None,
    accumulation_steps: int = 1,
    step_in_accumulation: int = 0,
    max_grad_norm: float = 1.0,
) -> Dict[str, float]:
    """
    Single training step for dynamics model with frozen Cosmos tokenizer.

    Args:
        cosmos_tokenizer: Frozen Cosmos tokenizer wrapper
        dynamics: Dynamics model (trainable)
        batch: Data batch with frames and actions
        loss_fn: Shortcut forcing loss function
        optimizer: Dynamics optimizer
        device: Training device
        scaler: Optional gradient scaler for mixed precision
        accumulation_steps: Number of gradient accumulation steps
        step_in_accumulation: Current step within accumulation
        max_grad_norm: Maximum gradient norm for clipping

    Returns:
        Dictionary of loss values
    """
    # Move data to device
    frames = batch["frames"].to(device)
    actions = batch["actions"].to(device)

    # Reshape frames for tokenizer: (B, T, C, H, W) -> (B, C, T, H, W)
    if frames.dim() == 5 and frames.shape[2] == 3:  # (B, T, C, H, W)
        frames = frames.permute(0, 2, 1, 3, 4)

    # Get latents from frozen Cosmos tokenizer (no gradient)
    with torch.no_grad():
        tokenizer_output = cosmos_tokenizer.encode(frames, mask_ratio=0.0)
        latents = tokenizer_output["latents"]  # (B, T_lat, num_latent, latent_dim)

    # Handle action format
    discrete_actions = actions.dim() == 2 or (actions.dim() == 3 and actions.shape[-1] == 1)
    if discrete_actions and actions.dim() == 3:
        actions = actions.squeeze(-1)

    # Adjust actions to match latent time dimension (T/8 due to temporal compression)
    T_lat = latents.shape[1]
    if actions.shape[1] > T_lat:
        # Subsample actions to match latent time steps
        # Take every 8th action (matching temporal compression)
        action_indices = torch.linspace(0, actions.shape[1] - 1, T_lat).long()
        actions = actions[:, action_indices]

    # Forward pass through dynamics with mixed precision
    if scaler is not None:
        with autocast(dtype=torch.bfloat16):
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

            loss = loss_dict["loss"] / accumulation_steps

        # Backward with scaling
        scaler.scale(loss).backward()

        # Only update on last accumulation step
        if (step_in_accumulation + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(dynamics.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
    else:
        output = dynamics(
            latents=latents,
            actions=actions,
            discrete_actions=discrete_actions,
        )

        loss_dict = loss_fn(
            predicted_latents=output["predicted_latents"],
            target_latents=output["target_latents"],
            signal_level=output["signal_level"],
            step_size=output["step_size"],
            d_is_min=output["d_is_min"],
        )

        loss = loss_dict["loss"] / accumulation_steps
        loss.backward()

        # Only update on last accumulation step
        if (step_in_accumulation + 1) % accumulation_steps == 0:
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(dynamics.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()

    return {k: v.item() * accumulation_steps for k, v in loss_dict.items()}


def train_phase1_cosmos(config: Dict, checkpoint_path: Optional[str] = None):
    """
    Train Phase 1 with Cosmos tokenizer: Dynamics only.

    The Cosmos tokenizer is frozen - we only train the dynamics model.
    This significantly reduces training time and memory requirements.

    Args:
        config: Training configuration
        checkpoint_path: Optional path to checkpoint to resume from
    """
    # Setup device
    requested_device = config["experiment"]["device"]
    if requested_device == "cuda" and not torch.cuda.is_available():
        print_flush("Warning: CUDA requested but not available. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(requested_device)

    print_flush(f"Using device: {device}")
    if device.type == "cuda":
        print_flush(f"  GPU: {torch.cuda.get_device_name(0)}")
        print_flush(f"  CUDA version: {torch.version.cuda}")
        torch.cuda.empty_cache()
        memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print_flush(f"  Total GPU Memory: {memory_total:.2f} GB")

    set_seed(config["experiment"]["seed"])

    # Create directories
    log_dir = Path(config["experiment"]["log_dir"]) / config["experiment"]["name"]
    ckpt_dir = Path(config["experiment"]["checkpoint_dir"]) / config["experiment"]["name"]
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Create Cosmos tokenizer (frozen)
    print_flush("\nLoading Cosmos tokenizer...")
    cosmos_tokenizer = create_cosmos_tokenizer_from_config(config, str(device))
    cosmos_tokenizer.eval()
    print_flush(f"  Cosmos tokenizer loaded")
    print_flush(f"  Pool tokens: {cosmos_tokenizer.pool_tokens}")
    print_flush(f"  Latent dim: {cosmos_tokenizer.latent_dim}")
    print_flush(f"  Input resolution: {cosmos_tokenizer.input_resolution}")

    # Create dynamics model
    print_flush("\nCreating dynamics model...")
    dynamics = create_dynamics_model_for_cosmos(config).to(device)

    # Load checkpoint if provided
    start_epoch = 0
    global_step = 0
    checkpoint = None
    if checkpoint_path:
        checkpoint_file = Path(checkpoint_path)
        if checkpoint_file.exists():
            print_flush(f"Loading checkpoint from {checkpoint_file}")
            checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=False)
            dynamics.load_state_dict(checkpoint["dynamics_state_dict"])
            start_epoch = checkpoint.get("epoch", 0)
            global_step = checkpoint.get("global_step", 0)
            print_flush(f"Resumed from epoch {start_epoch}, step {global_step}")
        else:
            print_flush(f"Warning: Checkpoint not found at {checkpoint_file}, starting from scratch")

    print_flush(f"\nModel Summary:")
    print_flush(f"  Dynamics parameters: {count_parameters(dynamics):,}")
    print_flush(f"  Cosmos tokenizer: FROZEN (pretrained)")

    # Create data loader
    print_flush("\nCreating data loader...")
    train_loader = create_dataloader(
        data_path=config["data"]["path"],
        batch_size=config["data"]["batch_size"],
        sequence_length=config["data"]["sequence_length"],
        image_size=(config["data"]["image_height"], config["data"]["image_width"]),
        num_workers=config["data"]["num_workers"],
        split="train",
        max_episodes=config["data"].get("max_episodes", None),
    )

    # Create loss function
    dynamics_loss_fn = ShortcutForcingLoss()

    # Training config
    phase1_config = config["training"]["phase1"]
    accumulation_steps = phase1_config.get("gradient_accumulation_steps", 1)
    use_amp = phase1_config.get("precision", "float32") in ["bfloat16", "float16"]

    print_flush(f"\nTraining Configuration:")
    print_flush(f"  Batch size: {config['data']['batch_size']}")
    print_flush(f"  Gradient accumulation: {accumulation_steps}")
    print_flush(f"  Effective batch size: {config['data']['batch_size'] * accumulation_steps}")
    print_flush(f"  Sequence length: {config['data']['sequence_length']}")
    print_flush(f"  Mixed precision: {use_amp}")
    print_flush(f"  Max steps: {phase1_config.get('steps', 'epoch-based')}")

    # Create optimizer (dynamics only)
    dynamics_optimizer = AdamW(
        dynamics.parameters(),
        lr=phase1_config["learning_rate"],
        weight_decay=phase1_config["weight_decay"],
        betas=tuple(config["optimizer"]["betas"]),
        eps=config["optimizer"]["eps"],
    )

    # Load optimizer state if resuming
    if checkpoint is not None and "dynamics_optimizer" in checkpoint:
        dynamics_optimizer.load_state_dict(checkpoint["dynamics_optimizer"])
        print_flush("Loaded dynamics optimizer state")

    # Create scheduler
    total_steps = phase1_config.get("steps", len(train_loader) * phase1_config.get("epochs", 100))
    dynamics_scheduler = CosineAnnealingLR(
        dynamics_optimizer,
        T_max=total_steps,
        eta_min=config["scheduler"]["min_lr"],
    )

    # Fast-forward scheduler if resuming
    if checkpoint is not None and global_step > 0:
        for _ in range(global_step):
            dynamics_scheduler.step()
        print_flush(f"Fast-forwarded scheduler to step {global_step}")

    # Gradient scaler for mixed precision
    scaler = GradScaler() if use_amp and device.type == "cuda" else None

    # Tensorboard
    writer = SummaryWriter(log_dir)

    # Training loop
    print_flush("\nStarting Phase 1 (Cosmos) training...")
    print_flush(f"Total batches per epoch: {len(train_loader)}")
    print_flush(f"Total steps: {total_steps}")
    print_flush()

    # Track losses
    epoch_losses = {
        "total": [],
        "d_min": [],
        "d_other": [],
        "mean_weight": [],
        "mean_signal": [],
        "mean_step": [],
    }

    max_steps = phase1_config.get("steps", None)
    max_epochs = phase1_config.get("epochs", 100)
    epoch = start_epoch

    dynamics.train()
    dynamics_optimizer.zero_grad()

    while True:
        if max_steps and global_step >= max_steps:
            break
        if not max_steps and epoch >= max_epochs:
            break

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}",
            file=sys.stdout,
            mininterval=1.0,
        )

        for batch_idx, batch in enumerate(pbar):
            # Train dynamics
            step_in_accumulation = batch_idx % accumulation_steps

            dyn_loss = train_dynamics_step(
                cosmos_tokenizer=cosmos_tokenizer,
                dynamics=dynamics,
                batch=batch,
                loss_fn=dynamics_loss_fn,
                optimizer=dynamics_optimizer,
                device=device,
                scaler=scaler,
                accumulation_steps=accumulation_steps,
                step_in_accumulation=step_in_accumulation,
                max_grad_norm=phase1_config["max_grad_norm"],
            )

            # Track losses
            epoch_losses["total"].append(dyn_loss["loss"])
            epoch_losses["d_min"].append(dyn_loss.get("d_min_loss", 0.0))
            epoch_losses["d_other"].append(dyn_loss.get("d_other_loss", 0.0))
            epoch_losses["mean_weight"].append(dyn_loss.get("mean_weight", 0.0))
            epoch_losses["mean_signal"].append(dyn_loss.get("mean_signal_level", 0.0))
            epoch_losses["mean_step"].append(dyn_loss.get("mean_step_size", 0.0))

            # Update scheduler (per effective step)
            if (step_in_accumulation + 1) % accumulation_steps == 0:
                dynamics_scheduler.step()
                global_step += 1

            dyn_lr = dynamics_scheduler.get_last_lr()[0]

            # Update progress bar
            pbar.set_postfix({
                "loss": f"{dyn_loss['loss']:.4f}",
                "d_min": f"{dyn_loss.get('d_min_loss', 0.0):.4f}",
                "d_other": f"{dyn_loss.get('d_other_loss', 0.0):.4f}",
                "lr": f"{dyn_lr:.2e}",
                "step": global_step,
            })

            # Log to tensorboard
            if global_step % config["logging"]["log_every"] == 0:
                writer.add_scalar("loss/dynamics/total", dyn_loss["loss"], global_step)
                writer.add_scalar("loss/dynamics/d_min", dyn_loss.get("d_min_loss", 0.0), global_step)
                writer.add_scalar("loss/dynamics/d_other", dyn_loss.get("d_other_loss", 0.0), global_step)
                writer.add_scalar("loss/dynamics/mean_weight", dyn_loss.get("mean_weight", 0.0), global_step)
                writer.add_scalar("loss/dynamics/mean_signal_level", dyn_loss.get("mean_signal_level", 0.0), global_step)
                writer.add_scalar("loss/dynamics/mean_step_size", dyn_loss.get("mean_step_size", 0.0), global_step)
                writer.add_scalar("lr/dynamics", dyn_lr, global_step)

                if device.type == "cuda":
                    memory_used = torch.cuda.memory_allocated(0) / 1024**3
                    writer.add_scalar("memory/gpu_gb", memory_used, global_step)

                writer.flush()

            # Save checkpoint periodically
            save_every = phase1_config.get("save_every", 10000)
            if global_step > 0 and global_step % save_every == 0:
                checkpoint = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "dynamics_state_dict": dynamics.state_dict(),
                    "dynamics_optimizer": dynamics_optimizer.state_dict(),
                    "config": config,
                }
                step_checkpoint_path = ckpt_dir / f"phase1_cosmos_step_{global_step}.pt"
                torch.save(checkpoint, step_checkpoint_path)
                print_flush(f"\nSaved checkpoint at step {global_step}")

            # Check max steps
            if max_steps and global_step >= max_steps:
                break

        # End of epoch
        epoch += 1

        # Epoch summary
        def avg(lst):
            return sum(lst) / len(lst) if lst else 0.0

        print_flush(f"\n{'='*60}")
        print_flush(f"Epoch {epoch} Summary")
        print_flush(f"{'='*60}")
        print_flush(f"  Total Loss:   {avg(epoch_losses['total']):.6f}")
        print_flush(f"  d_min Loss:   {avg(epoch_losses['d_min']):.6f}")
        print_flush(f"  d_other Loss: {avg(epoch_losses['d_other']):.6f}")
        print_flush(f"  Mean Weight:  {avg(epoch_losses['mean_weight']):.6f}")
        print_flush(f"  Mean Signal:  {avg(epoch_losses['mean_signal']):.6f}")
        print_flush(f"  Mean Step:    {avg(epoch_losses['mean_step']):.6f}")
        print_flush(f"  Global Step:  {global_step}")

        if device.type == "cuda":
            memory_used = torch.cuda.memory_allocated(0) / 1024**3
            print_flush(f"  GPU Memory:   {memory_used:.2f} GB")

        print_flush(f"{'='*60}\n")

        # Log epoch averages
        writer.add_scalar("epoch/loss", avg(epoch_losses['total']), epoch)
        writer.add_scalar("epoch/d_min", avg(epoch_losses['d_min']), epoch)
        writer.add_scalar("epoch/d_other", avg(epoch_losses['d_other']), epoch)

        # Reset epoch losses
        for key in epoch_losses:
            epoch_losses[key] = []

        # Save epoch checkpoint
        checkpoint = {
            "epoch": epoch,
            "global_step": global_step,
            "dynamics_state_dict": dynamics.state_dict(),
            "dynamics_optimizer": dynamics_optimizer.state_dict(),
            "config": config,
        }
        epoch_checkpoint_path = ckpt_dir / f"phase1_cosmos_epoch_{epoch}.pt"
        torch.save(checkpoint, epoch_checkpoint_path)
        print_flush(f"Saved epoch checkpoint to {epoch_checkpoint_path}")

    # Save final checkpoint
    final_checkpoint = {
        "epoch": epoch,
        "global_step": global_step,
        "dynamics_state_dict": dynamics.state_dict(),
        "dynamics_optimizer": dynamics_optimizer.state_dict(),
        "config": config,
    }
    torch.save(final_checkpoint, ckpt_dir / "phase1_cosmos_final.pt")
    print_flush(f"\nSaved final checkpoint (step {global_step}) to {ckpt_dir / 'phase1_cosmos_final.pt'}")

    writer.close()
    print_flush("Phase 1 (Cosmos) training complete!")


def main():
    parser = argparse.ArgumentParser(description="DreamerV4 Phase 1 Training with Cosmos Tokenizer")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/minerl_cosmos.yaml",
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

    # Verify Cosmos tokenizer is enabled
    if not config.get("cosmos_tokenizer", {}).get("enabled", False):
        print_flush("Warning: cosmos_tokenizer.enabled is not set to true in config")
        print_flush("This script is designed for Cosmos tokenizer. Use train_phase1.py for custom tokenizer.")

    # Override device if specified
    if args.device:
        config["experiment"]["device"] = args.device

    # Run training
    train_phase1_cosmos(config, checkpoint_path=args.checkpoint)


if __name__ == "__main__":
    main()
