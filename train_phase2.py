"""
DreamerV4 Phase 2: Agent Finetuning

Finetunes the policy and reward heads with frozen transformer.

Phase 2 consists of:
1. Freeze transformer weights (from tokenizer and dynamics)
2. Train policy head with behavior cloning (Eq. 9)
3. Train reward head with reward prediction
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
from dreamer.models import PolicyHead, ValueHead, RewardHead
from dreamer.losses import AgentFinetuningLoss
from dreamer.data import create_dataloader
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


def create_heads(config: Dict) -> Dict[str, nn.Module]:
    """Create agent heads from config."""
    # Input dimension is flattened latent tokens
    input_dim = config["tokenizer"]["num_latent_tokens"] * config["tokenizer"]["latent_dim"]
    
    # MTP configuration (default: True to match paper Equation 9)
    use_mtp = config["training"]["phase2"].get("use_mtp", True)
    mtp_length = config["training"]["phase2"].get("mtp_length", 8)
    
    # Check action space type
    action_space_type = config["dynamics"].get("action_space_type", "categorical")
    use_multi_discrete = (action_space_type == "multi_discrete")
    
    heads = {
        "policy": PolicyHead(
            input_dim=input_dim,
            hidden_dim=config["heads"]["hidden_dim"],
            num_discrete_actions=config["dynamics"]["num_discrete_actions"] if not use_multi_discrete else None,
            use_multi_discrete=use_multi_discrete,
            num_layers=config["heads"]["num_layers"],
            mtp_length=mtp_length,
            use_mtp=use_mtp,
        ),
        "value": ValueHead(
            input_dim=input_dim,
            hidden_dim=config["heads"]["hidden_dim"],
            num_layers=config["heads"]["num_layers"],
            num_bins=config["heads"]["num_bins"],
        ),
        "reward": RewardHead(
            input_dim=input_dim,
            hidden_dim=config["heads"]["hidden_dim"],
            num_layers=config["heads"]["num_layers"],
            num_bins=config["heads"]["num_bins"],
            mtp_length=mtp_length,
            use_mtp=use_mtp,
        ),
    }
    
    return heads


def train_agent_step(
    tokenizer: CausalTokenizer,
    dynamics: DynamicsModel,
    heads: Dict[str, nn.Module],
    batch: Dict[str, torch.Tensor],
    loss_fn: AgentFinetuningLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_grad_norm: float = 1.0,
    action_type: str = "discrete",
    debug: bool = False,
) -> Dict[str, float]:
    """Single training step for agent heads."""
    optimizer.zero_grad()
    
    # Move data to device
    frames = batch["frames"].to(device, non_blocking=True)  # non_blocking for faster transfer
    actions = batch["actions"].to(device, non_blocking=True)
    rewards = batch["rewards"].to(device, non_blocking=True)
    
    # Reshape frames for tokenizer
    if frames.dim() == 5 and frames.shape[2] != tokenizer.in_channels:
        frames = frames.permute(0, 2, 1, 3, 4)
    
    # Get latents from tokenizer (no gradient - frozen)
    # Use no_grad instead of inference_mode because latents need to be used in autograd
    # (even though tokenizer is frozen, the latents need to be regular tensors for head training)
    with torch.no_grad():
        tokenizer_output = tokenizer.encode(frames, mask_ratio=0.0)
        latents = tokenizer_output["latents"]  # (B, T, num_latent, latent_dim)
    
    # Clone latents to make them regular tensors (not inference tensors)
    # This allows them to be used in autograd for head training
    latents = latents.clone()
    
    # Flatten latents for heads
    batch_size, time_steps, num_latent, latent_dim = latents.shape
    latents_flat = latents.reshape(batch_size, time_steps, -1)  # (B, T, num_latent * latent_dim)

    # Handle action format
    if actions.dim() == 3 and actions.shape[-1] == 1:
        actions = actions.squeeze(-1)

    # Compute loss (heads forward pass happens inside loss_fn)
    loss_dict = loss_fn(
        policy_head=heads["policy"],
        reward_head=heads["reward"],
        latents=latents_flat,
        actions=actions,
        rewards=rewards,
        action_type=action_type,
    )
    
    loss = loss_dict["loss"]
    
    # Backward pass
    loss.backward()
    
    # Gradient clipping
    if max_grad_norm > 0:
        all_params = []
        for head in heads.values():
            all_params.extend(head.parameters())
        torch.nn.utils.clip_grad_norm_(all_params, max_grad_norm)
    
    optimizer.step()
    
    return {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in loss_dict.items()}


def train_phase2(config: Dict, checkpoint_path: Optional[str] = None):
    """
    Train Phase 2: Agent Finetuning.
    
    1. Load pretrained tokenizer and dynamics from Phase 1
    2. Freeze transformer weights
    3. Train policy and reward heads with behavior cloning
    """
    # Setup
    device = torch.device(config["experiment"]["device"])
    set_seed(config["experiment"]["seed"])
    
    # Enable optimizations for faster training
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True  # Faster convolutions
        torch.backends.cudnn.deterministic = False  # Allow non-deterministic for speed
        print_flush(f"CUDA optimizations enabled: cudnn.benchmark=True")
    
    # Create directories
    log_dir = Path(config["experiment"]["log_dir"]) / f"{config['experiment']['name']}_phase2"
    ckpt_dir = Path(config["experiment"]["checkpoint_dir"]) / config["experiment"]["name"]
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    # Create models
    print_flush("Creating models...")
    tokenizer = create_tokenizer(config).to(device)
    dynamics = create_dynamics_model(config).to(device)
    heads = {k: v.to(device) for k, v in create_heads(config).items()}
    
    # Load Phase 1 checkpoint BEFORE compiling (compilation changes state_dict keys)
    if checkpoint_path:
        print_flush(f"Loading Phase 1 checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        tokenizer.load_state_dict(checkpoint["tokenizer_state_dict"])
        dynamics.load_state_dict(checkpoint["dynamics_state_dict"])
    else:
        # Try to find latest Phase 1 checkpoint
        phase1_ckpt = ckpt_dir / "phase1_final.pt"
        if phase1_ckpt.exists():
            print_flush(f"Loading Phase 1 checkpoint from {phase1_ckpt}")
            checkpoint = torch.load(phase1_ckpt, map_location=device, weights_only=False)
            tokenizer.load_state_dict(checkpoint["tokenizer_state_dict"])
            dynamics.load_state_dict(checkpoint["dynamics_state_dict"])
        else:
            print_flush("Warning: No Phase 1 checkpoint found, training from scratch")
    
    # Compile tokenizer for faster inference (AFTER loading checkpoint)
    # This can speed up frozen tokenizer by 20-30%
    try:
        if hasattr(torch, 'compile') and device.type == "cuda":
            print_flush("Compiling tokenizer for faster inference...")
            tokenizer = torch.compile(tokenizer, mode="reduce-overhead")
            print_flush("✓ Tokenizer compiled successfully")
    except Exception as e:
        print_flush(f"Warning: Could not compile tokenizer: {e} (continuing without compilation)")
    
    # Freeze transformers if specified
    if config["training"]["phase2"]["freeze_transformer"]:
        print_flush("Freezing tokenizer and dynamics transformers...")
        freeze_module(tokenizer)
        freeze_module(dynamics)
    
    # Count parameters
    head_params = sum(count_parameters(h) for h in heads.values())
    print_flush(f"Trainable head parameters: {head_params:,}")
    
    # Create data loader
    print_flush("Creating data loader...")
    # Check action space type for dataloader
    action_space_type = config["dynamics"].get("action_space_type", "categorical")
    use_multi_discrete = (action_space_type == "multi_discrete")
    
    train_loader = create_dataloader(
        data_path=config["data"]["path"],
        batch_size=config["data"]["batch_size"],
        sequence_length=config["data"]["sequence_length"],
        image_size=(config["data"]["image_height"], config["data"]["image_width"]),
        num_workers=config["data"]["num_workers"],
        split="train",
        use_multi_discrete=use_multi_discrete,
        max_episodes=config["data"].get("max_episodes", None),
    )
    print_flush(f"Data loader created: {len(train_loader)} batches, batch_size={config['data']['batch_size']}, num_workers={config['data']['num_workers']}")
    sys.stdout.flush()
    sys.stderr.flush()
    
    # Create loss function
    # MTP configuration (default: True to match paper Equation 9)
    use_mtp = config["training"]["phase2"].get("use_mtp", True)
    mtp_length = config["training"]["phase2"].get("mtp_length", 8)
    
    loss_fn = AgentFinetuningLoss(
        reward_weight=config["training"]["phase2"].get("reward_weight", 1.0),
        num_prediction_steps=mtp_length,
        use_focal_loss=config["training"]["phase2"].get("use_focal_loss", False),
        focal_gamma=config["training"]["phase2"].get("focal_gamma", 2.0),
        focal_alpha=config["training"]["phase2"].get("focal_alpha", 0.25),
        use_mtp=use_mtp,
    )
    
    # Create optimizer (only for heads)
    phase2_config = config["training"]["phase2"]
    
    head_params = []
    for head in heads.values():
        head_params.extend(head.parameters())
    
    optimizer = AdamW(
        head_params,
        lr=phase2_config["learning_rate"],
        weight_decay=phase2_config["weight_decay"],
        betas=tuple(config["optimizer"]["betas"]),
        eps=config["optimizer"]["eps"],
    )
    
    # Create scheduler
    total_steps = len(train_loader) * phase2_config["epochs"]
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=config["scheduler"]["min_lr"],
    )
    
    # Tensorboard
    writer = SummaryWriter(log_dir)
    print_flush(f"Tensorboard log directory: {log_dir}")
    sys.stdout.flush()
    sys.stderr.flush()
    
    # Training loop
    print_flush("Starting Phase 2 training...")
    max_steps = phase2_config.get("max_steps", None)
    if max_steps:
        print_flush(f"Max steps: {max_steps} (will stop early if reached)")
    print_flush(f"Logging every {config['logging']['log_every']} steps")
    print_flush(f"Total batches per epoch: {len(train_loader)}")
    print_flush(f"Total epochs: {phase2_config['epochs']}")
    print_flush("")
    sys.stdout.flush()
    sys.stderr.flush()
    
    global_step = 0
    
    tokenizer.eval()  # Keep in eval mode
    dynamics.eval()
    print_flush("Models set to eval mode (tokenizer, dynamics) and train mode (heads)")
    sys.stdout.flush()
    sys.stderr.flush()
    
    for epoch in range(phase2_config["epochs"]):
        print_flush(f"\n{'='*60}")
        print_flush(f"Starting Epoch {epoch+1}/{phase2_config['epochs']}")
        print_flush(f"{'='*60}")
        sys.stdout.flush()
        sys.stderr.flush()
        
        for head in heads.values():
            head.train()
        
        epoch_losses = []
        epoch_accuracies = []
        
        print_flush(f"Entering training loop for epoch {epoch+1}...")
        sys.stdout.flush()
        sys.stderr.flush()
        
        # Configure tqdm to flush output immediately
        pbar = tqdm(
            train_loader, 
            desc=f"Epoch {epoch+1}/{phase2_config['epochs']}",
            file=sys.stdout,  # Explicitly use stdout
            mininterval=0.5,  # Update more frequently (every 0.5 seconds)
            maxinterval=2.0,  # Force update every 2 seconds max
        )
        
        batch_idx = 0
        for batch in pbar:
            batch_idx += 1
            
            # Log first batch info
            if batch_idx == 1:
                print_flush(f"✓ First batch received! Shapes: frames={batch['frames'].shape}, actions={batch['actions'].shape}, rewards={batch['rewards'].shape}")
                sys.stdout.flush()
                sys.stderr.flush()
            
            loss_dict = train_agent_step(
                tokenizer=tokenizer,
                dynamics=dynamics,
                heads=heads,
                batch=batch,
                loss_fn=loss_fn,
                optimizer=optimizer,
                device=device,
                max_grad_norm=phase2_config["max_grad_norm"],
                action_type="multi_discrete" if use_multi_discrete else "discrete",
                debug=False,  # Disable debug to reduce overhead
            )
            
            epoch_losses.append(loss_dict["loss"])
            epoch_accuracies.append(loss_dict["bc_accuracy"])
            
            scheduler.step()
            
            # Update progress bar
            pbar.set_postfix({
                "loss": f"{loss_dict['loss']:.4f}",
                "acc": f"{loss_dict['bc_accuracy']:.2%}",
            })
            # Flush progress bar output
            pbar.refresh()
            sys.stdout.flush()
            sys.stderr.flush()
            
            # Save checkpoint periodically during training (for max_steps scenarios)
            save_every_steps = phase2_config.get("save_every_steps", None)
            if save_every_steps and global_step % save_every_steps == 0:
                checkpoint = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "tokenizer_state_dict": tokenizer.state_dict(),
                    "dynamics_state_dict": dynamics.state_dict(),
                    "policy_state_dict": heads["policy"].state_dict(),
                    "value_state_dict": heads["value"].state_dict(),
                    "reward_state_dict": heads["reward"].state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": config,
                }
                step_checkpoint_path = ckpt_dir / f"phase2_step_{global_step}.pt"
                torch.save(checkpoint, step_checkpoint_path)
                print_flush(f"Saved checkpoint at step {global_step} to {step_checkpoint_path}")
            
            # Log to tensorboard
            if global_step % config["logging"]["log_every"] == 0:
                writer.add_scalar("loss/total", loss_dict["loss"], global_step)
                writer.add_scalar("loss/bc", loss_dict["bc_loss"], global_step)
                writer.add_scalar("loss/reward", loss_dict["reward_loss"], global_step)
                writer.add_scalar("metrics/bc_accuracy", loss_dict["bc_accuracy"], global_step)
                writer.add_scalar("lr", scheduler.get_last_lr()[0], global_step)
                
                # Reward prediction monitoring (Issue 4: Reward Collapse Detection)
                if "nonzero_ratio" in loss_dict:
                    writer.add_scalar("rewards/nonzero_ratio", loss_dict["nonzero_ratio"], global_step)
                if "pred_std" in loss_dict:
                    writer.add_scalar("rewards/pred_std", loss_dict["pred_std"], global_step)
                    # Warning if reward head collapses
                    if loss_dict["pred_std"] < 0.01:
                        print_flush(f"WARNING [Step {global_step}]: Reward head may have collapsed (std={loss_dict['pred_std']:.4f})")
                
                # Flush after logging
                writer.flush()
                sys.stdout.flush()
                sys.stderr.flush()
            
            global_step += 1
            
            # Check if we've reached max_steps
            if max_steps and global_step >= max_steps:
                print_flush(f"\nReached max_steps ({max_steps}), stopping training early...")
                # Save final checkpoint before stopping
                final_checkpoint = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "tokenizer_state_dict": tokenizer.state_dict(),
                    "dynamics_state_dict": dynamics.state_dict(),
                    "policy_state_dict": heads["policy"].state_dict(),
                    "value_state_dict": heads["value"].state_dict(),
                    "reward_state_dict": heads["reward"].state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": config,
                }
                final_step_path = ckpt_dir / f"phase2_step_{global_step}_final.pt"
                torch.save(final_checkpoint, final_step_path)
                print_flush(f"Saved final checkpoint at step {global_step} to {final_step_path}")
                break
        
        # Check if we should break out of epoch loop
        if max_steps and global_step >= max_steps:
            break
        
        # Epoch summary
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        avg_acc = sum(epoch_accuracies) / len(epoch_accuracies)
        
        print_flush(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}, Accuracy = {avg_acc:.2%}")
        
        writer.add_scalar("epoch/loss", avg_loss, epoch)
        writer.add_scalar("epoch/accuracy", avg_acc, epoch)
        
        # Save checkpoint
        if (epoch + 1) % phase2_config["save_every"] == 0:
            checkpoint = {
                "epoch": epoch,
                "tokenizer_state_dict": tokenizer.state_dict(),
                "dynamics_state_dict": dynamics.state_dict(),
                "policy_state_dict": heads["policy"].state_dict(),
                "value_state_dict": heads["value"].state_dict(),
                "reward_state_dict": heads["reward"].state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": config,
            }
            torch.save(checkpoint, ckpt_dir / f"phase2_epoch_{epoch+1}.pt")
            print_flush(f"Saved checkpoint to {ckpt_dir / f'phase2_epoch_{epoch+1}.pt'}")
    
    # Save final checkpoint
    final_checkpoint = {
        "epoch": epoch if max_steps and global_step >= max_steps else phase2_config["epochs"],
        "global_step": global_step,
        "tokenizer_state_dict": tokenizer.state_dict(),
        "dynamics_state_dict": dynamics.state_dict(),
        "policy_state_dict": heads["policy"].state_dict(),
        "value_state_dict": heads["value"].state_dict(),
        "reward_state_dict": heads["reward"].state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": config,
    }
    torch.save(final_checkpoint, ckpt_dir / "phase2_final.pt")
    print_flush(f"Saved final checkpoint (step {global_step}) to {ckpt_dir / 'phase2_final.pt'}")
    
    writer.close()
    print_flush("Phase 2 training complete!")


def main():
    parser = argparse.ArgumentParser(description="DreamerV4 Phase 2 Training")
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
        help="Path to Phase 1 checkpoint",
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
    train_phase2(config, args.checkpoint)


if __name__ == "__main__":
    main()
