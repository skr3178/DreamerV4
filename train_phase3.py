"""
DreamerV4 Phase 3: Imagination Training with PMPO

This script implements the third phase of DreamerV4 training:
- Generates imagined trajectories from the frozen world model
- Trains policy and value heads using PMPO and TD(λ)
- No environment interaction - all learning happens in imagination

Key features:
- Frozen world model (tokenizer + dynamics)
- Trainable policy, value, and reward heads
- PMPO for policy optimization (Eq. 11)
- TD(λ) for value learning (Eq. 10)
"""

import os
import argparse
from typing import Dict, Optional
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

# Import dreamer modules
from dreamer.models.tokenizer import CausalTokenizer
from dreamer.models.dynamics import DynamicsModel
from dreamer.models.heads import PolicyHead, ValueHead, RewardHead
from dreamer.imagination.rollout import ImaginationRollout
from dreamer.losses.pmpo_loss import PMPOLoss
from dreamer.losses.value_loss import TDLambdaLoss
from dreamer.data.minerl_dataset import MineRLDataset

# Optional Cosmos tokenizer import
try:
    from dreamer.models.cosmos_tokenizer_wrapper import create_cosmos_tokenizer
    _cosmos_available = True
except ImportError:
    _cosmos_available = False
    create_cosmos_tokenizer = None


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_models(config: Dict, device: torch.device):
    """Create all model components."""

    cosmos_config = config.get("cosmos_tokenizer", {})
    use_cosmos = cosmos_config.get("enabled", False)

    # Tokenizer (frozen)
    if use_cosmos:
        if not _cosmos_available:
            raise ImportError("Cosmos tokenizer requested but not available.")
        tokenizer = create_cosmos_tokenizer(
            checkpoint_path=cosmos_config.get("checkpoint_path", "cosmos_tokenizer/CV8x8x8"),
            pool_tokens=cosmos_config.get("pool_tokens", 16),
            input_resolution=cosmos_config.get("input_resolution", 256),
            device=str(device),
            dtype=cosmos_config.get("dtype", "bfloat16"),
        )
    else:
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
            dropout=config["tokenizer"].get("dropout", 0.0),
            num_registers=config["tokenizer"].get("num_registers", 4),
            mask_ratio=config["tokenizer"].get("mask_ratio", 0.75),
        ).to(device)

    # Dynamics (frozen) - read arch from dynamics config, fall back to tokenizer
    dynamics_cfg = config.get("dynamics", {})
    tokenizer_cfg = config.get("tokenizer", {})

    dynamics = DynamicsModel(
        latent_dim=tokenizer_cfg["latent_dim"],
        num_latent_tokens=tokenizer_cfg["num_latent_tokens"],
        embed_dim=dynamics_cfg.get("embed_dim", tokenizer_cfg.get("embed_dim", 256)),
        depth=dynamics_cfg.get("num_layers", tokenizer_cfg.get("depth", 6)),
        num_heads=dynamics_cfg.get("num_heads", tokenizer_cfg.get("num_heads", 8)),
        dropout=tokenizer_cfg.get("dropout", 0.0),
        num_discrete_actions=dynamics_cfg["num_discrete_actions"],
        num_registers=dynamics_cfg.get("num_registers", tokenizer_cfg.get("num_registers", 4)),
        max_shortcut_steps=dynamics_cfg.get("max_shortcut_steps", 6),
    ).to(device)
    
    # Calculate input dimension for heads
    input_dim = config["tokenizer"]["num_latent_tokens"] * config["tokenizer"]["latent_dim"]
    
    # Policy head (trainable)
    policy_head = PolicyHead(
        input_dim=input_dim,
        hidden_dim=config["heads"]["hidden_dim"],
        num_discrete_actions=config["dynamics"]["num_discrete_actions"],
        num_layers=config["heads"]["num_layers"],
    ).to(device)
    
    # Value head (trainable)
    value_head = ValueHead(
        input_dim=input_dim,
        hidden_dim=config["heads"]["hidden_dim"],
        num_layers=config["heads"]["num_layers"],
    ).to(device)
    
    # Reward head (trainable)
    reward_head = RewardHead(
        input_dim=input_dim,
        hidden_dim=config["heads"]["hidden_dim"],
        num_layers=config["heads"]["num_layers"],
    ).to(device)
    
    return tokenizer, dynamics, policy_head, value_head, reward_head


def strip_compiled_prefix(state_dict: Dict) -> Dict:
    """Strip '_orig_mod.' prefix from keys if checkpoint was saved with torch.compile()."""
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("_orig_mod."):
            new_state_dict[key[len("_orig_mod."):]] = value
        else:
            new_state_dict[key] = value
    return new_state_dict


def load_phase2_checkpoint(
    tokenizer: nn.Module,
    dynamics: nn.Module,
    policy_head: nn.Module,
    value_head: nn.Module,
    reward_head: nn.Module,
    checkpoint_path: str,
    device: torch.device,
    use_cosmos: bool = False,
):
    """Load Phase 2 checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Strip _orig_mod. prefix if checkpoint was saved with torch.compile()
    # Skip tokenizer state_dict for Cosmos (pretrained, not in checkpoint)
    if not use_cosmos and "tokenizer_state_dict" in checkpoint:
        tokenizer.load_state_dict(strip_compiled_prefix(checkpoint["tokenizer_state_dict"]))
    dynamics.load_state_dict(strip_compiled_prefix(checkpoint["dynamics_state_dict"]))
    policy_head.load_state_dict(strip_compiled_prefix(checkpoint["policy_state_dict"]))
    value_head.load_state_dict(strip_compiled_prefix(checkpoint["value_state_dict"]))
    reward_head.load_state_dict(strip_compiled_prefix(checkpoint["reward_state_dict"]))

    epoch = checkpoint.get("epoch", "unknown")
    step = checkpoint.get("global_step", "unknown")
    print(f"Loaded Phase 2 checkpoint from {checkpoint_path} (epoch {epoch}, step {step})")


def freeze_world_model(tokenizer: nn.Module, dynamics: nn.Module, reward_head: nn.Module):
    """Freeze world model components and reward head.

    In Phase 3, we freeze:
    - Tokenizer: encodes observations to latent space
    - Dynamics: predicts next latent states
    - Reward head: trained in Phase 2, used for reward prediction in imagination

    Only policy and value heads are trained with PMPO and TD(λ).
    """
    for param in tokenizer.parameters():
        param.requires_grad = False
    for param in dynamics.parameters():
        param.requires_grad = False
    for param in reward_head.parameters():
        param.requires_grad = False

    tokenizer.eval()
    dynamics.eval()
    reward_head.eval()

    print("Frozen world model (tokenizer + dynamics) and reward head")


def train_phase3(
    config: Dict,
    tokenizer: nn.Module,
    dynamics: nn.Module,
    policy_head: nn.Module,
    value_head: nn.Module,
    reward_head: nn.Module,
    dataset: MineRLDataset,
    device: torch.device,
    checkpoint_dir: str,
    log_interval: int = 100,
):
    """
    Run Phase 3 training: Imagination RL with PMPO.
    
    Training loop:
    1. Sample initial states from real data
    2. Encode to latent space
    3. Generate imagined rollouts
    4. Compute advantages and returns
    5. Update policy with PMPO
    6. Update value with TD(λ)
    """
    
    # Create data loader
    dataloader = DataLoader(
        dataset,
        batch_size=config["phase3"]["batch_size"],
        shuffle=True,
        num_workers=config["phase3"].get("num_workers", config["data"].get("num_workers", 4)),
        pin_memory=True,
    )
    
    # Create imagination rollout generator
    imagination = ImaginationRollout(
        dynamics_model=dynamics,
        policy_head=policy_head,
        value_head=value_head,
        reward_head=reward_head,
        horizon=config["phase3"]["imagination_horizon"],
        num_denoising_steps=config["phase3"]["num_denoising_steps"],
        discount=config["phase3"]["discount"],
        discrete_actions=True,
    ).to(device)
    
    # Create loss functions
    pmpo_loss_fn = PMPOLoss(
        alpha=config["phase3"]["pmpo_alpha"],
        beta_kl=config["phase3"]["pmpo_beta_kl"],
        entropy_coef=config["phase3"]["entropy_coef"],
        num_bins=config["phase3"]["advantage_bins"],
        discrete_actions=True,
        use_percentile_binning=config["phase3"].get("use_percentile_binning", False),
        percentile_threshold=config["phase3"].get("percentile_threshold", 10.0),
    ).to(device)
    
    value_loss_fn = TDLambdaLoss(
        discount=config["phase3"]["discount"],
        lambda_=config["phase3"]["lambda"],
        loss_scale=config["phase3"]["value_loss_scale"],
        use_distributional=True,
    ).to(device)
    
    # Calculate input dimension for heads (same as in create_models)
    input_dim = config["tokenizer"]["num_latent_tokens"] * config["tokenizer"]["latent_dim"]

    # EMA target network for value learning (Section 4.4)
    value_target_head = ValueHead(
        input_dim=input_dim,
        hidden_dim=config["heads"]["hidden_dim"],
        num_layers=config["heads"]["num_layers"],
        num_bins=config["heads"].get("num_bins", 255),
    ).to(device)
    
    # Initialize target network with main network weights
    value_target_head.load_state_dict(value_head.state_dict())
    value_target_head.eval()  # Target network is always in eval mode
    
    # EMA decay rate (default 0.999 per common practice, configurable)
    value_ema_decay = config["phase3"].get("value_ema_decay", 0.999)
    
    # Optimizers (only for trainable heads - policy and value)
    weight_decay = config["phase3"].get("weight_decay", 0.01)

    policy_optimizer = optim.AdamW(
        policy_head.parameters(),
        lr=config["phase3"]["policy_lr"],
        weight_decay=weight_decay,
    )

    value_optimizer = optim.AdamW(
        value_head.parameters(),
        lr=config["phase3"]["value_lr"],
        weight_decay=weight_decay,
    )

    # Note: reward_head is frozen (trained in Phase 2), no optimizer needed

    # Training state
    global_step = 0
    best_mean_return = float("-inf")
    
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    print("Starting Phase 3 training...")
    print(f"  Imagination horizon: {config['phase3']['imagination_horizon']}")
    print(f"  Batch size: {config['phase3']['batch_size']}")
    print(f"  Policy LR: {config['phase3']['policy_lr']}")
    print(f"  Value LR: {config['phase3']['value_lr']}")
    
    # Check for max_steps limit
    max_steps = config["phase3"].get("max_steps", None)
    if max_steps:
        print(f"  Max steps: {max_steps} (will stop early if reached)")
    
    # Initialize avg_return to avoid UnboundLocalError
    avg_return = 0.0
    
    for epoch in range(config["phase3"]["num_epochs"]):
        epoch_policy_loss = 0.0
        epoch_value_loss = 0.0
        epoch_entropy = 0.0
        epoch_mean_return = 0.0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}")
        
        for batch_idx, batch in enumerate(pbar):
            # Get initial frames from real data
            frames = batch["frames"].to(device)  # (B, T, C, H, W)

            # Encode first frame to get initial latent state
            with torch.no_grad():
                # Use first frame as starting point
                first_frame = frames[:, 0]  # (B, C, H, W) - first timestep
                first_frame = first_frame.unsqueeze(2)  # (B, C, 1, H, W)

                # Encode to latent
                tokenizer_out = tokenizer(first_frame, mask_ratio=0.0)
                initial_latents = tokenizer_out["latents"][:, 0]  # (B, num_latent, latent_dim)
            
            # Generate imagined rollouts
            rollout_data = imagination(
                initial_latents=initial_latents,
                lambda_=config["phase3"]["lambda"],
                normalize_advantages=True,
            )
            
            # Flatten latents for head input
            latents = rollout_data["latents"]
            batch_size, horizon = latents.shape[:2]
            flat_latents = latents.reshape(batch_size, horizon, -1)
            
            # ========== Policy Update (PMPO) ==========
            policy_optimizer.zero_grad()
            
            policy_result = pmpo_loss_fn(
                policy_head=policy_head,
                latents=flat_latents,
                actions=rollout_data["actions"],
                advantages=rollout_data["advantages"],
                prior_head=None,  # Could use Phase 2 policy as prior
            )
            
            policy_loss = policy_result["loss"]
            policy_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                policy_head.parameters(),
                config["phase3"]["grad_clip"],
            )
            
            policy_optimizer.step()
            
            # ========== Value Update (TD(λ)) ==========
            value_optimizer.zero_grad()
            
            # Use target network for bootstrap values (Section 4.4)
            # Get bootstrap latent from last timestep
            bootstrap_latent = flat_latents[:, -1] if flat_latents.shape[1] > 0 else None
            
            value_result = value_loss_fn(
                value_head=value_head,
                latents=flat_latents,
                rewards=rollout_data["rewards"],
                dones=rollout_data["dones"],
                bootstrap_latent=bootstrap_latent,
                target_head=value_target_head,  # Use EMA target for bootstrap
            )
            
            value_loss = value_result["loss"]
            value_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(
                value_head.parameters(),
                config["phase3"]["grad_clip"],
            )
            
            value_optimizer.step()
            
            # Update EMA target network (Section 4.4)
            with torch.no_grad():
                for target_param, main_param in zip(value_target_head.parameters(), value_head.parameters()):
                    target_param.data.mul_(value_ema_decay).add_(main_param.data, alpha=1.0 - value_ema_decay)
            
            # Track metrics
            epoch_policy_loss += policy_loss.item()
            epoch_value_loss += value_loss.item()
            epoch_entropy += policy_result["entropy"].item()
            epoch_mean_return += value_result["mean_return"].item()
            num_batches += 1
            global_step += 1
            
            # Update progress bar
            pbar.set_postfix({
                "p_loss": f"{policy_loss.item():.4f}",
                "v_loss": f"{value_loss.item():.4f}",
                "entropy": f"{policy_result['entropy'].item():.4f}",
                "return": f"{value_result['mean_return'].item():.2f}",
            })
            
            # Logging
            if global_step % log_interval == 0:
                print(f"\nStep {global_step}:")
                print(f"  Policy Loss: {policy_loss.item():.4f}")
                print(f"  Value Loss: {value_loss.item():.4f}")
                print(f"  Entropy: {policy_result['entropy'].item():.4f}")
                print(f"  Mean Return: {value_result['mean_return'].item():.2f}")
                print(f"  D+ samples: {policy_result['n_positive'].item():.0f}")
                print(f"  D- samples: {policy_result['n_negative'].item():.0f}")
                
                # Advantage statistics monitoring (Issue 3: Sparse Rewards Detection)
                if "adv_mean" in policy_result:
                    adv_mean = policy_result["adv_mean"].item()
                    adv_std = policy_result["adv_std"].item()
                    adv_min = policy_result["adv_min"].item()
                    adv_max = policy_result["adv_max"].item()
                    n_zero = policy_result.get("n_zero", torch.tensor(0.0)).item()
                    
                    print(f"  Advantage Stats: mean={adv_mean:.4f}, std={adv_std:.4f}, "
                          f"range=[{adv_min:.4f}, {adv_max:.4f}], zeros={n_zero:.0f}")
                    
                    # Warning if degenerate advantage distribution
                    if policy_result['n_positive'].item() < 10 and policy_result['n_negative'].item() < 10:
                        print(f"  WARNING: Degenerate advantage distribution - no learning signal!")
                        print(f"    Consider enabling percentile_binning or adding intrinsic rewards")
            
            # Check if we've reached max_steps (after processing batch)
            if max_steps and global_step >= max_steps:
                print(f"\nReached max_steps ({max_steps}), stopping training early.")
                # Calculate averages before breaking
                if num_batches > 0:
                    avg_policy_loss = epoch_policy_loss / num_batches
                    avg_value_loss = epoch_value_loss / num_batches
                    avg_entropy = epoch_entropy / num_batches
                    avg_return = epoch_mean_return / num_batches
                else:
                    avg_policy_loss = 0.0
                    avg_value_loss = 0.0
                    avg_entropy = 0.0
                    avg_return = 0.0
                # Break from inner loop
                break
        
        # Epoch summary (calculate if we didn't break early)
        if num_batches > 0:
            avg_policy_loss = epoch_policy_loss / num_batches
            avg_value_loss = epoch_value_loss / num_batches
            avg_entropy = epoch_entropy / num_batches
            avg_return = epoch_mean_return / num_batches
        else:
            # Fallback if no batches processed
            avg_policy_loss = 0.0
            avg_value_loss = 0.0
            avg_entropy = 0.0
            avg_return = 0.0
        
        # Break outer loop if max_steps reached
        if max_steps and global_step >= max_steps:
            break
        
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Avg Policy Loss: {avg_policy_loss:.4f}")
        print(f"  Avg Value Loss: {avg_value_loss:.4f}")
        print(f"  Avg Entropy: {avg_entropy:.4f}")
        print(f"  Avg Return: {avg_return:.2f}")
        
        # Save checkpoint
        if avg_return > best_mean_return:
            best_mean_return = avg_return
            save_checkpoint(
                policy_head, value_head, reward_head,
                epoch, global_step, avg_return,
                os.path.join(checkpoint_dir, "best_phase3.pt"),
            )
        
        # Regular checkpoint
        if (epoch + 1) % config["phase3"]["save_every"] == 0:
            save_checkpoint(
                policy_head, value_head, reward_head,
                epoch, global_step, avg_return,
                os.path.join(checkpoint_dir, f"phase3_epoch_{epoch + 1}.pt"),
            )
    
    # Final checkpoint
    save_checkpoint(
        policy_head, value_head, reward_head,
        epoch, global_step, avg_return,
        os.path.join(checkpoint_dir, "phase3_final.pt"),
    )
    
    print(f"\nPhase 3 training complete!")
    print(f"Best mean return: {best_mean_return:.2f}")


def save_checkpoint(
    policy_head: nn.Module,
    value_head: nn.Module,
    reward_head: nn.Module,
    epoch: int,
    global_step: int,
    mean_return: float,
    path: str,
):
    """Save training checkpoint."""
    torch.save({
        "policy_head": policy_head.state_dict(),
        "value_head": value_head.state_dict(),
        "reward_head": reward_head.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "mean_return": mean_return,
    }, path)
    print(f"Saved checkpoint to {path}")


def main():
    parser = argparse.ArgumentParser(description="DreamerV4 Phase 3: Imagination Training")
    parser.add_argument("--config", type=str, default="configs/minerl.yaml", help="Config file path")
    parser.add_argument("--phase2-checkpoint", type=str, required=True, help="Phase 2 checkpoint path")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/phase3", help="Checkpoint directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--log-interval", type=int, default=100, help="Logging interval")
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)

    # Map training.phase3 to phase3 for backward compatibility
    if "training" in config and "phase3" in config["training"]:
        config["phase3"] = config["training"]["phase3"]

    # Map epochs to num_epochs if present
    if "phase3" in config and "epochs" in config["phase3"] and "num_epochs" not in config["phase3"]:
        config["phase3"]["num_epochs"] = config["phase3"]["epochs"]

    # Add Phase 3 defaults if not in config
    if "phase3" not in config:
        config["phase3"] = {
            "batch_size": 16,
            "imagination_horizon": 15,
            "num_denoising_steps": 4,
            "discount": 0.997,
            "lambda": 0.95,
            "pmpo_alpha": 0.5,
            "pmpo_beta_kl": 0.1,
            "entropy_coef": 0.003,
            "advantage_bins": 16,
            "policy_lr": 3e-5,
            "value_lr": 1e-4,
            "value_loss_scale": 0.5,
            "grad_clip": 1.0,
            "num_epochs": 100,
            "save_every": 10,
        }
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Check if using Cosmos tokenizer
    cosmos_config = config.get("cosmos_tokenizer", {})
    use_cosmos = cosmos_config.get("enabled", False)

    # Create models
    tokenizer, dynamics, policy_head, value_head, reward_head = create_models(config, device)

    # Load Phase 2 checkpoint
    load_phase2_checkpoint(
        tokenizer, dynamics, policy_head, value_head, reward_head,
        args.phase2_checkpoint, device, use_cosmos=use_cosmos,
    )
    
    # Freeze world model and reward head
    freeze_world_model(tokenizer, dynamics, reward_head)
    
    # Create dataset
    dataset = MineRLDataset(
        data_path=config["data"]["path"],
        sequence_length=config["data"]["sequence_length"],
        image_size=(config["data"]["image_height"], config["data"]["image_width"]),
    )
    
    # Run training
    train_phase3(
        config=config,
        tokenizer=tokenizer,
        dynamics=dynamics,
        policy_head=policy_head,
        value_head=value_head,
        reward_head=reward_head,
        dataset=dataset,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        log_interval=args.log_interval,
    )


if __name__ == "__main__":
    main()
