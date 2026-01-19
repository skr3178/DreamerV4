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


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_models(config: Dict, device: torch.device):
    """Create all model components."""
    
    # Tokenizer (frozen)
    tokenizer = CausalTokenizer(
        image_height=config["tokenizer"]["image_height"],
        image_width=config["tokenizer"]["image_width"],
        in_channels=config["tokenizer"]["in_channels"],
        patch_size=config["tokenizer"]["patch_size"],
        embed_dim=config["model"]["embed_dim"],
        latent_dim=config["model"]["latent_dim"],
        num_latent_tokens=config["tokenizer"]["num_latent_tokens"],
        depth=config["model"]["depth"],
        num_heads=config["model"]["num_heads"],
    ).to(device)
    
    # Dynamics (frozen)
    dynamics = DynamicsModel(
        latent_dim=config["model"]["latent_dim"],
        num_latent_tokens=config["tokenizer"]["num_latent_tokens"],
        embed_dim=config["model"]["embed_dim"],
        depth=config["model"]["depth"],
        num_heads=config["model"]["num_heads"],
        num_discrete_actions=config["dynamics"]["num_discrete_actions"],
    ).to(device)
    
    # Calculate input dimension for heads
    input_dim = config["tokenizer"]["num_latent_tokens"] * config["model"]["latent_dim"]
    
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


def load_phase2_checkpoint(
    tokenizer: nn.Module,
    dynamics: nn.Module,
    policy_head: nn.Module,
    value_head: nn.Module,
    reward_head: nn.Module,
    checkpoint_path: str,
    device: torch.device,
):
    """Load Phase 2 checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    tokenizer.load_state_dict(checkpoint["tokenizer"])
    dynamics.load_state_dict(checkpoint["dynamics"])
    policy_head.load_state_dict(checkpoint["policy_head"])
    value_head.load_state_dict(checkpoint["value_head"])
    reward_head.load_state_dict(checkpoint["reward_head"])
    
    print(f"Loaded Phase 2 checkpoint from {checkpoint_path}")


def freeze_world_model(tokenizer: nn.Module, dynamics: nn.Module):
    """Freeze world model components."""
    for param in tokenizer.parameters():
        param.requires_grad = False
    for param in dynamics.parameters():
        param.requires_grad = False
    
    tokenizer.eval()
    dynamics.eval()
    
    print("Frozen world model (tokenizer + dynamics)")


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
        num_workers=config["training"]["num_workers"],
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
    ).to(device)
    
    value_loss_fn = TDLambdaLoss(
        discount=config["phase3"]["discount"],
        lambda_=config["phase3"]["lambda"],
        loss_scale=config["phase3"]["value_loss_scale"],
        use_distributional=True,
    ).to(device)
    
    # Optimizers (only for trainable heads)
    policy_optimizer = optim.AdamW(
        policy_head.parameters(),
        lr=config["phase3"]["policy_lr"],
        weight_decay=config["training"]["weight_decay"],
    )
    
    value_optimizer = optim.AdamW(
        value_head.parameters(),
        lr=config["phase3"]["value_lr"],
        weight_decay=config["training"]["weight_decay"],
    )
    
    reward_optimizer = optim.AdamW(
        reward_head.parameters(),
        lr=config["phase3"]["value_lr"],
        weight_decay=config["training"]["weight_decay"],
    )
    
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
    
    for epoch in range(config["phase3"]["num_epochs"]):
        epoch_policy_loss = 0.0
        epoch_value_loss = 0.0
        epoch_entropy = 0.0
        epoch_mean_return = 0.0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}")
        
        for batch_idx, batch in enumerate(pbar):
            # Get initial frames from real data
            frames = batch["frames"].to(device)  # (B, C, T, H, W)
            
            # Encode first frame to get initial latent state
            with torch.no_grad():
                # Use first frame as starting point
                first_frame = frames[:, :, 0]  # (B, C, H, W)
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
            
            value_result = value_loss_fn(
                value_head=value_head,
                latents=flat_latents,
                rewards=rollout_data["rewards"],
                dones=rollout_data["dones"],
            )
            
            value_loss = value_result["loss"]
            value_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(
                value_head.parameters(),
                config["phase3"]["grad_clip"],
            )
            
            value_optimizer.step()
            
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
        
        # Epoch summary
        avg_policy_loss = epoch_policy_loss / num_batches
        avg_value_loss = epoch_value_loss / num_batches
        avg_entropy = epoch_entropy / num_batches
        avg_return = epoch_mean_return / num_batches
        
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
    
    # Create models
    tokenizer, dynamics, policy_head, value_head, reward_head = create_models(config, device)
    
    # Load Phase 2 checkpoint
    load_phase2_checkpoint(
        tokenizer, dynamics, policy_head, value_head, reward_head,
        args.phase2_checkpoint, device,
    )
    
    # Freeze world model
    freeze_world_model(tokenizer, dynamics)
    
    # Create dataset
    dataset = MineRLDataset(
        data_dir=config["dataset"]["data_dir"],
        sequence_length=config["dataset"]["sequence_length"],
        image_size=(config["tokenizer"]["image_height"], config["tokenizer"]["image_width"]),
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
