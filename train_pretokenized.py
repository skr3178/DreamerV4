#!/usr/bin/env python3
"""
Training with Pretokenized Cosmos Latents

Trains DreamerV4 phases using pre-encoded latents (skips on-the-fly tokenization).
2-3× faster than on-the-fly tokenization.

Usage:
    # Phase 1: Dynamics training
    python train_pretokenized.py --phase 1 --data-path data/pretokenized_subset --config configs/minerl_cosmos.yaml

    # Phase 2: Agent finetuning
    python train_pretokenized.py --phase 2 --data-path data/pretokenized_subset --checkpoint checkpoints/.../phase1_final.pt

    # Phase 3: Imagination RL
    python train_pretokenized.py --phase 3 --data-path data/pretokenized_subset --checkpoint checkpoints/.../phase2_final.pt

    # Quick test (few steps)
    python train_pretokenized.py --phase 1 --data-path data/pretokenized_subset --max-steps 100 --test
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import yaml

# Add project root to path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from dreamer.models import DynamicsModel, PolicyHead, ValueHead, RewardHead
from dreamer.losses import ShortcutForcingLoss, AgentFinetuningLoss
from dreamer.data.pretokenized_dataset import create_pretokenized_dataloader
from dreamer.utils import set_seed, count_parameters


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_dynamics_model(config: Dict, device: str) -> DynamicsModel:
    """Create dynamics model from config."""
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
    ).to(device)


def create_heads(config: Dict, device: str) -> Dict[str, nn.Module]:
    """Create agent heads from config."""
    input_dim = config["tokenizer"]["num_latent_tokens"] * config["tokenizer"]["latent_dim"]
    heads_config = config.get("heads", {})
    dynamics_config = config.get("dynamics", {})

    return {
        "policy": PolicyHead(
            input_dim=input_dim,
            hidden_dim=heads_config.get("hidden_dim", 256),
            num_discrete_actions=dynamics_config.get("num_discrete_actions", 144),
            num_layers=heads_config.get("num_layers", 2),
            use_mtp=False,  # Disable MTP for pretokenized (already at latent level)
        ).to(device),
        "value": ValueHead(
            input_dim=input_dim,
            hidden_dim=heads_config.get("hidden_dim", 256),
            num_layers=heads_config.get("num_layers", 2),
            num_bins=heads_config.get("num_bins", 255),
        ).to(device),
        "reward": RewardHead(
            input_dim=input_dim,
            hidden_dim=heads_config.get("hidden_dim", 256),
            num_layers=heads_config.get("num_layers", 2),
            num_bins=heads_config.get("num_bins", 255),
            use_mtp=False,
        ).to(device),
    }


def train_phase1_pretokenized(
    config: Dict,
    data_path: str,
    device: str = "cuda",
    max_steps: Optional[int] = None,
    checkpoint_path: Optional[str] = None,
    test_mode: bool = False,
):
    """
    Phase 1: Train dynamics model on pretokenized latents.

    No tokenizer needed - latents are pre-computed.
    """
    print("=" * 60)
    print("Phase 1: Dynamics Training (Pretokenized)")
    print("=" * 60)

    set_seed(config["experiment"]["seed"])

    # Create dynamics model
    print("\nCreating dynamics model...")
    dynamics = create_dynamics_model(config, device)
    print(f"  Parameters: {count_parameters(dynamics):,}")

    # Load checkpoint if provided
    if checkpoint_path and Path(checkpoint_path).exists():
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        dynamics.load_state_dict(checkpoint["dynamics_state_dict"])

    # Create dataloader
    print(f"\nLoading pretokenized data from {data_path}...")
    sequence_length = config["data"].get("sequence_length", 32)
    # Convert frame sequence length to latent sequence length
    # T_lat = 1 + ceil((T - 1) / 8), so for 32 frames -> 5 latent steps
    latent_seq_length = 1 + ((sequence_length - 1) // 8) + (1 if (sequence_length - 1) % 8 > 0 else 0)

    dataloader = create_pretokenized_dataloader(
        data_path=data_path,
        batch_size=config["data"].get("batch_size", 4),
        sequence_length=latent_seq_length,
        num_workers=config["data"].get("num_workers", 4),
    )
    print(f"  Batches: {len(dataloader)}")
    print(f"  Latent sequence length: {latent_seq_length}")

    # Create loss and optimizer
    loss_fn = ShortcutForcingLoss()
    phase1_config = config["training"]["phase1"]

    optimizer = AdamW(
        dynamics.parameters(),
        lr=phase1_config["learning_rate"],
        weight_decay=phase1_config["weight_decay"],
    )

    total_steps = max_steps or phase1_config.get("steps", len(dataloader) * 10)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)

    # Training loop
    print(f"\nStarting training (max_steps={total_steps})...")
    dynamics.train()
    global_step = 0

    while global_step < total_steps:
        pbar = tqdm(dataloader, desc=f"Step {global_step}/{total_steps}")

        for batch in pbar:
            if global_step >= total_steps:
                break

            # Get pretokenized latents directly (no tokenization!)
            latents = batch["latents"].to(device)  # (B, T_lat, 16, 16)
            actions = batch["actions"].to(device)  # (B, T_lat)

            # Handle action format
            if actions.dim() == 3 and actions.shape[-1] == 1:
                actions = actions.squeeze(-1)

            # Forward pass
            optimizer.zero_grad()

            output = dynamics(
                latents=latents,
                actions=actions,
                discrete_actions=True,
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
            loss.backward()

            torch.nn.utils.clip_grad_norm_(dynamics.parameters(), phase1_config["max_grad_norm"])
            optimizer.step()
            scheduler.step()

            # Update progress
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}",
            })

            global_step += 1

            if test_mode and global_step >= 10:
                print("\n[TEST MODE] Completed 10 steps successfully!")
                return dynamics

    print(f"\nPhase 1 complete! Final loss: {loss.item():.4f}")
    return dynamics


def train_phase2_pretokenized(
    config: Dict,
    data_path: str,
    device: str = "cuda",
    max_steps: Optional[int] = None,
    checkpoint_path: Optional[str] = None,
    test_mode: bool = False,
):
    """
    Phase 2: Train policy/reward heads on pretokenized latents.
    """
    print("=" * 60)
    print("Phase 2: Agent Finetuning (Pretokenized)")
    print("=" * 60)

    set_seed(config["experiment"]["seed"])

    # Create models
    print("\nCreating models...")
    dynamics = create_dynamics_model(config, device)
    heads = create_heads(config, device)

    # Load checkpoint
    if checkpoint_path and Path(checkpoint_path).exists():
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        dynamics.load_state_dict(checkpoint["dynamics_state_dict"])
        if "policy_state_dict" in checkpoint:
            heads["policy"].load_state_dict(checkpoint["policy_state_dict"])
        if "value_state_dict" in checkpoint:
            heads["value"].load_state_dict(checkpoint["value_state_dict"])
        if "reward_state_dict" in checkpoint:
            heads["reward"].load_state_dict(checkpoint["reward_state_dict"])

    # Freeze dynamics
    dynamics.eval()
    for param in dynamics.parameters():
        param.requires_grad = False
    print("  Dynamics: FROZEN")

    # Create dataloader
    print(f"\nLoading pretokenized data from {data_path}...")
    sequence_length = config["data"].get("sequence_length", 32)
    latent_seq_length = 1 + ((sequence_length - 1) // 8) + (1 if (sequence_length - 1) % 8 > 0 else 0)

    dataloader = create_pretokenized_dataloader(
        data_path=data_path,
        batch_size=config["data"].get("batch_size", 4),
        sequence_length=latent_seq_length,
        num_workers=config["data"].get("num_workers", 4),
    )

    # Create loss and optimizer
    loss_fn = AgentFinetuningLoss(use_mtp=False)
    phase2_config = config["training"]["phase2"]

    head_params = []
    for head in heads.values():
        head_params.extend(head.parameters())

    optimizer = AdamW(
        head_params,
        lr=phase2_config["learning_rate"],
        weight_decay=phase2_config["weight_decay"],
    )

    total_steps = max_steps or len(dataloader) * phase2_config.get("epochs", 10)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)

    # Training loop
    print(f"\nStarting training (max_steps={total_steps})...")
    for head in heads.values():
        head.train()

    global_step = 0

    while global_step < total_steps:
        pbar = tqdm(dataloader, desc=f"Step {global_step}/{total_steps}")

        for batch in pbar:
            if global_step >= total_steps:
                break

            latents = batch["latents"].to(device)
            actions = batch["actions"].to(device)
            rewards = batch["rewards"].to(device)

            if actions.dim() == 3 and actions.shape[-1] == 1:
                actions = actions.squeeze(-1)

            # Flatten latents for heads: (B, T, 16, 16) -> (B, T, 256)
            B, T = latents.shape[:2]
            latents_flat = latents.reshape(B, T, -1)

            optimizer.zero_grad()

            loss_dict = loss_fn(
                policy_head=heads["policy"],
                reward_head=heads["reward"],
                latents=latents_flat,
                actions=actions,
                rewards=rewards,
                action_type="discrete",
            )

            loss = loss_dict["loss"]
            loss.backward()

            torch.nn.utils.clip_grad_norm_(head_params, phase2_config["max_grad_norm"])
            optimizer.step()
            scheduler.step()

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{loss_dict['bc_accuracy']:.2%}",
            })

            global_step += 1

            if test_mode and global_step >= 10:
                print("\n[TEST MODE] Completed 10 steps successfully!")
                return dynamics, heads

    print(f"\nPhase 2 complete! Final loss: {loss.item():.4f}")
    return dynamics, heads


def main():
    parser = argparse.ArgumentParser(
        description="Train DreamerV4 with pretokenized Cosmos latents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--phase", type=int, required=True, choices=[1, 2, 3],
                        help="Training phase (1=dynamics, 2=agent, 3=imagination)")
    parser.add_argument("--data-path", type=str, required=True,
                        help="Path to pretokenized dataset")
    parser.add_argument("--config", type=str, default="configs/minerl_cosmos.yaml",
                        help="Path to config file")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Maximum training steps")
    parser.add_argument("--test", action="store_true",
                        help="Test mode: run only 10 steps")

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Override device
    config["experiment"]["device"] = args.device

    # Set max steps for test mode
    max_steps = args.max_steps
    if args.test:
        max_steps = 10

    print(f"\nPretokenized Training")
    print(f"  Phase: {args.phase}")
    print(f"  Data: {args.data_path}")
    print(f"  Config: {args.config}")
    print(f"  Device: {args.device}")
    print(f"  Max steps: {max_steps or 'unlimited'}")
    print(f"  Test mode: {args.test}")

    if args.phase == 1:
        train_phase1_pretokenized(
            config=config,
            data_path=args.data_path,
            device=args.device,
            max_steps=max_steps,
            checkpoint_path=args.checkpoint,
            test_mode=args.test,
        )
    elif args.phase == 2:
        train_phase2_pretokenized(
            config=config,
            data_path=args.data_path,
            device=args.device,
            max_steps=max_steps,
            checkpoint_path=args.checkpoint,
            test_mode=args.test,
        )
    elif args.phase == 3:
        print("Phase 3 with pretokenized data requires initial latents from real frames.")
        print("Use train_phase3.py with --phase2-checkpoint instead.")
        print("(Phase 3 imagination rollouts generate latents internally)")


if __name__ == "__main__":
    main()
