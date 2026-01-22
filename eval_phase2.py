#!/usr/bin/env python3
"""Evaluate Phase 2 trained model (Agent Finetuning)."""

import torch
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys
from tqdm import tqdm
from typing import Dict

from dreamer.models import CausalTokenizer, DynamicsModel
from dreamer.models import PolicyHead, ValueHead, RewardHead
from dreamer.losses import AgentFinetuningLoss
from dreamer.data import create_dataloader
from dreamer.utils import set_seed


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


def create_heads(config: Dict) -> Dict[str, torch.nn.Module]:
    """Create agent heads from config."""
    input_dim = config["tokenizer"]["num_latent_tokens"] * config["tokenizer"]["latent_dim"]
    
    use_mtp = config["training"]["phase2"].get("use_mtp", True)
    mtp_length = config["training"]["phase2"].get("mtp_length", 8)
    
    heads = {
        "policy": PolicyHead(
            input_dim=input_dim,
            hidden_dim=config["heads"]["hidden_dim"],
            num_discrete_actions=config["dynamics"]["num_discrete_actions"],
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


def load_phase2_checkpoint(checkpoint_path: str, config: Dict, device: torch.device):
    """Load trained Phase 2 models."""
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Create models
    tokenizer = create_tokenizer(config).to(device)
    dynamics = create_dynamics_model(config).to(device)
    heads = {k: v.to(device) for k, v in create_heads(config).items()}
    
    # Helper function to strip _orig_mod prefix (from torch.compile)
    def strip_orig_mod(state_dict):
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("_orig_mod."):
                new_key = key[len("_orig_mod."):]
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        return new_state_dict
    
    # Load state dicts (handle compiled models)
    tokenizer_sd = strip_orig_mod(checkpoint["tokenizer_state_dict"])
    dynamics_sd = strip_orig_mod(checkpoint["dynamics_state_dict"])
    
    tokenizer.load_state_dict(tokenizer_sd, strict=False)
    dynamics.load_state_dict(dynamics_sd, strict=False)
    heads["policy"].load_state_dict(checkpoint["policy_state_dict"])
    heads["value"].load_state_dict(checkpoint["value_state_dict"])
    heads["reward"].load_state_dict(checkpoint["reward_state_dict"])
    
    # Set to eval mode
    tokenizer.eval()
    dynamics.eval()
    for head in heads.values():
        head.eval()
    
    print(f"✓ Checkpoint loaded successfully")
    if "global_step" in checkpoint:
        print(f"  Global step: {checkpoint['global_step']}")
    if "epoch" in checkpoint:
        print(f"  Epoch: {checkpoint['epoch']}")
    
    return tokenizer, dynamics, heads, checkpoint


@torch.inference_mode()  # Faster than no_grad for pure inference
def evaluate_phase2(
    tokenizer: CausalTokenizer,
    dynamics: DynamicsModel,
    heads: Dict[str, torch.nn.Module],
    dataloader,
    loss_fn: AgentFinetuningLoss,
    device: torch.device,
    num_batches: int = None,
    quick: bool = False,  # Quick mode: skip detailed predictions
) -> Dict:
    """Evaluate Phase 2 model on dataset (inference only, no training)."""
    print("\nEvaluating Phase 2 model (inference mode, no training)...")
    if quick:
        print("  Quick mode: Skipping detailed prediction collection")
    
    total_loss = 0.0
    total_bc_loss = 0.0
    total_reward_loss = 0.0
    total_bc_accuracy = 0.0
    total_reward_mse = 0.0
    
    all_predictions = []
    all_targets = []
    all_reward_preds = []
    all_reward_targets = []
    
    num_evaluated = 0
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
        if num_batches and batch_idx >= num_batches:
            break
        
        # Move data to device (non_blocking for faster transfer)
        frames = batch["frames"].to(device, non_blocking=True)
        actions = batch["actions"].to(device, non_blocking=True)
        rewards = batch["rewards"].to(device, non_blocking=True)
        
        # Reshape frames for tokenizer
        if frames.dim() == 5 and frames.shape[2] != tokenizer.in_channels:
            frames = frames.permute(0, 2, 1, 3, 4)
        
        # Get latents from tokenizer (frozen, no gradients needed)
        tokenizer_output = tokenizer.encode(frames, mask_ratio=0.0)
        latents = tokenizer_output["latents"]  # (B, T, num_latent, latent_dim)
        
        # Flatten latents for heads
        batch_size, time_steps, num_latent, latent_dim = latents.shape
        latents_flat = latents.reshape(batch_size, time_steps, -1)  # (B, T, num_latent * latent_dim)
        
        # Handle action format
        if actions.dim() == 3 and actions.shape[-1] == 1:
            actions = actions.squeeze(-1)
        
        # Compute loss (this is just for metrics, not training)
        loss_dict = loss_fn(
            policy_head=heads["policy"],
            reward_head=heads["reward"],
            latents=latents_flat,
            actions=actions,
            rewards=rewards,
            action_type="discrete",
        )
        
        # Accumulate metrics
        total_loss += loss_dict["loss"].item()
        total_bc_loss += loss_dict["bc_loss"].item()
        total_reward_loss += loss_dict["reward_loss"].item()
        total_bc_accuracy += loss_dict["bc_accuracy"]
        
        # Compute reward prediction MSE (only if not in quick mode)
        if not quick and "reward_predictions" in loss_dict and "reward_targets" in loss_dict:
            reward_preds = loss_dict["reward_predictions"]
            reward_targets = loss_dict["reward_targets"]
            reward_mse = ((reward_preds - reward_targets) ** 2).mean().item()
            total_reward_mse += reward_mse
            
            all_reward_preds.append(reward_preds.cpu().numpy())
            all_reward_targets.append(reward_targets.cpu().numpy())
        elif "reward_predictions" in loss_dict:
            # Still compute MSE but don't store arrays
            reward_preds = loss_dict["reward_predictions"]
            reward_targets = loss_dict["reward_targets"]
            reward_mse = ((reward_preds - reward_targets) ** 2).mean().item()
            total_reward_mse += reward_mse
        
        # Collect predictions for accuracy analysis (only if not in quick mode)
        if not quick and "policy_predictions" in loss_dict:
            preds = loss_dict["policy_predictions"].cpu().numpy()
            targets = actions.cpu().numpy()
            all_predictions.append(preds.flatten())
            all_targets.append(targets.flatten())
        
        num_evaluated += 1
    
    # Compute averages
    metrics = {
        "loss": total_loss / num_evaluated if num_evaluated > 0 else 0.0,
        "bc_loss": total_bc_loss / num_evaluated if num_evaluated > 0 else 0.0,
        "reward_loss": total_reward_loss / num_evaluated if num_evaluated > 0 else 0.0,
        "bc_accuracy": total_bc_accuracy / num_evaluated if num_evaluated > 0 else 0.0,
        "reward_mse": total_reward_mse / num_evaluated if num_evaluated > 0 else 0.0,
        "num_batches": num_evaluated,
    }
    
    # Compute action prediction statistics
    if all_predictions and all_targets:
        all_preds = np.concatenate(all_predictions)
        all_targs = np.concatenate(all_targets)
        metrics["action_accuracy"] = (all_preds == all_targs).mean()
        metrics["num_action_samples"] = len(all_preds)
    
    # Compute reward prediction statistics
    if all_reward_preds and all_reward_targets:
        all_reward_preds_flat = np.concatenate([r.flatten() for r in all_reward_preds])
        all_reward_targets_flat = np.concatenate([r.flatten() for r in all_reward_targets])
        metrics["reward_pred_mean"] = all_reward_preds_flat.mean()
        metrics["reward_target_mean"] = all_reward_targets_flat.mean()
        metrics["reward_pred_std"] = all_reward_preds_flat.std()
        metrics["reward_target_std"] = all_reward_targets_flat.std()
        metrics["num_reward_samples"] = len(all_reward_preds_flat)
    
    return metrics


def print_metrics(metrics: Dict):
    """Print evaluation metrics."""
    print("\n" + "="*60)
    print("Phase 2 Evaluation Results")
    print("="*60)
    print(f"\nOverall Metrics:")
    print(f"  Total Loss:        {metrics['loss']:.6f}")
    print(f"  BC Loss:           {metrics['bc_loss']:.6f}")
    print(f"  Reward Loss:       {metrics['reward_loss']:.6f}")
    print(f"  BC Accuracy:       {metrics['bc_accuracy']:.2%}")
    
    if "action_accuracy" in metrics:
        print(f"  Action Accuracy:   {metrics['action_accuracy']:.2%} ({metrics['num_action_samples']} samples)")
    
    if "reward_mse" in metrics:
        print(f"\nReward Prediction:")
        print(f"  Reward MSE:        {metrics['reward_mse']:.6f}")
        if "reward_pred_mean" in metrics:
            print(f"  Predicted Mean:    {metrics['reward_pred_mean']:.6f} ± {metrics['reward_pred_std']:.6f}")
            print(f"  Target Mean:       {metrics['reward_target_mean']:.6f} ± {metrics['reward_target_std']:.6f}")
            print(f"  Samples:           {metrics['num_reward_samples']}")
    
    print(f"\nEvaluation Details:")
    print(f"  Batches Evaluated: {metrics['num_batches']}")
    print(f"  Note: This is inference-only evaluation (no training)")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Phase 2 trained model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/minerl_subset.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to Phase 2 checkpoint"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to evaluation data (overrides config)"
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=10,
        help="Number of batches to evaluate (default: 10 for quick eval)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (overrides config)"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick evaluation mode: skip detailed prediction collection"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for evaluation (overrides config, use smaller for faster eval)"
    )
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Setup device
    device = torch.device(args.device or config["experiment"]["device"])
    set_seed(config["experiment"]["seed"])
    
    print(f"Device: {device}")
    print(f"Config: {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
    
    # Load checkpoint
    tokenizer, dynamics, heads, checkpoint = load_phase2_checkpoint(
        args.checkpoint, config, device
    )
    
    # Create data loader
    data_path = args.data_path or config["data"]["path"]
    eval_batch_size = args.batch_size or min(config["data"]["batch_size"], 8)  # Smaller batch for faster eval
    print(f"\nCreating data loader from: {data_path}")
    print(f"  Evaluation batch size: {eval_batch_size} (config: {config['data']['batch_size']})")
    
    # Create dataset directly to avoid timeout issues with num_workers=0
    from dreamer.data.minerl_dataset import MineRLDataset
    from torch.utils.data import DataLoader
    
    eval_dataset = MineRLDataset(
        data_path=data_path,
        sequence_length=config["data"]["sequence_length"],
        image_size=(config["data"]["image_height"], config["data"]["image_width"]),
        split="train",
        max_episodes=config["data"].get("max_episodes", None),
    )
    
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=eval_batch_size,
        shuffle=False,  # No shuffle for evaluation
        num_workers=0,  # Single process for faster startup
        pin_memory=True,
        drop_last=False,  # Don't drop last batch for evaluation
    )
    print(f"  Total batches: {len(eval_loader)}")
    print(f"  Will evaluate: {args.num_batches} batches")
    
    # Create loss function
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
    
    # Evaluate
    metrics = evaluate_phase2(
        tokenizer=tokenizer,
        dynamics=dynamics,
        heads=heads,
        dataloader=eval_loader,
        loss_fn=loss_fn,
        device=device,
        num_batches=args.num_batches,
        quick=args.quick,
    )
    
    # Print results
    print_metrics(metrics)
    
    print("\n✓ Evaluation complete!")


if __name__ == "__main__":
    main()
