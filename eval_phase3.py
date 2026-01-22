#!/usr/bin/env python3
"""Evaluate Phase 3 trained model (Imagination RL with PMPO).

This script:
1. Loads Phase 2 checkpoint for tokenizer and dynamics (world model)
2. Loads Phase 3 checkpoint for trained heads (policy, value, reward)
3. Evaluates the combined model on real data
"""

import torch
import yaml
from pathlib import Path
import numpy as np
import argparse
import sys
from tqdm import tqdm
from typing import Dict

from dreamer.models import CausalTokenizer, DynamicsModel
from dreamer.models import PolicyHead, ValueHead, RewardHead
from dreamer.losses import AgentFinetuningLoss
from dreamer.data import create_dataloader
from dreamer.utils import set_seed, load_phase2_world_model as _load_phase2_world_model, load_phase3_heads as _load_phase3_heads
from eval_phase2 import load_config, create_tokenizer, create_dynamics_model, create_heads


def load_phase2_world_model(checkpoint_path: str, config: Dict, device: torch.device):
    """Load Phase 2 checkpoint for tokenizer and dynamics only (not heads)."""
    return _load_phase2_world_model(
        checkpoint_path, config, device,
        create_tokenizer_fn=create_tokenizer,
        create_dynamics_fn=create_dynamics_model,
    )


def load_phase3_heads(checkpoint_path: str, config: Dict, device: torch.device):
    """Load Phase 3 checkpoint for heads (policy, value, reward)."""
    return _load_phase3_heads(
        checkpoint_path, config, device,
        create_heads_fn=create_heads,
    )


@torch.inference_mode()
def evaluate_phase3(
    tokenizer: CausalTokenizer,
    heads: Dict[str, torch.nn.Module],
    dataloader,
    loss_fn: AgentFinetuningLoss,
    device: torch.device,
    num_batches: int = None,
    quick: bool = False,
) -> Dict:
    """Evaluate Phase 3 model on dataset (inference only, no training).

    Note: dynamics model is not needed for dataset evaluation since we use
    real observations, not imagined rollouts.
    """
    print("\nEvaluating Phase 3 model (inference mode, no training)...")
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
        
        # Move data to device
        frames = batch["frames"].to(device, non_blocking=True)
        actions = batch["actions"].to(device, non_blocking=True)
        rewards = batch["rewards"].to(device, non_blocking=True)
        
        # Reshape frames for tokenizer
        if frames.dim() == 5 and frames.shape[2] != tokenizer.in_channels:
            frames = frames.permute(0, 2, 1, 3, 4)
        
        # Get latents from tokenizer
        tokenizer_output = tokenizer.encode(frames, mask_ratio=0.0)
        latents = tokenizer_output["latents"]  # (B, T, num_latent, latent_dim)
        
        # Flatten latents for heads
        batch_size, time_steps, num_latent, latent_dim = latents.shape
        latents_flat = latents.reshape(batch_size, time_steps, -1)  # (B, T, num_latent * latent_dim)
        
        # Handle action format
        if actions.dim() == 3 and actions.shape[-1] == 1:
            actions = actions.squeeze(-1)
        
        # Compute loss (for metrics only, not training)
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
        
        # Compute reward prediction MSE
        if not quick and "reward_predictions" in loss_dict and "reward_targets" in loss_dict:
            reward_preds = loss_dict["reward_predictions"]
            reward_targets = loss_dict["reward_targets"]
            reward_mse = ((reward_preds - reward_targets) ** 2).mean().item()
            total_reward_mse += reward_mse
            
            all_reward_preds.append(reward_preds.cpu().numpy())
            all_reward_targets.append(reward_targets.cpu().numpy())
        elif "reward_predictions" in loss_dict:
            reward_preds = loss_dict["reward_predictions"]
            reward_targets = loss_dict["reward_targets"]
            reward_mse = ((reward_preds - reward_targets) ** 2).mean().item()
            total_reward_mse += reward_mse
        
        # Collect predictions for accuracy analysis
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
    print("Phase 3 Evaluation Results")
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
    parser = argparse.ArgumentParser(description="Evaluate Phase 3 trained model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/minerl_subset.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--phase2-checkpoint",
        type=str,
        required=True,
        help="Path to Phase 2 checkpoint (for tokenizer and dynamics)"
    )
    parser.add_argument(
        "--phase3-checkpoint",
        type=str,
        required=True,
        help="Path to Phase 3 checkpoint (for heads)"
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
    print(f"Phase 2 checkpoint: {args.phase2_checkpoint}")
    print(f"Phase 3 checkpoint: {args.phase3_checkpoint}")
    
    # Load Phase 2 checkpoint (world model)
    tokenizer, dynamics, phase2_checkpoint = load_phase2_world_model(
        args.phase2_checkpoint, config, device
    )
    
    # Load Phase 3 checkpoint (heads)
    heads, phase3_checkpoint = load_phase3_heads(
        args.phase3_checkpoint, config, device
    )
    
    # Create data loader
    data_path = args.data_path or config["data"]["path"]
    eval_batch_size = args.batch_size or min(config["data"]["batch_size"], 8)
    print(f"\nCreating data loader from: {data_path}")
    print(f"  Evaluation batch size: {eval_batch_size} (config: {config['data']['batch_size']})")
    
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
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
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
    
    # Evaluate (dynamics not needed for dataset evaluation)
    metrics = evaluate_phase3(
        tokenizer=tokenizer,
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
