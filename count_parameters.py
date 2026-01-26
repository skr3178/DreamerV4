#!/usr/bin/env python3
"""
Count total parameters for DreamerV4 MineRL model configuration.
"""

import torch
import yaml
from pathlib import Path
from dreamer.models import CausalTokenizer, DynamicsModel, PolicyHead, ValueHead, RewardHead
from dreamer.utils import count_parameters

def main():
    # Load MineRL config
    config_path = Path("configs/minerl.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device("cpu")  # Use CPU for counting
    
    print("Creating models for parameter counting...")
    print("=" * 80)
    
    # Create tokenizer
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
        dropout=config["tokenizer"]["dropout"],
        num_registers=config["tokenizer"]["num_registers"],
        mask_ratio=config["tokenizer"]["mask_ratio"],
    ).to(device)
    
    tokenizer_params = count_parameters(tokenizer, trainable_only=False)
    print(f"Tokenizer parameters: {tokenizer_params:,} ({tokenizer_params/1e6:.2f}M)")
    
    # Create dynamics
    dynamics = DynamicsModel(
        latent_dim=config["tokenizer"]["latent_dim"],
        num_latent_tokens=config["tokenizer"]["num_latent_tokens"],
        embed_dim=config["tokenizer"]["embed_dim"],
        depth=config["tokenizer"]["depth"],
        num_heads=config["tokenizer"]["num_heads"],
        dropout=config["tokenizer"]["dropout"],
        num_discrete_actions=config["dynamics"]["num_discrete_actions"],
        num_registers=config["dynamics"]["num_registers"],
        max_shortcut_steps=config["dynamics"]["max_shortcut_steps"],
    ).to(device)
    
    dynamics_params = count_parameters(dynamics, trainable_only=False)
    print(f"Dynamics parameters: {dynamics_params:,} ({dynamics_params/1e6:.2f}M)")
    
    # Calculate input dimension for heads
    input_dim = config["tokenizer"]["num_latent_tokens"] * config["tokenizer"]["latent_dim"]
    
    # Create policy head
    policy_head = PolicyHead(
        input_dim=input_dim,
        hidden_dim=config["heads"]["hidden_dim"],
        num_discrete_actions=config["heads"]["num_actions"],
        num_layers=config["heads"]["num_layers"],
    ).to(device)
    
    policy_params = count_parameters(policy_head, trainable_only=False)
    print(f"Policy head parameters: {policy_params:,} ({policy_params/1e6:.2f}M)")
    
    # Create value head
    value_head = ValueHead(
        input_dim=input_dim,
        hidden_dim=config["heads"]["hidden_dim"],
        num_bins=config["heads"]["num_bins"],
        num_layers=config["heads"]["num_layers"],
    ).to(device)
    
    value_params = count_parameters(value_head, trainable_only=False)
    print(f"Value head parameters: {value_params:,} ({value_params/1e6:.2f}M)")
    
    # Create reward head
    reward_head = RewardHead(
        input_dim=input_dim,
        hidden_dim=config["heads"]["hidden_dim"],
        num_bins=config["heads"]["num_bins"],
        num_layers=config["heads"]["num_layers"],
    ).to(device)
    
    reward_params = count_parameters(reward_head, trainable_only=False)
    print(f"Reward head parameters: {reward_params:,} ({reward_params/1e6:.2f}M)")
    
    # Total
    total_params = tokenizer_params + dynamics_params + policy_params + value_params + reward_params
    print("=" * 80)
    print(f"TOTAL MODEL PARAMETERS: {total_params:,} ({total_params/1e6:.2f}M)")
    print("=" * 80)
    
    # Breakdown
    print("\nBreakdown:")
    print(f"  World Model (Tokenizer + Dynamics): {(tokenizer_params + dynamics_params)/1e6:.2f}M")
    print(f"  Agent Heads (Policy + Value + Reward): {(policy_params + value_params + reward_params)/1e6:.2f}M")

if __name__ == '__main__':
    main()
