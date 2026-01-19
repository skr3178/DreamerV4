#!/usr/bin/env python3
"""
Test DreamerV4 Pipeline with a Single Sample

Quick test to verify the entire pipeline works:
1. Data loading
2. Tokenizer encoding
3. Dynamics model
4. Policy/Value/Reward heads
"""

import torch
import yaml
from pathlib import Path
import sys

from dreamer.data import MineRLDataset, create_dataloader
from dreamer.models import CausalTokenizer, DynamicsModel, PolicyHead, ValueHead, RewardHead
from dreamer.utils import set_seed


def load_config(config_path: str = "configs/minerl.yaml"):
    """Load configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def test_data_loading(config):
    """Test data loading."""
    print("=" * 70)
    print("TEST 1: Data Loading")
    print("=" * 70)
    
    dataset = MineRLDataset(
        data_path=config["data"]["path"],
        sequence_length=config["data"]["sequence_length"],
        image_size=(config["data"]["image_height"], config["data"]["image_width"]),
    )
    
    print(f"✓ Dataset loaded: {len(dataset)} samples, {len(dataset.episodes)} episodes")
    
    # Get a single sample
    sample = dataset[0]
    
    print(f"\nSample shapes:")
    print(f"  frames: {sample['frames'].shape} (T, C, H, W)")
    print(f"  actions: {sample['actions'].shape}")
    print(f"  rewards: {sample['rewards'].shape}")
    print(f"  frames dtype: {sample['frames'].dtype}, range: [{sample['frames'].min():.3f}, {sample['frames'].max():.3f}]")
    
    return sample, dataset


def test_tokenizer(config, sample, device):
    """Test tokenizer."""
    print("\n" + "=" * 70)
    print("TEST 2: Tokenizer")
    print("=" * 70)
    
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
    
    # Prepare input: (B, T, C, H, W) -> (B, C, T, H, W) for tokenizer
    frames = sample["frames"].unsqueeze(0).to(device)  # (1, T, C, H, W)
    frames = frames.permute(0, 2, 1, 3, 4)  # (1, C, T, H, W)
    
    print(f"Input frames shape: {frames.shape}")
    
    with torch.no_grad():
        output = tokenizer(frames, mask_ratio=0.75)
    
    print(f"✓ Tokenizer forward pass successful")
    print(f"  Latents shape: {output['latents'].shape} (B, T, num_latent, latent_dim)")
    print(f"  Reconstructed shape: {output['reconstructed'].shape}")
    
    return tokenizer, output


def test_dynamics(config, latents, actions, device):
    """Test dynamics model."""
    print("\n" + "=" * 70)
    print("TEST 3: Dynamics Model")
    print("=" * 70)
    
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
    
    # Prepare inputs
    latents = latents.to(device)  # (B, T, num_latent, latent_dim)
    # Actions should be (B, T) for discrete actions
    if actions.dim() == 1:
        actions = actions.unsqueeze(0).to(device)  # (1, T)
    elif actions.dim() == 2 and actions.shape[-1] == 1:
        # Squeeze last dimension if it's (T, 1)
        actions = actions.squeeze(-1).unsqueeze(0).to(device)  # (1, T)
    else:
        actions = actions.to(device)
    
    # Ensure actions are long/int for discrete actions
    actions = actions.long()
    
    # Sample shortcut parameters
    batch_size = latents.shape[0]
    signal_level, step_size, d_is_min = dynamics.sample_shortcut_params(batch_size, device)
    
    print(f"Input latents shape: {latents.shape}")
    print(f"Input actions shape: {actions.shape}")
    print(f"Signal level: {signal_level.mean():.3f} (τ)")
    print(f"Step size: {step_size.mean():.3f} (d)")
    
    with torch.no_grad():
        # Add noise
        noisy_latents = dynamics.add_noise(latents, signal_level)
        
        # Forward pass
        output = dynamics(noisy_latents, actions, signal_level, step_size)
    
    print(f"✓ Dynamics forward pass successful")
    if isinstance(output, dict):
        print(f"  Output keys: {list(output.keys())}")
        if "predicted_latents" in output:
            print(f"  Predicted latents shape: {output['predicted_latents'].shape}")
        elif "latents" in output:
            print(f"  Latents shape: {output['latents'].shape}")
    else:
        print(f"  Predicted shape: {output.shape}")
    
    return dynamics


def test_heads(config, latents, device):
    """Test policy, value, and reward heads."""
    print("\n" + "=" * 70)
    print("TEST 4: Agent Heads")
    print("=" * 70)
    
    # Flatten latents for heads: (B, T, num_latent, latent_dim) -> (B*T, num_latent*latent_dim)
    B, T, num_latent, latent_dim = latents.shape
    flat_latents = latents.reshape(B * T, num_latent * latent_dim)
    
    input_dim = num_latent * latent_dim
    
    # Policy head
    policy_head = PolicyHead(
        input_dim=input_dim,
        hidden_dim=config["heads"]["hidden_dim"],
        num_layers=config["heads"]["num_layers"],
        num_discrete_actions=config["heads"].get("num_actions", 144),
    ).to(device)
    
    # Value head
    value_head = ValueHead(
        input_dim=input_dim,
        hidden_dim=config["heads"]["hidden_dim"],
        num_layers=config["heads"]["num_layers"],
        num_bins=config["heads"]["num_bins"],
    ).to(device)
    
    # Reward head
    reward_head = RewardHead(
        input_dim=input_dim,
        hidden_dim=config["heads"]["hidden_dim"],
        num_layers=config["heads"]["num_layers"],
        num_bins=config["heads"]["num_bins"],
    ).to(device)
    
    print(f"Input to heads: {flat_latents.shape}")
    
    with torch.no_grad():
        policy_out = policy_head(flat_latents)
        value_out = value_head(flat_latents)
        reward_out = reward_head(flat_latents)
    
    print(f"✓ Policy head output:")
    print(f"  Logits shape: {policy_out['logits'].shape}")
    print(f"  Probs shape: {policy_out['probs'].shape}")
    
    print(f"✓ Value head output:")
    print(f"  Logits shape: {value_out['logits'].shape}")
    print(f"  Value shape: {value_out['value'].shape}")
    
    print(f"✓ Reward head output:")
    print(f"  Logits shape: {reward_out['logits'].shape}")
    print(f"  Reward shape: {reward_out['reward'].shape}")
    
    return policy_head, value_head, reward_head


def main():
    print("=" * 70)
    print("DreamerV4 Pipeline Test")
    print("=" * 70)
    
    # Load config
    config_path = Path("configs/minerl.yaml")
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    
    config = load_config(config_path)
    print(f"\nConfig loaded from: {config_path}")
    
    # Set device
    device = torch.device(config["experiment"]["device"] if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Set seed
    set_seed(config["experiment"]["seed"])
    
    try:
        # Test 1: Data loading
        sample, dataset = test_data_loading(config)
        
        # Test 2: Tokenizer
        tokenizer, tokenizer_output = test_tokenizer(config, sample, device)
        latents = tokenizer_output["latents"]
        
        # Test 3: Dynamics
        actions = sample["actions"]
        dynamics = test_dynamics(config, latents, actions, device)
        
        # Test 4: Heads
        policy_head, value_head, reward_head = test_heads(config, latents, device)
        
        # Summary
        print("\n" + "=" * 70)
        print("✓ ALL TESTS PASSED!")
        print("=" * 70)
        print("\nPipeline Summary:")
        print(f"  Data: {sample['frames'].shape[0]} frames → Tokenizer → {latents.shape[2]} latent tokens")
        print(f"  Dynamics: {latents.shape} → {latents.shape} (predicted)")
        print(f"  Heads: {latents.shape} → Policy/Value/Reward predictions")
        print(f"\nReady for training!")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
