#!/usr/bin/env python3
"""
Comprehensive DreamerV4 Pipeline Test

Tests the entire DreamerV4 pipeline end-to-end:
1. Data loading and preprocessing
2. Model creation (tokenizer, dynamics, heads)
3. Phase 1: World model pretraining (tokenizer + dynamics)
4. Phase 2: Agent finetuning (policy + reward heads)
5. Phase 3: Imagination training (PMPO + TD(λ))
6. Checkpoint saving/loading

Usage:
    python test_pipeline.py [--config configs/minerl_small_batch.yaml] [--device cuda]
"""

import torch
import yaml
from pathlib import Path
import sys
import argparse
from typing import Dict, Optional

# Import dreamer modules
from dreamer.data import MineRLDataset, create_dataloader
from dreamer.models import CausalTokenizer, DynamicsModel, PolicyHead, ValueHead, RewardHead
from dreamer.losses import TokenizerLoss, ShortcutForcingLoss, AgentFinetuningLoss, PMPOLoss, TDLambdaLoss
from dreamer.imagination import ImaginationRollout
from dreamer.utils import set_seed, count_parameters, freeze_module


# =============================================================================
# Synthetic Data Generators (for unit tests without real data)
# =============================================================================

def make_video(B=2, T=4, C=3, H=64, W=64, device='cpu'):
    """Create synthetic video batch."""
    return torch.randn(B, C, T, H, W, device=device)


def make_latents(B=2, T=4, num_latent=16, latent_dim=32, device='cpu'):
    """Create synthetic latents in [-1, 1]."""
    return torch.randn(B, T, num_latent, latent_dim, device=device).tanh()


def make_actions(B=2, T=4, num_actions=144, device='cpu'):
    """Create synthetic discrete actions."""
    return torch.randint(0, num_actions, (B, T), device=device)


def make_rewards(B=2, T=4, device='cpu'):
    """Create synthetic rewards (mostly zeros, sparse)."""
    rewards = torch.zeros(B, T, device=device)
    # Add some sparse rewards
    rewards[:, -1] = torch.randn(B, device=device)
    return rewards


# =============================================================================
# Module Unit Tests
# =============================================================================

def test_embeddings(device: torch.device) -> bool:
    """Test embedding modules."""
    print("\n" + "=" * 80)
    print("TEST: Embeddings")
    print("=" * 80)

    try:
        from dreamer.models.embeddings import (
            PatchEmbedding, ActionEmbedding, SignalEmbedding,
            LatentTokenEmbedding, RegisterTokens
        )

        B, C, H, W = 2, 3, 64, 64
        embed_dim = 256
        patch_size = 8

        # PatchEmbedding: patchify/unpatchify round-trip
        patch_embed = PatchEmbedding(H, W, patch_size, C, embed_dim).to(device)
        images = torch.randn(B, C, H, W, device=device)
        patches = patch_embed.patchify(images)  # (B, 64, 192)
        assert patches.shape == (B, 64, patch_size * patch_size * C), \
            f"Expected (B, 64, {patch_size*patch_size*C}), got {patches.shape}"
        reconstructed = patch_embed.unpatchify(patches)  # (B, C, H, W)
        assert reconstructed.shape == images.shape, \
            f"Expected {images.shape}, got {reconstructed.shape}"
        print(f"  PatchEmbedding: patchify {images.shape} -> {patches.shape}, unpatchify -> {reconstructed.shape}")

        # ActionEmbedding: discrete actions
        action_embed = ActionEmbedding(embed_dim, num_discrete_actions=144).to(device)
        actions = torch.randint(0, 144, (B, 4), device=device)
        action_emb = action_embed(discrete_actions=actions)  # (B, T, embed_dim)
        assert action_emb.shape == (B, 4, embed_dim), \
            f"Expected (B, 4, {embed_dim}), got {action_emb.shape}"
        print(f"  ActionEmbedding: actions {actions.shape} -> embeddings {action_emb.shape}")

        # SignalEmbedding: tau and d
        signal_embed = SignalEmbedding(embed_dim).to(device)
        signal_level = torch.rand(B, device=device)
        step_size = torch.randint(1, 7, (B,), device=device).float()
        signal_emb = signal_embed(signal_level, step_size)  # (B, embed_dim)
        assert signal_emb.shape == (B, embed_dim), \
            f"Expected (B, {embed_dim}), got {signal_emb.shape}"
        print(f"  SignalEmbedding: tau {signal_level.shape}, d {step_size.shape} -> {signal_emb.shape}")

        # LatentTokenEmbedding: bottleneck in [-1, 1]
        latent_embed = LatentTokenEmbedding(16, embed_dim, 32).to(device)
        x = torch.randn(B, 16, embed_dim, device=device)
        bottleneck = latent_embed.to_bottleneck(x)  # (B, 16, 32)
        assert bottleneck.shape == (B, 16, 32), \
            f"Expected (B, 16, 32), got {bottleneck.shape}"
        assert bottleneck.min() >= -1 and bottleneck.max() <= 1, \
            f"Latents should be in [-1, 1], got [{bottleneck.min():.3f}, {bottleneck.max():.3f}]"
        print(f"  LatentTokenEmbedding: {x.shape} -> bottleneck {bottleneck.shape}, range [{bottleneck.min():.3f}, {bottleneck.max():.3f}]")

        # RegisterTokens
        register = RegisterTokens(4, embed_dim).to(device)
        reg_tokens = register(B)  # (B, 4, embed_dim)
        assert reg_tokens.shape == (B, 4, embed_dim), \
            f"Expected (B, 4, {embed_dim}), got {reg_tokens.shape}"
        print(f"  RegisterTokens: batch_size={B} -> {reg_tokens.shape}")

        print("Embeddings tests passed")
        return True

    except Exception as e:
        print(f"Embeddings tests failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_tokenizer_shapes(config: Dict, device: torch.device) -> bool:
    """Test tokenizer input/output shapes."""
    print("\n" + "=" * 80)
    print("TEST: Tokenizer Shapes")
    print("=" * 80)

    try:
        tokenizer = CausalTokenizer(
            image_height=64, image_width=64, in_channels=3,
            patch_size=8, embed_dim=256, latent_dim=32,
            num_latent_tokens=16, depth=2, num_heads=4,
            mask_ratio=0.75,
        ).to(device)

        B, C, T, H, W = 2, 3, 4, 64, 64
        video = make_video(B, T, C, H, W, device)

        with torch.no_grad():
            output = tokenizer(video, mask_ratio=0.75)

        # Shape checks
        latents = output['latents']
        assert latents.shape == (B, T, 16, 32), \
            f"Expected (B, T, 16, 32), got {latents.shape}"
        print(f"  Latents shape: {latents.shape} (expected: ({B}, {T}, 16, 32))")

        # Latent range check (tanh)
        assert latents.min() >= -1 and latents.max() <= 1, \
            f"Latents should be in [-1, 1], got [{latents.min():.3f}, {latents.max():.3f}]"
        print(f"  Latent range: [{latents.min():.3f}, {latents.max():.3f}] (expected: [-1, 1])")

        # Mask ratio check (~75% masked)
        mask = output['mask']
        mask_ratio_actual = mask.float().mean().item()
        assert 0.70 < mask_ratio_actual < 0.80, \
            f"Mask ratio {mask_ratio_actual:.3f} not ~0.75"
        print(f"  Mask ratio: {mask_ratio_actual:.3f} (expected: ~0.75)")

        # Check reconstructed patches shape
        reconstructed = output['reconstructed']
        print(f"  Reconstructed patches shape: {reconstructed.shape}")

        print("Tokenizer shape tests passed")
        return True

    except Exception as e:
        print(f"Tokenizer shape tests failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dynamics_shapes(config: Dict, device: torch.device) -> bool:
    """Test dynamics model shapes."""
    print("\n" + "=" * 80)
    print("TEST: Dynamics Shapes")
    print("=" * 80)

    try:
        dynamics = DynamicsModel(
            latent_dim=32, num_latent_tokens=16, embed_dim=256,
            depth=2, num_heads=4, num_discrete_actions=144,
            max_shortcut_steps=6,
        ).to(device)

        B, T = 2, 4
        latents = make_latents(B, T, 16, 32, device)
        actions = make_actions(B, T, 144, device)

        # Sample shortcut params
        signal_level, step_size, d_is_min = dynamics.sample_shortcut_params(B, device)
        print(f"  Sampled shortcut params:")
        print(f"    signal_level (tau): {signal_level}")
        print(f"    step_size (d): {step_size}")
        print(f"    d_is_min: {d_is_min}")

        # Check param ranges
        # tau is in [0, 1], d = 1/2^k where k in {0,...,max_shortcut_steps}
        # So d is in (0, 1] with values like 1, 0.5, 0.25, 0.125, ...
        assert (signal_level >= 0).all() and (signal_level <= 1).all(), \
            f"tau should be in [0,1], got [{signal_level.min():.3f}, {signal_level.max():.3f}]"
        assert (step_size > 0).all() and (step_size <= 1).all(), \
            f"d should be in (0,1], got [{step_size.min():.4f}, {step_size.max():.4f}]"
        print(f"  Param ranges valid: tau in [0,1], d in (0,1]")

        # Add noise
        noisy_latents = dynamics.add_noise(latents, signal_level)
        assert noisy_latents.shape == latents.shape, \
            f"Noisy latents shape mismatch: {noisy_latents.shape} vs {latents.shape}"
        print(f"  add_noise: {latents.shape} -> {noisy_latents.shape}")

        # Forward pass
        with torch.no_grad():
            output = dynamics(noisy_latents, actions, signal_level, step_size)

        predicted = output['predicted_latents']
        assert predicted.shape == (B, T, 16, 32), \
            f"Expected (B, T, 16, 32), got {predicted.shape}"
        print(f"  Forward pass: predicted_latents shape = {predicted.shape}")

        # Check target_latents
        if 'target_latents' in output:
            target = output['target_latents']
            assert target.shape == predicted.shape, \
                f"Target shape mismatch: {target.shape} vs {predicted.shape}"
            print(f"  target_latents shape = {target.shape}")

        print("Dynamics shape tests passed")
        return True

    except Exception as e:
        print(f"Dynamics shape tests failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_heads_shapes(config: Dict, device: torch.device) -> bool:
    """Test agent head outputs."""
    print("\n" + "=" * 80)
    print("TEST: Agent Heads Shapes")
    print("=" * 80)

    try:
        B = 8
        input_dim = 16 * 32  # 512

        # PolicyHead (use_mtp=False for simpler testing)
        policy = PolicyHead(input_dim, hidden_dim=256, num_discrete_actions=144, use_mtp=False).to(device)
        latents = torch.randn(B, input_dim, device=device)

        with torch.no_grad():
            policy_out = policy(latents)

        assert policy_out['logits'].shape == (B, 144), \
            f"Policy logits shape mismatch: expected ({B}, 144), got {policy_out['logits'].shape}"
        assert policy_out['probs'].shape == (B, 144), \
            f"Policy probs shape mismatch: expected ({B}, 144), got {policy_out['probs'].shape}"
        assert torch.allclose(policy_out['probs'].sum(dim=-1), torch.ones(B, device=device), atol=1e-5), \
            "Probs should sum to 1"
        print(f"  PolicyHead: latents {latents.shape} -> logits {policy_out['logits'].shape}, probs sum = {policy_out['probs'].sum(dim=-1).mean():.4f}")

        # ValueHead
        value = ValueHead(input_dim, hidden_dim=256, num_bins=255).to(device)

        with torch.no_grad():
            value_out = value(latents)

        assert value_out['logits'].shape == (B, 255), \
            f"Value logits shape mismatch: expected ({B}, 255), got {value_out['logits'].shape}"
        assert value_out['value'].shape == (B,), \
            f"Value shape mismatch: expected ({B},), got {value_out['value'].shape}"
        print(f"  ValueHead: latents {latents.shape} -> logits {value_out['logits'].shape}, value {value_out['value'].shape}")

        # RewardHead (use_mtp=False for simpler testing)
        reward = RewardHead(input_dim, hidden_dim=256, num_bins=255, use_mtp=False).to(device)

        with torch.no_grad():
            reward_out = reward(latents)

        assert reward_out['logits'].shape == (B, 255), \
            f"Reward logits shape mismatch: expected ({B}, 255), got {reward_out['logits'].shape}"
        assert reward_out['reward'].shape == (B,), \
            f"Reward shape mismatch: expected ({B},), got {reward_out['reward'].shape}"
        print(f"  RewardHead: latents {latents.shape} -> logits {reward_out['logits'].shape}, reward {reward_out['reward'].shape}")

        # Test target_to_bins for value head
        target_values = torch.randn(B, device=device) * 5  # Random values
        target_dist = value.target_to_bins(target_values)
        assert target_dist.shape == (B, 255), \
            f"target_to_bins shape mismatch: expected ({B}, 255), got {target_dist.shape}"
        assert torch.allclose(target_dist.sum(dim=-1), torch.ones(B, device=device), atol=1e-5), \
            "Target distribution should sum to 1"
        print(f"  target_to_bins: values {target_values.shape} -> distribution {target_dist.shape}, sum = {target_dist.sum(dim=-1).mean():.4f}")

        print("Heads shape tests passed")
        return True

    except Exception as e:
        print(f"Heads shape tests failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_losses(device: torch.device) -> bool:
    """Test loss computation."""
    print("\n" + "=" * 80)
    print("TEST: Loss Functions")
    print("=" * 80)

    try:
        from dreamer.losses import TokenizerLoss, ShortcutForcingLoss
        from dreamer.losses.value_loss import TDLambdaLoss

        B, T = 2, 4

        # TokenizerLoss (MSE only for unit test, LPIPS requires images)
        tokenizer_loss = TokenizerLoss(use_lpips=False).to(device)
        predicted = torch.randn(B * T, 64, 192, device=device)
        target = torch.randn(B * T, 64, 192, device=device)
        mask = torch.ones(B * T, 64, device=device).bool()

        loss_out = tokenizer_loss(predicted, target, mask)
        assert 'loss' in loss_out and loss_out['loss'].shape == (), \
            f"TokenizerLoss should return scalar loss"
        print(f"  TokenizerLoss: predicted {predicted.shape}, target {target.shape} -> loss = {loss_out['loss'].item():.4f}")

        # ShortcutForcingLoss: check ramp weight
        shortcut_loss = ShortcutForcingLoss().to(device)

        # ramp_weight(tau) = 0.9*tau + 0.1
        tau_0 = torch.tensor([0.0], device=device)
        tau_05 = torch.tensor([0.5], device=device)
        tau_1 = torch.tensor([1.0], device=device)

        w0 = shortcut_loss.ramp_weight(tau_0).item()
        w05 = shortcut_loss.ramp_weight(tau_05).item()
        w1 = shortcut_loss.ramp_weight(tau_1).item()

        assert abs(w0 - 0.1) < 0.01, f"ramp_weight(0) should be 0.1, got {w0}"
        assert abs(w05 - 0.55) < 0.01, f"ramp_weight(0.5) should be 0.55, got {w05}"
        assert abs(w1 - 1.0) < 0.01, f"ramp_weight(1) should be 1.0, got {w1}"
        print(f"  ShortcutForcingLoss ramp_weight: w(0)={w0:.2f}, w(0.5)={w05:.2f}, w(1)={w1:.2f}")

        # Full shortcut loss forward
        predicted_latents = torch.randn(B, T, 16, 32, device=device)
        target_latents = torch.randn(B, T, 16, 32, device=device)
        signal_level = torch.rand(B, device=device)
        step_size = torch.randint(1, 7, (B,), device=device).float()
        d_is_min = torch.zeros(B, device=device).bool()

        shortcut_out = shortcut_loss(predicted_latents, target_latents, signal_level, step_size, d_is_min)
        assert 'loss' in shortcut_out, "ShortcutForcingLoss should return loss"
        print(f"  ShortcutForcingLoss forward: loss = {shortcut_out['loss'].item():.4f}")

        # TDLambdaLoss: lambda returns computation
        td_loss = TDLambdaLoss(discount=0.99, lambda_=0.95).to(device)

        rewards = torch.randn(B, 5, device=device)
        values = torch.randn(B, 5, device=device)
        dones = torch.zeros(B, 5, device=device)
        bootstrap = torch.randn(B, device=device)

        returns = td_loss.compute_lambda_returns(rewards, values, dones, bootstrap)
        assert returns.shape == (B, 5), \
            f"Lambda returns shape mismatch: expected ({B}, 5), got {returns.shape}"
        print(f"  TDLambdaLoss.compute_lambda_returns: rewards {rewards.shape} -> returns {returns.shape}")

        print("Loss tests passed")
        return True

    except Exception as e:
        print(f"Loss tests failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_imagination(config: Dict, device: torch.device) -> bool:
    """Test imagination rollout."""
    print("\n" + "=" * 80)
    print("TEST: Imagination Rollout")
    print("=" * 80)

    try:
        B = 2
        input_dim = 16 * 32

        # Create modules
        dynamics = DynamicsModel(
            latent_dim=32, num_latent_tokens=16, embed_dim=256,
            depth=2, num_heads=4, num_discrete_actions=144,
        ).to(device)

        policy = PolicyHead(input_dim, hidden_dim=256, num_discrete_actions=144, use_mtp=False).to(device)
        value = ValueHead(input_dim, hidden_dim=256, num_bins=255).to(device)
        reward = RewardHead(input_dim, hidden_dim=256, num_bins=255, use_mtp=False).to(device)

        # Create rollout module
        imagination = ImaginationRollout(
            dynamics_model=dynamics,
            policy_head=policy,
            value_head=value,
            reward_head=reward,
            horizon=5,  # Short for testing
        ).to(device)

        # Test flatten_latents
        latents = torch.randn(B, 16, 32, device=device)
        flat = imagination.flatten_latents(latents)
        assert flat.shape == (B, 512), \
            f"Flattened shape mismatch: expected ({B}, 512), got {flat.shape}"
        print(f"  flatten_latents: {latents.shape} -> {flat.shape}")

        # Test generate_step
        with torch.no_grad():
            action = torch.randint(0, 144, (B,), device=device)
            next_latent = imagination.generate_step(latents, action)
        assert next_latent.shape == (B, 16, 32), \
            f"Next latent shape mismatch: expected ({B}, 16, 32), got {next_latent.shape}"
        print(f"  generate_step: latent {latents.shape} + action {action.shape} -> next_latent {next_latent.shape}")

        # Test full rollout
        initial_latents = torch.randn(B, 16, 32, device=device).tanh()
        with torch.no_grad():
            rollout_data = imagination(
                initial_latents=initial_latents,
                lambda_=0.95,
                normalize_advantages=True,
            )

        assert rollout_data['latents'].shape == (B, 5, 16, 32), \
            f"Rollout latents shape mismatch: {rollout_data['latents'].shape}"
        assert rollout_data['actions'].shape == (B, 5), \
            f"Rollout actions shape mismatch: {rollout_data['actions'].shape}"
        assert rollout_data['rewards'].shape == (B, 5), \
            f"Rollout rewards shape mismatch: {rollout_data['rewards'].shape}"
        assert rollout_data['advantages'].shape == (B, 5), \
            f"Rollout advantages shape mismatch: {rollout_data['advantages'].shape}"
        assert rollout_data['returns'].shape == (B, 5), \
            f"Rollout returns shape mismatch: {rollout_data['returns'].shape}"

        print(f"  Full rollout (horizon=5):")
        print(f"    latents: {rollout_data['latents'].shape}")
        print(f"    actions: {rollout_data['actions'].shape}")
        print(f"    rewards: {rollout_data['rewards'].shape}")
        print(f"    advantages: {rollout_data['advantages'].shape}")
        print(f"    returns: {rollout_data['returns'].shape}")

        print("Imagination tests passed")
        return True

    except Exception as e:
        print(f"Imagination tests failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# Configuration and Data Loading
# =============================================================================

def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def test_data_loading(config: Dict) -> tuple:
    """Test 1: Data loading and preprocessing."""
    print("\n" + "=" * 80)
    print("TEST 1: Data Loading")
    print("=" * 80)
    
    try:
        # Create dataset
        dataset = MineRLDataset(
            data_path=config["data"]["path"],
            sequence_length=config["data"]["sequence_length"],
            image_size=(config["data"]["image_height"], config["data"]["image_width"]),
        )
        
        print(f"✓ Dataset created: {len(dataset)} samples, {len(dataset.episodes)} episodes")
        
        # Get a sample
        sample = dataset[0]
        
        print(f"\nSample shapes:")
        print(f"  frames: {sample['frames'].shape} (T, C, H, W)")
        print(f"  actions: {sample['actions'].shape}")
        print(f"  rewards: {sample['rewards'].shape}")
        print(f"  frames dtype: {sample['frames'].dtype}, range: [{sample['frames'].min():.3f}, {sample['frames'].max():.3f}]")
        
        # Create dataloader
        dataloader = create_dataloader(
            data_path=config["data"]["path"],
            batch_size=min(4, config["data"]["batch_size"]),  # Small batch for testing
            sequence_length=config["data"]["sequence_length"],
            image_size=(config["data"]["image_height"], config["data"]["image_width"]),
            num_workers=0,  # Single worker for testing
            split="train",
            max_episodes=config["data"].get("max_episodes", 10),  # Limit for testing
        )
        
        batch = next(iter(dataloader))
        print(f"\nBatch shapes:")
        print(f"  frames: {batch['frames'].shape} (B, T, C, H, W)")
        print(f"  actions: {batch['actions'].shape}")
        print(f"  rewards: {batch['rewards'].shape}")
        
        return True, sample, batch, dataloader
        
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None, None


def test_model_creation(config: Dict, device: torch.device) -> tuple:
    """Test 2: Model creation."""
    print("\n" + "=" * 80)
    print("TEST 2: Model Creation")
    print("=" * 80)
    
    try:
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
        
        print(f"✓ Tokenizer created: {count_parameters(tokenizer):,} parameters")
        
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
        
        print(f"✓ Dynamics created: {count_parameters(dynamics):,} parameters")
        
        # Create heads
        input_dim = config["tokenizer"]["num_latent_tokens"] * config["tokenizer"]["latent_dim"]
        
        policy_head = PolicyHead(
            input_dim=input_dim,
            hidden_dim=config["heads"]["hidden_dim"],
            num_layers=config["heads"]["num_layers"],
            num_discrete_actions=config["dynamics"]["num_discrete_actions"],
        ).to(device)
        
        value_head = ValueHead(
            input_dim=input_dim,
            hidden_dim=config["heads"]["hidden_dim"],
            num_layers=config["heads"]["num_layers"],
            num_bins=config["heads"]["num_bins"],
        ).to(device)
        
        reward_head = RewardHead(
            input_dim=input_dim,
            hidden_dim=config["heads"]["hidden_dim"],
            num_layers=config["heads"]["num_layers"],
            num_bins=config["heads"]["num_bins"],
        ).to(device)
        
        print(f"✓ Policy head created: {count_parameters(policy_head):,} parameters")
        print(f"✓ Value head created: {count_parameters(value_head):,} parameters")
        print(f"✓ Reward head created: {count_parameters(reward_head):,} parameters")
        
        return True, tokenizer, dynamics, policy_head, value_head, reward_head
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None, None, None, None


def test_phase1_forward(config: Dict, tokenizer, dynamics, batch, device: torch.device) -> bool:
    """Test 3: Phase 1 forward passes (tokenizer + dynamics)."""
    print("\n" + "=" * 80)
    print("TEST 3: Phase 1 Forward Passes (World Model)")
    print("=" * 80)
    
    try:
        # Prepare data
        frames = batch["frames"].to(device)  # (B, T, C, H, W)
        actions = batch["actions"].to(device)
        
        # Reshape for tokenizer: (B, C, T, H, W)
        frames_tokenizer = frames.permute(0, 2, 1, 3, 4)
        
        print(f"Input frames shape: {frames_tokenizer.shape}")
        
        # Tokenizer forward pass
        tokenizer.eval()
        with torch.no_grad():
            tokenizer_output = tokenizer(frames_tokenizer, mask_ratio=0.75)
            latents = tokenizer_output["latents"]  # (B, T, num_latent, latent_dim)
        
        print(f"✓ Tokenizer forward pass successful")
        print(f"  Latents shape: {latents.shape}")
        print(f"  Reconstructed shape: {tokenizer_output['reconstructed'].shape}")
        
        # Dynamics forward pass
        dynamics.eval()
        with torch.no_grad():
            # Handle actions
            if actions.dim() == 3 and actions.shape[-1] == 1:
                actions = actions.squeeze(-1)
            actions = actions.long()
            
            # Sample shortcut parameters
            batch_size = latents.shape[0]
            signal_level, step_size, d_is_min = dynamics.sample_shortcut_params(batch_size, device)
            
            # Forward pass
            dynamics_output = dynamics(
                latents=latents,
                actions=actions,
                signal_level=signal_level,
                step_size=step_size,
            )
        
        print(f"✓ Dynamics forward pass successful")
        if isinstance(dynamics_output, dict):
            if "predicted_latents" in dynamics_output:
                print(f"  Predicted latents shape: {dynamics_output['predicted_latents'].shape}")
            elif "latents" in dynamics_output:
                print(f"  Latents shape: {dynamics_output['latents'].shape}")
        else:
            print(f"  Output shape: {dynamics_output.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Phase 1 forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_phase1_training_step(config: Dict, tokenizer, dynamics, batch, device: torch.device) -> bool:
    """Test 4: Phase 1 training step."""
    print("\n" + "=" * 80)
    print("TEST 4: Phase 1 Training Step")
    print("=" * 80)
    
    try:
        # Create optimizers
        tokenizer_optimizer = torch.optim.AdamW(tokenizer.parameters(), lr=1e-4)
        dynamics_optimizer = torch.optim.AdamW(dynamics.parameters(), lr=1e-4)
        
        # Create loss functions
        use_lpips = config["training"]["phase1"].get("use_lpips", False)
        tokenizer_loss_fn = TokenizerLoss(
            lpips_weight=config["training"]["phase1"].get("lpips_weight", 0.2),
            use_lpips=use_lpips,
        )
        if hasattr(tokenizer_loss_fn, 'lpips_fn') and tokenizer_loss_fn.lpips_fn is not None:
            tokenizer_loss_fn.lpips_fn = tokenizer_loss_fn.lpips_fn.to(device)
        
        dynamics_loss_fn = ShortcutForcingLoss()
        
        # Prepare data
        frames = batch["frames"].to(device)
        actions = batch["actions"].to(device)
        frames_tokenizer = frames.permute(0, 2, 1, 3, 4)
        
        # Tokenizer training step
        tokenizer.train()
        tokenizer_optimizer.zero_grad()
        
        tokenizer_output = tokenizer(frames_tokenizer, mask_ratio=0.75)
        tokenizer_loss_dict = tokenizer_loss_fn(
            predicted_patches=tokenizer_output["reconstructed"],
            target_patches=tokenizer_output["original_patches"],
            mask=tokenizer_output.get("mask"),
        )
        tokenizer_loss = tokenizer_loss_dict["loss"]
        tokenizer_loss.backward()
        tokenizer_optimizer.step()
        
        print(f"✓ Tokenizer training step successful")
        print(f"  Loss: {tokenizer_loss.item():.4f}")
        
        # Dynamics training step
        dynamics.train()
        dynamics_optimizer.zero_grad()
        
        with torch.no_grad():
            latents = tokenizer.encode(frames_tokenizer, mask_ratio=0.0)["latents"]
        
        if actions.dim() == 3 and actions.shape[-1] == 1:
            actions = actions.squeeze(-1)
        actions = actions.long()
        
        batch_size = latents.shape[0]
        signal_level, step_size, d_is_min = dynamics.sample_shortcut_params(batch_size, device)
        
        dynamics_output = dynamics(
            latents=latents,
            actions=actions,
            signal_level=signal_level,
            step_size=step_size,
        )
        
        dynamics_loss_dict = dynamics_loss_fn(
            predicted_latents=dynamics_output["predicted_latents"],
            target_latents=dynamics_output["target_latents"],
            signal_level=signal_level,
            step_size=step_size,
            d_is_min=d_is_min,
        )
        dynamics_loss = dynamics_loss_dict["loss"]
        dynamics_loss.backward()
        dynamics_optimizer.step()
        
        print(f"✓ Dynamics training step successful")
        print(f"  Loss: {dynamics_loss.item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Phase 1 training step failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_phase2_training_step(config: Dict, tokenizer, dynamics, policy_head, reward_head, batch, device: torch.device) -> bool:
    """Test 5: Phase 2 training step (agent finetuning)."""
    print("\n" + "=" * 80)
    print("TEST 5: Phase 2 Training Step (Agent Finetuning)")
    print("=" * 80)
    
    try:
        # Freeze tokenizer and dynamics
        freeze_module(tokenizer)
        freeze_module(dynamics)
        tokenizer.eval()
        dynamics.eval()
        
        # Create optimizer for heads
        head_params = list(policy_head.parameters()) + list(reward_head.parameters())
        optimizer = torch.optim.AdamW(head_params, lr=1e-4)
        
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
        
        # Prepare data
        frames = batch["frames"].to(device)
        actions = batch["actions"].to(device)
        rewards = batch["rewards"].to(device)
        frames_tokenizer = frames.permute(0, 2, 1, 3, 4)
        
        # Training step
        policy_head.train()
        reward_head.train()
        optimizer.zero_grad()
        
        # Get latents (frozen)
        with torch.no_grad():
            latents = tokenizer.encode(frames_tokenizer, mask_ratio=0.0)["latents"]
        
        # Flatten latents
        batch_size, time_steps, num_latent, latent_dim = latents.shape
        latents_flat = latents.reshape(batch_size, time_steps, -1)
        
        if actions.dim() == 3 and actions.shape[-1] == 1:
            actions = actions.squeeze(-1)
        
        # Compute loss
        loss_dict = loss_fn(
            policy_head=policy_head,
            reward_head=reward_head,
            latents=latents_flat,
            actions=actions,
            rewards=rewards,
            action_type="discrete",
        )
        
        loss = loss_dict["loss"]
        loss.backward()
        optimizer.step()
        
        print(f"✓ Phase 2 training step successful")
        print(f"  Total loss: {loss.item():.4f}")
        print(f"  BC loss: {loss_dict.get('bc_loss', 0.0):.4f}")
        print(f"  Reward loss: {loss_dict.get('reward_loss', 0.0):.4f}")
        print(f"  BC accuracy: {loss_dict.get('bc_accuracy', 0.0):.2%}")
        
        return True
        
    except Exception as e:
        print(f"❌ Phase 2 training step failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_phase3_imagination(config: Dict, tokenizer, dynamics, policy_head, value_head, reward_head, batch, device: torch.device) -> bool:
    """Test 6: Phase 3 imagination rollout."""
    print("\n" + "=" * 80)
    print("TEST 6: Phase 3 Imagination Rollout")
    print("=" * 80)
    
    try:
        # Freeze world model
        freeze_module(tokenizer)
        freeze_module(dynamics)
        tokenizer.eval()
        dynamics.eval()
        
        # Create imagination rollout
        imagination = ImaginationRollout(
            dynamics_model=dynamics,
            policy_head=policy_head,
            value_head=value_head,
            reward_head=reward_head,
            horizon=config["training"]["phase3"]["imagination_horizon"],
            num_denoising_steps=config["training"]["phase3"]["num_denoising_steps"],
            discount=config["training"]["phase3"]["discount"],
            discrete_actions=True,
        ).to(device)
        
        # Get initial latents from real data
        frames = batch["frames"].to(device)
        frames_tokenizer = frames.permute(0, 2, 1, 3, 4)
        
        with torch.no_grad():
            # Use first frame as starting point
            first_frame = frames_tokenizer[:, :, 0:1]  # (B, C, 1, H, W)
            tokenizer_output = tokenizer(first_frame, mask_ratio=0.0)
            initial_latents = tokenizer_output["latents"][:, 0]  # (B, num_latent, latent_dim)
        
        print(f"Initial latents shape: {initial_latents.shape}")
        
        # Generate imagined rollout
        with torch.no_grad():
            rollout_data = imagination(
                initial_latents=initial_latents,
                lambda_=config["training"]["phase3"]["lambda"],
                normalize_advantages=True,
            )
        
        print(f"✓ Imagination rollout successful")
        print(f"  Rollout latents shape: {rollout_data['latents'].shape}")
        print(f"  Rollout actions shape: {rollout_data['actions'].shape}")
        print(f"  Rollout rewards shape: {rollout_data['rewards'].shape}")
        print(f"  Advantages shape: {rollout_data['advantages'].shape}")
        print(f"  Returns shape: {rollout_data['returns'].shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Phase 3 imagination rollout failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_phase3_training_step(config: Dict, tokenizer, dynamics, policy_head, value_head, reward_head, batch, device: torch.device) -> bool:
    """Test 7: Phase 3 training step (PMPO + TD(λ))."""
    print("\n" + "=" * 80)
    print("TEST 7: Phase 3 Training Step (PMPO + TD(λ))")
    print("=" * 80)
    
    try:
        # Freeze world model
        freeze_module(tokenizer)
        freeze_module(dynamics)
        tokenizer.eval()
        dynamics.eval()
        
        # Create imagination rollout
        imagination = ImaginationRollout(
            dynamics_model=dynamics,
            policy_head=policy_head,
            value_head=value_head,
            reward_head=reward_head,
            horizon=config["training"]["phase3"]["imagination_horizon"],
            num_denoising_steps=config["training"]["phase3"]["num_denoising_steps"],
            discount=config["training"]["phase3"]["discount"],
            discrete_actions=True,
        ).to(device)
        
        # Create loss functions
        pmpo_loss_fn = PMPOLoss(
            alpha=config["training"]["phase3"]["pmpo_alpha"],
            beta_kl=config["training"]["phase3"]["pmpo_beta_kl"],
            entropy_coef=config["training"]["phase3"]["entropy_coef"],
            num_bins=config["training"]["phase3"]["advantage_bins"],
            discrete_actions=True,
            use_percentile_binning=config["training"]["phase3"].get("use_percentile_binning", False),
            percentile_threshold=config["training"]["phase3"].get("percentile_threshold", 10.0),
        ).to(device)
        
        value_loss_fn = TDLambdaLoss(
            discount=config["training"]["phase3"]["discount"],
            lambda_=config["training"]["phase3"]["lambda"],
            loss_scale=config["training"]["phase3"]["value_loss_scale"],
            use_distributional=True,
        ).to(device)
        
        # Create optimizers
        policy_optimizer = torch.optim.AdamW(policy_head.parameters(), lr=config["training"]["phase3"]["policy_lr"])
        value_optimizer = torch.optim.AdamW(value_head.parameters(), lr=config["training"]["phase3"]["value_lr"])
        
        # Create target network for value
        input_dim = config["tokenizer"]["num_latent_tokens"] * config["tokenizer"]["latent_dim"]
        value_target_head = ValueHead(
            input_dim=input_dim,
            hidden_dim=config["heads"]["hidden_dim"],
            num_layers=config["heads"]["num_layers"],
            num_bins=config["heads"]["num_bins"],
        ).to(device)
        value_target_head.load_state_dict(value_head.state_dict())
        value_target_head.eval()
        
        # Get initial latents
        frames = batch["frames"].to(device)
        frames_tokenizer = frames.permute(0, 2, 1, 3, 4)
        
        with torch.no_grad():
            first_frame = frames_tokenizer[:, :, 0:1]
            tokenizer_output = tokenizer(first_frame, mask_ratio=0.0)
            initial_latents = tokenizer_output["latents"][:, 0]
        
        # Generate rollout
        rollout_data = imagination(
            initial_latents=initial_latents,
            lambda_=config["training"]["phase3"]["lambda"],
            normalize_advantages=True,
        )
        
        # Flatten latents for heads
        latents = rollout_data["latents"]
        batch_size, horizon = latents.shape[:2]
        flat_latents = latents.reshape(batch_size, horizon, -1)
        
        # Policy update (PMPO)
        policy_head.train()
        policy_optimizer.zero_grad()
        
        policy_result = pmpo_loss_fn(
            policy_head=policy_head,
            latents=flat_latents,
            actions=rollout_data["actions"],
            advantages=rollout_data["advantages"],
            prior_head=None,
        )
        
        policy_loss = policy_result["loss"]
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_head.parameters(), config["training"]["phase3"]["grad_clip"])
        policy_optimizer.step()
        
        print(f"✓ Policy update (PMPO) successful")
        print(f"  Policy loss: {policy_loss.item():.4f}")
        print(f"  Entropy: {policy_result['entropy'].item():.4f}")
        
        # Value update (TD(λ))
        value_head.train()
        value_optimizer.zero_grad()
        
        bootstrap_latent = flat_latents[:, -1] if flat_latents.shape[1] > 0 else None
        
        value_result = value_loss_fn(
            value_head=value_head,
            latents=flat_latents,
            rewards=rollout_data["rewards"],
            dones=rollout_data["dones"],
            bootstrap_latent=bootstrap_latent,
            target_head=value_target_head,
        )
        
        value_loss = value_result["loss"]
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(value_head.parameters(), config["training"]["phase3"]["grad_clip"])
        value_optimizer.step()
        
        print(f"✓ Value update (TD(λ)) successful")
        print(f"  Value loss: {value_loss.item():.4f}")
        print(f"  Mean return: {value_result.get('mean_return', 0.0):.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Phase 3 training step failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_checkpoint_saving(config: Dict, tokenizer, dynamics, policy_head, value_head, reward_head, device: torch.device) -> bool:
    """Test 8: Checkpoint saving and loading."""
    print("\n" + "=" * 80)
    print("TEST 8: Checkpoint Saving/Loading")
    print("=" * 80)
    
    try:
        import tempfile
        import os
        
        # Create temporary checkpoint
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp_file:
            checkpoint_path = tmp_file.name
        
        # Save checkpoint
        checkpoint = {
            "tokenizer_state_dict": tokenizer.state_dict(),
            "dynamics_state_dict": dynamics.state_dict(),
            "policy_state_dict": policy_head.state_dict(),
            "value_state_dict": value_head.state_dict(),
            "reward_state_dict": reward_head.state_dict(),
            "config": config,
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"✓ Checkpoint saved to {checkpoint_path}")
        
        # Load checkpoint
        loaded_checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Create new models and load
        tokenizer_new = CausalTokenizer(
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
        
        tokenizer_new.load_state_dict(loaded_checkpoint["tokenizer_state_dict"])
        print(f"✓ Checkpoint loaded successfully")
        
        # Clean up
        os.unlink(checkpoint_path)
        print(f"✓ Temporary checkpoint deleted")
        
        return True
        
    except Exception as e:
        print(f"❌ Checkpoint saving/loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Test DreamerV4 Pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/minerl_small_batch.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (overrides config)",
    )
    args = parser.parse_args()
    
    print("=" * 80)
    print("DreamerV4 Pipeline Test Suite")
    print("=" * 80)
    
    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"❌ Error: Config file not found: {config_path}")
        sys.exit(1)
    
    config = load_config(config_path)
    print(f"\nConfig loaded from: {config_path}")
    
    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        requested_device = config["experiment"]["device"]
        if requested_device == "cuda" and not torch.cuda.is_available():
            print("⚠️  Warning: CUDA requested but not available. Using CPU.")
            device = torch.device("cpu")
        else:
            device = torch.device(requested_device)
    
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version: {torch.version.cuda}")
    
    # Set seed
    set_seed(config["experiment"]["seed"])

    # Track test results
    results = {}

    # =========================================================================
    # Module Unit Tests (no real data required)
    # =========================================================================
    print("\n" + "=" * 80)
    print("PHASE 0: Module Unit Tests (synthetic data)")
    print("=" * 80)

    # Test embeddings
    results["embeddings"] = test_embeddings(device)

    # Test tokenizer shapes
    results["tokenizer_shapes"] = test_tokenizer_shapes(config, device)

    # Test dynamics shapes
    results["dynamics_shapes"] = test_dynamics_shapes(config, device)

    # Test heads shapes
    results["heads_shapes"] = test_heads_shapes(config, device)

    # Test loss functions
    results["losses"] = test_losses(device)

    # Test imagination rollout
    results["imagination"] = test_imagination(config, device)

    # Check if module tests passed
    module_tests = ["embeddings", "tokenizer_shapes", "dynamics_shapes", "heads_shapes", "losses", "imagination"]
    module_passed = all(results.get(t, False) for t in module_tests)

    if not module_passed:
        print("\n" + "=" * 80)
        print("MODULE UNIT TESTS SUMMARY")
        print("=" * 80)
        for test_name in module_tests:
            status = "PASS" if results.get(test_name, False) else "FAIL"
            print(f"  {test_name:30s} {status}")
        print("\nModule tests failed. Skipping integration tests.")
        sys.exit(1)

    print("\nModule unit tests passed. Proceeding to integration tests...")

    # =========================================================================
    # Integration Tests (require real MineRL data)
    # =========================================================================
    print("\n" + "=" * 80)
    print("PHASE 1+: Integration Tests (real data)")
    print("=" * 80)

    # Test 1: Data loading
    success, sample, batch, dataloader = test_data_loading(config)
    results["data_loading"] = success
    if not success:
        print("\n❌ Pipeline test failed at data loading stage.")
        sys.exit(1)
    
    # Test 2: Model creation
    success, tokenizer, dynamics, policy_head, value_head, reward_head = test_model_creation(config, device)
    results["model_creation"] = success
    if not success:
        print("\n❌ Pipeline test failed at model creation stage.")
        sys.exit(1)
    
    # Test 3: Phase 1 forward passes
    results["phase1_forward"] = test_phase1_forward(config, tokenizer, dynamics, batch, device)
    
    # Test 4: Phase 1 training step
    results["phase1_training"] = test_phase1_training_step(config, tokenizer, dynamics, batch, device)
    
    # Test 5: Phase 2 training step
    results["phase2_training"] = test_phase2_training_step(
        config, tokenizer, dynamics, policy_head, reward_head, batch, device
    )
    
    # Test 6: Phase 3 imagination
    results["phase3_imagination"] = test_phase3_imagination(
        config, tokenizer, dynamics, policy_head, value_head, reward_head, batch, device
    )
    
    # Test 7: Phase 3 training step
    results["phase3_training"] = test_phase3_training_step(
        config, tokenizer, dynamics, policy_head, value_head, reward_head, batch, device
    )
    
    # Test 8: Checkpoint saving/loading
    results["checkpoint"] = test_checkpoint_saving(
        config, tokenizer, dynamics, policy_head, value_head, reward_head, device
    )
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "❌ FAIL"
        print(f"  {test_name:30s} {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 80)
    if all_passed:
        print("✅ ALL TESTS PASSED! Pipeline is ready for training.")
        print("\nNext steps:")
        print("  1. Run Phase 1: python train_phase1.py --config configs/minerl_small_batch.yaml")
        print("  2. Run Phase 2: python train_phase2.py --config configs/minerl_small_batch.yaml")
        print("  3. Run Phase 3: python train_phase3.py --config configs/minerl_small_batch.yaml --phase2-checkpoint <path>")
    else:
        print("❌ SOME TESTS FAILED. Please fix the issues above before training.")
        sys.exit(1)
    print("=" * 80)


if __name__ == "__main__":
    main()
