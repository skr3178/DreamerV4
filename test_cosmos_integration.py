"""
Test Cosmos Tokenizer Integration for DreamerV4

Tests:
  Phase A (Smoke tests):
    1. Wrapper shape test - Load Cosmos encoder, verify output (B, T/8, 16, 16)
    2. Dynamics compatibility - Verify dynamics accepts 16 tokens x 16 dim
    3. Head compatibility - Verify heads accept 256-dim input

  Phase B (Integration):
    4. Single Phase 1 training step - Full pipeline with subset data, check VRAM

Usage:
    python test_cosmos_integration.py                    # Run all tests
    python test_cosmos_integration.py --phase A          # Smoke tests only
    python test_cosmos_integration.py --phase B          # Integration only
    python test_cosmos_integration.py --data-path data/mineRL_subset
"""

import sys
import os
import argparse
import traceback
from pathlib import Path

import torch
import torch.nn as nn
import yaml


def print_header(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def print_pass(msg):
    print(f"  [PASS] {msg}")


def print_fail(msg):
    print(f"  [FAIL] {msg}")


def print_info(msg):
    print(f"  [INFO] {msg}")


def gpu_mem_mb():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2
    return 0.0


def gpu_mem_gb():
    return gpu_mem_mb() / 1024


# ─── Phase A: Smoke Tests ───────────────────────────────────────────────

def test_cosmos_wrapper_shapes(device="cuda"):
    """Test 1: Verify Cosmos wrapper produces correct output shapes."""
    print_header("Test 1: Cosmos Wrapper Shape Verification")

    from dreamer.models.cosmos_tokenizer_wrapper import create_cosmos_tokenizer

    tokenizer = create_cosmos_tokenizer(
        checkpoint_path="cosmos_tokenizer/CV8x8x8",
        pool_tokens=16,
        input_resolution=256,
        device=device,
        dtype="bfloat16",
    )
    print_info(f"Loaded Cosmos tokenizer (pool_tokens={tokenizer.pool_tokens}, "
               f"latent_dim={tokenizer.latent_dim})")

    mem_after_load = gpu_mem_gb()
    print_info(f"GPU memory after load: {mem_after_load:.2f} GB")

    # Test with synthetic video: (B=2, C=3, T=32, H=64, W=64)
    B, C, T, H, W = 2, 3, 32, 64, 64
    video = torch.rand(B, C, T, H, W, device=device)
    print_info(f"Input shape: {video.shape}")

    with torch.no_grad():
        output = tokenizer.encode(video)
        latents = output["latents"]

    mem_after_encode = gpu_mem_gb()
    print_info(f"GPU memory after encode: {mem_after_encode:.2f} GB")

    # Cosmos CV8x8x8 is causal: T_out = 1 + ceil((T-1)/8)
    # For T=32: 1 + ceil(31/8) = 1 + 4 = 5
    import math
    T_out = 1 + math.ceil((T - 1) / 8)
    expected_shape = (B, T_out, 16, 16)
    actual_shape = tuple(latents.shape)

    if actual_shape == expected_shape:
        print_pass(f"Output shape: {actual_shape} (expected {expected_shape})")
    else:
        print_fail(f"Output shape: {actual_shape} (expected {expected_shape})")
        return False

    # Check latent properties
    print_info(f"Latent dtype: {latents.dtype}")
    print_info(f"Latent range: [{latents.min().item():.3f}, {latents.max().item():.3f}]")
    print_info(f"Latent mean: {latents.mean().item():.4f}, std: {latents.std().item():.4f}")

    # Verify flattened dim = 256
    flat_dim = latents.shape[-2] * latents.shape[-1]
    if flat_dim == 256:
        print_pass(f"Flattened dim: {flat_dim} (16 tokens x 16 dim)")
    else:
        print_fail(f"Flattened dim: {flat_dim} (expected 256)")
        return False

    # Clean up
    del video, output, latents, tokenizer
    torch.cuda.empty_cache()

    return True


def test_dynamics_compatibility(device="cuda"):
    """Test 2: Verify dynamics model works with 16 tokens x 16 dim."""
    print_header("Test 2: Dynamics Model Compatibility")

    from dreamer.models.dynamics import DynamicsModel

    # Create dynamics matching cosmos config
    dynamics = DynamicsModel(
        latent_dim=16,
        num_latent_tokens=16,
        embed_dim=256,
        depth=12,
        num_heads=8,
        num_discrete_actions=144,
        num_registers=4,
        max_shortcut_steps=6,
    ).to(device)

    num_params = sum(p.numel() for p in dynamics.parameters())
    print_info(f"Dynamics parameters: {num_params:,}")
    print_info(f"GPU memory: {gpu_mem_gb():.2f} GB")

    # Test forward with synthetic latents
    B, T = 2, 4  # 4 latent timesteps (from 32 frames / 8)
    latents = torch.randn(B, T, 16, 16, device=device)
    actions = torch.randint(0, 144, (B, T), device=device)

    output = dynamics(latents=latents, actions=actions, discrete_actions=True)

    pred = output["predicted_latents"]
    expected_shape = (B, T, 16, 16)

    if tuple(pred.shape) == expected_shape:
        print_pass(f"Predicted latents shape: {tuple(pred.shape)}")
    else:
        print_fail(f"Predicted latents shape: {tuple(pred.shape)} (expected {expected_shape})")
        return False

    # Verify other outputs exist
    for key in ["target_latents", "signal_level", "step_size", "d_is_min"]:
        if key in output:
            print_pass(f"Output key '{key}' present")
        else:
            print_fail(f"Output key '{key}' missing")
            return False

    # Clean up
    del dynamics, latents, actions, output
    torch.cuda.empty_cache()

    return True


def test_head_compatibility(device="cuda"):
    """Test 3: Verify all heads accept 256-dim input."""
    print_header("Test 3: Head Compatibility (256-dim input)")

    from dreamer.models.heads import PolicyHead, ValueHead, RewardHead

    input_dim = 256  # 16 tokens x 16 dim
    B, T = 2, 4

    # Flattened latent input
    latents_flat = torch.randn(B, T, input_dim, device=device)

    results = {}

    # Policy head
    policy = PolicyHead(
        input_dim=input_dim,
        hidden_dim=256,
        num_discrete_actions=144,
        num_layers=2,
        use_mtp=False,  # Standard mode for testing
    ).to(device)
    policy_out = policy(latents_flat)
    expected_logit_shape = (B, T, 144)
    if tuple(policy_out["logits"].shape) == expected_logit_shape:
        print_pass(f"PolicyHead logits: {tuple(policy_out['logits'].shape)}")
        results["policy"] = True
    else:
        print_fail(f"PolicyHead logits: {tuple(policy_out['logits'].shape)} (expected {expected_logit_shape})")
        results["policy"] = False

    # Value head
    value = ValueHead(
        input_dim=input_dim,
        hidden_dim=256,
        num_layers=2,
        num_bins=255,
    ).to(device)
    value_out = value(latents_flat)
    if tuple(value_out["value"].shape) == (B, T):
        print_pass(f"ValueHead value: {tuple(value_out['value'].shape)}")
        results["value"] = True
    else:
        print_fail(f"ValueHead value: {tuple(value_out['value'].shape)} (expected {(B, T)})")
        results["value"] = False

    # Reward head
    reward = RewardHead(
        input_dim=input_dim,
        hidden_dim=256,
        num_layers=2,
        num_bins=255,
        use_mtp=False,
    ).to(device)
    reward_out = reward(latents_flat)
    if tuple(reward_out["reward"].shape) == (B, T):
        print_pass(f"RewardHead reward: {tuple(reward_out['reward'].shape)}")
        results["reward"] = True
    else:
        print_fail(f"RewardHead reward: {tuple(reward_out['reward'].shape)} (expected {(B, T)})")
        results["reward"] = False

    # Clean up
    del policy, value, reward, latents_flat
    torch.cuda.empty_cache()

    return all(results.values())


# ─── Phase B: Integration Tests ─────────────────────────────────────────

def test_single_training_step(device="cuda", data_path="data/mineRL_subset"):
    """Test 4: Run a single Phase 1 training step with real data."""
    print_header("Test 4: Single Phase 1 Training Step (Subset Data)")

    from dreamer.models.cosmos_tokenizer_wrapper import create_cosmos_tokenizer
    from dreamer.models.dynamics import DynamicsModel
    from dreamer.losses import ShortcutForcingLoss
    from dreamer.data import create_dataloader

    # Load config
    config_path = "configs/minerl_cosmos.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Override data path to subset
    config["data"]["path"] = data_path
    print_info(f"Data path: {data_path}")

    # Check data exists
    data_dir = Path(data_path)
    if not data_dir.exists():
        print_fail(f"Data directory not found: {data_path}")
        print_info("Run: python create_subset.py --source-dir data/mineRL_extracted "
                    "--output-dir data/mineRL_subset --num-episodes 5")
        return False

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    mem_start = gpu_mem_gb()
    print_info(f"GPU memory at start: {mem_start:.2f} GB")

    # Create Cosmos tokenizer (frozen)
    print_info("Loading Cosmos tokenizer...")
    cosmos_config = config.get("cosmos_tokenizer", {})
    tokenizer = create_cosmos_tokenizer(
        checkpoint_path=cosmos_config.get("checkpoint_path", "cosmos_tokenizer/CV8x8x8"),
        pool_tokens=cosmos_config.get("pool_tokens", 16),
        input_resolution=cosmos_config.get("input_resolution", 256),
        device=device,
        dtype=cosmos_config.get("dtype", "bfloat16"),
    )
    tokenizer.eval()
    mem_tokenizer = gpu_mem_gb()
    print_info(f"GPU memory after tokenizer: {mem_tokenizer:.2f} GB (+{mem_tokenizer - mem_start:.2f})")

    # Create dynamics model
    print_info("Creating dynamics model...")
    dynamics_config = config.get("dynamics", {})
    tokenizer_config = config.get("tokenizer", {})
    dynamics = DynamicsModel(
        latent_dim=tokenizer_config.get("latent_dim", 16),
        num_latent_tokens=tokenizer_config.get("num_latent_tokens", 16),
        embed_dim=dynamics_config.get("embed_dim", 256),
        depth=dynamics_config.get("num_layers", 12),
        num_heads=dynamics_config.get("num_heads", 8),
        num_discrete_actions=dynamics_config.get("num_discrete_actions", 144),
        num_registers=dynamics_config.get("num_registers", 4),
        max_shortcut_steps=dynamics_config.get("max_shortcut_steps", 6),
    ).to(device)
    dynamics.train()

    mem_dynamics = gpu_mem_gb()
    dyn_params = sum(p.numel() for p in dynamics.parameters())
    print_info(f"Dynamics parameters: {dyn_params:,}")
    print_info(f"GPU memory after dynamics: {mem_dynamics:.2f} GB (+{mem_dynamics - mem_tokenizer:.2f})")

    # Create data loader (small batch for test)
    print_info("Creating data loader...")
    train_loader = create_dataloader(
        data_path=config["data"]["path"],
        batch_size=2,  # Small batch for testing
        sequence_length=config["data"]["sequence_length"],
        image_size=(config["data"]["image_height"], config["data"]["image_width"]),
        num_workers=2,
        split="train",
        max_episodes=config["data"].get("max_episodes", None),
    )
    print_info(f"Data loader batches: {len(train_loader)}")

    # Get one batch
    batch = next(iter(train_loader))
    frames = batch["frames"].to(device)
    actions = batch["actions"].to(device)

    print_info(f"Batch frames shape: {frames.shape}")
    print_info(f"Batch actions shape: {actions.shape}")

    # Reshape frames for tokenizer: (B, T, C, H, W) -> (B, C, T, H, W)
    if frames.dim() == 5 and frames.shape[2] == 3:
        frames = frames.permute(0, 2, 1, 3, 4)

    # Encode with frozen Cosmos
    print_info("Encoding with Cosmos tokenizer...")
    with torch.no_grad():
        tok_output = tokenizer.encode(frames, mask_ratio=0.0)
        latents = tok_output["latents"]

    mem_after_encode = gpu_mem_gb()
    print_info(f"Latents shape: {latents.shape}")
    print_info(f"GPU memory after encode: {mem_after_encode:.2f} GB")

    # Align actions to latent time dimension
    T_lat = latents.shape[1]
    if actions.dim() == 3 and actions.shape[-1] == 1:
        actions = actions.squeeze(-1)
    if actions.shape[1] > T_lat:
        action_indices = torch.linspace(0, actions.shape[1] - 1, T_lat).long()
        actions = actions[:, action_indices]

    print_info(f"Actions aligned shape: {actions.shape}")

    # Forward through dynamics
    print_info("Forward pass through dynamics...")
    output = dynamics(
        latents=latents,
        actions=actions,
        discrete_actions=(actions.dim() == 2),
    )

    pred_latents = output["predicted_latents"]
    print_info(f"Predicted latents shape: {pred_latents.shape}")

    # Compute loss
    print_info("Computing shortcut forcing loss...")
    loss_fn = ShortcutForcingLoss()
    loss_dict = loss_fn(
        predicted_latents=output["predicted_latents"],
        target_latents=output["target_latents"],
        signal_level=output["signal_level"],
        step_size=output["step_size"],
        d_is_min=output["d_is_min"],
    )

    loss = loss_dict["loss"]
    print_info(f"Loss value: {loss.item():.4f}")

    # Backward pass
    print_info("Backward pass...")
    optimizer = torch.optim.AdamW(dynamics.parameters(), lr=3e-4)
    optimizer.zero_grad()
    loss.backward()

    # Check gradients
    grad_norm = torch.nn.utils.clip_grad_norm_(dynamics.parameters(), 1.0)
    print_info(f"Gradient norm: {grad_norm.item():.4f}")

    optimizer.step()

    mem_after_step = gpu_mem_gb()
    mem_peak = torch.cuda.max_memory_allocated() / 1024**3
    print_info(f"GPU memory after step: {mem_after_step:.2f} GB")
    print_info(f"GPU peak memory: {mem_peak:.2f} GB")

    # Verify VRAM budget
    if mem_peak < 12.0:
        print_pass(f"Peak VRAM: {mem_peak:.2f} GB < 12 GB budget")
    else:
        print_fail(f"Peak VRAM: {mem_peak:.2f} GB exceeds 12 GB budget!")
        return False

    # Verify loss is finite
    if torch.isfinite(loss):
        print_pass(f"Loss is finite: {loss.item():.4f}")
    else:
        print_fail(f"Loss is not finite: {loss.item()}")
        return False

    # Verify gradients are finite
    if torch.isfinite(grad_norm):
        print_pass(f"Gradients are finite (norm={grad_norm.item():.4f})")
    else:
        print_fail(f"Gradients are not finite")
        return False

    # Verify shapes match plan
    if tuple(latents.shape[2:]) == (16, 16):
        print_pass(f"Latent token shape: {tuple(latents.shape[2:])} = (16, 16)")
    else:
        print_fail(f"Latent token shape: {tuple(latents.shape[2:])} (expected (16, 16))")
        return False

    # Clean up
    del tokenizer, dynamics, optimizer, batch, frames, actions, latents, output
    torch.cuda.empty_cache()

    return True


# ─── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Test Cosmos Tokenizer Integration")
    parser.add_argument("--phase", choices=["A", "B", "all"], default="all",
                        help="Which test phase to run")
    parser.add_argument("--device", default="cuda", help="Device to use")
    parser.add_argument("--data-path", default="data/mineRL_subset",
                        help="Path to subset data for integration tests")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"

    if args.device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"Total GPU memory: {total_mem:.2f} GB")

    results = {}

    # Phase A: Smoke tests
    if args.phase in ("A", "all"):
        print_header("PHASE A: SMOKE TESTS")

        tests_a = [
            ("wrapper_shapes", test_cosmos_wrapper_shapes),
            ("dynamics_compat", test_dynamics_compatibility),
            ("head_compat", test_head_compatibility),
        ]

        for name, test_fn in tests_a:
            try:
                results[name] = test_fn(device=args.device)
            except Exception as e:
                print_fail(f"Exception in {name}: {e}")
                traceback.print_exc()
                results[name] = False

    # Phase B: Integration tests
    if args.phase in ("B", "all"):
        print_header("PHASE B: INTEGRATION TESTS")

        tests_b = [
            ("single_step", lambda device: test_single_training_step(device, args.data_path)),
        ]

        for name, test_fn in tests_b:
            try:
                results[name] = test_fn(device=args.device)
            except Exception as e:
                print_fail(f"Exception in {name}: {e}")
                traceback.print_exc()
                results[name] = False

    # Summary
    print_header("TEST SUMMARY")
    all_passed = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")
        if not passed:
            all_passed = False

    if all_passed:
        print(f"\n  All {len(results)} tests passed!")
    else:
        failed = sum(1 for v in results.values() if not v)
        print(f"\n  {failed}/{len(results)} tests FAILED")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
