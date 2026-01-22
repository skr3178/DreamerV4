"""
Checkpoint loading utilities for DreamerV4.

Shared functions for loading Phase 1, Phase 2, and Phase 3 checkpoints
across evaluation and video generation scripts.
"""

from typing import Dict, Tuple
import torch
import torch.nn as nn


def strip_compiled_prefix(state_dict: Dict) -> Dict:
    """Strip '_orig_mod.' prefix from keys if checkpoint was saved with torch.compile().

    Args:
        state_dict: Model state dict that may have _orig_mod. prefix

    Returns:
        State dict with prefix stripped from keys
    """
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("_orig_mod."):
            new_state_dict[key[len("_orig_mod."):]] = value
        else:
            new_state_dict[key] = value
    return new_state_dict


def load_state_dict_with_warnings(
    model: nn.Module,
    state_dict: Dict,
    model_name: str = "model",
    strict: bool = False,
) -> None:
    """Load state dict with warnings for missing/unexpected keys.

    Args:
        model: PyTorch model to load weights into
        state_dict: State dict to load
        model_name: Name for logging purposes
        strict: If True, raise error on mismatch; if False, log warnings
    """
    result = model.load_state_dict(state_dict, strict=strict)

    if result.missing_keys:
        print(f"  Warning: {model_name} missing keys: {result.missing_keys}")
    if result.unexpected_keys:
        print(f"  Warning: {model_name} unexpected keys: {result.unexpected_keys}")


def load_phase2_world_model(
    checkpoint_path: str,
    config: Dict,
    device: torch.device,
    create_tokenizer_fn,
    create_dynamics_fn,
) -> Tuple[nn.Module, nn.Module, Dict]:
    """Load Phase 2 checkpoint for tokenizer and dynamics only (not heads).

    Args:
        checkpoint_path: Path to Phase 2 checkpoint file
        config: Configuration dictionary
        device: Device to load models onto
        create_tokenizer_fn: Function to create tokenizer from config
        create_dynamics_fn: Function to create dynamics model from config

    Returns:
        Tuple of (tokenizer, dynamics, checkpoint_dict)
    """
    print(f"Loading Phase 2 checkpoint (world model) from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Create models
    tokenizer = create_tokenizer_fn(config).to(device)
    dynamics = create_dynamics_fn(config).to(device)

    # Load state dicts (handle compiled models)
    tokenizer_sd = strip_compiled_prefix(checkpoint["tokenizer_state_dict"])
    dynamics_sd = strip_compiled_prefix(checkpoint["dynamics_state_dict"])

    load_state_dict_with_warnings(tokenizer, tokenizer_sd, "tokenizer")
    load_state_dict_with_warnings(dynamics, dynamics_sd, "dynamics")

    # Set to eval mode
    tokenizer.eval()
    dynamics.eval()

    print(f"  Phase 2 world model loaded successfully")
    if "global_step" in checkpoint:
        print(f"  Global step: {checkpoint['global_step']}")
    if "epoch" in checkpoint:
        print(f"  Epoch: {checkpoint['epoch']}")

    return tokenizer, dynamics, checkpoint


def load_phase3_heads(
    checkpoint_path: str,
    config: Dict,
    device: torch.device,
    create_heads_fn,
) -> Tuple[Dict[str, nn.Module], Dict]:
    """Load Phase 3 checkpoint for heads (policy, value, reward).

    Args:
        checkpoint_path: Path to Phase 3 checkpoint file
        config: Configuration dictionary
        device: Device to load models onto
        create_heads_fn: Function to create heads dict from config

    Returns:
        Tuple of (heads_dict, checkpoint_dict)
    """
    print(f"Loading Phase 3 checkpoint (heads) from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Create heads
    heads = {k: v.to(device) for k, v in create_heads_fn(config).items()}

    # Load state dicts (handle compiled models)
    load_state_dict_with_warnings(
        heads["policy"],
        strip_compiled_prefix(checkpoint["policy_head"]),
        "policy_head"
    )
    load_state_dict_with_warnings(
        heads["value"],
        strip_compiled_prefix(checkpoint["value_head"]),
        "value_head"
    )
    load_state_dict_with_warnings(
        heads["reward"],
        strip_compiled_prefix(checkpoint["reward_head"]),
        "reward_head"
    )

    # Set to eval mode
    for head in heads.values():
        head.eval()

    print(f"  Phase 3 heads loaded successfully")
    if "epoch" in checkpoint:
        print(f"  Epoch: {checkpoint['epoch']}")
    if "global_step" in checkpoint:
        print(f"  Global step: {checkpoint['global_step']}")
    if "mean_return" in checkpoint:
        print(f"  Mean return: {checkpoint['mean_return']:.2f}")

    return heads, checkpoint
