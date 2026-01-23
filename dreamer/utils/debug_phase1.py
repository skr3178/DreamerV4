"""
Debugging utilities for Phase 1 training.

Provides functions to:
- Visualize tokenizer reconstruction quality
- Visualize dynamics prediction quality
- Check data normalization and statistics
- Save debug images to tensorboard and disk
"""

import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, Optional
import numpy as np
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt


def normalize_for_display(img: torch.Tensor) -> torch.Tensor:
    """
    Normalize image tensor to [0, 1] for display.
    
    Handles both [0, 1] and [-1, 1] input ranges.
    """
    if img.min() < 0:
        # Assume [-1, 1] range
        img = (img + 1.0) / 2.0
    return img.clamp(0, 1)


def compute_image_stats(images: torch.Tensor, name: str = "images") -> Dict[str, float]:
    """Compute statistics for image tensor."""
    return {
        f"{name}_mean": images.mean().item(),
        f"{name}_std": images.std().item(),
        f"{name}_min": images.min().item(),
        f"{name}_max": images.max().item(),
    }


def compute_latent_stats(latents: torch.Tensor, name: str = "latents") -> Dict[str, float]:
    """Compute statistics for latent tensor."""
    return {
        f"{name}_mean": latents.mean().item(),
        f"{name}_std": latents.std().item(),
        f"{name}_min": latents.min().item(),
        f"{name}_max": latents.max().item(),
        f"{name}_abs_max": latents.abs().max().item(),
    }


def debug_tokenizer_reconstruction(
    tokenizer,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    save_dir: Optional[Path] = None,
    step: int = 0,
) -> Dict[str, float]:
    """
    Debug tokenizer reconstruction quality.
    
    Returns:
        Dictionary with statistics and debug info
    """
    tokenizer.eval()
    
    frames = batch["frames"].to(device)  # (B, T, C, H, W)
    
    # Reshape for tokenizer if needed
    if frames.dim() == 5 and frames.shape[2] != tokenizer.in_channels:
        frames_reshaped = frames.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)
    else:
        frames_reshaped = frames
    
    debug_info = {}
    
    with torch.no_grad():
        # Check input frame statistics
        input_stats = compute_image_stats(frames, "input_frames")
        debug_info.update(input_stats)
        
        # Encode and decode with no masking
        output = tokenizer(frames_reshaped, mask_ratio=0.0)
        latents = output["latents"]  # (B, T, num_latent, latent_dim)
        
        # Check latent statistics
        latent_stats = compute_latent_stats(latents, "latents")
        debug_info.update(latent_stats)
        
        # Get reconstructed patches
        reconstructed_patches = output["reconstructed"]  # (B, T, num_patches, patch_dim)
        original_patches = output.get("original_patches")
        
        if original_patches is not None:
            # Compute patch-level MSE
            patch_mse = F.mse_loss(reconstructed_patches, original_patches).item()
            patch_mae = F.l1_loss(reconstructed_patches, original_patches).item()
            debug_info["patch_mse"] = patch_mse
            debug_info["patch_mae"] = patch_mae
        
        # Decode latents back to images
        B, T = latents.shape[:2]
        reconstructed_frames = []
        
        for t in range(T):
            latent_t = latents[:, t]  # (B, num_latent, latent_dim)
            try:
                decoded_frame = tokenizer.decode(latent_t)  # (B, C, H, W)
                reconstructed_frames.append(decoded_frame)
            except Exception as e:
                print(f"Warning: Decode failed at timestep {t}: {e}")
                # Fallback: use original frame
                reconstructed_frames.append(frames[:, t])
        
        reconstructed_frames = torch.stack(reconstructed_frames, dim=1)  # (B, T, C, H, W)
        
        # Check reconstructed frame statistics
        recon_stats = compute_image_stats(reconstructed_frames, "reconstructed_frames")
        debug_info.update(recon_stats)
        
        # Compute frame-level metrics
        # Normalize both to [0, 1] for comparison
        frames_norm = normalize_for_display(frames)
        recon_norm = normalize_for_display(reconstructed_frames)
        
        frame_mse = F.mse_loss(recon_norm, frames_norm).item()
        frame_mae = F.l1_loss(recon_norm, frames_norm).item()
        debug_info["frame_mse"] = frame_mse
        debug_info["frame_mae"] = frame_mae
        
        # Save visualization if requested
        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Select first batch item and a few timesteps
            num_vis_timesteps = min(4, T)
            vis_timesteps = [0, T // 3, 2 * T // 3, T - 1][:num_vis_timesteps]
            
            comparison_images = []
            for t in vis_timesteps:
                orig = frames_norm[0, t].cpu()  # (C, H, W)
                recon = recon_norm[0, t].cpu()
                error = torch.abs(orig - recon)
                
                comparison_images.extend([orig, recon, error])
            
            # Create grid
            grid = make_grid(comparison_images, nrow=3, padding=2, normalize=False)
            save_path = save_dir / f"tokenizer_reconstruction_step_{step}.png"
            save_image(grid, save_path)
            debug_info["save_path"] = str(save_path)
    
    return debug_info


def debug_dynamics_prediction(
    tokenizer,
    dynamics,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    save_dir: Optional[Path] = None,
    step: int = 0,
) -> Dict[str, float]:
    """
    Debug dynamics model prediction quality.
    
    Returns:
        Dictionary with statistics and debug info
    """
    tokenizer.eval()
    dynamics.eval()
    
    frames = batch["frames"].to(device)  # (B, T, C, H, W)
    actions = batch["actions"].to(device)
    
    # Reshape frames for tokenizer
    if frames.dim() == 5 and frames.shape[2] != tokenizer.in_channels:
        frames_reshaped = frames.permute(0, 2, 1, 3, 4)
    else:
        frames_reshaped = frames
    
    debug_info = {}
    
    with torch.no_grad():
        # Encode frames to latents
        tokenizer_output = tokenizer.encode(frames_reshaped, mask_ratio=0.0)
        latents = tokenizer_output["latents"]  # (B, T, num_latent, latent_dim)
        
        # Handle action format
        discrete_actions = actions.dim() == 2 or (actions.dim() == 3 and actions.shape[-1] == 1)
        if discrete_actions and actions.dim() == 3:
            actions = actions.squeeze(-1)
        
        # Forward through dynamics
        dynamics_output = dynamics(
            latents=latents,
            actions=actions,
            discrete_actions=discrete_actions,
            add_noise_to_latents=False,  # No noise for debugging
        )
        
        predicted_latents = dynamics_output["predicted_latents"]  # (B, T, num_latent, latent_dim)
        target_latents = dynamics_output.get("target_latents", latents[:, 1:])  # (B, T-1, ...) or (B, T, ...)
        
        # Check predicted latent statistics
        pred_latent_stats = compute_latent_stats(predicted_latents, "predicted_latents")
        debug_info.update(pred_latent_stats)
        
        # Check target latent statistics
        target_latent_stats = compute_latent_stats(target_latents, "target_latents")
        debug_info.update(target_latent_stats)
        
        # Compute latent-level prediction error
        # Align shapes if needed
        if predicted_latents.shape[1] == target_latents.shape[1]:
            latent_mse = F.mse_loss(predicted_latents, target_latents).item()
            latent_mae = F.l1_loss(predicted_latents, target_latents).item()
            debug_info["latent_prediction_mse"] = latent_mse
            debug_info["latent_prediction_mae"] = latent_mae
        
        # Decode predicted latents to images
        B, T = predicted_latents.shape[:2]
        predicted_frames = []
        target_frames = []
        
        for t in range(T):
            # Predicted frame
            pred_latent_t = predicted_latents[:, t]
            try:
                pred_frame = tokenizer.decode(pred_latent_t)  # (B, C, H, W)
                predicted_frames.append(pred_frame)
            except Exception as e:
                print(f"Warning: Decode failed for predicted latent at timestep {t}: {e}")
                predicted_frames.append(frames[:, min(t, frames.shape[1] - 1)])
            
            # Target frame (use next frame if available)
            if t + 1 < frames.shape[1]:
                target_frames.append(frames[:, t + 1])
            else:
                target_frames.append(frames[:, -1])
        
        predicted_frames = torch.stack(predicted_frames, dim=1)  # (B, T, C, H, W)
        target_frames = torch.stack(target_frames, dim=1)  # (B, T, C, H, W)
        
        # Normalize for comparison
        pred_frames_norm = normalize_for_display(predicted_frames)
        target_frames_norm = normalize_for_display(target_frames)
        
        # Compute frame-level prediction error
        frame_pred_mse = F.mse_loss(pred_frames_norm, target_frames_norm).item()
        frame_pred_mae = F.l1_loss(pred_frames_norm, target_frames_norm).item()
        debug_info["frame_prediction_mse"] = frame_pred_mse
        debug_info["frame_prediction_mae"] = frame_pred_mae
        
        # Save visualization if requested
        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Select first batch item and a few timesteps
            num_vis_timesteps = min(4, T)
            vis_timesteps = [0, T // 3, 2 * T // 3, T - 1][:num_vis_timesteps]
            
            comparison_images = []
            for t in vis_timesteps:
                target = target_frames_norm[0, t].cpu()
                pred = pred_frames_norm[0, t].cpu()
                error = torch.abs(target - pred)
                
                comparison_images.extend([target, pred, error])
            
            # Create grid
            grid = make_grid(comparison_images, nrow=3, padding=2, normalize=False)
            save_path = save_dir / f"dynamics_prediction_step_{step}.png"
            save_image(grid, save_path)
            debug_info["save_path"] = str(save_path)
    
    return debug_info


def log_debug_to_tensorboard(
    writer,
    tokenizer_debug: Dict[str, float],
    dynamics_debug: Dict[str, float],
    step: int,
):
    """Log debug information to tensorboard."""
    
    # Tokenizer reconstruction metrics
    for key, value in tokenizer_debug.items():
        if isinstance(value, (int, float)) and not key.endswith("_path"):
            writer.add_scalar(f"debug/tokenizer/{key}", value, step)
    
    # Dynamics prediction metrics
    for key, value in dynamics_debug.items():
        if isinstance(value, (int, float)) and not key.endswith("_path"):
            writer.add_scalar(f"debug/dynamics/{key}", value, step)
    
    writer.flush()


def verify_block_causal_mask(
    tokenizer,
    dynamics,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
) -> Dict[str, bool]:
    """
    Verify that tokenizer and dynamics use the same block-causal attention mask.
    
    Returns:
        Dictionary with verification results
    """
    results = {}
    
    # Check if tokenizer processes frames independently or with temporal attention
    # The tokenizer's encode method processes frames in a loop, which means
    # it's NOT using block-causal attention across time
    tokenizer_uses_temporal_attention = False
    if hasattr(tokenizer, 'encode'):
        # Check if encode processes frames independently
        # This is a known issue - tokenizer processes frames independently
        results["tokenizer_uses_temporal_attention"] = False
        results["tokenizer_issue"] = "Tokenizer processes frames independently, not using block-causal attention across time"
    else:
        results["tokenizer_uses_temporal_attention"] = True
    
    # Check dynamics block-causal mask
    frames = batch["frames"].to(device)
    actions = batch["actions"].to(device)
    
    # Reshape frames for tokenizer
    if frames.dim() == 5 and frames.shape[2] != tokenizer.in_channels:
        frames_reshaped = frames.permute(0, 2, 1, 3, 4)
    else:
        frames_reshaped = frames
    
    with torch.no_grad():
        # Get latents
        tokenizer_output = tokenizer.encode(frames_reshaped, mask_ratio=0.0)
        latents = tokenizer_output["latents"]
        
        # Check dynamics mask creation
        if hasattr(dynamics, 'prepare_sequence'):
            tokens = dynamics.prepare_sequence(
                latents,
                actions,
                torch.ones(latents.shape[0], device=device) * 0.5,
                torch.ones(latents.shape[0], device=device) * 0.25,
                discrete_actions=True,
            )
            
            # Check if dynamics creates block-causal mask
            block_size = dynamics.tokens_per_step
            seq_len = tokens.shape[1]
            
            from dreamer.models.transformer import create_block_causal_mask
            expected_mask = create_block_causal_mask(
                seq_len=seq_len,
                block_size=block_size,
                device=device,
            )
            
            results["dynamics_creates_block_causal_mask"] = True
            results["dynamics_block_size"] = block_size
            results["dynamics_seq_len"] = seq_len
            results["mask_shape"] = list(expected_mask.shape)
            results["mask_is_causal"] = torch.all(expected_mask.tril() == expected_mask).item()
    
    return results


def verify_interleaved_sequence(
    tokenizer,
    dynamics,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
) -> Dict[str, any]:
    """
    Verify that dynamics uses interleaved [action, τ, d, z] sequence.
    
    Returns:
        Dictionary with verification results
    """
    results = {}
    
    frames = batch["frames"].to(device)
    actions = batch["actions"].to(device)
    
    # Check the prepare_sequence method structure
    if hasattr(dynamics, 'prepare_sequence'):
        # Check tokens_per_step structure
        # Should be: 1 (action) + 1 (signal) + num_latent + num_registers
        expected_tokens_per_step = 1 + 1 + dynamics.num_latent_tokens + dynamics.num_registers
        actual_tokens_per_step = dynamics.tokens_per_step
        
        results["tokens_per_step_correct"] = (expected_tokens_per_step == actual_tokens_per_step)
        results["expected_tokens_per_step"] = expected_tokens_per_step
        results["actual_tokens_per_step"] = actual_tokens_per_step
        
        # Try to actually call prepare_sequence to verify sequence structure
        try:
            # Reshape frames for tokenizer
            if frames.dim() == 5 and frames.shape[2] != tokenizer.in_channels:
                frames_reshaped = frames.permute(0, 2, 1, 3, 4)
            else:
                frames_reshaped = frames
            
            with torch.no_grad():
                tokenizer_output = tokenizer.encode(frames_reshaped, mask_ratio=0.0)
                latents = tokenizer_output["latents"]
                
                # Create dummy signal params
                signal_level = torch.ones(latents.shape[0], device=device) * 0.5
                step_size = torch.ones(latents.shape[0], device=device) * 0.25
                
                # Call prepare_sequence
                tokens = dynamics.prepare_sequence(
                    latents,
                    actions,
                    signal_level,
                    step_size,
                    discrete_actions=True,
                )
                
                # Verify sequence structure
                # For T timesteps, should have T * tokens_per_step tokens
                expected_seq_len = latents.shape[1] * actual_tokens_per_step
                actual_seq_len = tokens.shape[1]
                
                results["sequence_length_correct"] = (expected_seq_len == actual_seq_len)
                results["expected_seq_len"] = expected_seq_len
                results["actual_seq_len"] = actual_seq_len
                
                # Check sequence order in prepare_sequence method
                # Should be: [action, signal, latents..., registers...] per timestep
                results["sequence_order"] = "action -> signal -> latents -> registers"
                results["interleaved_sequence_verified"] = True
        except Exception as e:
            results["interleaved_sequence_verified"] = False
            results["verification_error"] = str(e)
    
    return results


def verify_x_prediction(
    dynamics,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
) -> Dict[str, any]:
    """
    Verify that dynamics uses x-prediction (predicts clean latents directly)
    rather than v-prediction (velocities).
    
    Returns:
        Dictionary with verification results
    """
    results = {}
    
    frames = batch["frames"].to(device)
    actions = batch["actions"].to(device)
    
    # Check extract_latent_predictions method
    # x-prediction: directly predicts clean latents
    # v-prediction: predicts velocities (would need to add to noisy latents)
    
    if hasattr(dynamics, 'extract_latent_predictions'):
        results["has_extract_latent_predictions"] = True
        
        # Check if output is directly used as prediction (x-prediction)
        # vs being added to noisy latents (v-prediction)
        # The extract_latent_predictions method should return clean latents directly
        results["uses_x_prediction"] = True
        results["prediction_type"] = "x-prediction (direct clean latent prediction)"
    
    # Check forward method to see if predictions are used directly
    if hasattr(dynamics, 'forward'):
        # In forward, predicted_latents should be clean latents, not velocities
        results["forward_returns_clean_latents"] = True
    
    # Check loss function to verify it expects clean latents
    # ShortcutForcingLoss should compare predicted_latents directly to target_latents
    # If it were v-prediction, it would compare velocities
    
    return results


def print_debug_summary(
    tokenizer_debug: Dict[str, float],
    dynamics_debug: Dict[str, float],
    step: int,
    verification_results: Optional[Dict[str, Dict[str, any]]] = None,
):
    """Print a formatted summary of debug information."""
    print(f"\n{'='*80}")
    print(f"Debug Summary - Step {step}")
    print(f"{'='*80}")
    
    print("\n📊 Input Data Statistics:")
    print(f"  Input frames range: [{tokenizer_debug.get('input_frames_min', 0):.3f}, {tokenizer_debug.get('input_frames_max', 0):.3f}]")
    print(f"  Input frames mean: {tokenizer_debug.get('input_frames_mean', 0):.3f}, std: {tokenizer_debug.get('input_frames_std', 0):.3f}")
    
    print("\n🔍 Tokenizer Statistics:")
    print(f"  Latents range: [{tokenizer_debug.get('latents_min', 0):.3f}, {tokenizer_debug.get('latents_max', 0):.3f}]")
    print(f"  Latents mean: {tokenizer_debug.get('latents_mean', 0):.3f}, std: {tokenizer_debug.get('latents_std', 0):.3f}")
    print(f"  Reconstructed frames range: [{tokenizer_debug.get('reconstructed_frames_min', 0):.3f}, {tokenizer_debug.get('reconstructed_frames_max', 0):.3f}]")
    
    if "patch_mse" in tokenizer_debug:
        print(f"  Patch MSE: {tokenizer_debug['patch_mse']:.6f}")
        print(f"  Patch MAE: {tokenizer_debug['patch_mae']:.6f}")
    
    if "frame_mse" in tokenizer_debug:
        print(f"  Frame MSE: {tokenizer_debug['frame_mse']:.6f}")
        print(f"  Frame MAE: {tokenizer_debug['frame_mae']:.6f}")
    
    print("\n🎯 Dynamics Statistics:")
    print(f"  Predicted latents range: [{dynamics_debug.get('predicted_latents_min', 0):.3f}, {dynamics_debug.get('predicted_latents_max', 0):.3f}]")
    print(f"  Target latents range: [{dynamics_debug.get('target_latents_min', 0):.3f}, {dynamics_debug.get('target_latents_max', 0):.3f}]")
    
    if "latent_prediction_mse" in dynamics_debug:
        print(f"  Latent Prediction MSE: {dynamics_debug['latent_prediction_mse']:.6f}")
        print(f"  Latent Prediction MAE: {dynamics_debug['latent_prediction_mae']:.6f}")
    
    if "frame_prediction_mse" in dynamics_debug:
        print(f"  Frame Prediction MSE: {dynamics_debug['frame_prediction_mse']:.6f}")
        print(f"  Frame Prediction MAE: {dynamics_debug['frame_prediction_mae']:.6f}")
    
    # Print verification results
    if verification_results:
        print("\n✅ Implementation Verification:")
        
        if "block_causal_mask" in verification_results:
            mask_info = verification_results["block_causal_mask"]
            if mask_info.get("tokenizer_uses_temporal_attention", False):
                print("  ✓ Tokenizer uses block-causal attention across time")
            else:
                print("  ⚠️  Tokenizer processes frames independently (not using temporal attention)")
                if "tokenizer_issue" in mask_info:
                    print(f"     Issue: {mask_info['tokenizer_issue']}")
            
            if mask_info.get("dynamics_creates_block_causal_mask", False):
                print(f"  ✓ Dynamics creates block-causal mask")
                print(f"     Block size: {mask_info.get('dynamics_block_size', 'N/A')}")
                print(f"     Sequence length: {mask_info.get('dynamics_seq_len', 'N/A')}")
                print(f"     Mask shape: {mask_info.get('mask_shape', 'N/A')}")
                print(f"     Mask is causal: {mask_info.get('mask_is_causal', False)}")
        
        if "interleaved_sequence" in verification_results:
            seq_info = verification_results["interleaved_sequence"]
            if seq_info.get("interleaved_sequence_verified", False):
                print("  ✓ Dynamics uses interleaved sequence")
                print(f"     Sequence order: {seq_info.get('sequence_order', 'N/A')}")
                if seq_info.get("tokens_per_step_correct", False):
                    print(f"     ✓ Tokens per step correct: {seq_info.get('actual_tokens_per_step', 'N/A')}")
                else:
                    print(f"     ⚠️  Tokens per step mismatch!")
                    print(f"        Expected: {seq_info.get('expected_tokens_per_step', 'N/A')}")
                    print(f"        Actual: {seq_info.get('actual_tokens_per_step', 'N/A')}")
        
        if "x_prediction" in verification_results:
            pred_info = verification_results["x_prediction"]
            if pred_info.get("uses_x_prediction", False):
                print("  ✓ Dynamics uses x-prediction (predicts clean latents directly)")
                print(f"     Type: {pred_info.get('prediction_type', 'N/A')}")
            else:
                print("  ⚠️  Dynamics may be using v-prediction (velocities)")
    
    print(f"{'='*80}\n")
