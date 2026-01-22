"""
Action space utilities for MineRL dataset.

Converts MineRL action format (binary + camera + discrete strings) 
into multi-discrete action space for DreamerV4:
- Keyboard: 8 binary distributions (independent Bernoulli)
- Mouse/Camera: Categorical with foveated discretization (121 classes)

As per Training.md: keyboard uses binary distributions, mouse uses categorical with foveated discretization.
"""

import numpy as np
import torch
from typing import Dict, Tuple, Optional, Union
from collections import defaultdict


# MineRL action space constants (verified from dataset)
BINARY_ACTIONS = ['forward', 'back', 'left', 'right', 'jump', 'sneak', 'sprint', 'attack']
NUM_BINARY = len(BINARY_ACTIONS)

CAMERA_BINS = 11  # 11×11 = 121 classes (foveated discretization)
# Camera range based on actual MineRL dataset analysis:
# Pitch: [-62.25, 60.60] (fits within range)
# Yaw: [-146.85, 69.75] (needs wider range)
# Using wider range to accommodate all values with margin
CAMERA_MIN = -150.0  # Updated to accommodate yaw min of -146.85
CAMERA_MAX = 75.0    # Updated to accommodate yaw max of 69.75 (with margin)
# Foveated discretization: μ-law compression for finer resolution near center
MU_LAW_MU = 255.0  # μ-law compression parameter

# Discrete action value mappings (from dataset inspection)
CRAFT_VALUES = ['none', 'crafting_table', 'planks', 'stick', 'torch']
EQUIP_VALUES = ['none', 'iron_pickaxe', 'stone_pickaxe', 'wooden_axe', 'wooden_pickaxe']
NEARBY_CRAFT_VALUES = ['none', 'furnace', 'iron_pickaxe', 'stone_pickaxe', 'wooden_axe', 'wooden_pickaxe']
NEARBY_SMELT_VALUES = ['none', 'iron_ingot']
PLACE_VALUES = ['none', 'cobblestone', 'crafting_table', 'dirt', 'furnace', 'stone', 'torch']

# Create mapping from string values to indices
CRAFT_TO_IDX = {v: i for i, v in enumerate(CRAFT_VALUES)}
EQUIP_TO_IDX = {v: i for i, v in enumerate(EQUIP_VALUES)}
NEARBY_CRAFT_TO_IDX = {v: i for i, v in enumerate(NEARBY_CRAFT_VALUES)}
NEARBY_SMELT_TO_IDX = {v: i for i, v in enumerate(NEARBY_SMELT_VALUES)}
PLACE_TO_IDX = {v: i for i, v in enumerate(PLACE_VALUES)}

# Total discrete combinations: 5×5×6×2×7 = 2,100
# But we'll only use the most common ~15 combinations
NUM_DISCRETE_COMBINATIONS = 15


def foveated_discretize(value: np.ndarray, min_val: float, max_val: float, num_bins: int) -> np.ndarray:
    """
    Foveated discretization using μ-law compression.
    
    Provides finer resolution near the center (0) and coarser resolution at extremes.
    This is similar to foveated vision - more detail where it matters most.
    
    Args:
        value: Array of values to discretize
        min_val: Minimum value
        max_val: Maximum value
        num_bins: Number of discrete bins
    
    Returns:
        Discrete bin indices in [0, num_bins-1]
    """
    # Normalize to [-1, 1] centered at 0
    center = (min_val + max_val) / 2.0
    range_val = (max_val - min_val) / 2.0
    normalized = (value - center) / range_val
    
    # Clamp to [-1, 1]
    normalized = np.clip(normalized, -1.0, 1.0)
    
    # Apply μ-law compression: sign(x) * ln(1 + μ|x|) / ln(1 + μ)
    # This compresses large values more, giving finer resolution near 0
    mu = MU_LAW_MU
    compressed = np.sign(normalized) * np.log1p(mu * np.abs(normalized)) / np.log1p(mu)
    
    # Map from [-1, 1] to [0, num_bins-1]
    # Compressed values are in [-1, 1], map linearly to bins
    bin_idx = ((compressed + 1.0) / 2.0) * (num_bins - 1)
    bin_idx = np.clip(bin_idx, 0, num_bins - 1).astype(np.int32)
    
    return bin_idx


def discretize_camera(camera: np.ndarray, use_foveated: bool = True) -> np.ndarray:
    """
    Discretize continuous camera (pitch, yaw) to 121 classes (11×11 grid).
    
    Uses foveated discretization by default (as per Training.md) for finer resolution
    near the center where most camera movements occur.
    
    Args:
        camera: (T, 2) or (2,) array of [pitch, yaw] values
        use_foveated: If True, use foveated (μ-law) discretization; else uniform
    
    Returns:
        Discrete camera action indices (T,) or scalar in [0, 120]
    """
    camera = np.asarray(camera)
    if camera.ndim == 1:
        camera = camera.reshape(1, -1)
        squeeze = True
    else:
        squeeze = False
    
    # Clamp to valid range
    pitch = np.clip(camera[:, 0], CAMERA_MIN, CAMERA_MAX)
    yaw = np.clip(camera[:, 1], CAMERA_MIN, CAMERA_MAX)
    
    if use_foveated:
        # Foveated discretization: finer resolution near center
        pitch_bin = foveated_discretize(pitch, CAMERA_MIN, CAMERA_MAX, CAMERA_BINS)
        yaw_bin = foveated_discretize(yaw, CAMERA_MIN, CAMERA_MAX, CAMERA_BINS)
    else:
        # Uniform discretization (backward compatibility)
        pitch_norm = (pitch - CAMERA_MIN) / (CAMERA_MAX - CAMERA_MIN)
        yaw_norm = (yaw - CAMERA_MIN) / (CAMERA_MAX - CAMERA_MIN)
        pitch_bin = np.clip((pitch_norm * CAMERA_BINS).astype(np.int32), 0, CAMERA_BINS - 1)
        yaw_bin = np.clip((yaw_norm * CAMERA_BINS).astype(np.int32), 0, CAMERA_BINS - 1)
    
    # Combine into single index: 11×11 grid
    camera_idx = pitch_bin * CAMERA_BINS + yaw_bin
    
    if squeeze:
        return camera_idx[0]
    return camera_idx


def combine_discrete_actions(
    craft: np.ndarray,
    equip: np.ndarray,
    nearby_craft: np.ndarray,
    nearby_smelt: np.ndarray,
    place: np.ndarray,
) -> np.ndarray:
    """
    Combine discrete string actions into single categorical index.
    
    Uses a hash-based approach to map (craft, equip, nearby_craft, nearby_smelt, place)
    combinations to indices in [0, NUM_DISCRETE_COMBINATIONS-1].
    
    Args:
        craft, equip, nearby_craft, nearby_smelt, place: String arrays or indices
    
    Returns:
        Combined discrete action indices
    """
    # Convert strings to indices if needed
    if craft.dtype.kind == 'U':  # Unicode string
        craft_idx = np.array([CRAFT_TO_IDX.get(str(v), 0) for v in craft])
    else:
        craft_idx = craft
    
    if equip.dtype.kind == 'U':
        equip_idx = np.array([EQUIP_TO_IDX.get(str(v), 0) for v in equip])
    else:
        equip_idx = equip
    
    if nearby_craft.dtype.kind == 'U':
        nearby_craft_idx = np.array([NEARBY_CRAFT_TO_IDX.get(str(v), 0) for v in nearby_craft])
    else:
        nearby_craft_idx = nearby_craft
    
    if nearby_smelt.dtype.kind == 'U':
        nearby_smelt_idx = np.array([NEARBY_SMELT_TO_IDX.get(str(v), 0) for v in nearby_smelt])
    else:
        nearby_smelt_idx = nearby_smelt
    
    if place.dtype.kind == 'U':
        place_idx = np.array([PLACE_TO_IDX.get(str(v), 0) for v in place])
    else:
        place_idx = place
    
    # Combine into single index using hash
    # Formula: craft * (5*6*2*7) + equip * (6*2*7) + nearby_craft * (2*7) + nearby_smelt * 7 + place
    # This gives unique indices in [0, 2099], then we map to [0, NUM_DISCRETE_COMBINATIONS-1]
    max_combinations = len(CRAFT_VALUES) * len(EQUIP_VALUES) * len(NEARBY_CRAFT_VALUES) * len(NEARBY_SMELT_VALUES) * len(PLACE_VALUES)
    
    combined = (
        craft_idx * (len(EQUIP_VALUES) * len(NEARBY_CRAFT_VALUES) * len(NEARBY_SMELT_VALUES) * len(PLACE_VALUES)) +
        equip_idx * (len(NEARBY_CRAFT_VALUES) * len(NEARBY_SMELT_VALUES) * len(PLACE_VALUES)) +
        nearby_craft_idx * (len(NEARBY_SMELT_VALUES) * len(PLACE_VALUES)) +
        nearby_smelt_idx * len(PLACE_VALUES) +
        place_idx
    )
    
    # Map to reduced space [0, NUM_DISCRETE_COMBINATIONS-1] using modulo
    # This ensures we stay within the configured action space
    discrete_idx = combined % NUM_DISCRETE_COMBINATIONS
    
    return discrete_idx


def combine_binary_actions(binary_dict: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Combine 8 binary actions into single categorical index.
    
    Args:
        binary_dict: Dictionary with keys from BINARY_ACTIONS, values are binary arrays
    
    Returns:
        Combined binary action indices in [0, 255] (2^8 combinations)
    """
    # Get binary values in order
    binary_values = []
    for action_name in BINARY_ACTIONS:
        if action_name in binary_dict:
            binary_values.append(binary_dict[action_name].astype(np.int32))
        else:
            # Default to 0 if missing
            shape = list(binary_dict.values())[0].shape if binary_dict else (1,)
            binary_values.append(np.zeros(shape, dtype=np.int32))
    
    # Combine: sum of (value * 2^position)
    combined = np.zeros_like(binary_values[0], dtype=np.int32)
    for i, val in enumerate(binary_values):
        combined += val * (2 ** i)
    
    return combined


def mineRL_actions_to_multi_discrete(actions_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Convert MineRL action dictionary to multi-discrete format.
    
    As per Training.md:
    - Keyboard: 8 binary distributions (independent Bernoulli)
    - Mouse/Camera: Categorical with foveated discretization (121 classes)
    
    Args:
        actions_dict: Dictionary with keys like 'action$forward', 'action$camera', etc.
    
    Returns:
        Dictionary with:
            - 'keyboard': (T, 8) binary array - one per keyboard action
            - 'camera': (T,) categorical indices in [0, 120]
    """
    # Extract binary actions
    binary_dict = {}
    for action_name in BINARY_ACTIONS:
        key = f'action${action_name}'
        if key in actions_dict:
            binary_dict[action_name] = actions_dict[key]
    
    # Extract camera
    camera = actions_dict.get('action$camera', None)
    
    # Get sequence length from first available array
    seq_len = None
    for arr in actions_dict.values():
        if arr is not None and hasattr(arr, 'shape'):
            seq_len = arr.shape[0] if arr.ndim > 0 else 1
            break
    
    if seq_len is None:
        raise ValueError("No valid actions found in actions_dict")
    
    # Build keyboard binary array (T, 8)
    keyboard_actions = np.zeros((seq_len, NUM_BINARY), dtype=np.int64)
    for i, action_name in enumerate(BINARY_ACTIONS):
        if action_name in binary_dict:
            keyboard_actions[:, i] = binary_dict[action_name].astype(np.int64)
    
    # Discretize camera with foveated discretization
    if camera is not None:
        camera_idx = discretize_camera(camera, use_foveated=True)
    else:
        # Default to center bin if no camera action
        center_bin = (CAMERA_BINS // 2) * CAMERA_BINS + (CAMERA_BINS // 2)
        camera_idx = np.full(seq_len, center_bin, dtype=np.int64)
    
    return {
        'keyboard': keyboard_actions,  # (T, 8) binary
        'camera': camera_idx,  # (T,) categorical in [0, 120]
    }


def mineRL_actions_to_categorical(actions_dict: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Convert MineRL action dictionary to single categorical action space.
    
    Maps to 144 actions:
    - Actions 0-7: Binary keyboard actions (8 actions)
    - Actions 8-128: Camera actions (121 actions, 11×11 grid)
    - Actions 129-143: Discrete action combinations (15 actions)
    
    Priority: discrete > camera > binary (if multiple are active)
    
    Args:
        actions_dict: Dictionary with keys like 'action$forward', 'action$camera', etc.
    
    Returns:
        Categorical action indices in [0, 143]
    """
    # Extract binary actions
    binary_dict = {}
    for action_name in BINARY_ACTIONS:
        key = f'action${action_name}'
        if key in actions_dict:
            binary_dict[action_name] = actions_dict[key]
    
    # Extract camera
    camera = actions_dict.get('action$camera', None)
    
    # Extract discrete actions
    craft = actions_dict.get('action$craft', None)
    equip = actions_dict.get('action$equip', None)
    nearby_craft = actions_dict.get('action$nearbyCraft', None)
    nearby_smelt = actions_dict.get('action$nearbySmelt', None)
    place = actions_dict.get('action$place', None)
    
    # Get sequence length from first available array
    seq_len = None
    for arr in actions_dict.values():
        if arr is not None and hasattr(arr, 'shape'):
            seq_len = arr.shape[0] if arr.ndim > 0 else 1
            break
    
    if seq_len is None:
        raise ValueError("No valid actions found in actions_dict")
    
    # Initialize output
    categorical_actions = np.zeros(seq_len, dtype=np.int64)
    
    # Priority 1: Discrete actions (129-143)
    if all(x is not None for x in [craft, equip, nearby_craft, nearby_smelt, place]):
        discrete_idx = combine_discrete_actions(craft, equip, nearby_craft, nearby_smelt, place)
        # Check if discrete action is active (not all 'none')
        has_discrete = (
            (craft != 'none') | 
            (equip != 'none') | 
            (nearby_craft != 'none') | 
            (nearby_smelt != 'none') | 
            (place != 'none')
        )
        if isinstance(has_discrete, np.ndarray):
            categorical_actions[has_discrete] = discrete_idx[has_discrete] + NUM_BINARY + (CAMERA_BINS * CAMERA_BINS)
    
    # Priority 2: Camera actions (8-128)
    if camera is not None:
        camera_idx = discretize_camera(camera)
        # Use camera if discrete is not active
        if all(x is not None for x in [craft, equip, nearby_craft, nearby_smelt, place]):
            has_discrete = (
                (craft == 'none') & 
                (equip == 'none') & 
                (nearby_craft == 'none') & 
                (nearby_smelt == 'none') & 
                (place == 'none')
            )
            if isinstance(has_discrete, np.ndarray):
                categorical_actions[has_discrete] = camera_idx[has_discrete] + NUM_BINARY
        else:
            # No discrete actions, use camera
            categorical_actions = camera_idx + NUM_BINARY
    
    # Priority 3: Binary actions (0-7)
    # Use binary if neither discrete nor camera is active
    if binary_dict:
        # Find first active binary action for each timestep
        for i, action_name in enumerate(BINARY_ACTIONS):
            if action_name in binary_dict:
                active = binary_dict[action_name] == 1
                if isinstance(active, np.ndarray):
                    # Only use binary if no discrete/camera action
                    use_binary = (
                        (categorical_actions == 0) | 
                        (categorical_actions < NUM_BINARY)
                    )
                    categorical_actions[active & use_binary] = i
    
    return categorical_actions
