"""
MineRL Dataset Loader for DreamerV4

Loads video sequences with actions and rewards from MineRL/VPT data.
"""

import os
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Union

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class MineRLDataset(Dataset):
    """
    Dataset for loading MineRL/VPT trajectory data.
    
    Expects data in format:
    - Video frames: (T, C, H, W) or stored as individual images
    - Actions: Discrete keyboard/mouse actions
    - Rewards: Scalar rewards per timestep
    """
    
    def __init__(
        self,
        data_path: str,
        sequence_length: int = 16,
        image_size: Tuple[int, int] = (64, 64),
        frame_skip: int = 1,
        split: str = "train",
        transform: Optional[callable] = None,
        max_episodes: Optional[int] = None,
        use_multi_discrete: bool = False,  # If True, use multi-discrete format (keyboard binary + camera categorical)
    ):
        """
        Args:
            data_path: Path to dataset directory
            sequence_length: Number of frames per sequence
            image_size: (height, width) to resize images
            frame_skip: Number of frames to skip between samples
            split: "train" or "val"
            transform: Optional transform to apply to images
            max_episodes: Maximum number of episodes to load (None = all)
        """
        self.data_path = Path(data_path)
        self.sequence_length = sequence_length
        self.image_size = image_size
        self.frame_skip = frame_skip
        self.split = split
        self.transform = transform
        self.max_episodes = max_episodes
        self.use_multi_discrete = use_multi_discrete
        
        # Load episode metadata
        self.episodes = self._load_episodes()
        
        # Create index mapping (episode_idx, start_frame)
        self.samples = self._create_sample_index()
    
    def _load_episodes(self) -> List[Dict]:
        """Load episode metadata from data directory."""
        episodes = []
        
        # Look for .pt files (trajectory data) at root level
        pt_files = sorted(self.data_path.glob("*.pt"))
        
        if pt_files:
            # Load from .pt files
            for pt_file in pt_files:
                try:
                    data = torch.load(pt_file, map_location="cpu", weights_only=False)
                    episodes.append({
                        "path": pt_file,
                        "data": data,
                        "length": len(data) if hasattr(data, "__len__") else data.shape[0],
                    })
                except Exception as e:
                    print(f"Warning: Could not load {pt_file}: {e}")
        
        # Look for episode directories - support nested structure (environment/episode)
        # First, try direct episode directories
        episode_dirs = sorted([d for d in self.data_path.iterdir() if d.is_dir()])
        
        # Also search recursively for directories containing frames.npy (nested structure)
        frames_npy_files = sorted(self.data_path.rglob("frames.npy"))
        
        # Collect unique episode directories
        episode_dirs_set = set()
        for frames_file in frames_npy_files:
            episode_dirs_set.add(frames_file.parent)
        
        # Also add direct episode directories that have frames
        for ep_dir in episode_dirs:
            frames = sorted(ep_dir.glob("*.png")) + sorted(ep_dir.glob("*.jpg"))
            frames_npy = ep_dir / "frames.npy"
            if frames or frames_npy.exists():
                episode_dirs_set.add(ep_dir)
        
        # Load episodes from all found directories
        for ep_dir in sorted(episode_dirs_set):
            # Check for frames.npy (preferred format)
            frames_npy = ep_dir / "frames.npy"
            action_file = ep_dir / "actions.npy"
            reward_file = ep_dir / "rewards.npy"
            
            if frames_npy.exists():
                # Load from frames.npy
                try:
                    frames_array = np.load(frames_npy)
                    num_frames = frames_array.shape[0]
                    
                    episodes.append({
                        "path": ep_dir,
                        "frames_npy": frames_npy,
                        "actions": action_file if action_file.exists() else None,
                        "rewards": reward_file if reward_file.exists() else None,
                        "length": num_frames,
                    })
                except Exception as e:
                    print(f"Warning: Could not load {frames_npy}: {e}")
            
            else:
                # Fallback: Check for image files
                frames = sorted(ep_dir.glob("*.png")) + sorted(ep_dir.glob("*.jpg"))
                if frames:
                    episodes.append({
                        "path": ep_dir,
                        "frames": frames,
                        "actions": np.load(action_file) if action_file.exists() else None,
                        "rewards": np.load(reward_file) if reward_file.exists() else None,
                        "length": len(frames),
                    })
        
        # Limit number of episodes if specified
        if self.max_episodes is not None and len(episodes) > self.max_episodes:
            print(f"Limiting to {self.max_episodes} episodes (found {len(episodes)})")
            episodes = episodes[:self.max_episodes]
        
        print(f"Loaded {len(episodes)} episodes from {len(episode_dirs_set)} directories")
        return episodes
    
    def _create_sample_index(self) -> List[Tuple[int, int]]:
        """Create index of (episode_idx, start_frame) pairs."""
        samples = []
        
        for ep_idx, episode in enumerate(self.episodes):
            ep_length = episode["length"]
            
            # Calculate valid starting positions
            max_start = ep_length - self.sequence_length * self.frame_skip
            
            for start in range(0, max(1, max_start), self.sequence_length // 2):
                samples.append((ep_idx, start))
        
        return samples
    
    def _load_frame(self, episode: Dict, frame_idx: int) -> torch.Tensor:
        """Load a single frame from an episode."""
        if "data" in episode:
            # Data is stored in .pt file
            data = episode["data"]
            
            if hasattr(data, "__getitem__"):
                # Check bounds
                if frame_idx < len(data):
                    item = data[frame_idx]
                    
                    if isinstance(item, dict):
                        # TensorDict format
                        if "obs" in item:
                            obs = item["obs"]
                            if obs.dim() == 1:
                                # State observation, need to handle differently
                                return obs
                            return obs
                        elif "image" in item:
                            return item["image"]
                    
                    return item
        
        elif "frames_npy" in episode:
            # Load from frames.npy file
            frames_array = np.load(episode["frames_npy"])
            # Clamp frame_idx to frames array bounds
            frame_idx = min(frame_idx, len(frames_array) - 1) if len(frames_array) > 0 else 0
            frame = frames_array[frame_idx]  # (H, W, C) uint8
            
            # Convert to tensor and normalize
            frame_tensor = torch.from_numpy(frame).permute(2, 0, 1)  # (C, H, W)
            frame_tensor = frame_tensor.float() / 255.0  # Normalize to [0, 1]
            
            # Resize if needed
            if frame_tensor.shape[1:] != self.image_size:
                import torchvision.transforms.functional as TF
                frame_tensor = TF.resize(frame_tensor.unsqueeze(0), self.image_size).squeeze(0)
            
            return frame_tensor
        
        elif "frames" in episode:
            # Load from image file
            from PIL import Image
            import torchvision.transforms.functional as TF
            
            # Clamp frame_idx to frames list bounds
            frames_list = episode["frames"]
            frame_idx = min(frame_idx, len(frames_list) - 1) if len(frames_list) > 0 else 0
            frame_path = frames_list[frame_idx]
            image = Image.open(frame_path).convert("RGB")
            image = TF.resize(image, self.image_size)
            image = TF.to_tensor(image)
            return image
        
        raise ValueError("Unknown episode format")
    
    def _load_action(self, episode: Dict, frame_idx: int, use_multi_discrete: bool = False) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Load action for a frame.
        
        Args:
            episode: Episode dictionary
            frame_idx: Frame index
            use_multi_discrete: If True, return multi-discrete format (keyboard binary + camera categorical)
        
        Returns:
            If use_multi_discrete: Dict with 'keyboard' (8,) and 'camera' (1,)
            Else: Single categorical action tensor
        """
        if "data" in episode:
            data = episode["data"]
            if hasattr(data, "__getitem__"):
                # Check bounds
                if frame_idx < len(data):
                    item = data[frame_idx]
                    if isinstance(item, dict) and "action" in item:
                        return item["action"]
        
        # Load from actions.npz file
        if "actions" in episode:
            if isinstance(episode["actions"], Path):
                # Path to actions.npz file
                actions_dict = np.load(episode["actions"])
                
                if use_multi_discrete:
                    # Use multi-discrete format (keyboard binary + camera categorical)
                    from .action_utils import mineRL_actions_to_multi_discrete
                    
                    try:
                        multi_discrete = mineRL_actions_to_multi_discrete(actions_dict)
                        # Clamp frame_idx to action array bounds
                        frame_idx = min(frame_idx, len(multi_discrete["keyboard"]) - 1) if len(multi_discrete["keyboard"]) > 0 else 0
                        return {
                            "keyboard": torch.from_numpy(multi_discrete["keyboard"][frame_idx]).long(),
                            "camera": torch.tensor(multi_discrete["camera"][frame_idx], dtype=torch.long),
                        }
                    except Exception as e:
                        print(f"Warning: Could not convert to multi-discrete, using fallback: {e}")
                        # Fallback to default
                        return {
                            "keyboard": torch.zeros(8, dtype=torch.long),
                            "camera": torch.tensor(60, dtype=torch.long),  # Center bin
                        }
                else:
                    # Combine all action components into single discrete action (backward compatibility)
                    from .action_utils import mineRL_actions_to_categorical
                    
                    try:
                        # Convert to categorical action space
                        categorical_actions = mineRL_actions_to_categorical(actions_dict)
                        # Clamp frame_idx to action array bounds
                        frame_idx = min(frame_idx, len(categorical_actions) - 1) if len(categorical_actions) > 0 else 0
                        action_val = categorical_actions[frame_idx]
                        return torch.tensor(action_val, dtype=torch.long)
                    except Exception as e:
                        # Fallback: use first action component if combination fails
                        print(f"Warning: Could not combine actions, using fallback: {e}")
                        if len(actions_dict.keys()) > 0:
                            first_key = list(actions_dict.keys())[0]
                            action_array = actions_dict[first_key]
                            frame_idx = min(frame_idx, len(action_array) - 1) if len(action_array) > 0 else 0
                            action_val = action_array[frame_idx]
                            return torch.tensor(action_val, dtype=torch.long)
            elif episode["actions"] is not None:
                # Already loaded numpy array
                actions = episode["actions"]
                # Clamp frame_idx to actions array bounds
                frame_idx = min(frame_idx, len(actions) - 1) if len(actions) > 0 else 0
                return torch.from_numpy(actions[frame_idx])
        
        # Default action (no-op)
        if use_multi_discrete:
            return {
                "keyboard": torch.zeros(8, dtype=torch.long),
                "camera": torch.tensor(60, dtype=torch.long),  # Center bin
            }
        return torch.zeros(1, dtype=torch.long)
    
    def _load_reward(self, episode: Dict, frame_idx: int) -> torch.Tensor:
        """Load reward for a frame."""
        if "data" in episode:
            data = episode["data"]
            if hasattr(data, "__getitem__"):
                # Check bounds
                if frame_idx < len(data):
                    item = data[frame_idx]
                    if isinstance(item, dict) and "reward" in item:
                        return item["reward"]
        
        # Load from rewards.npy file
        if "rewards" in episode:
            if isinstance(episode["rewards"], Path):
                # Path to rewards.npy file
                rewards = np.load(episode["rewards"])
                # Clamp frame_idx to rewards array bounds
                frame_idx = min(frame_idx, len(rewards) - 1) if len(rewards) > 0 else 0
                return torch.tensor(rewards[frame_idx], dtype=torch.float32)
            elif episode["rewards"] is not None:
                # Already loaded numpy array
                rewards = episode["rewards"]
                # Clamp frame_idx to rewards array bounds
                frame_idx = min(frame_idx, len(rewards) - 1) if len(rewards) > 0 else 0
                return torch.tensor(rewards[frame_idx], dtype=torch.float32)
        
        return torch.tensor(0.0)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sequence of frames with actions and rewards.
        
        Returns:
            Dictionary containing:
                - frames: (T, C, H, W) or (T, obs_dim) for state observations
                - actions: (T,) or (T, action_dim)
                - rewards: (T,)
        """
        ep_idx, start_frame = self.samples[idx]
        episode = self.episodes[ep_idx]
        
        frames = []
        actions = []
        rewards = []
        
        for i in range(self.sequence_length):
            frame_idx = start_frame + i * self.frame_skip
            frame_idx = min(frame_idx, episode["length"] - 1)  # Clamp
            
            frame = self._load_frame(episode, frame_idx)
            action = self._load_action(episode, frame_idx, use_multi_discrete=self.use_multi_discrete)
            reward = self._load_reward(episode, frame_idx)
            
            frames.append(frame)
            actions.append(action)
            rewards.append(reward)
        
        # Stack into tensors
        frames = torch.stack(frames, dim=0)  # (T, ...)
        
        # Handle actions - could be discrete, continuous, or multi-discrete
        if isinstance(actions[0], dict):
            # Multi-discrete format
            keyboard_actions = torch.stack([a["keyboard"] for a in actions], dim=0)  # (T, 8)
            camera_actions = torch.stack([a["camera"] for a in actions], dim=0)  # (T,)
            actions = {
                "keyboard": keyboard_actions,
                "camera": camera_actions,
            }
        elif actions[0].dim() == 0:
            actions = torch.stack(actions, dim=0)  # (T,)
        else:
            actions = torch.stack(actions, dim=0)  # (T, action_dim)
        
        rewards = torch.stack(rewards, dim=0)  # (T,)
        
        # Apply transforms if provided
        if self.transform is not None:
            frames = self.transform(frames)
        
        # Normalize frames to [0, 1] if they're images
        if frames.dim() == 4 and frames.max() > 1:
            frames = frames.float() / 255.0
        
        return {
            "frames": frames,
            "actions": actions,
            "rewards": rewards,
        }


def create_dataloader(
    data_path: str,
    batch_size: int = 16,
    sequence_length: int = 16,
    image_size: Tuple[int, int] = (64, 64),
    num_workers: int = 4,
    split: str = "train",
    use_multi_discrete: bool = False,
    **kwargs,
) -> DataLoader:
    """
    Create a DataLoader for MineRL data.
    
    Args:
        data_path: Path to dataset
        batch_size: Batch size
        sequence_length: Frames per sequence
        image_size: (height, width)
        num_workers: Number of data loading workers
        split: "train" or "val"
    
    Returns:
        DataLoader instance
    """
    dataset = MineRLDataset(
        data_path=data_path,
        sequence_length=sequence_length,
        image_size=image_size,
        split=split,
        use_multi_discrete=use_multi_discrete,
        **kwargs,
    )
    
    # Disable persistent_workers to avoid hanging issues
    # Can re-enable if data loading is stable
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=1 if num_workers > 0 else None,  # Reduced from 2 to 1 for faster first batch
        persistent_workers=False,  # Disabled to avoid hanging
        timeout=30,  # Add timeout to prevent hanging
    )
