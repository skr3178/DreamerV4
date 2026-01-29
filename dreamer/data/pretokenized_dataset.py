"""
Pretokenized Dataset Loader for DreamerV4

Loads pre-encoded latents from Cosmos tokenizer for faster training.
Avoids on-the-fly tokenization overhead (2-3× speedup).

Expected format:
    pretokenized_data/
    ├── episode_00000/
    │   ├── latents.pt      # (T_lat, 16, 16) float16
    │   ├── actions.pt      # (T,) or dict
    │   ├── rewards.pt      # (T,)
    │   └── info.pt         # {num_frames, num_latent_steps}
    ├── episode_00001/
    │   └── ...
    └── metadata.pt         # {num_episodes, pool_tokens, latent_dim, ...}
"""

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader


class PretokenizedDataset(Dataset):
    """
    Dataset for loading pretokenized Cosmos latents.

    Provides sequences of latents with aligned actions and rewards.
    Much faster than on-the-fly tokenization.
    """

    def __init__(
        self,
        data_path: str,
        sequence_length: int = 5,  # In latent steps (not frames!)
        split: str = "train",
        max_episodes: Optional[int] = None,
        use_multi_discrete: bool = False,
    ):
        """
        Args:
            data_path: Path to pretokenized dataset directory
            sequence_length: Number of latent steps per sequence (default: 5 for 32-frame equivalent)
            split: "train" or "val"
            max_episodes: Maximum episodes to load (None = all)
            use_multi_discrete: If True, expect multi-discrete action format
        """
        self.data_path = Path(data_path)
        self.sequence_length = sequence_length
        self.split = split
        self.use_multi_discrete = use_multi_discrete

        # Load metadata
        metadata_path = self.data_path / "metadata.pt"
        if metadata_path.exists():
            self.metadata = torch.load(metadata_path, map_location="cpu", weights_only=False)
        else:
            self.metadata = {}

        # Verify dimensions
        self.pool_tokens = self.metadata.get("pool_tokens", 16)
        self.latent_dim = self.metadata.get("latent_dim", 16)

        # Find and load episodes
        self.episodes = self._load_episodes(max_episodes)

        # Create sample index
        self.samples = self._create_sample_index()

        print(f"PretokenizedDataset loaded:")
        print(f"  Episodes: {len(self.episodes)}")
        print(f"  Samples: {len(self.samples)}")
        print(f"  Sequence length: {sequence_length} latent steps")
        print(f"  Latent shape: ({self.pool_tokens}, {self.latent_dim})")

    def _load_episodes(self, max_episodes: Optional[int]) -> List[Dict]:
        """Load all episode data into memory."""
        episodes = []

        # Find episode directories
        episode_dirs = sorted(
            [d for d in self.data_path.iterdir() if d.is_dir() and d.name.startswith("episode_")]
        )

        if max_episodes:
            episode_dirs = episode_dirs[:max_episodes]

        for ep_dir in episode_dirs:
            latents_path = ep_dir / "latents.pt"
            if not latents_path.exists():
                continue

            try:
                # Load latents (float16 -> float32)
                latents = torch.load(latents_path, map_location="cpu", weights_only=False)
                latents = latents.float()  # Convert to float32 for training

                # Load actions
                actions_path = ep_dir / "actions.pt"
                if actions_path.exists():
                    actions = torch.load(actions_path, map_location="cpu", weights_only=False)
                else:
                    actions = torch.zeros(latents.shape[0], dtype=torch.long)

                # Load rewards
                rewards_path = ep_dir / "rewards.pt"
                if rewards_path.exists():
                    rewards = torch.load(rewards_path, map_location="cpu", weights_only=False)
                else:
                    rewards = torch.zeros(latents.shape[0], dtype=torch.float32)

                # Load info
                info_path = ep_dir / "info.pt"
                if info_path.exists():
                    info = torch.load(info_path, map_location="cpu", weights_only=False)
                else:
                    info = {"num_frames": latents.shape[0] * 8, "num_latent_steps": latents.shape[0]}

                # Align actions/rewards to latent steps
                actions, rewards = self._align_to_latent_steps(
                    actions, rewards, latents.shape[0], info.get("num_frames", latents.shape[0] * 8)
                )

                episodes.append({
                    "latents": latents,
                    "actions": actions,
                    "rewards": rewards,
                    "length": latents.shape[0],
                })

            except Exception as e:
                print(f"Warning: Failed to load {ep_dir}: {e}")
                continue

        return episodes

    def _align_to_latent_steps(
        self,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        num_latent_steps: int,
        num_frames: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Align frame-level actions/rewards to latent steps.

        Uses linspace sampling to evenly distribute across latent steps.
        """
        # Handle dict actions (multi-discrete)
        if isinstance(actions, dict):
            aligned_actions = {}
            for key, val in actions.items():
                if val.shape[0] > num_latent_steps:
                    indices = torch.linspace(0, val.shape[0] - 1, num_latent_steps).long()
                    aligned_actions[key] = val[indices]
                else:
                    aligned_actions[key] = val
            actions = aligned_actions
        else:
            # Handle tensor actions
            if actions.shape[0] > num_latent_steps:
                indices = torch.linspace(0, actions.shape[0] - 1, num_latent_steps).long()
                actions = actions[indices]

        # Align rewards (sum within each latent step window)
        if rewards.shape[0] > num_latent_steps:
            # Sum rewards within each latent step's temporal window
            aligned_rewards = torch.zeros(num_latent_steps, dtype=rewards.dtype)
            frames_per_step = num_frames / num_latent_steps

            for i in range(num_latent_steps):
                start = int(i * frames_per_step)
                end = int((i + 1) * frames_per_step)
                aligned_rewards[i] = rewards[start:end].sum()

            rewards = aligned_rewards

        return actions, rewards

    def _create_sample_index(self) -> List[Tuple[int, int]]:
        """Create index of (episode_idx, start_step) tuples."""
        samples = []

        for ep_idx, episode in enumerate(self.episodes):
            length = episode["length"]

            # Create samples with stride 1
            for start in range(length - self.sequence_length + 1):
                samples.append((ep_idx, start))

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ep_idx, start = self.samples[idx]
        episode = self.episodes[ep_idx]

        end = start + self.sequence_length

        # Get latent sequence
        latents = episode["latents"][start:end]  # (seq_len, 16, 16)

        # Get aligned actions
        actions = episode["actions"]
        if isinstance(actions, dict):
            seq_actions = {k: v[start:end] for k, v in actions.items()}
        else:
            seq_actions = actions[start:end]

        # Get aligned rewards
        rewards = episode["rewards"][start:end]

        result = {
            "latents": latents,
            "actions": seq_actions,
            "rewards": rewards,
        }

        return result


def create_pretokenized_dataloader(
    data_path: str,
    batch_size: int = 16,
    sequence_length: int = 5,
    num_workers: int = 4,
    split: str = "train",
    max_episodes: Optional[int] = None,
    use_multi_discrete: bool = False,
    pin_memory: bool = True,
) -> DataLoader:
    """
    Create DataLoader for pretokenized dataset.

    Args:
        data_path: Path to pretokenized dataset
        batch_size: Batch size
        sequence_length: Number of latent steps per sequence
        num_workers: Number of data loading workers
        split: "train" or "val"
        max_episodes: Maximum episodes to load
        use_multi_discrete: If True, use multi-discrete action format
        pin_memory: Pin memory for faster GPU transfer

    Returns:
        DataLoader instance
    """
    dataset = PretokenizedDataset(
        data_path=data_path,
        sequence_length=sequence_length,
        split=split,
        max_episodes=max_episodes,
        use_multi_discrete=use_multi_discrete,
    )

    # Custom collate for dict actions
    def collate_fn(batch):
        latents = torch.stack([b["latents"] for b in batch])
        rewards = torch.stack([b["rewards"] for b in batch])

        # Handle actions
        first_actions = batch[0]["actions"]
        if isinstance(first_actions, dict):
            actions = {
                k: torch.stack([b["actions"][k] for b in batch])
                for k in first_actions.keys()
            }
        else:
            actions = torch.stack([b["actions"] for b in batch])

        return {
            "latents": latents,
            "actions": actions,
            "rewards": rewards,
        }

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        drop_last=True,
    )
