#!/usr/bin/env python3
"""
Pretokenize Dataset with Cosmos Tokenizer

Pre-encodes all video frames using the frozen Cosmos CV8x8x8 tokenizer
and saves the latents to disk. This provides 2-3× training speedup by
avoiding on-the-fly tokenization.

Output format:
    data/pretokenized/
    ├── episode_0000/
    │   ├── latents.pt      # (T_lat, 16, 16) float16
    │   ├── actions.pt      # (T,) or dict
    │   └── rewards.pt      # (T,)
    ├── episode_0001/
    │   └── ...
    └── metadata.pt         # {num_episodes, T_lat_formula, pool_tokens, ...}

Usage:
    python pretokenize_dataset.py --data-path data/mineRL_extracted --output-path data/pretokenized
    python pretokenize_dataset.py --data-path data/mineRL_extracted --output-path data/pretokenized --max-episodes 100
"""

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, Optional, List
from tqdm import tqdm

import torch
import numpy as np


def compute_temporal_latent_steps(num_frames: int) -> int:
    """
    Compute T_latent for causal Cosmos tokenizer.

    Formula: T_latent = 1 + ceil((T_frames - 1) / 8)
    """
    if num_frames <= 0:
        raise ValueError(f"num_frames must be positive, got {num_frames}")
    if num_frames == 1:
        return 1
    return 1 + math.ceil((num_frames - 1) / 8)


def load_cosmos_tokenizer(checkpoint_path: str, device: str = "cuda", pool_tokens: Optional[int] = None, input_resolution: int = 64):
    """Load the Cosmos tokenizer wrapper."""
    # Add project root to path
    project_root = Path(__file__).parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from dreamer.models.cosmos_tokenizer_wrapper import create_cosmos_tokenizer

    tokenizer = create_cosmos_tokenizer(
        checkpoint_path=checkpoint_path,
        pool_tokens=pool_tokens,      # None = no pooling (best quality)
        input_resolution=input_resolution,  # 64 = native MineRL (no upsampling)
        device=device,
        dtype="bfloat16",
    )
    return tokenizer


def load_episode_data(episode_path: Path) -> Optional[Dict[str, torch.Tensor]]:
    """
    Load episode data from various formats.

    Returns:
        Dict with 'frames', 'actions', 'rewards' or None if failed
    """
    try:
        # Format 1: Single .pt file with all data
        if episode_path.suffix == ".pt":
            data = torch.load(episode_path, map_location="cpu", weights_only=False)
            if isinstance(data, dict):
                return data
            # Handle list of transitions
            if isinstance(data, list):
                frames = torch.stack([d.get("frame", d.get("obs", d.get("observation"))) for d in data])
                actions = torch.stack([d.get("action", torch.tensor(0)) for d in data])
                rewards = torch.stack([d.get("reward", torch.tensor(0.0)) for d in data])
                return {"frames": frames, "actions": actions, "rewards": rewards}

        # Format 2: Directory with separate files
        if episode_path.is_dir():
            result = {}

            # Load frames
            frames_npy = episode_path / "frames.npy"
            frames_pt = episode_path / "frames.pt"

            if frames_npy.exists():
                frames = torch.from_numpy(np.load(frames_npy))
            elif frames_pt.exists():
                frames = torch.load(frames_pt, map_location="cpu", weights_only=False)
            else:
                # Try loading individual images
                import cv2
                png_files = sorted(episode_path.glob("*.png"))
                jpg_files = sorted(episode_path.glob("*.jpg"))
                img_files = png_files if png_files else jpg_files

                if not img_files:
                    return None

                frames_list = []
                for img_file in img_files:
                    img = cv2.imread(str(img_file))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    frames_list.append(torch.from_numpy(img).permute(2, 0, 1))
                frames = torch.stack(frames_list)

            result["frames"] = frames

            # Load actions
            actions_npy = episode_path / "actions.npy"
            actions_pt = episode_path / "actions.pt"

            if actions_npy.exists():
                result["actions"] = torch.from_numpy(np.load(actions_npy, allow_pickle=True))
            elif actions_pt.exists():
                result["actions"] = torch.load(actions_pt, map_location="cpu", weights_only=False)
            else:
                # Default: zeros
                result["actions"] = torch.zeros(len(result["frames"]), dtype=torch.long)

            # Load rewards
            rewards_npy = episode_path / "rewards.npy"
            rewards_pt = episode_path / "rewards.pt"

            if rewards_npy.exists():
                result["rewards"] = torch.from_numpy(np.load(rewards_npy))
            elif rewards_pt.exists():
                result["rewards"] = torch.load(rewards_pt, map_location="cpu", weights_only=False)
            else:
                # Default: zeros
                result["rewards"] = torch.zeros(len(result["frames"]), dtype=torch.float32)

            return result

        return None

    except Exception as e:
        print(f"Warning: Failed to load {episode_path}: {e}")
        return None


def find_episodes(data_path: Path) -> List[Path]:
    """Find all episode paths in the data directory."""
    episodes = []

    # .pt files at root
    pt_files = sorted(data_path.glob("*.pt"))
    episodes.extend(pt_files)

    # Directories with frames.npy
    for frames_file in sorted(data_path.rglob("frames.npy")):
        episodes.append(frames_file.parent)

    # Directories with images
    for subdir in sorted(data_path.iterdir()):
        if subdir.is_dir() and subdir not in episodes:
            # Check if it has images
            if list(subdir.glob("*.png")) or list(subdir.glob("*.jpg")):
                episodes.append(subdir)

    # Remove duplicates while preserving order
    seen = set()
    unique_episodes = []
    for ep in episodes:
        ep_str = str(ep.resolve())
        if ep_str not in seen:
            seen.add(ep_str)
            unique_episodes.append(ep)

    return unique_episodes


def preprocess_frames(frames: torch.Tensor) -> torch.Tensor:
    """
    Preprocess frames for Cosmos tokenizer.

    Args:
        frames: (T, C, H, W) or (T, H, W, C) uint8 or float

    Returns:
        frames: (1, C, T, H, W) float32 in [0, 1]
    """
    # Handle channel-last format
    if frames.dim() == 4 and frames.shape[-1] in [1, 3]:
        frames = frames.permute(0, 3, 1, 2)

    # Convert to float [0, 1]
    if frames.dtype == torch.uint8:
        frames = frames.float() / 255.0
    elif frames.max() > 1.0:
        frames = frames.float() / 255.0
    else:
        frames = frames.float()

    # Add batch dimension and rearrange: (T, C, H, W) -> (1, C, T, H, W)
    frames = frames.permute(1, 0, 2, 3).unsqueeze(0)

    return frames


@torch.no_grad()
def tokenize_episode(
    tokenizer,
    frames: torch.Tensor,
    device: str = "cuda",
    chunk_size: int = 64,
) -> torch.Tensor:
    """
    Tokenize video frames with Cosmos.

    Args:
        tokenizer: CosmosTokenizerWrapper
        frames: (1, C, T, H, W) float32 in [0, 1]
        device: Device to use
        chunk_size: Max frames per chunk (for memory)

    Returns:
        latents: (T_lat, 16, 16) float16
    """
    frames = frames.to(device)
    T = frames.shape[2]

    if T <= chunk_size:
        # Process all at once
        output = tokenizer.encode(frames)
        latents = output["latents"].squeeze(0)  # (T_lat, 16, 16)
    else:
        # Process in chunks with overlap for continuity
        latents_chunks = []
        overlap = 8  # Overlap frames for smooth transitions

        start = 0
        while start < T:
            end = min(start + chunk_size, T)
            chunk = frames[:, :, start:end]

            output = tokenizer.encode(chunk)
            chunk_latents = output["latents"].squeeze(0)  # (T_lat_chunk, 16, 16)

            # Skip overlapping latents except for first chunk
            if start > 0:
                # Skip first latent steps (corresponds to overlap region)
                overlap_latent_steps = compute_temporal_latent_steps(overlap) - 1
                chunk_latents = chunk_latents[overlap_latent_steps:]

            latents_chunks.append(chunk_latents)
            start = end - overlap if end < T else T

        latents = torch.cat(latents_chunks, dim=0)

    # Convert to float16 for storage efficiency
    return latents.half().cpu()


def pretokenize_dataset(
    data_path: str,
    output_path: str,
    cosmos_checkpoint: str = "cosmos_tokenizer/CV8x8x8",
    device: str = "cuda",
    max_episodes: Optional[int] = None,
    chunk_size: int = 64,
):
    """
    Pretokenize entire dataset with Cosmos tokenizer.

    Args:
        data_path: Path to input dataset
        output_path: Path to save pretokenized data
        cosmos_checkpoint: Path to Cosmos tokenizer checkpoint
        device: Device to use for encoding
        max_episodes: Maximum episodes to process (None = all)
        chunk_size: Max frames per encoding chunk
    """
    data_path = Path(data_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading Cosmos tokenizer from {cosmos_checkpoint}...")
    tokenizer = load_cosmos_tokenizer(cosmos_checkpoint, device)
    print(f"  Pool tokens: {tokenizer.pool_tokens}")
    print(f"  Latent dim: {tokenizer.latent_dim}")

    print(f"\nFinding episodes in {data_path}...")
    episodes = find_episodes(data_path)
    print(f"  Found {len(episodes)} episodes")

    if max_episodes:
        episodes = episodes[:max_episodes]
        print(f"  Processing first {max_episodes} episodes")

    # Track statistics
    stats = {
        "total_frames": 0,
        "total_latent_steps": 0,
        "num_episodes": 0,
        "failed_episodes": 0,
    }

    print(f"\nPretokenizing to {output_path}...")

    for idx, episode_path in enumerate(tqdm(episodes, desc="Tokenizing")):
        episode_name = f"episode_{idx:05d}"
        episode_output_dir = output_path / episode_name

        # Skip if already processed
        if (episode_output_dir / "latents.pt").exists():
            continue

        # Load episode data
        data = load_episode_data(episode_path)
        if data is None or "frames" not in data:
            stats["failed_episodes"] += 1
            continue

        frames = data["frames"]
        T = frames.shape[0]

        # Preprocess frames
        frames_preprocessed = preprocess_frames(frames)

        # Tokenize
        try:
            latents = tokenize_episode(
                tokenizer,
                frames_preprocessed,
                device=device,
                chunk_size=chunk_size,
            )
        except Exception as e:
            print(f"\nWarning: Failed to tokenize episode {idx}: {e}")
            stats["failed_episodes"] += 1
            continue

        # Save outputs
        episode_output_dir.mkdir(parents=True, exist_ok=True)

        torch.save(latents, episode_output_dir / "latents.pt")
        torch.save(data["actions"], episode_output_dir / "actions.pt")
        torch.save(data["rewards"], episode_output_dir / "rewards.pt")

        # Save original frame count for action alignment
        torch.save({"num_frames": T, "num_latent_steps": latents.shape[0]},
                   episode_output_dir / "info.pt")

        # Update stats
        stats["total_frames"] += T
        stats["total_latent_steps"] += latents.shape[0]
        stats["num_episodes"] += 1

        # Clear GPU cache periodically
        if idx % 100 == 0:
            torch.cuda.empty_cache()

    # Save metadata
    metadata = {
        "num_episodes": stats["num_episodes"],
        "total_frames": stats["total_frames"],
        "total_latent_steps": stats["total_latent_steps"],
        "pool_tokens": tokenizer.pool_tokens,
        "latent_dim": tokenizer.latent_dim,
        "temporal_formula": "T_lat = 1 + ceil((T - 1) / 8)",
        "failed_episodes": stats["failed_episodes"],
        "source_path": str(data_path),
    }
    torch.save(metadata, output_path / "metadata.pt")

    # Print summary
    print(f"\n{'='*60}")
    print("Pretokenization Complete!")
    print(f"{'='*60}")
    print(f"  Episodes processed: {stats['num_episodes']}")
    print(f"  Failed episodes: {stats['failed_episodes']}")
    print(f"  Total frames: {stats['total_frames']:,}")
    print(f"  Total latent steps: {stats['total_latent_steps']:,}")
    print(f"  Compression ratio: {stats['total_frames'] / max(1, stats['total_latent_steps']):.2f}x")
    print(f"  Output path: {output_path}")

    # Estimate storage
    latent_size_mb = stats["total_latent_steps"] * 16 * 16 * 2 / 1024 / 1024  # float16
    print(f"  Estimated latent storage: {latent_size_mb:.1f} MB")


def main():
    parser = argparse.ArgumentParser(
        description="Pretokenize dataset with Cosmos tokenizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Tokenize full dataset
    python pretokenize_dataset.py --data-path data/mineRL_extracted --output-path data/pretokenized

    # Tokenize subset for testing
    python pretokenize_dataset.py --data-path data/mineRL_extracted --output-path data/pretokenized_test --max-episodes 10

    # Use custom Cosmos checkpoint
    python pretokenize_dataset.py --data-path data/mineRL_extracted --output-path data/pretokenized --cosmos-checkpoint /path/to/CV8x8x8
        """
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to input dataset directory",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Path to save pretokenized data",
    )
    parser.add_argument(
        "--cosmos-checkpoint",
        type=str,
        default="cosmos_tokenizer/CV8x8x8",
        help="Path to Cosmos tokenizer checkpoint",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for encoding",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Maximum episodes to process (default: all)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=64,
        help="Max frames per encoding chunk (for memory)",
    )

    args = parser.parse_args()

    pretokenize_dataset(
        data_path=args.data_path,
        output_path=args.output_path,
        cosmos_checkpoint=args.cosmos_checkpoint,
        device=args.device,
        max_episodes=args.max_episodes,
        chunk_size=args.chunk_size,
    )


if __name__ == "__main__":
    main()
