#!/usr/bin/env python3
"""
Test Cosmos latent encode → decode → compare with original.

Modes:
  1. Synthetic: create frames, encode, decode, compare (default).
  2. Pretokenized (single): --episode-dir data/pretokenized_subset/episode_00000
  3. Pretokenized (all): --pretokenized-dir data/pretokenized_subset  (reconstructed + original video per episode)

Usage:
    python test_latent_decode.py
    python test_latent_decode.py --save-dir data/latent_decode_test
    python test_latent_decode.py --episode-dir data/pretokenized_subset/episode_00000
    python test_latent_decode.py --pretokenized-dir data/pretokenized_subset
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

# Project root
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_tokenizer(checkpoint_path: str = "cosmos_tokenizer/CV8x8x8", device: str = "cuda", pool_tokens=None, input_resolution: int = 64):
    from dreamer.models.cosmos_tokenizer_wrapper import create_cosmos_tokenizer
    tokenizer = create_cosmos_tokenizer(
        checkpoint_path=checkpoint_path,
        pool_tokens=pool_tokens,      # None = no pooling (best quality)
        input_resolution=input_resolution,  # 64 = native MineRL (no upsampling)
        device=device,
        dtype="bfloat16",
    )
    return tokenizer


def get_test_frames(device: str, batch_size: int = 1, num_frames: int = 32, H: int = 64, W: int = 64):
    """Synthetic video in [0, 1], shape (B, C, T, H, W)."""
    B, C = batch_size, 3
    video = torch.rand(B, C, num_frames, H, W, device=device)
    return video


def compare_and_report(original: torch.Tensor, decoded: torch.Tensor, save_dir: Optional[Path]):
    """
    original: (B, C, T_orig, H, W) e.g. (1, 3, 32, 256, 256)
    decoded:  (B, C, T_dec, H, W) - Cosmos decoder returns (B, C, T, H, W), T may differ (e.g. 33)
    Both in [0, 1], same H,W. Compare first min(T_orig, T_dec) frames frame-by-frame.
    """
    B, C, T_orig, H, W = original.shape
    T_dec = decoded.shape[2]
    T_compare = min(T_orig, T_dec)
    orig_c = original[:, :, :T_compare, :, :]   # (B, C, T_compare, H, W)
    dec_c = decoded[:, :, :T_compare, :, :]     # (B, C, T_compare, H, W)

    mse_overall = (dec_c - orig_c).pow(2).mean().item()
    mae_overall = (dec_c - orig_c).abs().mean().item()
    mse_per_frame = (dec_c - orig_c).pow(2).mean(dim=(0, 1, 3, 4))  # (T_compare,)
    mae_per_frame = (dec_c - orig_c).abs().mean(dim=(0, 1, 3, 4))

    print("  Latent decode vs original (frame-by-frame, first {} frames):".format(T_compare))
    print(f"    MSE overall: {mse_overall:.6f}")
    print(f"    MAE overall: {mae_overall:.6f}")
    for t in [0, T_compare // 2, T_compare - 1]:
        if t < len(mse_per_frame):
            print(f"    frame {t}: MSE={mse_per_frame[t].item():.6f}, MAE={mae_per_frame[t].item():.6f}")
    return mse_overall, mae_overall


def _save_video(frames: torch.Tensor, output_path: Path, fps: float) -> None:
    """Write (B, C, T, H, W) or (T, C, H, W) to .mp4. Frames in [0, 1] or uint8 [0,255] (normalized)."""
    if frames.dtype == torch.uint8:
        frames = frames.float() / 255.0
    if frames.dim() == 5:
        frames = frames[0]  # (C, T, H, W)
        frames = frames.permute(1, 0, 2, 3)  # (T, C, H, W)
    T, C, H, W = frames.shape
    frames_np = (frames.float().clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy() * 255).astype("uint8")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio
        writer = imageio.get_writer(
            str(output_path),
            fps=fps,
            codec="libx264",
            quality=8,
            pixelformat="yuv420p",
        )
        for frame in frames_np:
            writer.append_data(frame)
        writer.close()
        print(f"  Saved video: {output_path} ({T} frames, {fps:.2f} FPS, {T/fps:.2f}s)")
    except Exception:
        import cv2
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (W, H))
        if out.isOpened():
            for frame in frames_np:
                out.write(frame[..., ::-1])
            out.release()
            print(f"  Saved video: {output_path} ({T} frames, {fps:.2f} FPS)")
        else:
            raise RuntimeError("Could not open video writer. Try: pip install imageio imageio-ffmpeg")
    return


def _decode_latents_chunked(tokenizer, latents: torch.Tensor, chunk_size: int = 200, device: str = "cuda") -> torch.Tensor:
    """Decode latents in chunks to avoid OOM; returns (B, C, T, H, W)."""
    B, T_lat, N, D = latents.shape
    if T_lat <= chunk_size:
        return tokenizer.decode(latents)
    parts = []
    for start in range(0, T_lat, chunk_size):
        end = min(start + chunk_size, T_lat)
        chunk = latents[:, start:end].to(device)
        with torch.no_grad():
            out = tokenizer.decode(chunk)
        parts.append(out.cpu())
        del chunk
        if device == "cuda":
            torch.cuda.empty_cache()
    decoded = torch.cat(parts, dim=2)
    del parts
    return decoded.to(device)


def run_pretokenized_episode(
    episode_dir: Path,
    tokenizer,
    save_dir: Optional[Path],
    device: str,
    fps_override: Optional[float] = None,
    decode_chunk_size: int = 200,
) -> None:
    """Load latents from episode_dir/latents.pt, decode, and save as video (same duration as original)."""
    latents_path = episode_dir / "latents.pt"
    if not latents_path.exists():
        raise FileNotFoundError(f"Latents not found: {latents_path}")

    latents = torch.load(latents_path, map_location=device, weights_only=False)
    # Pretokenized format: (T_lat, 16, 16), possibly float16
    if latents.dim() == 3:
        latents = latents.unsqueeze(0)  # (1, T_lat, 16, 16)
    latents = latents.float()
    T_lat = latents.shape[1]
    print(f"  Loaded latents shape: {latents.shape} (T_lat = {T_lat})")

    tokenizer._load_decoder_if_needed()
    if tokenizer._decoder is None:
        raise RuntimeError("Decoder not available; cannot decode pretokenized latents.")

    with torch.no_grad():
        if T_lat > decode_chunk_size:
            decoded = _decode_latents_chunked(tokenizer, latents, chunk_size=decode_chunk_size, device=device)
        else:
            decoded = tokenizer.decode(latents)
    print(f"  Decoded shape: {decoded.shape} (B, C, T, H, W)")

    B, C, T_dec, H, W = decoded.shape
    # FPS: match original video duration when info.pt has num_frames (e.g. MineRL extraction ~20 FPS)
    original_fps = 20.0
    if fps_override is not None:
        fps = fps_override
        print(f"  Using FPS: {fps:.2f} (from --fps)")
    else:
        info_path = episode_dir / "info.pt"
        if info_path.exists():
            info = torch.load(info_path, map_location="cpu", weights_only=False)
            num_original_frames = info.get("num_frames")
            if num_original_frames is not None and num_original_frames > 0:
                # Same duration as original: duration = num_original_frames / original_fps
                duration_sec = num_original_frames / original_fps
                fps = T_dec / duration_sec
                print(f"  Original: {num_original_frames} frames @ {original_fps} FPS ({duration_sec:.2f}s)")
                print(f"  Output FPS: {fps:.2f} (same duration)")
            else:
                fps = original_fps
                print(f"  Using FPS: {fps:.2f} (default)")
        else:
            fps = original_fps
            print(f"  Using FPS: {fps:.2f} (default, no info.pt)")

    if save_dir is None:
        save_dir = episode_dir / "reconstructed"
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    output_path = save_dir / "reconstructed.mp4"
    _save_video(decoded, output_path, fps)

    # Load and save original video from source dataset (same episode index)
    try:
        meta_path = episode_dir.parent / "metadata.pt"
        if meta_path.exists():
            meta = torch.load(meta_path, map_location="cpu", weights_only=False)
            source_path = meta.get("source_path")
            if source_path:
                # Episode index from dir name: episode_00000 -> 0
                ep_name = episode_dir.name
                if ep_name.startswith("episode_") and ep_name[8:].isdigit():
                    ep_idx = int(ep_name[8:])
                else:
                    ep_idx = 0
                from pretokenize_dataset import find_episodes, load_episode_data
                source_episodes = find_episodes(Path(source_path))
                if ep_idx < len(source_episodes):
                    data = load_episode_data(source_episodes[ep_idx])
                    if data is not None and "frames" in data:
                        orig_frames = data["frames"]
                        if orig_frames.dim() == 4 and orig_frames.shape[-1] in [1, 3]:
                            orig_frames = orig_frames.permute(0, 3, 1, 2)
                        if orig_frames.dtype == torch.uint8:
                            orig_frames = orig_frames.float() / 255.0
                        orig_path = save_dir / "original.mp4"
                        _save_video(orig_frames, orig_path, original_fps)
                        print(f"  Original video saved: {orig_path}")
                    else:
                        print("  Original frames not found in source episode; skipping original.mp4")
                else:
                    print(f"  Source has only {len(source_episodes)} episodes (need index {ep_idx}); skipping original.mp4")
            else:
                print("  metadata.pt has no source_path; skipping original.mp4")
        else:
            print("  No metadata.pt in pretokenized dir; skipping original.mp4")
    except Exception as e:
        print(f"  Could not load original video: {e}")


def main():
    parser = argparse.ArgumentParser(description="Test Cosmos encode → decode → compare")
    parser.add_argument("--checkpoint-path", type=str, default="cosmos_tokenizer/CV8x8x8")
    parser.add_argument("--save-dir", type=str, default=None, help="Save comparison or reconstructed images here")
    parser.add_argument("--episode-dir", type=str, default=None, help="Single pretokenized episode dir; decode and save reconstructed + original video")
    parser.add_argument("--pretokenized-dir", type=str, default=None, help="Pretokenized root (e.g. data/pretokenized_subset); process all episode_* dirs, save reconstructed.mp4 + original.mp4 per episode")
    parser.add_argument("--decode-chunk-size", type=int, default=200, help="Decode latents in chunks of this size to avoid OOM (default 200)")
    parser.add_argument("--fps", type=float, default=None, help="Output video FPS (default: match original duration using info.pt num_frames and 20 FPS)")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-frames", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        print("CUDA not available, using CPU")

    print("Loading Cosmos tokenizer (encoder + decoder if present)...")
    tokenizer = load_tokenizer(checkpoint_path=args.checkpoint_path, device=device)
    tokenizer._load_decoder_if_needed()
    has_decoder = tokenizer._decoder is not None
    print(f"  Decoder available: {has_decoder}")

    if args.pretokenized_dir:
        root = Path(args.pretokenized_dir)
        if not root.is_dir():
            raise FileNotFoundError(f"Pretokenized dir not found: {root}")
        episode_dirs = sorted(root.glob("episode_*"), key=lambda p: p.name)
        episode_dirs = [d for d in episode_dirs if d.is_dir() and (d / "latents.pt").exists()]
        print(f"Processing {len(episode_dirs)} episodes from {root}...")
        for i, episode_dir in enumerate(episode_dirs):
            print(f"\n[{i+1}/{len(episode_dirs)}] {episode_dir.name}")
            try:
                run_pretokenized_episode(
                    episode_dir,
                    tokenizer,
                    None,
                    device,
                    fps_override=args.fps,
                    decode_chunk_size=args.decode_chunk_size,
                )
            except Exception as e:
                print(f"  Error: {e}")
        print("\nDone.")
        return

    if args.episode_dir:
        episode_dir = Path(args.episode_dir)
        if not episode_dir.is_dir():
            raise FileNotFoundError(f"Episode dir not found: {episode_dir}")
        print(f"Decoding pretokenized latents from {episode_dir}...")
        run_pretokenized_episode(
            episode_dir,
            tokenizer,
            Path(args.save_dir) if args.save_dir else None,
            device,
            fps_override=args.fps,
            decode_chunk_size=args.decode_chunk_size,
        )
        print("Done.")
        return

    print("Creating test frames (B, C, T, H, W)...")
    video = get_test_frames(device, batch_size=args.batch_size, num_frames=args.num_frames)
    print(f"  Input shape: {video.shape}")

    print("Encode...")
    with torch.no_grad():
        out = tokenizer.encode(video)
    latents = out["latents"]
    T_lat = latents.shape[1]
    print(f"  Latent shape: {latents.shape} (T_lat = {T_lat})")

    if not has_decoder:
        print("No decoder available; skipping decode and comparison.")
        print("Place decoder.jit in the Cosmos checkpoint directory to test reconstruction.")
        return

    print("Decode...")
    with torch.no_grad():
        decoded = tokenizer.decode(latents)
    print(f"  Decoded shape: {decoded.shape} (B, C, T, H, W)")

    # Upsample original to 256x256 to match decoded spatial size
    B, C, T, H, W = video.shape
    video_256 = F.interpolate(
        video.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W),
        size=(256, 256),
        mode="bicubic",
        align_corners=False,
    ).reshape(B, T, C, 256, 256).permute(0, 2, 1, 3, 4)

    print("Compare decoded vs original (aligned timesteps)...")
    save_dir = Path(args.save_dir) if args.save_dir else None
    compare_and_report(video_256, decoded, save_dir)
    print("Done.")


if __name__ == "__main__":
    main()
