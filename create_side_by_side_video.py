#!/usr/bin/env python3
"""Create a side-by-side video: original | reconstructed. Same FPS as original."""
import argparse
from pathlib import Path

import numpy as np

try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False

try:
    import cv2
except ImportError:
    cv2 = None


def read_video(path: Path):
    if HAS_IMAGEIO:
        reader = imageio.get_reader(str(path))
        frames = [np.asarray(f) for f in reader]
        fps = reader.get_meta_data().get("fps", 20.0)
        reader.close()
        return frames, fps
    if cv2 is not None:
        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        return frames, fps
    raise RuntimeError("Need imageio or opencv-python")


def write_video(frames, path: Path, fps: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        raise ValueError("No frames")
    H, W = frames[0].shape[:2]
    if HAS_IMAGEIO:
        writer = imageio.get_writer(str(path), fps=fps, codec="libx264", quality=8, pixelformat="yuv420p")
        for f in frames:
            writer.append_data(np.asarray(f))
        writer.close()
    elif cv2 is not None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(path), fourcc, fps, (W, H))
        for f in frames:
            out.write(cv2.cvtColor(np.asarray(f), cv2.COLOR_RGB2BGR))
        out.release()
    else:
        raise RuntimeError("Need imageio or opencv-python")
    print(f"Saved: {path} ({len(frames)} frames, {fps:.2f} FPS)")


def main():
    parser = argparse.ArgumentParser(description="Side-by-side video: original | reconstructed")
    parser.add_argument("--original", type=str, default=None, help="Path to original.mp4")
    parser.add_argument("--reconstructed", type=str, default=None, help="Path to reconstructed.mp4")
    parser.add_argument("--output", type=str, default=None, help="Output path (default: same dir as original, side_by_side.mp4)")
    parser.add_argument("--dir", type=str, default=None, dest="pretokenized_dir", help="Pretokenized root; create side_by_side.mp4 for each episode_*/reconstructed/")
    args = parser.parse_args()

    if args.pretokenized_dir:
        root = Path(args.pretokenized_dir)
        if not root.is_dir():
            raise FileNotFoundError(root)
        recon_dirs = sorted(root.glob("episode_*/reconstructed"))
        for rec_dir in recon_dirs:
            orig_path = rec_dir / "original.mp4"
            recon_path = rec_dir / "reconstructed.mp4"
            if not orig_path.exists() or not recon_path.exists():
                print(f"Skipping {rec_dir}: missing original.mp4 or reconstructed.mp4")
                continue
            print(f"[{rec_dir.parent.name}] Creating side_by_side.mp4...")
            orig_frames, orig_fps = read_video(orig_path)
            recon_frames, _ = read_video(recon_path)
            n_orig = len(orig_frames)
            n_recon = len(recon_frames)
            n_out = n_orig
            out_frames = []
            for i in range(n_out):
                j = round(i * (n_recon - 1) / max(1, n_orig - 1)) if n_orig > 1 else 0
                j = min(j, n_recon - 1)
                o = np.asarray(orig_frames[i])
                r = np.asarray(recon_frames[j])
                if o.shape[0] != r.shape[0] or o.shape[1] != r.shape[1]:
                    from PIL import Image
                    r_pil = Image.fromarray(r)
                    r_pil = r_pil.resize((o.shape[1], o.shape[0]), Image.Resampling.NEAREST)
                    r = np.array(r_pil)
                side = np.concatenate([o, r], axis=1)
                out_frames.append(side)
            write_video(out_frames, rec_dir / "side_by_side.mp4", orig_fps)
        print("Done.")
        return

    orig_path = Path(args.original)
    recon_path = Path(args.reconstructed)
    if not orig_path or not recon_path or not orig_path.exists():
        raise FileNotFoundError(orig_path or "original")
    if not recon_path.exists():
        raise FileNotFoundError(recon_path)

    print("Reading original...")
    orig_frames, orig_fps = read_video(orig_path)
    print("Reading reconstructed...")
    recon_frames, _ = read_video(recon_path)

    n_orig = len(orig_frames)
    n_recon = len(recon_frames)
    n_out = n_orig
    out_frames = []
    for i in range(n_out):
        j = round(i * (n_recon - 1) / max(1, n_orig - 1)) if n_orig > 1 else 0
        j = min(j, n_recon - 1)
        o = np.asarray(orig_frames[i])
        r = np.asarray(recon_frames[j])
        if o.shape[0] != r.shape[0] or o.shape[1] != r.shape[1]:
            from PIL import Image
            r_pil = Image.fromarray(r)
            r_pil = r_pil.resize((o.shape[1], o.shape[0]), Image.Resampling.NEAREST)
            r = np.array(r_pil)
        side = np.concatenate([o, r], axis=1)
        out_frames.append(side)

    out_path = Path(args.output) if args.output else orig_path.parent / "side_by_side.mp4"
    print("Writing side-by-side...")
    write_video(out_frames, out_path, orig_fps)
    print("Done.")


if __name__ == "__main__":
    main()
