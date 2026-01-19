# MineRL Dataset Extraction

## Overview

The `extract_minerl_frames.py` script extracts POV (point-of-view) frames from MineRL dataset video files and prepares them for DreamerV4 training.

## Dataset Inspection Results

Based on inspection of the MineRL datasets:

### Action Space
- **8 binary keyboard actions**: forward, left, back, right, jump, sneak, sprint, attack
- **Camera**: continuous 2D (pitch, yaw) - will be discretized to 121 classes (11×11 bins)
- **Additional discrete actions**: place, equip, craft, nearbyCraft, nearbySmelt (varies by environment)
- **Total for DreamerV4**: ~144 discrete actions (combined into single categorical space)

### Observation Space
- **POV images**: 64×64×3 RGB (stored in .mp4 video files)
- **Additional observations**: inventory items, compass angle, equipped items (in .npz files)

### Reward
- Scalar float32 per timestep
- Range varies by environment (0-1024 for ObtainDiamond, 0-100 for Navigate)

## Usage

### Basic Extraction

Extract all episodes from all environments:

```bash
conda activate dreamerV4
python extract_minerl_frames.py \
    --input-dir /media/skr/storage/dreamerv4/data/mineRL \
    --output-dir /media/skr/storage/dreamerv4/data/mineRL_extracted \
    --target-size 64 64 \
    --target-fps 20
```

### Extract Specific Environments

```bash
python extract_minerl_frames.py \
    --input-dir /media/skr/storage/dreamerv4/data/mineRL \
    --output-dir /media/skr/storage/dreamerv4/data/mineRL_extracted \
    --environments MineRLObtainDiamond-v0 MineRLNavigate-v0 \
    --target-size 64 64
```

### Extract Limited Episodes (for testing)

```bash
python extract_minerl_frames.py \
    --input-dir /media/skr/storage/dreamerv4/data/mineRL \
    --output-dir /media/skr/storage/dreamerv4/data/mineRL_extracted \
    --max-episodes 10 \
    --target-size 64 64
```

## Output Structure

After extraction, the output directory will have the following structure:

```
mineRL_extracted/
├── MineRLObtainDiamond-v0/
│   ├── episode_1/
│   │   ├── frames.npy          # (T, 64, 64, 3) RGB frames
│   │   ├── actions.npz          # Dictionary of action arrays
│   │   ├── rewards.npy          # (T,) reward array
│   │   ├── observations.npz     # Dictionary of observation arrays
│   │   └── metadata.json        # Episode metadata
│   └── episode_2/
│       └── ...
├── MineRLNavigate-v0/
│   └── ...
└── ...
```

## Configuration

The config file `configs/minerl.yaml` has been updated with:

- **Image size**: 64×64×3 (standard MineRL resolution)
- **Action space**: 144 discrete actions
- **Data path**: `data/mineRL_extracted` (after extraction)

## Notes

1. **Frame extraction**: Frames are extracted from .mp4 video files using OpenCV
2. **Action format**: Actions are stored as separate arrays in .npz format
3. **Memory**: Each episode's frames are saved as numpy arrays (uint8, 0-255)
4. **Processing time**: Extraction can take a while for large datasets (several GB)

## Dependencies

- `opencv-python` (for video processing)
- `numpy`
- `PIL` (Pillow)
- `tqdm` (optional, for progress bars)

Install with:
```bash
conda activate dreamerV4
pip install opencv-python tqdm
```
