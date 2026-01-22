# MNIST Shortcut Model Testing

This folder contains tests for the shortcut model implementation on the MNIST dataset.

## Overview

The shortcut model code from `dreamer/losses/shortcut_loss.py` and `dreamer/models/dynamics.py` is tested here without modification. This test setup:

1. Uses MNIST sequences as a simple test case
2. References the core shortcut model code
3. Tests the shortcut forcing loss and dynamics model

## Files

- `dataset.py`: MNIST sequence dataset loader
- `test_shortcut.py`: Main test script that trains the shortcut model on MNIST
- `generate_images.py`: Script to load checkpoints and generate/compute images for comparison
- `README.md`: This file

## Usage

### Setup

Activate the conda environment:
```bash
conda activate dreamerV4
cd MNIST
```

### Training

```bash
python test_shortcut.py
```

This will:
- Train the dynamics model using shortcut forcing
- Save checkpoints to `checkpoints/` directory
- Evaluate on the test set

### Generate and Compare Images

After training, use the checkpoint to generate images and compare with originals:

```bash
# Auto-detect latest checkpoint
python generate_images.py --num_samples 10

# Or specify a checkpoint explicitly
python generate_images.py --checkpoint checkpoints/final_checkpoint.pt --num_samples 10
```

Options:
- `--checkpoint`: Path to checkpoint file (default: `checkpoints/final_checkpoint.pt`)
- `--num_samples`: Number of samples to generate (default: 10)
- `--output_dir`: Output directory for generated images (default: `generated_images`)
- `--device`: Device to use (default: `cuda` if available, else `cpu`)

The script will:
- Load the trained models from checkpoint
- Generate predicted images from the dynamics model
- Compare with original MNIST images
- Save comparison visualizations to `generated_images/` directory

Output files:
- `comparison_batch_*.png`: Individual batch comparisons
- `grid_comparison.png`: Grid of all comparisons
- `comparison_visualization.png`: Matplotlib visualization with error maps

## Configuration

The test uses a simplified configuration:
- Small model (embed_dim=128, depth=4, num_heads=4)
- MNIST images (28x28, grayscale)
- Sequence length: 8 frames
- No actions (uses digit labels as dummy actions for conditioning)

## Notes

- The tokenizer is frozen (not trained) in this test
- The dynamics model learns to predict future latents using shortcut forcing
- Bootstrap targets are computed for non-minimum step sizes
- This is a simplified test - in practice, you'd want proper action conditioning
