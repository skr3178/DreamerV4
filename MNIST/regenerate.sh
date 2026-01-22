#!/bin/bash
# Quick script to regenerate images after training

cd /media/skr/storage/dreamerv4/MNIST
source /home/skr/miniconda3/etc/profile.d/conda.sh
conda activate dreamerV4

echo "Regenerating comparison images with latest checkpoint..."
python generate_images.py --num_samples 10

echo ""
echo "✓ Done! Check generated_images/ directory"
