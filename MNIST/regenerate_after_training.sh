#!/bin/bash
# Script to wait for training to complete and regenerate images

echo "Waiting for training to complete..."
echo "Checking if training process is still running..."

# Wait for training to finish (check if python process with test_shortcut.py is running)
while pgrep -f "test_shortcut.py" > /dev/null; do
    echo "Training still running... (checking every 10 seconds)"
    sleep 10
done

echo ""
echo "Training appears to be complete!"
echo "Regenerating comparison images..."

# Activate conda environment and run image generation
source /home/skr/miniconda3/etc/profile.d/conda.sh
conda activate dreamerV4

cd /media/skr/storage/dreamerv4/MNIST
python generate_images.py --num_samples 10

echo ""
echo "✓ Image regeneration complete!"
echo "Check the generated_images/ directory for results."
