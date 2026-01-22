#!/bin/bash
# Run complete DreamerV4 pipeline on subset dataset
# This script creates a subset, then runs Phase 1, 2, and 3

set -e  # Exit on error

# Configuration
SOURCE_DIR="${1:-data/mineRL_extracted}"
SUBSET_DIR="${2:-data/mineRL_subset}"
NUM_EPISODES="${3:-10}"
CONFIG_FILE="${4:-configs/minerl_subset.yaml}"

echo "=========================================="
echo "DreamerV4 Subset Pipeline"
echo "=========================================="
echo "Source dataset: $SOURCE_DIR"
echo "Subset directory: $SUBSET_DIR"
echo "Number of episodes: $NUM_EPISODES"
echo "Config file: $CONFIG_FILE"
echo ""

# Step 1: Create subset
echo "=========================================="
echo "Step 1: Creating subset dataset"
echo "=========================================="
python create_subset.py \
    --source-dir "$SOURCE_DIR" \
    --output-dir "$SUBSET_DIR" \
    --num-episodes "$NUM_EPISODES"

echo ""
echo "✓ Subset created successfully"
echo ""

# Step 2: Run Phase 1
echo "=========================================="
echo "Step 2: Phase 1 - World Model Pretraining"
echo "=========================================="
python train_phase1.py --config "$CONFIG_FILE"

# Find Phase 1 checkpoint
CHECKPOINT_BASE=$(python -c "import yaml; f=open('$CONFIG_FILE'); c=yaml.safe_load(f); print(c['experiment']['checkpoint_dir'] + '/' + c['experiment']['name'])")
PHASE1_CHECKPOINT=$(find "$CHECKPOINT_BASE/phase1" -name "*.pt" -type f | head -1)

if [ -z "$PHASE1_CHECKPOINT" ]; then
    echo "Error: Could not find Phase 1 checkpoint"
    exit 1
fi

echo "✓ Phase 1 checkpoint: $PHASE1_CHECKPOINT"
echo ""

# Step 3: Run Phase 2
echo "=========================================="
echo "Step 3: Phase 2 - Agent Finetuning"
echo "=========================================="
python train_phase2.py \
    --config "$CONFIG_FILE" \
    --checkpoint "$PHASE1_CHECKPOINT"

# Find Phase 2 checkpoint
PHASE2_CHECKPOINT=$(find "$CHECKPOINT_BASE/phase2" -name "*.pt" -type f | head -1)

if [ -z "$PHASE2_CHECKPOINT" ]; then
    echo "Error: Could not find Phase 2 checkpoint"
    exit 1
fi

echo "✓ Phase 2 checkpoint: $PHASE2_CHECKPOINT"
echo ""

# Step 4: Run Phase 3
echo "=========================================="
echo "Step 4: Phase 3 - Imagination Training"
echo "=========================================="
python train_phase3.py \
    --config "$CONFIG_FILE" \
    --phase2-checkpoint "$PHASE2_CHECKPOINT"

echo ""
echo "=========================================="
echo "✓ Full pipeline completed successfully!"
echo "=========================================="
