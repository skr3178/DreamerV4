#!/bin/bash
# Run DreamerV4 training with nohup (runs in background, survives disconnection)

set -e

# Configuration
CONFIG_FILE="${1:-configs/minerl_subset.yaml}"
PHASE="${2:-phase1}"  # phase1, phase2, or phase3
LOG_DIR="logs"

# Create log directory
mkdir -p "$LOG_DIR"

# Activate conda environment and run training
echo "Starting Phase ${PHASE} training with nohup..."
echo "Config: $CONFIG_FILE"
echo "Log file: $LOG_DIR/train_${PHASE}.log"
echo ""

case "$PHASE" in
    phase1)
        # Use Python with unbuffered output (-u) for immediate log updates
        nohup python -u train_phase1.py --config "$CONFIG_FILE" > "$LOG_DIR/train_phase1.log" 2>&1 &
        echo "Phase 1 training started in background (PID: $!)"
        ;;
    phase2)
        if [ -z "$3" ]; then
            echo "Error: Phase 2 requires a checkpoint path"
            echo "Usage: $0 <config> phase2 <checkpoint_path>"
            exit 1
        fi
        CHECKPOINT="$3"
        # Use Python with unbuffered output (-u) for immediate log updates
        nohup python -u train_phase2.py --config "$CONFIG_FILE" --checkpoint "$CHECKPOINT" > "$LOG_DIR/train_phase2.log" 2>&1 &
        echo "Phase 2 training started in background (PID: $!)"
        ;;
    phase3)
        if [ -z "$3" ]; then
            echo "Error: Phase 3 requires a checkpoint path"
            echo "Usage: $0 <config> phase3 <phase2_checkpoint_path>"
            exit 1
        fi
        CHECKPOINT="$3"
        # Use Python with unbuffered output (-u) for immediate log updates
        nohup python -u train_phase3.py --config "$CONFIG_FILE" --phase2-checkpoint "$CHECKPOINT" > "$LOG_DIR/train_phase3.log" 2>&1 &
        echo "Phase 3 training started in background (PID: $!)"
        ;;
    *)
        echo "Error: Unknown phase '$PHASE'"
        echo "Usage: $0 <config> <phase> [checkpoint_path]"
        echo "  phase: phase1, phase2, or phase3"
        exit 1
        ;;
esac

echo ""
echo "To monitor training:"
echo "  tail -f $LOG_DIR/train_${PHASE}.log"
echo ""
echo "To check if training is running:"
echo "  ps aux | grep train_${PHASE}"
echo ""
echo "To stop training:"
echo "  pkill -f train_${PHASE}"
