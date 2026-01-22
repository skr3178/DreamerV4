#!/usr/bin/env python3
"""
Run the complete DreamerV4 pipeline on a subset dataset.

This script runs Phase 1, Phase 2, and Phase 3 sequentially,
using checkpoints from previous phases.

Usage:
    python run_full_pipeline.py --config configs/minerl_subset.yaml
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
from typing import Optional


def find_latest_checkpoint(checkpoint_dir: Path, prefix: str = "") -> Optional[Path]:
    """Find the latest checkpoint in a directory."""
    if not checkpoint_dir.exists():
        return None
    
    checkpoints = list(checkpoint_dir.glob(f"{prefix}*.pt"))
    if not checkpoints:
        return None
    
    # Sort by modification time
    checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return checkpoints[0]


def main():
    parser = argparse.ArgumentParser(
        description="Run complete DreamerV4 pipeline (Phase 1 → Phase 2 → Phase 3)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/minerl_subset.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--skip-phase1",
        action="store_true",
        help="Skip Phase 1 (use existing checkpoint)"
    )
    parser.add_argument(
        "--skip-phase2",
        action="store_true",
        help="Skip Phase 2 (use existing checkpoint)"
    )
    parser.add_argument(
        "--skip-phase3",
        action="store_true",
        help="Skip Phase 3"
    )
    parser.add_argument(
        "--phase1-checkpoint",
        type=str,
        default=None,
        help="Path to Phase 1 checkpoint (if skipping Phase 1)"
    )
    parser.add_argument(
        "--phase2-checkpoint",
        type=str,
        default=None,
        help="Path to Phase 2 checkpoint (if skipping Phase 2)"
    )
    
    args = parser.parse_args()
    
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"❌ Error: Config file not found: {config_path}")
        sys.exit(1)
    
    print("=" * 80)
    print("DreamerV4 Full Pipeline Test")
    print("=" * 80)
    print(f"Config: {config_path}")
    print()
    
    # Phase 1: World Model Pretraining
    phase1_checkpoint = None
    if not args.skip_phase1:
        print("=" * 80)
        print("PHASE 1: World Model Pretraining")
        print("=" * 80)
        try:
            # Run Phase 1 training script
            result = subprocess.run(
                ["python", "train_phase1.py", "--config", str(config_path)],
                check=True,
                cwd=Path(__file__).parent
            )
            
            # Try to find Phase 1 checkpoint
            # Checkpoints are saved in checkpoint_dir/experiment_name/phase1/
            import yaml
            with open(config_path) as f:
                config = yaml.safe_load(f)
            checkpoint_base = Path(config["experiment"]["checkpoint_dir"]) / config["experiment"]["name"]
            phase1_dir = checkpoint_base / "phase1"
            
            phase1_checkpoint = find_latest_checkpoint(phase1_dir, prefix="checkpoint_")
            if phase1_checkpoint is None:
                phase1_checkpoint = find_latest_checkpoint(phase1_dir, prefix="best_")
            
            if phase1_checkpoint:
                print(f"✓ Phase 1 checkpoint saved: {phase1_checkpoint}")
            else:
                print("⚠️  Warning: Could not find Phase 1 checkpoint")
                
        except subprocess.CalledProcessError as e:
            print(f"❌ Phase 1 failed with exit code {e.returncode}")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Phase 1 failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        # Use provided checkpoint
        if args.phase1_checkpoint:
            phase1_checkpoint = Path(args.phase1_checkpoint)
        else:
            # Try to find existing checkpoint
            import yaml
            with open(config_path) as f:
                config = yaml.safe_load(f)
            checkpoint_base = Path(config["experiment"]["checkpoint_dir"]) / config["experiment"]["name"]
            phase1_dir = checkpoint_base / "phase1"
            
            phase1_checkpoint = find_latest_checkpoint(phase1_dir, prefix="checkpoint_")
            if phase1_checkpoint is None:
                phase1_checkpoint = find_latest_checkpoint(phase1_dir, prefix="best_")
        
        if phase1_checkpoint and phase1_checkpoint.exists():
            print(f"✓ Using existing Phase 1 checkpoint: {phase1_checkpoint}")
        else:
            print(f"❌ Error: Phase 1 checkpoint not found")
            sys.exit(1)
    
    print()
    
    # Phase 2: Agent Finetuning
    phase2_checkpoint = None
    if not args.skip_phase2:
        print("=" * 80)
        print("PHASE 2: Agent Finetuning")
        print("=" * 80)
        try:
            # Run Phase 2 training script
            cmd = [
                "python", "train_phase2.py",
                "--config", str(config_path),
                "--checkpoint", str(phase1_checkpoint)
            ]
            result = subprocess.run(cmd, check=True, cwd=Path(__file__).parent)
            
            # Try to find Phase 2 checkpoint
            import yaml
            with open(config_path) as f:
                config = yaml.safe_load(f)
            checkpoint_base = Path(config["experiment"]["checkpoint_dir"]) / config["experiment"]["name"]
            phase2_dir = checkpoint_base / "phase2"
            
            phase2_checkpoint = find_latest_checkpoint(phase2_dir, prefix="checkpoint_")
            if phase2_checkpoint is None:
                phase2_checkpoint = find_latest_checkpoint(phase2_dir, prefix="best_")
            
            if phase2_checkpoint:
                print(f"✓ Phase 2 checkpoint saved: {phase2_checkpoint}")
            else:
                print("⚠️  Warning: Could not find Phase 2 checkpoint")
                
        except subprocess.CalledProcessError as e:
            print(f"❌ Phase 2 failed with exit code {e.returncode}")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Phase 2 failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        # Use provided checkpoint
        if args.phase2_checkpoint:
            phase2_checkpoint = Path(args.phase2_checkpoint)
        else:
            # Try to find existing checkpoint
            import yaml
            with open(config_path) as f:
                config = yaml.safe_load(f)
            checkpoint_base = Path(config["experiment"]["checkpoint_dir"]) / config["experiment"]["name"]
            phase2_dir = checkpoint_base / "phase2"
            
            phase2_checkpoint = find_latest_checkpoint(phase2_dir, prefix="checkpoint_")
            if phase2_checkpoint is None:
                phase2_checkpoint = find_latest_checkpoint(phase2_dir, prefix="best_")
        
        if phase2_checkpoint and phase2_checkpoint.exists():
            print(f"✓ Using existing Phase 2 checkpoint: {phase2_checkpoint}")
        else:
            print(f"❌ Error: Phase 2 checkpoint not found")
            sys.exit(1)
    
    print()
    
    # Phase 3: Imagination Training
    if not args.skip_phase3:
        print("=" * 80)
        print("PHASE 3: Imagination Training (RL)")
        print("=" * 80)
        try:
            # Run Phase 3 training script
            cmd = [
                "python", "train_phase3.py",
                "--config", str(config_path),
                "--phase2-checkpoint", str(phase2_checkpoint)
            ]
            result = subprocess.run(cmd, check=True, cwd=Path(__file__).parent)
            
            # Try to find Phase 3 checkpoint
            import yaml
            with open(config_path) as f:
                config = yaml.safe_load(f)
            checkpoint_base = Path(config["experiment"]["checkpoint_dir"]) / config["experiment"]["name"]
            phase3_dir = checkpoint_base / "phase3"
            
            phase3_checkpoint = find_latest_checkpoint(phase3_dir, prefix="checkpoint_")
            if phase3_checkpoint is None:
                phase3_checkpoint = find_latest_checkpoint(phase3_dir, prefix="best_")
            
            if phase3_checkpoint:
                print(f"✓ Phase 3 checkpoint saved: {phase3_checkpoint}")
            else:
                print("⚠️  Warning: Could not find Phase 3 checkpoint")
                
        except subprocess.CalledProcessError as e:
            print(f"❌ Phase 3 failed with exit code {e.returncode}")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Phase 3 failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    print()
    print("=" * 80)
    print("✅ FULL PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"Phase 1 checkpoint: {phase1_checkpoint}")
    if phase2_checkpoint:
        print(f"Phase 2 checkpoint: {phase2_checkpoint}")
    if not args.skip_phase3:
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f)
        checkpoint_base = Path(config["experiment"]["checkpoint_dir"]) / config["experiment"]["name"]
        phase3_dir = checkpoint_base / "phase3"
        phase3_checkpoint = find_latest_checkpoint(phase3_dir, prefix="checkpoint_")
        if phase3_checkpoint is None:
            phase3_checkpoint = find_latest_checkpoint(phase3_dir, prefix="best_")
        if phase3_checkpoint:
            print(f"Phase 3 checkpoint: {phase3_checkpoint}")


if __name__ == "__main__":
    main()
