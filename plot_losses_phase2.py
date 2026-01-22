#!/usr/bin/env python3
"""
Plot loss curves from Phase 2 training log file.

Parses the training log and extracts loss and accuracy values from tqdm progress bars,
then plots them as curves over training steps.
"""

import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse


def parse_log_file(log_path: str):
    """
    Parse Phase 2 training log file to extract loss and accuracy values.
    
    Returns:
        dict with keys: steps, loss, accuracy
    """
    with open(log_path, 'r') as f:
        lines = f.readlines()
    
    # Pattern to match tqdm progress bar lines with metrics
    # Example: Epoch 1/2:   0%|          | 0/555 [00:19<?, ?it/s, loss=10.6577, acc=0.00%]
    pattern = r'(\d+)/\d+.*loss=([\d.]+).*acc=([\d.]+)%'
    
    steps = []
    losses = []
    accuracies = []
    
    step_counter = 0  # Track actual step number
    
    for line in lines:
        match = re.search(pattern, line)
        if match:
            batch_num = int(match.group(1))
            loss = float(match.group(2))
            acc = float(match.group(3))
            
            # Use step counter to track unique steps (avoid duplicates from progress bar updates)
            # Only increment if this is a new batch number or if values changed
            if len(steps) == 0 or batch_num != steps[-1] or loss != losses[-1]:
                steps.append(step_counter)
                losses.append(loss)
                accuracies.append(acc)
                step_counter += 1
    
    return {
        'steps': np.array(steps),
        'loss': np.array(losses),
        'accuracy': np.array(accuracies),
    }


def plot_losses(data: dict, output_path: str = None):
    """Plot loss and accuracy curves from parsed data."""
    steps = data['steps']
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Training Curves - Phase 2 (Agent Finetuning)', fontsize=16, fontweight='bold')
    
    # Plot 1: Total Loss
    ax1 = axes[0, 0]
    ax1.plot(steps, data['loss'], label='Total Loss', linewidth=2, alpha=0.8, color='blue')
    ax1.set_xlabel('Step', fontsize=11)
    ax1.set_ylabel('Loss', fontsize=11)
    ax1.set_title('Total Loss', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Accuracy
    ax2 = axes[0, 1]
    ax2.plot(steps, data['accuracy'], label='BC Accuracy', linewidth=2, alpha=0.8, color='green')
    ax2.set_xlabel('Step', fontsize=11)
    ax2.set_ylabel('Accuracy (%)', fontsize=11)
    ax2.set_title('Behavior Cloning Accuracy', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 105])  # Accuracy from 0-100%
    
    # Plot 3: Loss (log scale for better visualization)
    ax3 = axes[1, 0]
    ax3.semilogy(steps, data['loss'], label='Total Loss', linewidth=2, alpha=0.8, color='blue')
    ax3.set_xlabel('Step', fontsize=11)
    ax3.set_ylabel('Loss (log scale)', fontsize=11)
    ax3.set_title('Total Loss (Log Scale)', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Combined view (loss and accuracy on same plot with dual y-axis)
    ax4 = axes[1, 1]
    ax4_twin = ax4.twinx()
    
    line1 = ax4.plot(steps, data['loss'], label='Loss', linewidth=2, alpha=0.8, color='blue')
    line2 = ax4_twin.plot(steps, data['accuracy'], label='Accuracy', linewidth=2, alpha=0.8, color='green')
    
    ax4.set_xlabel('Step', fontsize=11)
    ax4.set_ylabel('Loss', fontsize=11, color='blue')
    ax4_twin.set_ylabel('Accuracy (%)', fontsize=11, color='green')
    ax4.set_title('Loss & Accuracy (Dual Axis)', fontsize=12, fontweight='bold')
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax4.legend(lines, labels, loc='upper left')
    
    ax4.tick_params(axis='y', labelcolor='blue')
    ax4_twin.tick_params(axis='y', labelcolor='green')
    ax4.grid(True, alpha=0.3)
    ax4_twin.set_ylim([0, 105])
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved plot to {output_path}")
    else:
        plt.savefig('loss_curves_phase2.png', dpi=150, bbox_inches='tight')
        print("✓ Saved plot to loss_curves_phase2.png")
    
    plt.close()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("Phase 2 Training Summary Statistics")
    print("="*60)
    print(f"Total Loss:")
    print(f"  Initial: {data['loss'][0]:.6f}")
    print(f"  Final:   {data['loss'][-1]:.6f}")
    print(f"  Change:  {data['loss'][-1] - data['loss'][0]:.6f} ({((data['loss'][-1] - data['loss'][0]) / data['loss'][0] * 100):.2f}%)")
    print(f"  Min:     {data['loss'].min():.6f}")
    print(f"  Max:     {data['loss'].max():.6f}")
    print()
    print(f"Behavior Cloning Accuracy:")
    print(f"  Initial: {data['accuracy'][0]:.2f}%")
    print(f"  Final:   {data['accuracy'][-1]:.2f}%")
    print(f"  Change:  {data['accuracy'][-1] - data['accuracy'][0]:.2f}%")
    print(f"  Min:     {data['accuracy'].min():.2f}%")
    print(f"  Max:     {data['accuracy'].max():.2f}%")
    print()
    print(f"Training Steps: {len(data['steps'])}")
    print(f"Steps Range: {data['steps'].min()} to {data['steps'].max()}")


def main():
    parser = argparse.ArgumentParser(description="Plot loss curves from Phase 2 training log")
    parser.add_argument(
        "--log-file",
        type=str,
        default="logs/train_phase2.log",
        help="Path to training log file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for plot (default: loss_curves_phase2.png)"
    )
    
    args = parser.parse_args()
    
    log_path = Path(args.log_file)
    if not log_path.exists():
        print(f"❌ Error: Log file not found: {log_path}")
        return
    
    print(f"Parsing log file: {log_path}")
    data = parse_log_file(str(log_path))
    
    if len(data['steps']) == 0:
        print("❌ Error: No loss data found in log file. Make sure training has started.")
        return
    
    print(f"✓ Found {len(data['steps'])} data points")
    print(f"  Steps range: {data['steps'].min()} to {data['steps'].max()}")
    
    plot_losses(data, args.output)


if __name__ == "__main__":
    main()
