#!/usr/bin/env python3
"""
Parse training log files and plot loss curves.
"""

import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def parse_log_file(log_path):
    """Parse a log file and extract training metrics."""
    metrics = {
        'step': [],
        'epoch': [],
        'batches_per_epoch': [],
        'tok': [],
        'tok_mse': [],
        'tok_lpips': [],
        'dyn': [],
        'd_min': [],
        'd_other': [],
        'lr_tok': [],
        'lr_dyn': []
    }
    
    batches_per_epoch = 890  # default, will be updated from log
    
    # First pass: find batches_per_epoch
    with open(log_path, 'r') as f:
        for line in f:
            match = re.search(r'Total batches per epoch: (\d+)', line)
            if match:
                batches_per_epoch = int(match.group(1))
                break
    
    # Pattern to match tqdm progress bar lines with metrics
    # Example: Epoch 2/20:   1%|▏         | 10/890 [00:38<54:41,  3.73s/it, tok=1.3000, tok_mse=0.0214, tok_lpips=0.4243, dyn=0.0045, d_min=0.0149, d_other=0.0149, lr_tok=3.00e-04, lr_dyn=3.00e-04]
    pattern = r'Epoch (\d+)/\d+:\s+\d+%\s*\|\s*\S+\s*\|\s*(\d+)/(\d+).*?tok=([\d.]+),\s*tok_mse=([\d.]+),\s*tok_lpips=([\d.]+),\s*dyn=([\d.]+),\s*d_min=([\d.]+),\s*d_other=([\d.]+),\s*lr_tok=([\d.e-]+),\s*lr_dyn=([\d.e-]+)'
    
    with open(log_path, 'r') as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                epoch = int(match.group(1))
                step = int(match.group(2))
                total_batches = int(match.group(3))
                tok = float(match.group(4))
                tok_mse = float(match.group(5))
                tok_lpips = float(match.group(6))
                dyn = float(match.group(7))
                d_min = float(match.group(8))
                d_other = float(match.group(9))
                lr_tok = float(match.group(10))
                lr_dyn = float(match.group(11))
                
                # Update batches_per_epoch if found in the line
                if total_batches > 0:
                    batches_per_epoch = total_batches
                
                metrics['step'].append(step)
                metrics['epoch'].append(epoch)
                metrics['batches_per_epoch'].append(batches_per_epoch)
                metrics['tok'].append(tok)
                metrics['tok_mse'].append(tok_mse)
                metrics['tok_lpips'].append(tok_lpips)
                metrics['dyn'].append(dyn)
                metrics['d_min'].append(d_min)
                metrics['d_other'].append(d_other)
                metrics['lr_tok'].append(lr_tok)
                metrics['lr_dyn'].append(lr_dyn)
    
    # Convert to numpy arrays
    for key in metrics:
        metrics[key] = np.array(metrics[key])
    
    return metrics

def plot_loss_curves(metrics1, label1, metrics2, label2, output_path='loss_curves.png'):
    """Plot loss curves from two log files."""
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Training Loss Curves Comparison', fontsize=16, fontweight='bold')
    
    # Calculate global step (assuming steps are per epoch, need to account for epoch number)
    def get_global_step(metrics):
        if len(metrics['step']) == 0:
            return np.array([])
        # Calculate global step: (epoch - 1) * batches_per_epoch + step
        batches_per_epoch = metrics['batches_per_epoch'][0] if len(metrics['batches_per_epoch']) > 0 else 890
        global_steps = []
        for i in range(len(metrics['step'])):
            epoch = metrics['epoch'][i] - 1  # 0-indexed (epoch 1 -> 0, epoch 2 -> 1, etc.)
            step = metrics['step'][i]
            global_step = epoch * batches_per_epoch + step
            global_steps.append(global_step)
        return np.array(global_steps)
    
    steps1 = get_global_step(metrics1)
    steps2 = get_global_step(metrics2)
    
    # Plot 1: Tokenizer Loss
    ax = axes[0, 0]
    if len(steps1) > 0:
        ax.plot(steps1, metrics1['tok'], label=label1, alpha=0.7, linewidth=1.5)
    if len(steps2) > 0:
        ax.plot(steps2, metrics2['tok'], label=label2, alpha=0.7, linewidth=1.5)
    ax.set_xlabel('Global Step')
    ax.set_ylabel('Tokenizer Loss')
    ax.set_title('Tokenizer Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Tokenizer MSE
    ax = axes[0, 1]
    if len(steps1) > 0:
        ax.plot(steps1, metrics1['tok_mse'], label=label1, alpha=0.7, linewidth=1.5)
    if len(steps2) > 0:
        ax.plot(steps2, metrics2['tok_mse'], label=label2, alpha=0.7, linewidth=1.5)
    ax.set_xlabel('Global Step')
    ax.set_ylabel('Tokenizer MSE')
    ax.set_title('Tokenizer MSE')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Tokenizer LPIPS
    ax = axes[0, 2]
    if len(steps1) > 0:
        ax.plot(steps1, metrics1['tok_lpips'], label=label1, alpha=0.7, linewidth=1.5)
    if len(steps2) > 0:
        ax.plot(steps2, metrics2['tok_lpips'], label=label2, alpha=0.7, linewidth=1.5)
    ax.set_xlabel('Global Step')
    ax.set_ylabel('Tokenizer LPIPS')
    ax.set_title('Tokenizer LPIPS')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Dynamics Loss
    ax = axes[1, 0]
    if len(steps1) > 0:
        ax.plot(steps1, metrics1['dyn'], label=label1, alpha=0.7, linewidth=1.5)
    if len(steps2) > 0:
        ax.plot(steps2, metrics2['dyn'], label=label2, alpha=0.7, linewidth=1.5)
    ax.set_xlabel('Global Step')
    ax.set_ylabel('Dynamics Loss')
    ax.set_title('Dynamics Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 5: d_min
    ax = axes[1, 1]
    if len(steps1) > 0:
        ax.plot(steps1, metrics1['d_min'], label=label1, alpha=0.7, linewidth=1.5)
    if len(steps2) > 0:
        ax.plot(steps2, metrics2['d_min'], label=label2, alpha=0.7, linewidth=1.5)
    ax.set_xlabel('Global Step')
    ax.set_ylabel('d_min')
    ax.set_title('Dynamics d_min')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 6: d_other
    ax = axes[1, 2]
    if len(steps1) > 0:
        ax.plot(steps1, metrics1['d_other'], label=label1, alpha=0.7, linewidth=1.5)
    if len(steps2) > 0:
        ax.plot(steps2, metrics2['d_other'], label=label2, alpha=0.7, linewidth=1.5)
    ax.set_xlabel('Global Step')
    ax.set_ylabel('d_other')
    ax.set_title('Dynamics d_other')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {output_path}")
    return fig

def main():
    log1_path = Path('logs/phase1_training_resume_20260126_065102.log')
    log2_path = Path('train_test.log')
    
    print(f"Parsing {log1_path}...")
    metrics1 = parse_log_file(log1_path)
    print(f"  Found {len(metrics1['step'])} data points")
    if len(metrics1['step']) > 0:
        batches1 = metrics1['batches_per_epoch'][0] if len(metrics1['batches_per_epoch']) > 0 else 890
        print(f"  Epochs: {metrics1['epoch'].min()} to {metrics1['epoch'].max()}")
        print(f"  Batches per epoch: {batches1}")
        print(f"  Tokenizer loss range: {metrics1['tok'].min():.4f} - {metrics1['tok'].max():.4f}")
        print(f"  Dynamics loss range: {metrics1['dyn'].min():.4f} - {metrics1['dyn'].max():.4f}")
    
    print(f"\nParsing {log2_path}...")
    metrics2 = parse_log_file(log2_path)
    print(f"  Found {len(metrics2['step'])} data points")
    if len(metrics2['step']) > 0:
        batches2 = metrics2['batches_per_epoch'][0] if len(metrics2['batches_per_epoch']) > 0 else 890
        print(f"  Epochs: {metrics2['epoch'].min()} to {metrics2['epoch'].max()}")
        print(f"  Batches per epoch: {batches2}")
        print(f"  Tokenizer loss range: {metrics2['tok'].min():.4f} - {metrics2['tok'].max():.4f}")
        print(f"  Dynamics loss range: {metrics2['dyn'].min():.4f} - {metrics2['dyn'].max():.4f}")
    
    if len(metrics1['step']) == 0 and len(metrics2['step']) == 0:
        print("Error: No metrics found in either log file!")
        return
    
    print("\nPlotting loss curves...")
    plot_loss_curves(
        metrics1, 
        'phase1_training_resume',
        metrics2, 
        'train_test',
        output_path='loss_curves.png'
    )
    
    print("\nDone!")

if __name__ == '__main__':
    main()
