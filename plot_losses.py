#!/usr/bin/env python3
"""
Plot loss curves from training log file.

Parses the training log and extracts loss values from tqdm progress bars,
then plots them as curves over training steps.
"""

import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse


def parse_log_file(log_path: str):
    """
    Parse training log file to extract loss values.
    
    Returns:
        dict with keys: steps, tok_loss, tok_mse, tok_lpips, dyn_loss, d_min, d_other
    """
    with open(log_path, 'r') as f:
        lines = f.readlines()
    
    # Pattern to match tqdm progress bar lines with metrics
    # Example: Epoch 1/1:   0%|          | 1/277 [00:54<4:08:14, 53.97s/it, tok=1.0000, tok_mse=0.1198, tok_lpips=0.0000, dyn=0.0318, d_min=0.1021, d_other=0.1019, lr_tok=3.00e-04, lr_dyn=3.00e-04]
    # Use a flexible pattern that matches the key parts
    pattern = r'(\d+)/\d+.*tok=([\d.]+).*tok_mse=([\d.]+).*tok_lpips=([\d.]+).*dyn=([\d.]+).*d_min=([\d.]+).*d_other=([\d.]+)'
    
    steps = []
    tok_loss = []
    tok_mse = []
    tok_lpips = []
    dyn_loss = []
    d_min = []
    d_other = []
    
    step_counter = 0  # Track actual step number (since batch_num might repeat)
    
    for line in lines:
        match = re.search(pattern, line)
        if match:
            batch_num = int(match.group(1))
            tok = float(match.group(2))
            tok_m = float(match.group(3))
            tok_l = float(match.group(4))
            dyn = float(match.group(5))
            d_m = float(match.group(6))
            d_o = float(match.group(7))
            
            # Use step counter to track unique steps (avoid duplicates from progress bar updates)
            # Only increment if this is a new batch number or if values changed
            if len(steps) == 0 or batch_num != steps[-1] or tok != tok_loss[-1]:
                steps.append(step_counter)
                tok_loss.append(tok)
                tok_mse.append(tok_m)
                tok_lpips.append(tok_l)
                dyn_loss.append(dyn)
                d_min.append(d_m)
                d_other.append(d_o)
                step_counter += 1
    
    return {
        'steps': np.array(steps),
        'tok_loss': np.array(tok_loss),
        'tok_mse': np.array(tok_mse),
        'tok_lpips': np.array(tok_lpips),
        'dyn_loss': np.array(dyn_loss),
        'd_min': np.array(d_min),
        'd_other': np.array(d_other),
    }


def plot_losses(data: dict, output_path: str = None):
    """Plot loss curves from parsed data."""
    steps = data['steps']
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Training Loss Curves - Phase 1', fontsize=16, fontweight='bold')
    
    # Plot 1: Tokenizer losses
    ax1 = axes[0, 0]
    ax1.plot(steps, data['tok_loss'], label='Total', linewidth=2, alpha=0.8)
    ax1.plot(steps, data['tok_mse'], label='MSE', linewidth=2, alpha=0.8)
    ax1.plot(steps, data['tok_lpips'], label='LPIPS', linewidth=2, alpha=0.8)
    ax1.set_xlabel('Step', fontsize=11)
    ax1.set_ylabel('Loss', fontsize=11)
    ax1.set_title('Tokenizer Losses', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Dynamics losses
    ax2 = axes[0, 1]
    ax2.plot(steps, data['dyn_loss'], label='Total', linewidth=2, alpha=0.8, color='orange')
    ax2.plot(steps, data['d_min'], label='d_min', linewidth=2, alpha=0.8, color='red')
    ax2.plot(steps, data['d_other'], label='d_other', linewidth=2, alpha=0.8, color='purple')
    ax2.set_xlabel('Step', fontsize=11)
    ax2.set_ylabel('Loss', fontsize=11)
    ax2.set_title('Dynamics Losses', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Combined view (log scale for better visualization)
    ax3 = axes[1, 0]
    ax3.semilogy(steps, data['tok_loss'], label='Tokenizer Total', linewidth=2, alpha=0.8)
    ax3.semilogy(steps, data['dyn_loss'], label='Dynamics Total', linewidth=2, alpha=0.8)
    ax3.set_xlabel('Step', fontsize=11)
    ax3.set_ylabel('Loss (log scale)', fontsize=11)
    ax3.set_title('Combined Losses (Log Scale)', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Tokenizer MSE and Dynamics components
    ax4 = axes[1, 1]
    ax4.plot(steps, data['tok_mse'], label='Tokenizer MSE', linewidth=2, alpha=0.8, color='blue')
    ax4.plot(steps, data['d_min'], label='Dynamics d_min', linewidth=2, alpha=0.8, color='red')
    ax4.plot(steps, data['d_other'], label='Dynamics d_other', linewidth=2, alpha=0.8, color='purple')
    ax4.set_xlabel('Step', fontsize=11)
    ax4.set_ylabel('Loss', fontsize=11)
    ax4.set_title('Key Loss Components', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved plot to {output_path}")
    else:
        plt.savefig('loss_curves.png', dpi=150, bbox_inches='tight')
        print("✓ Saved plot to loss_curves.png")
    
    plt.close()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("Loss Summary Statistics")
    print("="*60)
    print(f"Tokenizer Total Loss:")
    print(f"  Initial: {data['tok_loss'][0]:.6f}")
    print(f"  Final:   {data['tok_loss'][-1]:.6f}")
    print(f"  Change:  {data['tok_loss'][-1] - data['tok_loss'][0]:.6f} ({((data['tok_loss'][-1] - data['tok_loss'][0]) / data['tok_loss'][0] * 100):.2f}%)")
    print(f"  Min:     {data['tok_loss'].min():.6f}")
    print(f"  Max:     {data['tok_loss'].max():.6f}")
    print()
    print(f"Tokenizer MSE Loss:")
    print(f"  Initial: {data['tok_mse'][0]:.6f}")
    print(f"  Final:   {data['tok_mse'][-1]:.6f}")
    print(f"  Change:  {data['tok_mse'][-1] - data['tok_mse'][0]:.6f} ({((data['tok_mse'][-1] - data['tok_mse'][0]) / data['tok_mse'][0] * 100):.2f}%)")
    print()
    print(f"Dynamics Total Loss:")
    print(f"  Initial: {data['dyn_loss'][0]:.6f}")
    print(f"  Final:   {data['dyn_loss'][-1]:.6f}")
    print(f"  Change:  {data['dyn_loss'][-1] - data['dyn_loss'][0]:.6f} ({((data['dyn_loss'][-1] - data['dyn_loss'][0]) / data['dyn_loss'][0] * 100):.2f}%)")
    print(f"  Min:     {data['dyn_loss'].min():.6f}")
    print(f"  Max:     {data['dyn_loss'].max():.6f}")


def main():
    parser = argparse.ArgumentParser(description="Plot loss curves from training log")
    parser.add_argument(
        "--log-file",
        type=str,
        default="logs/train_phase1.log",
        help="Path to training log file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for plot (default: loss_curves.png)"
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
