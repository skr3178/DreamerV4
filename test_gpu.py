#!/usr/bin/env python3
"""Quick test to verify GPU is being used."""

import torch
import torch.nn as nn

print("=" * 60)
print("GPU Availability Test")
print("=" * 60)

# Check CUDA
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"PyTorch version: {torch.__version__}")
    
    # Test device assignment
    device = torch.device("cuda")
    print(f"\nDevice object: {device}")
    
    # Create a test tensor
    x = torch.randn(1000, 1000).to(device)
    print(f"Test tensor device: {x.device}")
    
    # Create a test model
    model = nn.Linear(1000, 1000).to(device)
    model_device = next(model.parameters()).device
    print(f"Test model device: {model_device}")
    
    # Run a simple computation
    y = model(x)
    print(f"Output tensor device: {y.device}")
    
    print("\n✅ GPU is working correctly!")
else:
    print("\n❌ CUDA is not available. Training will use CPU.")
    print("   Make sure:")
    print("   1. NVIDIA drivers are installed")
    print("   2. CUDA toolkit is installed")
    print("   3. PyTorch was installed with CUDA support")

print("=" * 60)
