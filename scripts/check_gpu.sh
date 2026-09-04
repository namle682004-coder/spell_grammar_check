#!/usr/bin/env bash
echo "========================================="
echo "GPU Information"
echo "========================================="

uv run python << EOF
import torch
if torch.cuda.is_available():
    print(f"✓ CUDA available")
    print(f"  Device: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"  CUDA version: {torch.version.cuda}")
else:
    print("✗ CUDA not available")
    print("  Will run on CPU (slow)")
EOF