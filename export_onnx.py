#!/usr/bin/env python3
"""
Export the PyTorch checkpoint to ONNX format for lightweight deployment.

Usage:
    python export_onnx.py                     # uses default paths
    python export_onnx.py --ckpt path/to/checkpoint.pt --out model.onnx

This script requires torch + the model definitions.
The exported .onnx file can be used with onnxruntime (no torch needed at runtime).
"""

import argparse
import os
import sys

import torch

# Add brain_app to path so we can import the model definitions
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "brain_app"))
from analyzer.nn_models import SmallUNet, AttentionUNet


def export(checkpoint_path: str, output_path: str):
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    model_class = ckpt.get("model_class", "SmallUNet")
    base_ch = ckpt.get("base_ch", 48)
    img_size = ckpt.get("img_size", 128)
    threshold = ckpt.get("threshold", 0.5)

    print(f"Model: {model_class}, base_ch={base_ch}, img_size={img_size}, threshold={threshold}")

    if model_class == "AttentionUNet":
        model = AttentionUNet(base_ch=base_ch, deep_supervision=False)
    else:
        model = SmallUNet(base_ch=base_ch)

    # Prefer EMA weights
    if "ema_state_dict" in ckpt:
        model.load_state_dict(ckpt["ema_state_dict"])
        print("Loaded EMA weights")
    else:
        model.load_state_dict(ckpt["model_state_dict"])
        print("Loaded model weights")

    model.eval()

    # Dummy input: [batch=1, channels=1, H, W]
    dummy = torch.randn(1, 1, img_size, img_size)

    print(f"Exporting to ONNX: {output_path}")
    torch.onnx.export(
        model,
        dummy,
        output_path,
        opset_version=17,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input": {0: "batch"},
            "logits": {0: "batch"},
        },
    )

    # Save metadata as a companion JSON
    import json
    meta_path = output_path.replace(".onnx", "_meta.json")
    meta = {
        "model_class": model_class,
        "base_ch": base_ch,
        "img_size": img_size,
        "threshold": threshold,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata saved to: {meta_path}")

    # Report sizes
    pt_size = os.path.getsize(checkpoint_path) / (1024 * 1024)
    onnx_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\nCheckpoint size: {pt_size:.1f} MB")
    print(f"ONNX model size: {onnx_size:.1f} MB")
    print(f"Size reduction:  {(1 - onnx_size / pt_size) * 100:.0f}%")
    print("\nDone! You can now deploy with onnxruntime instead of torch.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export PyTorch model to ONNX")
    parser.add_argument("--ckpt", default="checkpoints/unet_resse_best.pt",
                        help="Path to .pt checkpoint")
    parser.add_argument("--out", default="checkpoints/model.onnx",
                        help="Output ONNX file path")
    args = parser.parse_args()
    export(args.ckpt, args.out)
