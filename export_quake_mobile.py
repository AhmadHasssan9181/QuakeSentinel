"""
QuakeSentinel - Mobile ONNX & TFLite Export & Latency Benchmarking
Exports the trained mobile models (MobileNetV4-FPN or EfficientViT-FPN)
to production ONNX format with static shapes for Mobile NPUs.

Usage:
    python export_quake_mobile.py --arch mobilenetv4 --img_size 512
    python export_quake_mobile.py --arch efficientvit --img_size 512
"""

import os
import time
import argparse
import torch
import numpy as np

from quake_mobile_model import QuakeMobileNet
from quake_vit_model import QuakeViT

ARCH_MAP = {
    "mobilenetv4": "mobilenetv4_conv_small.e1200_r224_in1k",
    "quakevit": "efficientvit_b2",
    "efficientvit": "efficientvit_b0",
}

def parse_args():
    p = argparse.ArgumentParser(description="Export QuakeSentinel model to Mobile ONNX")
    p.add_argument("--arch", type=str, default="quakevit", choices=["mobilenetv4", "quakevit", "efficientvit"])
    p.add_argument("--img_size", type=int, default=512)
    p.add_argument("--ckpt", type=str, default="")
    p.add_argument("--out_dir", type=str, default="exports")
    return p.parse_args()

def benchmark_inference(model, dummy_input, device="cuda", warmup=10, runs=50):
    model.eval()
    model.to(device)
    dummy_input = dummy_input.to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)

    if device == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(runs):
            _ = model(dummy_input)
            if device == "cuda":
                torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    avg_ms = (elapsed / runs) * 1000.0
    fps = 1000.0 / avg_ms
    return avg_ms, fps

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 65)
    print(f">> Exporting QuakeSentinel [{args.arch}] to Mobile ONNX")
    print(f"   Target Input Resolution: {args.img_size}x{args.img_size}")
    print("=" * 65)

    # 1. Instantiate Model
    num_classes = 3
    if args.arch == "quakevit":
        model = QuakeViT(
            num_classes=num_classes,
            backbone_name=ARCH_MAP[args.arch],
            fpn_dim=128,
            pretrained=False,
        )
    else:
        model = QuakeMobileNet(
            num_classes=num_classes,
            backbone_name=ARCH_MAP[args.arch],
            fpn_dim=64,
            pretrained=False,
        )

    triage_ckpt = f"checkpoints/best_quake_triage_mobile_{args.arch}.pth"
    ckpt_path = args.ckpt or (triage_ckpt if os.path.exists(triage_ckpt) else f"checkpoints/best_quake_mobile_{args.arch}.pth")
    if os.path.exists(ckpt_path):
        print(f"Loading weights from {ckpt_path}...")
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        print("[OK] Checkpoint loaded successfully!")
    else:
        print(f"[INFO] Checkpoint {ckpt_path} not found. Exporting base architecture for shape & latency verification.")

    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params / 1e6:.2f}M")

    # 2. Benchmarking Latency
    dummy_input = torch.randn(1, 3, args.img_size, args.img_size)

    # Benchmark CPU (Simulates Mobile CPU)
    print("\nBenchmarking Latency...")
    cpu_ms, cpu_fps = benchmark_inference(model, dummy_input, device="cpu", warmup=5, runs=20)
    print(f"   Desktop CPU   : {cpu_ms:.1f} ms / frame  ({cpu_fps:.1f} FPS)")

    if torch.cuda.is_available():
        gpu_ms, gpu_fps = benchmark_inference(model, dummy_input, device="cuda", warmup=10, runs=50)
        print(f"   RTX 3090 GPU  : {gpu_ms:.2f} ms / frame  ({gpu_fps:.1f} FPS)")

    # 3. Export to PyTorch Mobile Lite (.ptl) and TorchScript (.pt)
    from torch.utils.mobile_optimizer import optimize_for_mobile
    ptl_filename = os.path.join(args.out_dir, f"quake_{args.arch}_{args.img_size}.ptl")
    pt_filename  = os.path.join(args.out_dir, f"quake_{args.arch}_{args.img_size}.pt")
    
    print(f"\nExporting to PyTorch Mobile ({ptl_filename})...")
    traced = torch.jit.trace(model.cpu(), dummy_input.cpu())
    traced.save(pt_filename)
    opt = optimize_for_mobile(traced)
    opt._save_for_lite_interpreter(ptl_filename)
    
    ptl_size = os.path.getsize(ptl_filename) / (1024 * 1024)
    print(f"[OK] PyTorch Mobile Lite (.ptl) exported: {ptl_size:.2f} MB")
    print(f"[OK] TorchScript (.pt) exported: {os.path.getsize(pt_filename)/(1024*1024):.2f} MB")

    # 4. Optional ONNX export if onnx is available
    onnx_filename = os.path.join(args.out_dir, f"quake_{args.arch}_{args.img_size}.onnx")
    try:
        print(f"\nExporting to ONNX ({onnx_filename})...")
        torch.onnx.export(
            model.cpu(),
            dummy_input.cpu(),
            onnx_filename,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["seg_logits", "damage_score"],
            dynamic_axes=None,  # Static shapes required for optimal mobile NPU compilation
        )
        file_size_mb = os.path.getsize(onnx_filename) / (1024 * 1024)
        print(f"[OK] ONNX export complete! File size: {file_size_mb:.2f} MB")
    except Exception as e:
        print(f"[INFO] ONNX export skipped ({e}). PyTorch Mobile Lite (.ptl) is ready for direct mobile deployment.")

    print("=" * 65)

if __name__ == "__main__":
    main()
