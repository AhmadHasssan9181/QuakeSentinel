"""
QuakeSentinel Mobile Inference & Visualizer
Runs high-speed inference using the lightweight mobile model and generates
an alpha-blended multi-class damage overlay.

Usage:
    python quake_mobile_inference.py <input_image> <output_image>
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cv2
import torch
import numpy as np

from quake_mobile_model import QuakeMobileNet
from quake_vit_model import QuakeViT

NUM_CLASSES = 3
DEFAULT_CKPT = "checkpoints/best_quake_triage_mobile_quakevit.pth"
FALLBACK_CKPT = "checkpoints/best_quake_triage_mobile_mobilenetv4.pth"

COLOR_MAP = {
    0: (0, 0, 0),        # Background / Terrain (clean / transparent)
    1: (0, 220, 0),      # Intact Building (Vibrant Green)
    2: (0, 30, 255),     # Damaged / Destroyed Building (Vivid Red)
}

CLASS_NAMES = {
    0: "Background",
    1: "Intact Building",
    2: "Damaged Building",
}

def infer(img_path: str, model_path: str = None, output_path: str = "outputs/mobile_out.png", 
          img_size: int = 512, conf_dmg: float = 0.72, conf_intact: float = 0.40, min_area: int = 200):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if model_path is None:
        if os.path.exists(DEFAULT_CKPT):
            model_path = DEFAULT_CKPT
        elif os.path.exists(FALLBACK_CKPT):
            model_path = FALLBACK_CKPT
        else:
            model_path = DEFAULT_CKPT

    print(f">> Running Mobile Inference on {img_path} ({device})...")
    print(f"   Model Checkpoint: {model_path}")
    print(f"   Settings: conf_dmg={conf_dmg}, conf_intact={conf_intact}, min_area={min_area}px")

    # 1. Load Image
    orig_img = cv2.imread(img_path)
    if orig_img is None:
        raise FileNotFoundError(f"Could not read image from {img_path}")
    orig_h, orig_w = orig_img.shape[:2]

    # Preprocessing
    rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    norm = (resized.astype(np.float32) / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    tensor = torch.from_numpy(norm).permute(2, 0, 1).unsqueeze(0).float().to(device)

    # 2. Load Model
    if "quakevit" in str(model_path).lower():
        model = QuakeViT(num_classes=NUM_CLASSES, backbone_name="efficientvit_b2", fpn_dim=128, pretrained=False).to(device)
    else:
        model = QuakeMobileNet(num_classes=NUM_CLASSES, pretrained=False).to(device)

    if os.path.exists(model_path):
        state = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        print(f"   [OK] Loaded weights from {model_path}")
    else:
        print(f"   [WARN] Checkpoint not found at {model_path}! Using initialized weights.")
    model.eval()

    # 3. Predict with Calibrated Probabilities
    with torch.no_grad():
        seg_logits, damage_score = model(tensor)
        probs = torch.softmax(seg_logits, dim=1).squeeze(0).cpu().numpy()

    prob_intact = probs[1]
    prob_dmg    = probs[2]

    pred_512 = np.zeros((img_size, img_size), dtype=np.uint8)
    # Intact building detection
    pred_512[(prob_intact >= conf_intact) & (prob_intact > prob_dmg)] = 1
    # Damaged building detection (calibrated threshold avoids false positive driveways/gravel)
    pred_512[(prob_dmg >= conf_dmg) & (prob_dmg > prob_intact)] = 2

    # 4. Upsample to original image resolution
    mask_orig = cv2.resize(pred_512, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    # 5. Connected Component Area Filtering (eliminates isolated noise specks)
    if min_area > 0:
        for c_id in [1, 2]:
            c_mask = (mask_orig == c_id).astype(np.uint8)
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(c_mask)
            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] < min_area:
                    mask_orig[labels == i] = 0

    total_px = mask_orig.size
    bg_pct = (mask_orig == 0).sum() / total_px * 100
    intact_pct = (mask_orig == 1).sum() / total_px * 100
    dmg_pct = (mask_orig == 2).sum() / total_px * 100

    print(f"   Predicted Damage Threat Score: {damage_score.item():.4f}")
    print(f"   Triage Distribution: Clean Bg={bg_pct:.1f}% | Intact={intact_pct:.1f}% | Damaged={dmg_pct:.1f}%")

    # 6. Generate Alpha-Blended Overlay
    overlay = np.zeros_like(orig_img)
    for c_id, color in COLOR_MAP.items():
        if c_id == 0: continue
        overlay[mask_orig == c_id] = color

    alpha = 0.55
    mask_bool = mask_orig > 0
    final_img = orig_img.copy()
    if mask_bool.any():
        blended = cv2.addWeighted(overlay, alpha, orig_img, 1.0 - alpha, 0)
        final_img[mask_bool] = blended[mask_bool]

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cv2.imwrite(output_path, final_img)
    print(f"   [OK] Saved prediction overlay to {output_path}")

if __name__ == "__main__":
    img_in = sys.argv[1] if len(sys.argv) > 1 else "QuakeSentinel/testclean.png"
    img_out = sys.argv[2] if len(sys.argv) > 2 else "outputs/mobile_testclean_out.png"
    ckpt = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_CKPT
    infer(img_in, model_path=ckpt, output_path=img_out)
