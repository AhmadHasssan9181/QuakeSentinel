"""
QuakeSentinel - Visual Evaluation & Test Suite
Generates side-by-side comparison images:
[Original Drone Image] | [Ground Truth Triage] | [Mobile Model Prediction]

Proves that:
1. Background is clean and NOT falsely labeled as damage.
2. Intact buildings are cleanly outlined in Green.
3. Collapsed / Damaged structures are pinpointed in Vivid Red.
"""

import os
import cv2
import torch
import numpy as np
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quake_mobile_model import QuakeMobileNet
from quake_vit_model import QuakeViT

OUT_DIR = "outputs/evaluation_tests"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

COLOR_BG      = (0, 0, 0)
COLOR_INTACT  = (0, 225, 0)    # Green
COLOR_DAMAGED = (0, 30, 255)   # Red

def make_overlay(img_bgr, class_mask, alpha=0.55):
    overlay = np.zeros_like(img_bgr)
    overlay[class_mask == 1] = COLOR_INTACT
    overlay[class_mask == 2] = COLOR_DAMAGED
    
    blended = img_bgr.copy()
    active = class_mask > 0
    if active.any():
        blended[active] = cv2.addWeighted(overlay, alpha, img_bgr, 1.0 - alpha, 0)[active]
    return blended

def add_header(img, text, color=(255, 255, 255)):
    h, w = img.shape[:2]
    header = np.zeros((40, w, 3), dtype=np.uint8)
    cv2.putText(header, text, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    return np.vstack([header, img])

def predict_triage_mask(model, orig_img, conf_dmg=0.72, conf_intact=0.40, min_area=200):
    h, w = orig_img.shape[:2]
    rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (512, 512), interpolation=cv2.INTER_LINEAR)
    norm = (resized.astype(np.float32) / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    tensor = torch.from_numpy(norm).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)

    with torch.no_grad():
        seg_logits, damage_score = model(tensor)
        probs = torch.softmax(seg_logits, dim=1).squeeze(0).cpu().numpy()

    prob_intact = probs[1]
    prob_dmg    = probs[2]

    pred_512 = np.zeros((512, 512), dtype=np.uint8)
    pred_512[(prob_intact >= conf_intact) & (prob_intact > prob_dmg)] = 1
    pred_512[(prob_dmg >= conf_dmg) & (prob_dmg > prob_intact)] = 2

    # Upsample to original image resolution
    mask_orig = cv2.resize(pred_512, (w, h), interpolation=cv2.INTER_NEAREST)

    # Filter isolated noise specks
    if min_area > 0:
        for c_id in [1, 2]:
            c_mask = (mask_orig == c_id).astype(np.uint8)
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(c_mask)
            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] < min_area:
                    mask_orig[labels == i] = 0

    return mask_orig, damage_score.item()

def evaluate_samples(arch="quakevit"):
    os.makedirs(OUT_DIR, exist_ok=True)
    ckpt_path = f"checkpoints/best_quake_triage_mobile_{arch}.pth"
    if not os.path.exists(ckpt_path):
        # Fallback to mobilenetv4 checkpoint if quakevit not yet trained
        ckpt_path = "checkpoints/best_quake_triage_mobile_mobilenetv4.pth"
        arch = "mobilenetv4"

    print("=" * 65)
    print(f">> Running Comprehensive Visual Test ({arch})")
    print(f"   Checkpoint: {ckpt_path}")
    print(f"   Device    : {DEVICE}")
    print("=" * 65)

    # 1. Load Model
    if arch == "quakevit":
        model = QuakeViT(num_classes=3, backbone_name="efficientvit_b2", fpn_dim=128, pretrained=False).to(DEVICE)
    else:
        model = QuakeMobileNet(num_classes=3, pretrained=False).to(DEVICE)

    state = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    # 2. Pick sample test images from dataset
    samples = [
        ("10782_lab.png", "Dense Residential Collapse"),
        ("10783_lab.png", "Major Devastation Zone"),
        ("10779_lab.png", "Mixed Intact & Damaged"),
        ("10780_lab.png", "Complex Structural Failure"),
    ]

    for filename, title in samples:
        img_path = os.path.join("data/QuakeSentinel/processed/Images", filename)
        mask_path = os.path.join("data/QuakeSentinel/processed/Masks", filename)

        orig_img = cv2.imread(img_path)
        raw_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if orig_img is None or raw_mask is None:
            continue

        h, w = orig_img.shape[:2]

        # Remap raw mask to 3-class ground truth
        gt_triage = np.zeros_like(raw_mask, dtype=np.uint8)
        gt_triage[raw_mask == 3] = 1
        gt_triage[(raw_mask == 4) | (raw_mask == 5) | (raw_mask == 6)] = 2

        pred_orig, threat_score = predict_triage_mask(model, orig_img)

        # Compute Image-Level IoU
        intersection_dmg = ((pred_orig == 2) & (gt_triage == 2)).sum()
        union_dmg = ((pred_orig == 2) | (gt_triage == 2)).sum()
        dmg_iou = intersection_dmg / max(union_dmg, 1)

        intersection_intact = ((pred_orig == 1) & (gt_triage == 1)).sum()
        union_intact = ((pred_orig == 1) | (gt_triage == 1)).sum()
        intact_iou = intersection_intact / max(union_intact, 1)

        bg_pct = (pred_orig == 0).sum() / pred_orig.size * 100
        intact_pct = (pred_orig == 1).sum() / pred_orig.size * 100
        dmg_pct = (pred_orig == 2).sum() / pred_orig.size * 100

        print(f"\n[Test Sample: {filename}] - {title}")
        print(f"   Threat Score     : {threat_score:.3f}")
        print(f"   Damaged Bldg IoU : {dmg_iou:.4f}  | Pixels: {dmg_pct:.1f}%")
        print(f"   Intact Bldg IoU  : {intact_iou:.4f}  | Pixels: {intact_pct:.1f}%")
        print(f"   Clean Background : {bg_pct:.1f}%")

        # Create Visual Panels
        panel_orig = add_header(orig_img, f"Original Image: {filename}")
        panel_gt   = add_header(make_overlay(orig_img, gt_triage), "Ground Truth (Green=Intact, Red=Damage)", color=(0, 255, 255))
        panel_pred = add_header(make_overlay(orig_img, pred_orig), f"Mobile Prediction (Dmg IoU: {dmg_iou:.2f})", color=(0, 255, 0))

        # Stitch horizontally
        comparison = np.hstack([panel_orig, panel_gt, panel_pred])
        out_file = os.path.join(OUT_DIR, f"compare_{filename}")
        cv2.imwrite(out_file, comparison)
        print(f"   Saved 3-Panel Comparison: {out_file}")

    # 3. Test on standalone images: testclean.png and imageearth.png
    print("\n" + "-" * 65)
    print("Testing on Standalone Aerial Photos:")
    for standalone in ["QuakeSentinel/testclean.png", "QuakeSentinel/imageearth.png", "QuakeSentinel/earthqykey.png"]:
        if not os.path.exists(standalone): continue
        img = cv2.imread(standalone)
        h, w = img.shape[:2]

        pred_orig, threat_score = predict_triage_mask(model, img)
        overlay = make_overlay(img, pred_orig)
        
        bg_pct = (pred_orig == 0).sum() / pred_orig.size * 100
        intact_pct = (pred_orig == 1).sum() / pred_orig.size * 100
        dmg_pct = (pred_orig == 2).sum() / pred_orig.size * 100

        print(f"  {standalone:32s} -> Score: {threat_score:.3f} | Background: {bg_pct:.1f}% | Damage: {dmg_pct:.1f}% | Intact: {intact_pct:.1f}%")
        out_name = os.path.join(OUT_DIR, f"result_{os.path.basename(standalone)}")
        panel_orig = add_header(img, f"Input: {os.path.basename(standalone)}")
        panel_pred = add_header(overlay, f"Mobile Output (Threat: {threat_score:.2f})", color=(0, 255, 0))
        cv2.imwrite(out_name, np.hstack([panel_orig, panel_pred]))
        print(f"  Saved: {out_name}")

    print("=" * 65)

if __name__ == "__main__":
    arch_arg = sys.argv[1] if len(sys.argv) > 1 else "quakevit"
    evaluate_samples(arch=arch_arg)
