"""
QuakeSentinel - Real-World Web Aerial Evaluation Suite
Tests QuakeViT on unseen aerial images from the 2023 Turkey-Syria Earthquake:
Pre-Earthquake (Undamaged Baseline) vs. Post-Earthquake (Devastation Zone).
Generates side-by-side triage overlays and comparative analytics.
"""

import os
import sys
import glob
import cv2
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quake_vit_model import QuakeViT
from quake_mobile_model import QuakeMobileNet

CKPT_QUAKEVIT = "checkpoints/best_quake_triage_mobile_quakevit.pth"
CKPT_MOBILENET = "checkpoints/best_quake_triage_mobile_mobilenetv4.pth"
OUT_DIR = "outputs/web_tests"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

COLOR_MAP = {
    0: (0, 0, 0),        # Background (Terrain/Roads)
    1: (0, 220, 0),      # Intact Building (Vibrant Green)
    2: (0, 30, 255),     # Damaged / Collapsed Building (Vivid Red)
}

def make_overlay(img_bgr, class_mask, alpha=0.55):
    overlay = np.zeros_like(img_bgr)
    overlay[class_mask == 1] = COLOR_MAP[1]
    overlay[class_mask == 2] = COLOR_MAP[2]

    blended = img_bgr.copy()
    active = class_mask > 0
    if active.any():
        blended[active] = cv2.addWeighted(overlay, alpha, img_bgr, 1.0 - alpha, 0)[active]
    return blended

def add_header(img, text, color=(255, 255, 255)):
    h, w = img.shape[:2]
    header = np.zeros((35, w, 3), dtype=np.uint8)
    cv2.putText(header, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
    return np.vstack([header, img])

def predict_mask(model, orig_img, conf_dmg=0.65, conf_intact=0.40, min_area=150):
    h, w = orig_img.shape[:2]
    rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (512, 512), interpolation=cv2.INTER_LINEAR)
    norm = (resized.astype(np.float32) / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    tensor = torch.from_numpy(norm).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)

    with torch.no_grad():
        seg_logits, damage_score = model(tensor)
        probs = torch.softmax(seg_logits, dim=1).squeeze(0).cpu().numpy()

    prob_intact = cv2.resize(probs[1], (w, h), interpolation=cv2.INTER_LINEAR)
    prob_dmg    = cv2.resize(probs[2], (w, h), interpolation=cv2.INTER_LINEAR)

    pred = np.zeros((h, w), dtype=np.uint8)
    pred[(prob_intact >= conf_intact) & (prob_intact > prob_dmg)] = 1
    pred[(prob_dmg >= conf_dmg) & (prob_dmg > prob_intact)] = 2

    # Noise filter
    if min_area > 0:
        for c in [1, 2]:
            mask_c = (pred == c).astype(np.uint8)
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_c)
            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] < min_area:
                    pred[labels == i] = 0

    return pred, damage_score.item()

def run_evaluation():
    print("=" * 70)
    print(">> Evaluating QuakeViT on Real-World Web Aerial Imagery (2023 Turkey Quake)")
    print(f"   Checkpoint: {CKPT_QUAKEVIT}")
    print(f"   Device    : {DEVICE}")
    print("=" * 70)

    # Load Model
    model = QuakeViT(num_classes=3, backbone_name="efficientvit_b2", fpn_dim=128, pretrained=False).to(DEVICE)
    state = torch.load(CKPT_QUAKEVIT, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    test_pairs = [
        ("turkey_2023_site0_PRE_undamaged.png", "turkey_2023_site0_POST_disaster.png", "Site 0: Dense Urban District"),
        ("turkey_2023_site5_PRE_undamaged.png", "turkey_2023_site5_POST_disaster.png", "Site 5: Commercial & Residential Zone"),
        ("turkey_2023_site12_PRE_undamaged.png", "turkey_2023_site12_POST_disaster.png", "Site 12: High-Density Building Cluster"),
        ("turkey_2023_site20_PRE_undamaged.png", "turkey_2023_site20_POST_disaster.png", "Site 20: Collapsed Perimeter Zone"),
    ]

    results = []

    for pre_name, post_name, title in test_pairs:
        pre_path = os.path.join(OUT_DIR, pre_name)
        post_path = os.path.join(OUT_DIR, post_name)

        if not os.path.exists(pre_path) or not os.path.exists(post_path):
            continue

        pre_img = cv2.imread(pre_path)
        post_img = cv2.imread(post_path)

        # Run predictions
        pred_pre, score_pre = predict_mask(model, pre_img)
        pred_post, score_post = predict_mask(model, post_img)

        # Calculate metrics
        pre_bg = (pred_pre == 0).mean() * 100
        pre_intact = (pred_pre == 1).mean() * 100
        pre_dmg = (pred_pre == 2).mean() * 100

        post_bg = (pred_post == 0).mean() * 100
        post_intact = (pred_post == 1).mean() * 100
        post_dmg = (pred_post == 2).mean() * 100

        print(f"\n[{title}]")
        print(f"  PRE-QUAKE  -> Threat Score: {score_pre:.3f} | Intact (Green): {pre_intact:4.1f}% | Damage (Red): {pre_dmg:4.1f}% | Background: {pre_bg:4.1f}%")
        print(f"  POST-QUAKE -> Threat Score: {score_post:.3f} | Intact (Green): {post_intact:4.1f}% | Damage (Red): {post_dmg:4.1f}% | Background: {post_bg:4.1f}%")

        # Generate 4-panel comprehensive comparison:
        # [Pre Input] | [Pre Triage Overlay] | [Post Input] | [Post Triage Overlay]
        p_pre_in   = add_header(pre_img, f"Pre-Quake Aerial Input")
        p_pre_out  = add_header(make_overlay(pre_img, pred_pre), f"Pre-Quake Triage (Intact: {pre_intact:.1f}%)", color=(0, 255, 0))
        p_post_in  = add_header(post_img, f"Post-Quake Disaster Aerial")
        p_post_out = add_header(make_overlay(post_img, pred_post), f"Post-Quake Triage (Damage: {post_dmg:.1f}%)", color=(0, 30, 255))

        pair_comp = np.hstack([p_pre_in, p_pre_out, p_post_in, p_post_out])
        out_comp_path = os.path.join(OUT_DIR, f"compare_{pre_name.replace('_PRE_undamaged.png', '')}.png")
        cv2.imwrite(out_comp_path, pair_comp)
        print(f"  Saved 4-Panel Comparison: {out_comp_path}")

        results.append({
            "title": title,
            "pre_intact": pre_intact,
            "pre_dmg": pre_dmg,
            "post_intact": post_intact,
            "post_dmg": post_dmg,
            "comp_path": out_comp_path,
        })

    print("\n" + "=" * 70)
    print(">> Evaluation complete on unseen web imagery!")
    print("=" * 70)
    return results

if __name__ == "__main__":
    run_evaluation()
