"""
QuakeSentinel - Real-World Web Aerial Comparison Visualizer
Generates high-definition side-by-side triage visualizations on real-world aerial
earthquake drone & aerial imagery downloaded from international disaster repositories:
- INGV (Italy National Institute of Geophysics & Volcanology - Central Italy Earthquakes)
- Post-Disaster Drone Ruins Aerial Survey Dataset
"""

import os
import sys
import cv2
import torch
import numpy as np
import shutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quake_vit_model import QuakeViT
from test_and_visualize import predict_triage_mask, make_overlay, add_header

CKPT_PATH = "checkpoints/best_quake_triage_mobile_quakevit.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR = "outputs/web_tests"
ART_DIR = r"C:\Users\Ahmad\.gemini\antigravity-ide\brain\1534726f-11b1-4699-8e0d-776a09139bbc"

def main():
    print("=" * 70)
    print(">> Generating High-Definition Web Aerial Triage Visualizations")
    print("=" * 70)

    # 1. Load QuakeViT
    model = QuakeViT(num_classes=3, backbone_name="efficientvit_b2", fpn_dim=128, pretrained=False).to(DEVICE)
    state = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    test_targets = [
        {
            "id": "ingv_intact_building",
            "path": "outputs/web_tests/ingv_sample_5_cat1_deg1.png",
            "title": "Central Italy Aerial - Standing Intact Residential Building",
            "expected": "Intact (Green)",
        },
        {
            "id": "ingv_collapsed_structure",
            "path": "outputs/web_tests/ingv_sample_0_cat2_deg3.png",
            "title": "Central Italy Aerial - Severe Structural Collapse & Debris",
            "expected": "Damaged (Red)",
        },
        {
            "id": "ingv_rubble_perimeter",
            "path": "outputs/web_tests/ingv_sample_20_cat2_deg3.png",
            "title": "Central Italy Aerial - Heavy Rubble & Wall Failure",
            "expected": "Damaged (Red)",
        },
        {
            "id": "drone_ruins_siteA",
            "path": "outputs/web_tests/drone_ruins/55863e3d40ae8aa918738a103a13c67c.jpg",
            "title": "UAV Drone Disaster Survey - Collapsed Ruin Site A",
            "expected": "Damage Localization",
        },
        {
            "id": "drone_ruins_siteB",
            "path": "outputs/web_tests/drone_ruins/38e01dcdf4cf120cee0cff45cca24cfc.jpg",
            "title": "UAV Drone Disaster Survey - Ruin Perimeter Site B",
            "expected": "Damage Localization",
        },
    ]

    for item in test_targets:
        img_path = item["path"]
        if not os.path.exists(img_path):
            print(f"Skipping {img_path} (not found)")
            continue

        orig_img = cv2.imread(img_path)
        h, w = orig_img.shape[:2]

        # Resize if huge for display clarity
        target_display_w = 960
        target_display_h = int(h * (target_display_w / w))
        display_img = cv2.resize(orig_img, (target_display_w, target_display_h), interpolation=cv2.INTER_AREA)

        # Run triage inference
        mask, score = predict_triage_mask(model, display_img, conf_dmg=0.70, conf_intact=0.40, min_area=200)

        bg_pct = (mask == 0).mean() * 100
        intact_pct = (mask == 1).mean() * 100
        dmg_pct = (mask == 2).mean() * 100

        print(f"\n[{item['title']}]")
        print(f"  Threat Score: {score:.3f}")
        print(f"  Intact (Green): {intact_pct:5.1f}% | Damage (Red): {dmg_pct:5.1f}% | Background: {bg_pct:5.1f}%")

        # Create overlay
        overlay = make_overlay(display_img, mask, alpha=0.55)

        # Headers
        h1 = add_header(display_img, f"Raw Aerial Input: {item['title']}", color=(255, 255, 255))
        hdr_color = (0, 255, 0) if intact_pct > dmg_pct else (0, 30, 255)
        h2 = add_header(overlay, f"QuakeViT Triage [Intact: {intact_pct:.1f}% | Damage: {dmg_pct:.1f}% | Threat: {score:.2f}]", color=hdr_color)

        side_by_side = np.hstack([h1, h2])

        out_name = f"web_eval_{item['id']}.png"
        out_path = os.path.join(OUT_DIR, out_name)
        cv2.imwrite(out_path, side_by_side)
        print(f"  Saved: {out_path}")

        # Copy to artifact dir
        dst_art = os.path.join(ART_DIR, out_name)
        shutil.copy2(out_path, dst_art)
        print(f"  Artifact: {dst_art}")

    print("\n" + "=" * 70)
    print(">> All visual comparisons successfully generated!")
    print("=" * 70)

if __name__ == "__main__":
    main()
