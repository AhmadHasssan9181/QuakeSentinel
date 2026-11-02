"""
QuakeSentinel - RescueNet Dataset Processor

Processes the RescueNet UAV semantic segmentation dataset.
Maps all damaged structures (minor/major/destroyed buildings and blocked roads)
to class 1 (Damage). All other classes (background, healthy buildings, water, etc.) to 0.

Usage:
    python build_dataset.py --source "D:/big aahhh datasets/rescuenet/RescueNet/train"
    python build_dataset.py --dummy  # Generate fake data for testing

RescueNet Classes:
  0: Background, 1: Water, 2: Building No Damage, 3: Building Minor Damage,
  4: Building Major Damage, 5: Building Total Destruction, 6: Vehicle,
  7: Road-Clear, 8: Road-Blocked/Damaged
"""

import os
import argparse
import numpy as np
import cv2
import shutil
from pathlib import Path
from tqdm import tqdm

DATA_ROOT = Path(__file__).resolve().parent.parent / "data" / "QuakeSentinel" / "processed"
IMG_DIR = DATA_ROOT / "Images"
MASK_DIR = DATA_ROOT / "Masks"

def setup_dirs():
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    MASK_DIR.mkdir(parents=True, exist_ok=True)

def generate_dummy_data(samples=10):
    setup_dirs()
    print(f"🛠️ Generating {samples} DUMMY samples for smoke testing...")
    for i in range(samples):
        img_name = f"dummy_uav_damage_{i:04d}.png"
        img = np.random.randint(0, 255, (384, 384, 3), dtype=np.uint8)
        mask = np.zeros((384, 384), dtype=np.uint8)
        cv2.rectangle(mask, (100, 100), (200, 200), 1, -1)  # fake damage
        cv2.imwrite(str(IMG_DIR / img_name), img)
        cv2.imwrite(str(MASK_DIR / img_name), mask)
    print("✅ Dummy data generation complete!")

def process_rescuenet(source_dir):
    source_dir = Path(source_dir)
    images_dir = source_dir / "train-org-img"
    labels_dir = source_dir / "train-label-img"
    
    if not images_dir.exists() or not labels_dir.exists():
        print(f"❌ Error: Cannot find 'train-org-img' and 'train-label-img' in {source_dir}")
        print("Please ensure your external drive is plugged in or the path is correct.")
        return

    setup_dirs()
    
    labels = list(labels_dir.glob("*.png"))
    print(f"🚁 Found {len(labels)} RescueNet UAV samples. Processing masks...")
    
    # Corrected classes: 3 (Minor), 4 (Major), 5 (Destroyed), 8 (Road-Blocked)
    damage_classes = {3, 4, 5, 8}
    
    for label_path in tqdm(labels):
        base_name = label_path.stem.replace("_lab", "")
        src_img_path = images_dir / (base_name + '.jpg')
        if not src_img_path.exists():
            src_img_path = images_dir / (base_name + '.png')
            if not src_img_path.exists():
                continue
                
        # Read exact pixel values (raw mask)
        raw_mask = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
        if raw_mask is None:
            continue
            
        # Multiclass: retain all original labels.
            
        # Only save if there is visible damage in the image, or optionally save all
        img = cv2.imread(str(src_img_path))
        if img is None:
            continue
            
        img = cv2.resize(img, (768, 768), interpolation=cv2.INTER_LINEAR)
        final_mask = cv2.resize(raw_mask, (768, 768), interpolation=cv2.INTER_NEAREST)
        
        dest_img_path = IMG_DIR / (label_path.stem + ".png")
        dest_mask_path = MASK_DIR / (label_path.stem + ".png")
        
        # Save processed outputs
        cv2.imwrite(str(dest_mask_path), final_mask)
        cv2.imwrite(str(dest_img_path), img)

    print(f"\n✅ RescueNet binary damage processing complete!")
    print(f"Data ready at: {DATA_ROOT}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, help="Path to raw RescueNet train directory")
    parser.add_argument("--dummy", action="store_true", help="Generate dummy data for smoke test")
    args = parser.parse_args()
    
    if args.dummy:
        generate_dummy_data()
    elif args.source:
        process_rescuenet(args.source)
    else:
        print("Please provide --source <path> or use --dummy for testing.")
        print('Example: python build_dataset.py --source "D:/big aahhh datasets/rescuenet/RescueNet/train"')
