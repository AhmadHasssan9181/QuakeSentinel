"""
QuakeSentinel - EIDSeg Multi-Earthquake Dataset Pipeline
Downloads huilihuang413/EIDSeg (covers 9 major global earthquakes: 2008-2023)
from Hugging Face and converts CVAT XML annotations into 3-class triage masks:
  0: Background / Terrain / Roads
  1: Intact Building (UD_Building)
  2: Damaged Building / Debris / Rubble (D_Building, Debris, D_Road)
"""

import os
import sys
import glob
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm
from huggingface_hub import snapshot_download

DATA_DIR = Path("data")
EIDSEG_RAW = DATA_DIR / "EIDSeg_raw"
PROCESSED_DIR = DATA_DIR / "QuakeSentinel" / "processed"
IMG_OUT = PROCESSED_DIR / "Images"
MASK_OUT = PROCESSED_DIR / "Masks"

TARGET_SIZE = (768, 768)

LABEL_MAP = {
    "UD_Building": 1,     # Intact Building -> 1
    "D_Building": 2,      # Damaged Building -> 2
    "Debris": 2,          # Pulverized Rubble / Collapsed Structure -> 2
    "D_Road": 2,          # Heavily Damaged / Fissured Road -> 2
    "UD_Road": 0,         # Undamaged Road -> 0 (Background)
    "Undesignated": 0,    # Background -> 0
}

def clean_low_res_clips():
    """Remove 64x64 pixel road clips from previous test downloads to avoid dataset contamination."""
    print(">> Cleaning low-resolution 64x64 road clips from dataset...")
    removed = 0
    for img_p in glob.glob(str(IMG_OUT / "uav_damage_*.png")):
        mask_p = MASK_OUT / os.path.basename(img_p)
        try:
            os.remove(img_p)
            if os.path.exists(mask_p):
                os.remove(mask_p)
            removed += 1
        except Exception:
            pass
    print(f"   [OK] Removed {removed} tiny 64x64 road clips.")

def download_dataset():
    print("=" * 65)
    print(">> Step 1: Downloading EIDSeg from Hugging Face (~789 MB)...")
    print("   Repository: huilihuang413/EIDSeg (9 Global Earthquakes)")
    print("=" * 65)
    EIDSEG_RAW.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id="huilihuang413/EIDSeg",
        repo_type="dataset",
        local_dir=str(EIDSEG_RAW),
        resume_download=True,
        max_workers=16,
    )
    print("   [OK] EIDSeg raw download complete!")

def process_split(split_name: str):
    split_dir = EIDSEG_RAW / "data" / split_name
    xml_path = split_dir / f"{split_name}.xml"
    images_dir = split_dir / "images" / "default"

    if not xml_path.exists():
        print(f"   [WARN] XML not found: {xml_path}")
        return 0

    print(f"\n>> Processing EIDSeg [{split_name}] annotations from {xml_path}...")
    tree = ET.parse(xml_path)
    root = tree.getroot()

    count = 0
    skipped = 0

    for img_node in tqdm(root.findall("image"), desc=f"Rasterizing {split_name}"):
        img_name = img_node.get("name")
        w_orig = int(img_node.get("width", 0))
        h_orig = int(img_node.get("height", 0))

        src_path = images_dir / img_name
        if not src_path.exists():
            skipped += 1
            continue

        orig_img = cv2.imread(str(src_path))
        if orig_img is None:
            skipped += 1
            continue

        h_img, w_img = orig_img.shape[:2]
        if w_orig == 0 or h_orig == 0:
            w_orig, h_orig = w_img, h_img

        # Initialize full-resolution mask with 0 (Background)
        full_mask = np.zeros((h_img, w_img), dtype=np.uint8)

        # Parse polygons
        for poly_node in img_node.findall("polygon"):
            label = poly_node.get("label", "")
            target_class = LABEL_MAP.get(label, 0)
            if target_class == 0:
                continue

            pts_str = poly_node.get("points", "")
            if not pts_str:
                continue

            try:
                pts = np.array([
                    [float(coord) for coord in pt.split(",")]
                    for pt in pts_str.split(";") if pt
                ], dtype=np.int32)
                cv2.fillPoly(full_mask, [pts], color=target_class)
            except Exception:
                continue

        # Resize both image and mask to standard 768x768
        resized_img = cv2.resize(orig_img, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)
        resized_mask = cv2.resize(full_mask, TARGET_SIZE, interpolation=cv2.INTER_NEAREST)

        base_stem = Path(img_name).stem
        out_name = f"eidseg_{split_name}_{base_stem}.png"

        dest_img = IMG_OUT / out_name
        dest_mask = MASK_OUT / out_name

        cv2.imwrite(str(dest_img), resized_img)
        cv2.imwrite(str(dest_mask), resized_mask)
        count += 1

    print(f"   [OK] Processed {count} samples from {split_name} (Skipped {skipped}).")
    return count

def main():
    IMG_OUT.mkdir(parents=True, exist_ok=True)
    MASK_OUT.mkdir(parents=True, exist_ok=True)

    # 1. Clean out the 64x64 low-res road clips
    clean_low_res_clips()

    # 2. Download EIDSeg
    download_dataset()

    # 3. Process all 3 splits: train, val, test
    total_new = 0
    for split in ["train", "val", "test"]:
        total_new += process_split(split)

    # 4. Total dataset summary
    all_imgs = list(IMG_OUT.glob("*.png"))
    all_masks = list(MASK_OUT.glob("*.png"))

    print("\n" + "=" * 65)
    print(">> DATASET UPGRADE COMPLETE!")
    print(f"   Newly Ingested Earthquake Samples : {total_new}")
    print(f"   Total High-Res Aerial Images      : {len(all_imgs)}")
    print(f"   Total Triage Masks                : {len(all_masks)}")
    print("=" * 65)

if __name__ == "__main__":
    main()
