import os
import cv2
import numpy as np

mask_dir = r"H:\quakequack\data\QuakeSentinel\processed\Masks"
mask_files = [f for f in os.listdir(mask_dir) if f.endswith('.png')]

print(f"Found {len(mask_files)} masks.")
if len(mask_files) > 0:
    for i in range(min(3, len(mask_files))):
        mask_path = os.path.join(mask_dir, mask_files[i])
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        unique_vals = np.unique(mask)
        print(f"Mask {mask_files[i]} unique values: {unique_vals}")
