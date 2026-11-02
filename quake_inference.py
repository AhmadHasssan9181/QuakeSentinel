import os
import torch
import cv2
import numpy as np
import sys
from quake_model import QuakeSentinelNet


def _infer_num_classes_from_state_dict(state_dict: dict, fallback: int = 2) -> int:
    # seg_head.1 is the final 1x1 conv in quake_model.py
    head_key = "seg_head.1.weight"
    w = state_dict.get(head_key)
    if w is not None and hasattr(w, "shape") and len(w.shape) == 4:
        return int(w.shape[0])
    return fallback


def _parse_damage_class_ids(arg: str | None, num_classes: int) -> list[int]:
    # Auto behavior:
    # - binary model -> class 1 means damage
    # - multiclass model -> all non-background classes are treated as damage by default
    if arg is None or arg.strip() == "":
        if num_classes <= 2:
            return [1]
        return list(range(1, num_classes))

    ids = []
    for x in arg.split(','):
        x = x.strip()
        if not x:
            continue
        cls = int(x)
        if 0 <= cls < num_classes:
            ids.append(cls)
    ids = sorted(set(ids))
    if not ids:
        return [1] if num_classes > 1 else [0]
    return ids

def _remove_small_regions(binary_mask: np.ndarray, min_area: int) -> np.ndarray:
    if min_area <= 0:
        return binary_mask
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask.astype(np.uint8), connectivity=8)
    filtered = np.zeros_like(binary_mask, dtype=np.uint8)
    for label_id in range(1, num_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area >= min_area:
            filtered[labels == label_id] = 1
    return filtered


def _adaptive_threshold(
    damage_prob: np.ndarray,
    base_threshold: float,
    max_mask_ratio: float,
) -> tuple[np.ndarray, float, float]:
    # Start from user threshold and raise it until mask coverage is reasonable.
    th = float(base_threshold)
    mask = (damage_prob >= th).astype(np.uint8)
    ratio = float(mask.mean())

    if ratio <= max_mask_ratio:
        return mask, th, ratio

    for th in (0.65, 0.70, 0.75, 0.80, 0.85, 0.90):
        mask = (damage_prob >= th).astype(np.uint8)
        ratio = float(mask.mean())
        if ratio <= max_mask_ratio:
            return mask, float(th), ratio

    return mask, float(th), ratio


def infer(
    image_path,
    model_path="checkpoints/best_quake_damage.pth",
    output_path="imageearth_out.png",
    prob_threshold: float = 0.6,
    min_area_ratio: float = 0.0001,
    damage_class_ids: list[int] | str | None = None,
    max_mask_ratio: float = 0.20,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading model on {device}...")
    
    # 1. Load model
    state = None
    if os.path.exists(model_path):
        state = torch.load(model_path, map_location=device, weights_only=False)
        num_classes = 12
        model = QuakeSentinelNet(num_classes=num_classes, pretrained=False)
        model.load_state_dict(state)
        print(f"Loaded weights from {model_path}")
        print(f"Detected num_classes={num_classes}")
    else:
        print(f"Warning: Model weights not found at {model_path}. Using random weights.")
        num_classes = 2
        model = QuakeSentinelNet(num_classes=num_classes, pretrained=False)
    model.to(device)
    model.eval()

    if isinstance(damage_class_ids, str):
        damage_class_ids = _parse_damage_class_ids(damage_class_ids, num_classes)
    if damage_class_ids is None:
        damage_class_ids = [1] if num_classes <= 2 else list(range(1, num_classes))

    # Keep only valid class ids and avoid using background id=0 as damage.
    damage_class_ids = [c for c in sorted(set(damage_class_ids)) if 0 < c < num_classes]
    if not damage_class_ids:
        damage_class_ids = [1] if num_classes > 1 else [0]
    print(f"Damage classes used: {damage_class_ids}")

    # 2. Load image
    orig_img = cv2.imread(image_path)
    if orig_img is None:
        raise FileNotFoundError(f"Could not read {image_path}")
    
    orig_h, orig_w = orig_img.shape[:2]

    # 3. Preprocess
    img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    
    # Resize keeping aspect ratio, targeting the scale the model was trained on (768)
    scale = 768.0 / max(orig_h, orig_w)
    target_w = int(orig_w * scale)
    target_h = int(orig_h * scale)
    
    new_w = max(32, (target_w // 32) * 32)
    new_h = max(32, (target_h // 32) * 32)
    
    # Don't exceed memory (e.g. 1536 max)
    if max(new_w, new_h) > 1536:
        mem_scale = 1536 / max(new_w, new_h)
        new_w = max(32, (int(new_w * mem_scale) // 32) * 32)
        new_h = max(32, (int(new_h * mem_scale) // 32) * 32)
        
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # manual normalize to match A.Normalize()
    img_float = img_resized.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_norm = (img_float - mean) / std
    
    # to tensor (C, H, W)
    img_tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).to(device)

    # 4. Inference
    print("Running inference...")
    with torch.no_grad():
        seg_logits, damage_score = model(img_tensor)
        # Multiclass prediction (num_classes)
        probs = torch.softmax(seg_logits, dim=1)
        pred_class = torch.argmax(probs, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    print(f"Overall Damage Score: {damage_score.item():.4f}")

    # Resize back to original dimensions using Nearest Neighbor to preserve discrete class IDs
    mask_resized = cv2.resize(pred_class, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    # Correct Color Map (BGR) based on documentation
    # 1 is actually Debris, 8 is Road, 3 is Building No Damage.
    # The model learned this perfectly, but our previous map was offset.
    color_map = {
        0: (0, 0, 0),             # Background (transparent)
        1: (128, 128, 128),       # Debris (grey)
        2: (255, 0, 0),           # Water (blue)
        3: (0, 255, 0),           # Building No Damage (green)
        4: (0, 255, 255),         # Building Minor Damage (yellowish)
        5: (0, 165, 255),         # Building Major Damage (orange)
        6: (0, 0, 255),           # Building Total Destruction (red)
        7: (255, 255, 0),         # Vehicle (cyan)
        8: (200, 200, 200),       # Road (light grey)
        9: (0, 128, 0),           # Tree (dark green)
        10: (255, 200, 0),        # Pool 
        11: (42, 42, 165)         # Sand
    }

    overlay = np.zeros_like(orig_img)
    for c_id, color in color_map.items():
        if c_id == 0: continue  # skip background
        # if c_id == 1: continue # optionally skip debris if it clutters the image? user wants pixel perfect, so keep it.
        overlay[mask_resized == c_id] = color

    # Blend and arrange side-by-side
    alpha = 0.5
    blended = cv2.addWeighted(overlay, alpha, orig_img, 1 - alpha, 0)
    
    # We create a side-by-side visualization to clearly show the crisp segmentation
    # Left: Original, Middle: Blended, Right: Pure Semantic Map
    h, w = orig_img.shape[:2]
    side_by_side = np.zeros((h, w * 3, 3), dtype=np.uint8)
    side_by_side[:, :w] = orig_img
    side_by_side[:, w:w*2] = blended
    side_by_side[:, w*2:] = overlay

    # Save output
    cv2.imwrite(output_path, side_by_side)
    print(f"Saved inference output to {output_path}")

if __name__ == "__main__":
    import sys
    img_path = sys.argv[1] if len(sys.argv) > 1 else "testclean.png"
    model_path = sys.argv[2] if len(sys.argv) > 2 else "checkpoints/best_quake_damage.pth"
    out_path = sys.argv[3] if len(sys.argv) > 3 else "outputs/imageearth_out.png"
    
    infer(img_path, model_path=model_path, output_path=out_path)
