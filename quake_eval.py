import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from quake_model import QuakeSentinelNet
from quake_dataset import make_split_datasets

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "PYRO")))
from utils import compute_confusion_matrix

DATA_ROOT = os.environ.get("QUAKE_DATA_ROOT", r"H:\quakequack\data\QuakeSentinel\processed")
NUM_CLASSES = 12
IGNORE_INDEX = 255
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH = 8
IMG_SIZE = 768

def evaluate():
    print(f"QuakeSentinel - Evaluation on Validation Set")
    print(f"   Data root: {DATA_ROOT}")
    print(f"   Device: {DEVICE}")
    print("-" * 60)
    
    # Load data
    _, val_ds = make_split_datasets(
        images_dir=os.path.join(DATA_ROOT, "Images"),
        masks_dir=os.path.join(DATA_ROOT, "Masks"),
        img_size=IMG_SIZE,
    )
    val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=0)
    
    # Load model
    model = QuakeSentinelNet(num_classes=NUM_CLASSES, pretrained=False).to(DEVICE)
    ckpt_path = os.path.join("..", "checkpoints", "best_quake_damage.pth")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join("..", "checkpoints", "best_quake.pth")
        
    if not os.path.exists(ckpt_path):
        print(f"[ERROR] No checkpoint found in {ckpt_path}")
        return
        
    print(f"Loading checkpoint {ckpt_path}...")
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.eval()
    
    hist = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.float64)
    
    print("Running inference...")
    with torch.no_grad():
        for imgs, masks in tqdm(val_loader, desc="Evaluating"):
            imgs = imgs.to(DEVICE)
            masks = masks.numpy()
            
            seg_logits, _ = model(imgs)
            preds = torch.argmax(seg_logits, dim=1).cpu().numpy()
            
            hist += compute_confusion_matrix(preds, masks, NUM_CLASSES, ignore_index=IGNORE_INDEX)
            
    # Calculate metrics
    TP = np.diag(hist)
    FP = hist.sum(axis=0) - TP
    FN = hist.sum(axis=1) - TP
    
    Precision = TP / (TP + FP + 1e-9)
    Recall = TP / (TP + FN + 1e-9)
    F1 = 2 * Precision * Recall / (Precision + Recall + 1e-9)
    IoU = TP / (TP + FP + FN + 1e-9)
    
    Overall_Accuracy = TP.sum() / (hist.sum() + 1e-9)
    
    print("\n" + "="*80)
    print(f"{'Class':<15} | {'Precision':<10} | {'Recall':<10} | {'F1 Score':<10} | {'IoU':<10}")
    print("-" * 80)
    
    valid_classes = []
    for i in range(NUM_CLASSES):
        # Only print classes that exist in ground truth or predictions
        if hist.sum(axis=1)[i] > 0 or hist.sum(axis=0)[i] > 0:
            c_name = f"Class {i}"
            print(f"{c_name:<15} | {Precision[i]:.4f}     | {Recall[i]:.4f}     | {F1[i]:.4f}     | {IoU[i]:.4f}")
            valid_classes.append(i)
            
    print("-" * 80)
    
    macro_precision = np.mean(Precision[valid_classes])
    macro_recall = np.mean(Recall[valid_classes])
    macro_f1 = np.mean(F1[valid_classes])
    macro_iou = np.mean(IoU[valid_classes])
    
    print(f"{'Macro-Average':<15} | {macro_precision:.4f}     | {macro_recall:.4f}     | {macro_f1:.4f}     | {macro_iou:.4f}")
    print(f"Overall Accuracy: {Overall_Accuracy:.4f}")
    print("="*80)

if __name__ == "__main__":
    evaluate()
