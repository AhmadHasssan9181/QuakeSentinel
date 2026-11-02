"""
QuakeSentinel Training Script

Usage:
    python quake_train.py
    python quake_train.py --epochs 80 --batch 8 --img_size 384
"""

import os
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
import numpy as np
import cv2
from tqdm import tqdm

from quake_model import QuakeSentinelNet, QuakeLoss
from quake_dataset import QuakeDataset, make_split_datasets

# Assume utils from PYRO are reused or we can write our own simple metrics.
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "PYRO")))
from utils import compute_confusion_matrix, compute_miou

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────

DATA_ROOT = os.environ.get(
    "QUAKE_DATA_ROOT",
    r"H:\quakequack\data\QuakeSentinel\processed"
)

NUM_CLASSES  = 12
IMG_SIZE     = 768     
BATCH        = 8       
EPOCHS       = 300     
LR           = 3e-4
WARMUP_EPOCHS= 5       
WEIGHT_DECAY = 1e-4
DAMAGE_WEIGHT= -1.0
IGNORE_INDEX = 255
SAVE_DIR     = "checkpoints"
USE_AMP      = True
RESUME_CKPT  = "" 
EARLY_STOP_PATIENCE = 20   
NUM_WORKERS  = 0       

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def parse_args():
    p = argparse.ArgumentParser(description="Train QuakeSentinel damage segmentation model")
    p.add_argument("--epochs",   type=int,   default=EPOCHS)
    p.add_argument("--batch",    type=int,   default=BATCH)
    p.add_argument("--img_size", type=int,   default=IMG_SIZE)
    p.add_argument("--lr",       type=float, default=LR)
    p.add_argument("--data",     type=str,   default=DATA_ROOT)
    p.add_argument("--damage_weight", type=float, default=DAMAGE_WEIGHT,
                   help="Positive class weight. Use <=0 to auto-estimate from train split.")
    return p.parse_args()


def get_loaders(data_root: str, img_size: int, batch: int):
    train_ds, val_ds = make_split_datasets(
        images_dir=os.path.join(data_root, "Images"),
        masks_dir=os.path.join(data_root, "Masks"),
        img_size=img_size,
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=(DEVICE == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=(DEVICE == "cuda"),
    )
    return train_loader, val_loader, train_ds


def estimate_class_weights(train_subset, num_classes=12, max_samples: int = 400) -> torch.Tensor:
    import numpy as np
    import cv2
    import torch
    base_ds = train_subset.dataset if hasattr(train_subset, 'dataset') else train_subset
    indices = list(train_subset.indices) if hasattr(train_subset, 'indices') else list(range(len(base_ds)))
    
    counts = np.zeros(num_classes, dtype=np.float64)
    if not hasattr(base_ds, 'mask_paths') or len(indices) == 0:
        return torch.ones(num_classes)

    if len(indices) > max_samples:
        pick = np.linspace(0, len(indices) - 1, num=max_samples, dtype=int)
        sample_indices = [indices[i] for i in pick]
    else:
        sample_indices = indices

    for idx in sample_indices:
        if idx >= len(base_ds.mask_paths): continue
        mask = cv2.imread(str(base_ds.mask_paths[idx]), cv2.IMREAD_GRAYSCALE)
        if mask is None: continue
        c = np.bincount(mask.flatten(), minlength=num_classes)
        counts += c[:num_classes]

    counts = np.maximum(counts, 1.0)
    
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes
    weights = np.clip(weights, 0.1, 10.0)
    
    # Class 0 doesn't exist natively, zero its weight
    if counts[0] < 10000:
        weights[0] = 0.0
        
    return torch.tensor(weights, dtype=torch.float32)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    hist = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    total_score = 0.0
    n = 0
    for imgs, masks in loader:
        imgs  = imgs.to(DEVICE, non_blocking=True)
        masks = masks.to(DEVICE, non_blocking=True)
        seg_logits, damage_score = model(imgs)
        preds = torch.argmax(seg_logits, dim=1).cpu().numpy()
        labs  = masks.cpu().numpy()
        hist += compute_confusion_matrix(preds, labs, NUM_CLASSES, ignore_index=IGNORE_INDEX)
        total_score += damage_score.sum().item()
        n += imgs.size(0)
    miou, per_class = compute_miou(hist)
    avg_score = total_score / max(n, 1)
    return miou, per_class, avg_score


def get_warmup_lr(epoch, warmup_epochs, base_lr):
    if epoch < warmup_epochs: return base_lr * (epoch + 1) / warmup_epochs
    return base_lr


def train():
    args = parse_args()
    os.makedirs(SAVE_DIR, exist_ok=True)

    print(f"🌍 QuakeSentinel — Damage Segmentation Training")
    print(f"   Device   : {DEVICE}")
    print(f"   Data root: {args.data}")
    print(f"   Img size : {args.img_size}px  |  Batch: {args.batch}  |  Epochs: {args.epochs}")
    print("─" * 60)

    train_loader, val_loader, train_ds = get_loaders(args.data, args.img_size, args.batch)

    model     = QuakeSentinelNet(num_classes=NUM_CLASSES, pretrained=True).to(DEVICE)
    class_weights = estimate_class_weights(train_ds, num_classes=NUM_CLASSES)
    print(f"Class Weights: {class_weights.numpy().round(3)}")
    criterion = QuakeLoss(num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX, weights=class_weights.to(DEVICE))

    encoder_params = list(model.model.backbone.parameters())
    decoder_params = [p for p in model.parameters() if not any(p is ep for ep in encoder_params)]
    optimizer = optim.AdamW([
        {'params': encoder_params, 'lr': args.lr * 0.1},
        {'params': decoder_params, 'lr': args.lr},
    ], weight_decay=WEIGHT_DECAY)

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=1, eta_min=1e-6
    )
    scaler = GradScaler("cuda", enabled=USE_AMP)

    best_miou = 0.0
    best_dmg_iou = 0.0
    no_improve_epochs = 0

    if RESUME_CKPT and os.path.isfile(RESUME_CKPT):
        ckpt = torch.load(RESUME_CKPT, map_location=DEVICE)
        missing, unexpected = model.load_state_dict(ckpt, strict=False)
        print(f"   ✅ Resumed weights from {RESUME_CKPT}")

    for epoch in range(1, args.epochs + 1):
        if epoch <= WARMUP_EPOCHS:
            warmup_lr = get_warmup_lr(epoch - 1, WARMUP_EPOCHS, args.lr)
            for i, pg in enumerate(optimizer.param_groups):
                pg['lr'] = warmup_lr * (0.1 if i == 0 else 1.0)

        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:>3}/{args.epochs}", ncols=110)
        running_loss = 0.0
        running_tv = 0.0
        running_focal = 0.0

        for imgs, masks in pbar:
            imgs  = imgs.to(DEVICE, non_blocking=True)
            masks = masks.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=USE_AMP):
                seg_logits, _ = model(imgs)
                loss, tv_l, fc_l = criterion(seg_logits, masks)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

            running_loss  += loss.item()
            running_tv    += tv_l.item()
            running_focal += fc_l.item()
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'tv': f'{tv_l.item():.4f}',
                'focal': f'{fc_l.item():.4f}',
            })

        if epoch > WARMUP_EPOCHS: scheduler.step(epoch - WARMUP_EPOCHS)
        avg_loss = running_loss / len(train_loader)
        cur_lr   = optimizer.param_groups[1]['lr']

        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'loss': avg_loss}, os.path.join(SAVE_DIR, "latest_quake.pth"))

        miou, per_class, avg_score = evaluate(model, val_loader)
        dmg_iou = float(np.mean([per_class[i] for i in [4, 5, 6, 8] if i < len(per_class)])) if len(per_class) > 8 else 0.0
        bg_iou  = per_class[0] if len(per_class) > 0 else 0.0

        print(f"  Ep {epoch:>3} | Loss {avg_loss:.4f} | mIoU {miou:.4f} | DmgIoU {dmg_iou:.4f} | BgIoU {bg_iou:.4f} | Score {avg_score:.3f} | LR {cur_lr:.2e}")

        if miou > best_miou:
            best_miou = miou
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_quake.pth"))

        if dmg_iou > best_dmg_iou:
            best_dmg_iou = dmg_iou
            no_improve_epochs = 0
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_quake_damage.pth"))
            print(f"  🌍 New best DmgIoU: {best_dmg_iou:.4f} → saved best_quake_damage.pth")
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= EARLY_STOP_PATIENCE:
                print(f"\n  ⏹  Early stop: DmgIoU flat for {EARLY_STOP_PATIENCE} epochs.")
                break

    print("─" * 60)
    print(f"✅ Training complete. Best mIoU: {best_miou:.4f}  |  Best DmgIoU: {best_dmg_iou:.4f}")

if __name__ == "__main__":
    train()
