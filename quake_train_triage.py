"""
QuakeSentinel - 3-Class Triage Mobile Training Pipeline
Trains the Anti-Tunnel-Vision MobileNetV4 / EfficientViT from scratch
on the 3-Class Disaster Triage task using the RTX 3090.

Classes:
    0: Background / Terrain / Roads / Trees / Water
    1: Intact Building (Standing, undamaged roofs)
    2: Damaged & Destroyed Building (Rubble, collapsed walls, structural damage)

Anti-Tunnel-Vision features:
    - Macro global context gating across pyramid levels.
    - Multi-scale dilated depthwise receptive fields (local crack + neighborhood context).
    - Multi-scale altitude jitter augmentations (scale 0.4x - 1.25x).

Usage:
    python quake_train_triage.py --epochs 35 --batch 16 --img_size 512 --arch mobilenetv4
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
import numpy as np
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "PYRO")))

from quake_mobile_model import QuakeMobileNet
from quake_vit_model import QuakeViT
from quake_dataset import make_split_datasets
from utils import compute_confusion_matrix, compute_miou

DATA_ROOT = os.environ.get(
    "QUAKE_DATA_ROOT",
    os.path.join(os.path.dirname(__file__), "..", "data", "QuakeSentinel", "processed")
)
SAVE_DIR = os.path.join(os.path.dirname(__file__), "..", "checkpoints")

NUM_CLASSES = 3
IGNORE_INDEX = 255
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ARCH_MAP = {
    "mobilenetv4": "mobilenetv4_conv_small.e1200_r224_in1k",
    "quakevit": "efficientvit_b2",
    "efficientvit": "efficientvit_b0",
}

class TriageLoss(nn.Module):
    """
    Focal Cross-Entropy + Multiclass Dice Loss tailored for building & damage imbalance.
    """
    def __init__(self, weights=None, gamma=2.0):
        super().__init__()
        self.weights = weights
        self.gamma = gamma

    def forward(self, logits, targets):
        # 1. Focal Cross-Entropy
        ce = F.cross_entropy(logits, targets, weight=self.weights, reduction='none')
        pt = torch.exp(-ce)
        focal_ce = (((1.0 - pt) ** self.gamma) * ce).mean()

        # 2. Multi-Class Dice (specifically focus on Building classes 1 and 2)
        probs = torch.softmax(logits, dim=1)
        targets_onehot = F.one_hot(targets.clamp(0, NUM_CLASSES - 1), num_classes=NUM_CLASSES).permute(0, 3, 1, 2).float()

        intersection = (probs * targets_onehot).sum(dim=(2, 3))
        cardinality  = probs.sum(dim=(2, 3)) + targets_onehot.sum(dim=(2, 3))
        dice_scores  = (2.0 * intersection + 1e-5) / (cardinality + 1e-5)

        # Weight building classes (1 & 2) more in Dice loss
        dice_loss = 1.0 - dice_scores
        # Dice focus: 0.2*Bg + 0.4*Intact + 0.4*Damaged
        dice_weights = torch.tensor([0.2, 0.4, 0.4], device=logits.device).unsqueeze(0)
        weighted_dice = (dice_loss * dice_weights).sum(dim=1).mean()

        total = focal_ce + 1.2 * weighted_dice
        return total, focal_ce.detach(), weighted_dice.detach()


def parse_args():
    p = argparse.ArgumentParser(description="Train QuakeSentinel 3-Class Triage Mobile Model")
    p.add_argument("--arch", type=str, default="quakevit", choices=["mobilenetv4", "quakevit", "efficientvit"])
    p.add_argument("--epochs", type=int, default=35)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--img_size", type=int, default=512)
    p.add_argument("--lr", type=float, default=4e-4)
    p.add_argument("--data", type=str, default=DATA_ROOT)
    return p.parse_args()


@torch.no_grad()
def evaluate_triage(model, loader):
    model.eval()
    hist = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    total_score = 0.0
    n = 0
    for imgs, masks in loader:
        imgs = imgs.to(DEVICE, non_blocking=True)
        masks = masks.to(DEVICE, non_blocking=True)
        with autocast("cuda"):
            seg_logits, damage_score = model(imgs)
        preds = torch.argmax(seg_logits, dim=1).cpu().numpy()
        labs = masks.cpu().numpy()
        hist += compute_confusion_matrix(preds, labs, NUM_CLASSES, ignore_index=IGNORE_INDEX)
        total_score += damage_score.sum().item()
        n += imgs.size(0)

    miou, per_class = compute_miou(hist)
    avg_score = total_score / max(n, 1)
    return miou, per_class, avg_score


def main():
    args = parse_args()
    os.makedirs(SAVE_DIR, exist_ok=True)

    print("=" * 70)
    print(f">> QuakeSentinel 3-Class Triage Training Pipeline (Anti-Tunnel-Vision)")
    print(f"   Compute Device : {DEVICE} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"   Architecture   : {args.arch} ({ARCH_MAP[args.arch]})")
    print(f"   Resolution     : {args.img_size}x{args.img_size} | Batch Size: {args.batch} | Epochs: {args.epochs}")
    print(f"   Classes        : 0=Background, 1=Intact Building, 2=Damaged Structure")
    print("=" * 70)

    # 1. Load Split Datasets (using remap_to_triage)
    train_ds, val_ds = make_split_datasets(
        images_dir=os.path.join(args.data, "Images"),
        masks_dir=os.path.join(args.data, "Masks"),
        img_size=args.img_size,
        val_fraction=0.15,
        seed=42,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0, pin_memory=True)

    # 2. Instantiate Anti-Tunnel-Vision Model
    if args.arch == "quakevit":
        model = QuakeViT(
            num_classes=NUM_CLASSES,
            backbone_name=ARCH_MAP[args.arch],
            fpn_dim=128,
            pretrained=True,
        ).to(DEVICE)
    else:
        model = QuakeMobileNet(
            num_classes=NUM_CLASSES,
            backbone_name=ARCH_MAP[args.arch],
            fpn_dim=64,
            pretrained=True,
        ).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[OK] Initialized {args.arch} model with {total_params / 1e6:.2f}M parameters")

    # 3. Class Weights: Background=0.4, Intact=2.5, Damaged=3.5 (Counter imbalance)
    class_weights = torch.tensor([0.4, 2.5, 3.5], device=DEVICE)
    criterion = TriageLoss(weights=class_weights, gamma=2.0)

    # Differentiated learning rates: smaller for pretrained backbone, higher for decoder
    encoder_params = list(model.encoder.parameters())
    head_params = [p for p in model.parameters() if not any(p is ep for ep in encoder_params)]
    optimizer = optim.AdamW([
        {"params": encoder_params, "lr": args.lr * 0.2},
        {"params": head_params, "lr": args.lr},
    ], weight_decay=1e-4)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = GradScaler("cuda")

    best_dmg_iou = 0.0
    best_miou = 0.0
    save_path = os.path.join(SAVE_DIR, f"best_quake_triage_mobile_{args.arch}.pth")

    print("\nStarting Training on RTX 3090...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_focal = 0.0
        running_dice = 0.0

        pbar = tqdm(train_loader, desc=f"Ep {epoch:>2}/{args.epochs}", ncols=115)
        for imgs, masks in pbar:
            imgs  = imgs.to(DEVICE, non_blocking=True)
            masks = masks.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast("cuda"):
                seg_logits, damage_score = model(imgs)
                loss, focal_l, dice_l = criterion(seg_logits, masks)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

            running_loss  += loss.item()
            running_focal += focal_l.item()
            running_dice  += dice_l.item()

            pbar.set_postfix({
                "loss": f"{loss.item():.3f}",
                "focal": f"{focal_l.item():.3f}",
                "dice": f"{dice_l.item():.3f}",
            })

        scheduler.step()
        avg_loss = running_loss / len(train_loader)

        # Validation
        miou, per_class, avg_score = evaluate_triage(model, val_loader)
        bg_iou     = per_class[0] if len(per_class) > 0 else 0.0
        intact_iou = per_class[1] if len(per_class) > 1 else 0.0
        dmg_iou    = per_class[2] if len(per_class) > 2 else 0.0

        print(f"  Ep {epoch:>2} | Loss {avg_loss:.4f} | mIoU {miou:.4f} | DmgIoU {dmg_iou:.4f} | IntactIoU {intact_iou:.4f} | BgIoU {bg_iou:.4f} | Score {avg_score:.3f}")

        # Combined building detection score: DmgIoU + 0.5*IntactIoU
        combined_score = dmg_iou + 0.5 * intact_iou
        if dmg_iou > best_dmg_iou:
            best_dmg_iou = dmg_iou
            best_miou = miou
            torch.save(model.state_dict(), save_path)
            print(f"  * New Best DmgIoU: {best_dmg_iou:.4f} (mIoU: {best_miou:.4f}) -> Saved to {save_path}")

    print("\n" + "=" * 70)
    print(f"[OK] 3-Class Triage Training Complete!")
    print(f"   Best DmgIoU    : {best_dmg_iou:.4f}")
    print(f"   Best mIoU      : {best_miou:.4f}")
    print(f"   Saved Checkpoint: {save_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
