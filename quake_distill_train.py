"""
QuakeSentinel - Knowledge Distillation Training Pipeline
Leverages the RTX 3090 (24GB VRAM) to distill the 158MB DeepLabV3-ResNet50 Teacher
into an ultra-fast, mobile-deployable Student (MobileNetV4-FPN or EfficientViT-FPN).

Usage:
    python quake_distill_train.py --arch mobilenetv4 --epochs 40 --batch 16 --img_size 512
    python quake_distill_train.py --arch efficientvit --epochs 40 --batch 16 --img_size 512
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

from quake_model import QuakeSentinelNet, QuakeLoss
from quake_mobile_model import QuakeMobileNet
from quake_dataset import make_split_datasets

# Import evaluation metrics from PYRO utils
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "PYRO")))
from utils import compute_confusion_matrix, compute_miou

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG & HYPERPARAMETERS
# ──────────────────────────────────────────────────────────────────────────────

DATA_ROOT = os.environ.get(
    "QUAKE_DATA_ROOT",
    os.path.join(os.path.dirname(__file__), "..", "data", "QuakeSentinel", "processed")
)
TEACHER_CKPT = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "best_quake_damage.pth")
SAVE_DIR = os.path.join(os.path.dirname(__file__), "..", "checkpoints")

NUM_CLASSES = 12
IGNORE_INDEX = 255
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ARCH_MAP = {
    "mobilenetv4": "mobilenetv4_conv_small.e1200_r224_in1k",
    "efficientvit": "efficientvit_b0",
}

class DistillationLoss(nn.Module):
    """
    Blends hard ground-truth supervision with soft teacher dark knowledge.
    """
    def __init__(self, num_classes=12, temperature=4.0, alpha=0.6, ignore_index=255, class_weights=None):
        super().__init__()
        self.temp = temperature
        self.alpha = alpha  # Weight for ground truth vs teacher
        self.hard_loss = QuakeLoss(num_classes=num_classes, ignore_index=ignore_index, weights=class_weights)
        self.kl_div = nn.KLDivLoss(reduction="batchmean")
        self.score_mse = nn.MSELoss()

    def forward(self, student_logits, student_score, teacher_logits, teacher_score, targets):
        # 1. Hard label loss (Focal + Dice)
        hard_l, ce_l, dice_l = self.hard_loss(student_logits, targets)

        # 2. Soft logit distillation (KL Divergence with Temperature T, normalized per pixel)
        t_soft = F.log_softmax(student_logits / self.temp, dim=1)
        s_soft = F.softmax(teacher_logits / self.temp, dim=1)
        spatial_pixels = student_logits.size(2) * student_logits.size(3)
        kd_l = (self.kl_div(t_soft, s_soft) * (self.temp ** 2)) / spatial_pixels

        # 3. Global image threat score distillation
        score_l = self.score_mse(student_score, teacher_score)

        # Total combined loss
        total_loss = (self.alpha * hard_l) + ((1.0 - self.alpha) * kd_l) + (0.2 * score_l)
        return total_loss, hard_l.detach(), kd_l.detach(), score_l.detach()


def parse_args():
    p = argparse.ArgumentParser(description="Distill ResNet50 into Mobile Architecture on RTX 3090")
    p.add_argument("--arch", type=str, default="mobilenetv4", choices=["mobilenetv4", "efficientvit"],
                   help="Mobile backbone architecture")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=16, help="Batch size (RTX 3090 easily fits 16-32 at 512px)")
    p.add_argument("--img_size", type=int, default=512)
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--data", type=str, default=DATA_ROOT)
    p.add_argument("--teacher_ckpt", type=str, default=TEACHER_CKPT)
    p.add_argument("--temp", type=float, default=4.0, help="Distillation temperature")
    return p.parse_args()


@torch.no_grad()
def evaluate_student(model, loader):
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
    print(f">> QuakeSentinel Mobile Distillation Pipeline")
    print(f"   Compute Device : {DEVICE} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"   Architecture   : {args.arch} ({ARCH_MAP[args.arch]})")
    print(f"   Resolution     : {args.img_size}x{args.img_size} | Batch Size: {args.batch} | Epochs: {args.epochs}")
    print(f"   Dataset Root   : {args.data}")
    print("=" * 70)

    # 1. Load Datasets
    train_ds, val_ds = make_split_datasets(
        images_dir=os.path.join(args.data, "Images"),
        masks_dir=os.path.join(args.data, "Masks"),
        img_size=args.img_size,
        val_fraction=0.15,
        seed=42,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0, pin_memory=True)

    # 2. Load Teacher (Frozen ResNet50 DeepLabV3)
    print(f"\n[INFO] Loading Frozen Teacher from {args.teacher_ckpt}...")
    teacher = QuakeSentinelNet(num_classes=NUM_CLASSES, pretrained=False).to(DEVICE)
    if os.path.exists(args.teacher_ckpt):
        teacher_state = torch.load(args.teacher_ckpt, map_location=DEVICE, weights_only=True)
        teacher.load_state_dict(teacher_state)
        print("   [OK] Teacher weights successfully loaded!")
    else:
        print(f"   [WARN] Teacher checkpoint not found at {args.teacher_ckpt}! Running without pretrained teacher.")
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # 3. Initialize Student (MobileNetV4-FPN or EfficientViT-FPN)
    print(f"[INFO] Initializing Student [{args.arch}]...")
    student = QuakeMobileNet(
        num_classes=NUM_CLASSES,
        backbone_name=ARCH_MAP[args.arch],
        fpn_dim=64,
        pretrained=True,
    ).to(DEVICE)
    student_params = sum(p.numel() for p in student.parameters())
    print(f"   [OK] Student model created ({student_params / 1e6:.2f}M parameters)")

    # 4. Class Weights & Criterion
    criterion = DistillationLoss(
        num_classes=NUM_CLASSES,
        temperature=args.temp,
        alpha=0.5,
        ignore_index=IGNORE_INDEX,
    )

    optimizer = optim.AdamW(student.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = GradScaler("cuda")

    best_dmg_iou = 0.0
    best_miou = 0.0
    save_path = os.path.join(SAVE_DIR, f"best_quake_mobile_{args.arch}.pth")

    print("\nStarting Distillation Training...")
    for epoch in range(1, args.epochs + 1):
        student.train()
        running_loss = 0.0
        running_hard = 0.0
        running_kd = 0.0

        pbar = tqdm(train_loader, desc=f"Ep {epoch:>2}/{args.epochs}", ncols=115)
        for imgs, masks in pbar:
            imgs  = imgs.to(DEVICE, non_blocking=True)
            masks = masks.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast("cuda"):
                # Forward pass student
                s_logits, s_score = student(imgs)
                # Forward pass teacher (no gradients)
                with torch.no_grad():
                    t_logits, t_score = teacher(imgs)

                loss, hard_l, kd_l, score_l = criterion(
                    s_logits, s_score, t_logits, t_score, masks
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            running_hard += hard_l.item()
            running_kd   += kd_l.item()

            pbar.set_postfix({
                "loss": f"{loss.item():.3f}",
                "hard": f"{hard_l.item():.3f}",
                "kd": f"{kd_l.item():.3f}",
            })

        scheduler.step()
        avg_loss = running_loss / len(train_loader)

        # Validation
        miou, per_class, avg_score = evaluate_student(student, val_loader)
        # Damage classes: 1=Debris, 4=Minor Dmg, 5=Major Dmg, 6=Destroyed
        damage_indices = [1, 4, 5, 6]
        dmg_iou = float(np.mean([per_class[i] for i in damage_indices if i < len(per_class)]))
        bg_iou  = per_class[0] if len(per_class) > 0 else 0.0

        print(f"  Ep {epoch:>2} | Loss {avg_loss:.4f} | mIoU {miou:.4f} | DmgIoU {dmg_iou:.4f} | BgIoU {bg_iou:.4f} | Score {avg_score:.3f}")

        if dmg_iou > best_dmg_iou:
            best_dmg_iou = dmg_iou
            best_miou = miou
            torch.save(student.state_dict(), save_path)
            print(f"  * New Best DmgIoU: {best_dmg_iou:.4f} -> Saved to {save_path}")

    print("\n" + "=" * 70)
    print(f"[OK] Distillation Training Complete!")
    print(f"   Best DmgIoU : {best_dmg_iou:.4f} | Best mIoU : {best_miou:.4f}")
    print(f"   Checkpoint  : {save_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
