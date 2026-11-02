"""
QuakeSentinel - QuakeViT: Vision Transformer Disaster Segmentation
Architecture:
  - Backbone: EfficientViT-B2 (Linear Multi-Head Self-Attention Vision Transformer)
  - Capacity: ~16.8 Million Parameters (10.7x larger brain than MobileNetV4-Small)
  - Anti-Tunnel-Vision Feature Pyramid Network (FPN) with 128-dim feature channels
  - Dual Receptive Field Depthwise Convolutions (d=1 local rubble, d=3 building neighborhood)
  - Global Context Squeeze-and-Excitation (SE) Modulation
  - Dual Output Heads:
      1. Dense 3-Class Triage Segmentation (Background, Intact, Damaged)
      2. Scene-Level Damage Threat Score (Global Devastation Index)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class DualReceptiveBlock(nn.Module):
    """
    Anti-Tunnel-Vision Block: Captures both micro-level crack/rubble details (d=1)
    and macro-level building footprints (d=3) simultaneously.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.dw_local = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, dilation=1,
            groups=channels, bias=False
        )
        self.dw_context = nn.Conv2d(
            channels, channels, kernel_size=3, padding=3, dilation=3,
            groups=channels, bias=False
        )
        self.pw = nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local_feat = self.dw_local(x)
        context_feat = self.dw_context(x)
        cat = torch.cat([local_feat, context_feat], dim=1)
        return self.act(self.bn(self.pw(cat))) + x

class GlobalContextModulation(nn.Module):
    """
    Global Context Module: Squeezes whole-image statistics to re-weight feature maps,
    preventing localized hallucinations on normal rooftops or driveways.
    """
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w

class QuakeViT(nn.Module):
    def __init__(
        self,
        num_classes: int = 3,
        backbone_name: str = "efficientvit_b2",
        fpn_dim: int = 128,
        pretrained: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.fpn_dim = fpn_dim

        # 1. Linear Multi-Head Self-Attention Vision Transformer Backbone
        self.encoder = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
        )

        # Get feature channels from encoder
        in_channels = self.encoder.feature_info.channels()
        # For efficientvit_b2: [48, 96, 192, 384]

        # 2. Lateral 1x1 Projections
        self.lat0 = nn.Sequential(
            nn.Conv2d(in_channels[0], fpn_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(fpn_dim),
            nn.SiLU(inplace=True),
        )
        self.lat1 = nn.Sequential(
            nn.Conv2d(in_channels[1], fpn_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(fpn_dim),
            nn.SiLU(inplace=True),
        )
        self.lat2 = nn.Sequential(
            nn.Conv2d(in_channels[2], fpn_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(fpn_dim),
            nn.SiLU(inplace=True),
        )
        self.lat3 = nn.Sequential(
            nn.Conv2d(in_channels[3], fpn_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(fpn_dim),
            nn.SiLU(inplace=True),
        )

        # 3. Dual-Receptive Blocks for anti-tunnel vision
        self.smooth0 = DualReceptiveBlock(fpn_dim)
        self.smooth1 = DualReceptiveBlock(fpn_dim)
        self.smooth2 = DualReceptiveBlock(fpn_dim)
        self.smooth3 = DualReceptiveBlock(fpn_dim)

        # 4. Multi-Scale Fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(fpn_dim * 4, fpn_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_dim),
            nn.SiLU(inplace=True),
            GlobalContextModulation(fpn_dim),
        )

        # 5. Output Heads
        # Segmentation Head (upsamples back to full resolution)
        self.seg_head = nn.Sequential(
            nn.Conv2d(fpn_dim, fpn_dim // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_dim // 2),
            nn.SiLU(inplace=True),
            nn.Conv2d(fpn_dim // 2, num_classes, kernel_size=1),
        )

        # Global Threat Score Head (scene-level damage probability)
        self.threat_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(fpn_dim, 64),
            nn.SiLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor):
        input_size = x.shape[2:]

        # Backbone Features
        feats = self.encoder(x)
        c0, c1, c2, c3 = feats[0], feats[1], feats[2], feats[3]

        # Top-Down FPN
        p3 = self.lat3(c3)
        p2 = self.lat2(c2) + F.interpolate(p3, size=c2.shape[2:], mode="bilinear", align_corners=False)
        p1 = self.lat1(c1) + F.interpolate(p2, size=c1.shape[2:], mode="bilinear", align_corners=False)
        p0 = self.lat0(c0) + F.interpolate(p1, size=c0.shape[2:], mode="bilinear", align_corners=False)

        # Anti-Tunnel-Vision Smoothing
        p3 = self.smooth3(p3)
        p2 = self.smooth2(p2)
        p1 = self.smooth1(p1)
        p0 = self.smooth0(p0)

        # Align all pyramids to stage 0 resolution (H/4, W/4)
        target_h, target_w = p0.shape[2:]
        p3_up = F.interpolate(p3, size=(target_h, target_w), mode="bilinear", align_corners=False)
        p2_up = F.interpolate(p2, size=(target_h, target_w), mode="bilinear", align_corners=False)
        p1_up = F.interpolate(p1, size=(target_h, target_w), mode="bilinear", align_corners=False)

        # Fused multi-scale representation
        fused = self.fusion(torch.cat([p0, p1_up, p2_up, p3_up], dim=1))

        # Prediction Heads
        seg_logits = self.seg_head(fused)
        if seg_logits.shape[2:] != input_size:
            seg_logits = F.interpolate(seg_logits, size=input_size, mode="bilinear", align_corners=False)

        damage_score = self.threat_head(fused)

        return seg_logits, damage_score

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Testing QuakeViT instantiation...")
    model = QuakeViT(num_classes=3, backbone_name="efficientvit_b2", fpn_dim=128, pretrained=False).to(device)
    dummy = torch.randn(2, 3, 512, 512).to(device)
    logits, score = model(dummy)
    params = sum(p.numel() for p in model.parameters())
    print(f"QuakeViT initialized successfully on {device}!")
    print(f"  Total Parameters : {params / 1e6:.2f}M")
    print(f"  Logits Shape     : {logits.shape}")
    print(f"  Threat Score     : {score.shape}")
