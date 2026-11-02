"""
QuakeSentinel Mobile Architecture
High-efficiency post-disaster / earthquake damage segmentation designed for Mobile & Edge NPUs.

Backbones supported via timm:
  1. mobilenetv4_conv_small (Google 2024 - Universal Inverted Bottleneck, pure NPU-accelerated convs)
  2. efficientvit_b0 (MIT HAN Lab - Cascaded Group Attention, mobile-first linear transformer)

Decoder:
  Lightweight Feature Pyramid Network (FPN) + Bilinear Upsampling.
  Avoids heavy ConvTranspose2d (no checkerboard artifacts, 100% supported by TFLite/CoreML/ONNX).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class FPNBlock(nn.Module):
    """
    Anti-Tunnel-Vision FPN Block:
    Combines 1x1 projection with dual-scale depthwise convolutions:
      - Branch A: dilation=1 for fine rubble cracks and crisp building borders.
      - Branch B: dilation=3 for wide neighborhood receptive field (anti-tunnel vision).
    """
    def __init__(self, in_ch: int, out_ch: int = 64):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.norm = nn.BatchNorm2d(out_ch)
        self.act  = nn.ReLU6(inplace=True)

        mid_ch = out_ch // 2
        # Local branch (dilation 1)
        self.dw_local = nn.Conv2d(mid_ch, mid_ch, kernel_size=3, padding=1, dilation=1, groups=mid_ch, bias=False)
        # Contextual branch (dilation 3) - broad receptive field
        self.dw_context = nn.Conv2d(mid_ch, mid_ch, kernel_size=3, padding=3, dilation=3, groups=mid_ch, bias=False)
        self.dw_norm = nn.BatchNorm2d(out_ch)

        self.pw = nn.Conv2d(out_ch, out_ch, kernel_size=1, bias=False)
        self.pw_norm = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        proj = self.act(self.norm(self.proj(x)))
        # Split channels for dual receptive fields
        c_split = proj.shape[1] // 2
        x1 = self.dw_local(proj[:, :c_split])
        x2 = self.dw_context(proj[:, c_split:])
        merged = torch.cat([x1, x2], dim=1)
        refined = self.act(self.dw_norm(merged))
        out = self.act(self.pw_norm(self.pw(refined)))
        return out + proj

class QuakeMobileNet(nn.Module):
    def __init__(
        self,
        num_classes: int = 3,
        backbone_name: str = "mobilenetv4_conv_small.e1200_r224_in1k",
        fpn_dim: int = 64,
        pretrained: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.backbone_name = backbone_name

        # Extract 4 hierarchical stages: H/4, H/8, H/16, H/32
        self.encoder = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(1, 2, 3, 4) if "mobilenetv4" in backbone_name else (0, 1, 2, 3),
        )

        channels = self.encoder.feature_info.channels()
        c1, c2, c3, c4 = channels[-4:]

        # Lateral FPN connections with dual receptive field
        self.lat4 = FPNBlock(c4, fpn_dim)
        self.lat3 = FPNBlock(c3, fpn_dim)
        self.lat2 = FPNBlock(c2, fpn_dim)
        self.lat1 = FPNBlock(c1, fpn_dim)

        # Anti-Tunnel-Vision: Global Context Injector
        # Captures macro scene context across the entire image to prevent localized false alarms
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c4, fpn_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(fpn_dim),
            nn.ReLU6(inplace=True),
            nn.Conv2d(fpn_dim, fpn_dim, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

        # Multi-scale fusion head: fuses all 4 pyramid stages at H/4 resolution
        self.fuse = nn.Sequential(
            nn.Conv2d(fpn_dim * 4, fpn_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_dim),
            nn.ReLU6(inplace=True),
            nn.Dropout2d(0.1),
        )

        # Final segmentation head (H/4 -> H via 4x bilinear upsample)
        self.seg_head = nn.Conv2d(fpn_dim, num_classes, kernel_size=1)

        # Mobile threat / global damage score classifier
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.score_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fpn_dim, 32),
            nn.ReLU6(inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        input_size = x.shape[2:]
        feats = self.encoder(x)
        f1, f2, f3, f4 = feats[-4:]

        # Anti-tunnel-vision global context modulation
        g_ctx = self.global_context(f4)

        # Top-down FPN pathway
        p4 = self.lat4(f4)
        p3 = self.lat3(f3) + F.interpolate(p4, size=f3.shape[2:], mode='bilinear', align_corners=False)
        p2 = self.lat2(f2) + F.interpolate(p3, size=f2.shape[2:], mode='bilinear', align_corners=False)
        p1 = self.lat1(f1) + F.interpolate(p2, size=f1.shape[2:], mode='bilinear', align_corners=False)

        # Upsample all stages to H/4 (size of p1) and concatenate
        target_size = p1.shape[2:]
        p2_up = F.interpolate(p2, size=target_size, mode='bilinear', align_corners=False)
        p3_up = F.interpolate(p3, size=target_size, mode='bilinear', align_corners=False)
        p4_up = F.interpolate(p4, size=target_size, mode='bilinear', align_corners=False)

        fused = self.fuse(torch.cat([p1, p2_up, p3_up, p4_up], dim=1))
        # Modulate fused features with global context: eliminates localized tunnel vision!
        fused = fused * (1.0 + g_ctx)

        # Predict logits and upsample to full input resolution
        seg_logits = self.seg_head(fused)
        if seg_logits.shape[2:] != input_size:
            seg_logits = F.interpolate(seg_logits, size=input_size, mode='bilinear', align_corners=False)

        # Image-level threat score
        damage_score = self.score_head(self.global_pool(fused)).squeeze(1)

        return seg_logits, damage_score
