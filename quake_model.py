import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation import deeplabv3_resnet50

class QuakeSentinelNet(nn.Module):
    def __init__(self, num_classes=12, pretrained=True):
        super().__init__()
        
        # Upgraded to DeepLabV3 with ResNet50 backbone for pixel-perfect segmentation mask generation
        weights_backbone = "DEFAULT" if pretrained else None
        self.model = deeplabv3_resnet50(
            num_classes=num_classes, 
            weights=None, 
            weights_backbone=weights_backbone
        )
        
        # Global pooling and classification head for the image-level damage score
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_classes, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.model(x)['out']
        # Global damage score based on the semantic segmentation logits
        pooled = self.global_pool(out)
        score = self.global_classifier(pooled).squeeze(1)
        return out, score

class QuakeLoss(nn.Module):
    def __init__(self, num_classes=12, ignore_index: int = 255, weights=None):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.ce = nn.CrossEntropyLoss(weight=weights, ignore_index=ignore_index)

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        
        # Multiclass Dice
        probs = torch.softmax(logits, dim=1)
        valid_mask = (targets != self.ignore_index).unsqueeze(1)
        
        targets_clamped = targets.clone()
        targets_clamped[targets == self.ignore_index] = 0
        t_onehot = torch.nn.functional.one_hot(targets_clamped, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        
        probs = probs * valid_mask
        t_onehot = t_onehot * valid_mask
        
        intersection = (probs * t_onehot).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + t_onehot.sum(dim=(2, 3))
        dice_scores = (2.0 * intersection + 1e-5) / (union + 1e-5)
        
        if self.ce.weight is not None:
            dice_loss = ((1.0 - dice_scores) * self.ce.weight.unsqueeze(0)).mean()
        else:
            dice_loss = (1.0 - dice_scores).mean()
            
        loss = ce_loss + 0.5 * dice_loss
        return loss, ce_loss.detach(), dice_loss.detach()
