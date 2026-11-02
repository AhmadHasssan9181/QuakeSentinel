import torch
import torchvision.models as models

mbv3 = models.mobilenet_v3_large(weights=None).features
x = torch.randn(1, 3, 384, 384)

print("Layer sizes for MobileNetV3-Large:")
for i, layer in enumerate(mbv3):
    x = layer(x)
    print(f"Layer {i:2d}: {list(x.shape)}")
