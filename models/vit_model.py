import torch.nn as nn
import timm


class VisionTransformer(nn.Module):
    def __init__(self, num_classes=3, freeze_backbone=True):
        super(VisionTransformer, self).__init__()

        
        self.vit = timm.create_model(
            'vit_base_patch16_224',
            pretrained=True
        )

        # Freeze transformer encoder layers
        if freeze_backbone:
            for name, param in self.vit.named_parameters():
                if not name.startswith("head"):
                    param.requires_grad = False

        # Replace classification head
        in_features = self.vit.head.in_features
        self.vit.head = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.vit(x)
