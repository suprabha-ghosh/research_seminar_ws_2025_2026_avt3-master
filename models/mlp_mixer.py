import torch.nn as nn
import timm


class MLPMixerModel(nn.Module):
    def __init__(self, num_classes=3, freeze_backbone=True):
        super(MLPMixerModel, self).__init__()

      
        self.mixer = timm.create_model(
            "mixer_b16_224",
            pretrained=True
        )

        # Freeze backbone parameters
        if freeze_backbone:
            for name, param in self.mixer.named_parameters():
                if not name.startswith("head"):
                    param.requires_grad = False

        # Replace classification head
        in_features = self.mixer.head.in_features
        self.mixer.head = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.mixer(x)
