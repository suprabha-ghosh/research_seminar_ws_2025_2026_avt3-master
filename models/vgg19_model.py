import torch.nn as nn
from torchvision import models


class VGG19Model(nn.Module):
    def __init__(self, num_classes=3, freeze_backbone=True):
        super(VGG19Model, self).__init__()

        
        self.vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)

        # Freeze convolutional layers if transfer learning
        if freeze_backbone:
            for param in self.vgg.features.parameters():
                param.requires_grad = False

        # Replace classifier head
        in_features = self.vgg.classifier[6].in_features
        self.vgg.classifier[6] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.vgg(x)
