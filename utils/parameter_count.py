import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
from models.vgg19_model import VGG19Model
from models.vit_model import VisionTransformer
from models.mlp_mixer import MLPMixerModel

NUM_CLASSES = 6  # or 3 — doesn't affect backbone size much

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

models = {
    "VGG19": VGG19Model(num_classes=NUM_CLASSES),
    "ViT": VisionTransformer(num_classes=NUM_CLASSES),
    "MLP-Mixer": MLPMixerModel(num_classes=NUM_CLASSES),
}

for name, model in models.items():
    total, trainable = count_parameters(model)
    print(f"{name}:")
    print(f"  Total parameters     : {total:,}")
    print(f"  Trainable parameters : {trainable:,}\n")
