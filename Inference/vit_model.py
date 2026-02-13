import torch
import torch.nn as nn
import timm
from PIL import Image
from torchvision import transforms
import requests


class VisionTransformer(nn.Module):
    def __init__(self):
        super(VisionTransformer, self).__init__()

        # Load pretrained Vision Transformer (ViT-Base, patch size 16x16)
        # Keep the original 1000-class ImageNet head
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True)
        # in_features = self.vit.head.in_features  # 768 for ViT-Base
        # self.vit.head = nn.Sequential(
        #     nn.Linear(in_features, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 5)
        # )

    def forward(self, x):
        return self.vit(x)


if __name__ == "__main__":
    print("Setting up Vision Transformer network...")

    # Initialize model (no num_classes since we use original head)
    model = VisionTransformer()
    model.eval()
    for name, param in model.named_parameters():
     print(name)

    # Load your goldfish image
    image_path = r"E:\Research_seminar\small_mammal_classification\n01518878_ostrich.JPEG"
    image = Image.open(image_path).convert("RGB")

    # Choose device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device) 
    print("layers",model.vit.head)

    # Preprocess image
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )
    ])

    input_tensor = preprocess(image).unsqueeze(0).to(device)
    print("Running on device:", device)
    print("Image tensor shape:", input_tensor.shape)

    # Run inference
    with torch.no_grad():
        patch_output = model.vit.patch_embed(input_tensor)
        output = model(input_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)
        top5 = torch.topk(probs, 5)

    # Load ImageNet labels
    labels = requests.get("https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt").text.strip().split("\n")

    print("ViT inference successful.")
    print("Patch embedding output shape:", patch_output.shape)
    print("Final output shape:", output.shape)

    print("\nTop-5 Predictions:")
    for idx, score in zip(top5.indices[0], top5.values[0]):
        print(f"{labels[idx]}: {score.item():.4f}")
