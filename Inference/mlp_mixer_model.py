import torch
import torch.nn as nn
import timm
from torchvision import transforms
from PIL import Image
import requests


class MLPMixer(nn.Module):
    def __init__(self, pretrained=True, num_classes=1000):
        super(MLPMixer, self).__init__()

        # Load pretrained MLP-Mixer (Base variant, patch size 16x16)
        if pretrained:
            self.mixer = timm.create_model("mixer_b16_224", pretrained=True)
        else:
            self.mixer = timm.create_model("mixer_b16_224", pretrained=False)

        # Keep the original 1000-class head for ImageNet predictions
        # To fine-tune later, uncomment these lines:
        # in_features = self.mixer.head.in_features 

        # add a new layer to the model 
        # in_features = self.mixer.head.in_features 
        # self.mixer.head = nn.Sequential(
        #     nn.Linear(in_features, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 5)
        # )
        # self.mixer.head = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.mixer(x)


if __name__ == "__main__":
    print("Setting up pretrained MLP-Mixer network...")

    # Initialize model
    model = MLPMixer(pretrained=True, num_classes=1000)
    model.eval()
    # for name, param in model.named_parameters():
    #  print(name)

    # Load test image (your goldfish)
    image_path = r"E:\Research_seminar\small_mammal_classification\test_image1.jpg"
    image = Image.open(image_path).convert("RGB")

    # Choose device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    # print(f"Running on device: {device}")
   

    # print("layers",model.mixer.head)
    # Preprocess (ImageNet normalization)
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    input_tensor = preprocess(image).unsqueeze(0).to(device)
    # print("Image tensor shape:", input_tensor.shape)

    # Inference
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)
        top5 = torch.topk(probs, 5)

    # Load ImageNet labels
    labels = requests.get("https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt").text.strip().split("\n")

    # Print predictions
    print("\nTop-5 Predictions:")
    for idx, score in zip(top5.indices[0], top5.values[0]):
        print(f"{labels[idx]}: {score.item():.4f}")

    print("\nMLP-Mixer inference successful (pretrained weights).")
