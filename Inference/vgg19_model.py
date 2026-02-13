import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import requests 
from utils.helper import resize_images


class VGG19Model(nn.Module):
    def __init__(self, pretrained=True, num_classes=1000):
        super(VGG19Model, self).__init__()

        # Load pretrained VGG19 model
        if pretrained:
            self.vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        else:
            self.vgg = models.vgg19(weights=None)

        # Keep the original 1000-class head for ImageNet predictions
        # (for testing pretrained performance)
        # If fine-tuning later, replace this with your dataset classes:
        # in_features = self.vgg.classifier[6].in_feclearatures
        # self.vgg.classifier[6] = nn.Linear(in_features, num_classes)
        # add a new layer to the model
        # in_features = self.vgg.classifier[6].in_features
        # self.vgg.classifier[6] = nn.Sequential(
        #       nn.Linear(in_features, 512),  # new layer 
        #     nn.ReLU(),
        #     nn.Linear(512, num_classes)   # original output layer
        # )

    def forward(self, x):
        return self.vgg(x)


if __name__ == "__main__":
    print("Setting up pretrained VGG19 network...")

    # Initialize model with pretrained weights
    model = VGG19Model(pretrained=True, num_classes=1000)
    model.eval() 
    # for name, param in model.named_parameters():
    #  print(name)
        

    resize_images(
        input_folder=r"E:\Research_seminar\small_mammal_classification\raw_images",
        output_folder=r"E:\Research_seminar\small_mammal_classification\resized_images"
    )    
    # Load test image (goldfish)
    image_path = r"E:\Research_seminar\small_mammal_classification\_Field Lina 09_BM-253_1_DSCF2455.jpg"
    image = Image.open(image_path).convert("RGB")

    # Select device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device) 
    # print("layers",model.vgg.classifier)
 
    # Preprocess image
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    input_tensor = preprocess(image).unsqueeze(0).to(device)
    # print("Running on device:", device)
    # print("Image tensor shape:", input_tensor.shape)

    # Run inference
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)
        top5 = torch.topk(probs, 5)

    # Load ImageNet labels
    labels = requests.get("https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt").text.strip().split("\n")

    # Display results
    print("\nTop-5 Predictions:")
    for idx, score in zip(top5.indices[0], top5.values[0]):
        print(f"{labels[idx]}: {score.item():.4f}")

    # print("\nVGG19 inference successful (pretrained weights).")
