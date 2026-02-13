import os
import torch
from torchvision import transforms
from PIL import Image
import csv
from tqdm import tqdm

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import models
from models.vgg19_model import VGG19Model
from models.vit_model import VisionTransformer
from models.mlp_mixer import MLPMixerModel

# -----------------------------
# 6-class setup
# -----------------------------
class_names = [
    "Big_mammals",
    "Birds",
    "Opossums",
    "Small_mammals",
    "empty",
    "insects"
]
NUM_CLASSES = 6

# Preprocessing (same as training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def load_model(model_type, checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_type == "vgg":
        model = VGG19Model(num_classes=NUM_CLASSES, freeze_backbone=True)
    elif model_type == "vit":
        model = VisionTransformer(num_classes=NUM_CLASSES, freeze_backbone=True)
    elif model_type == "mlp":
        model = MLPMixerModel(num_classes=NUM_CLASSES, freeze_backbone=True)
    else:
        raise ValueError("Unknown model type. Choose: vgg / vit / mlp")

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def predict_single(model, img_path, device):
    img = Image.open(img_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)

    return pred.item(), conf.item()


def run_batch_inference(
    model_type,
    checkpoint_path,
    data_root=r"E:\Research_seminar\small_mammal_classification\data_subclass\val",
    save_csv="batch_results.csv"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_type, checkpoint_path)

    results = []
    correct = 0
    total = 0

    print(f"\nRunning batch inference for model: {model_type.upper()}")
    print(f"Scanning directory: {data_root}\n")

    # Loop over each class folder
    for class_index, class_name in enumerate(class_names):
        class_folder = os.path.join(data_root, class_name)

        files = [f for f in os.listdir(class_folder)
                 if f.lower().endswith((".jpg", ".jpeg", ".png"))]

        print(f"Class {class_name} - {len(files)} images")

        for fname in tqdm(files):
            img_path = os.path.join(class_folder, fname)

            pred, conf = predict_single(model, img_path, device)
            is_correct = (pred == class_index)

            results.append([
                fname,
                class_name,
                class_names[pred],
                round(conf, 4),
                is_correct
            ])

            if is_correct:
                correct += 1
            total += 1

    accuracy = correct / total
    print(f"\n Total images: {total}")
    print(f" Correct predictions: {correct}")
    print(f" Accuracy: {accuracy:.4f}")

    # Save results to CSV
    os.makedirs("sub_class_results", exist_ok=True)
    save_path = os.path.join("sub_class_results", save_csv)

    with open(save_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Image", "TrueClass", "PredictedClass", "Confidence", "Correct"])
        writer.writerows(results)

    print(f"\nResults saved to: {save_path}\n")
    return accuracy


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    run_batch_inference("vgg", "checkpoints/vgg19_best_6class.pth", save_csv="results_vgg19_6class.csv")
    run_batch_inference("vit", "checkpoints/vit_best_6class.pth", save_csv="results_vit_6class.csv")
    run_batch_inference("mlp", "checkpoints/mlp_best_6class.pth", save_csv="results_mlp_6class.csv")
