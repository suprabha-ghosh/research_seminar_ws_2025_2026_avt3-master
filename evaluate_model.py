import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from tqdm import tqdm
import numpy as np

from utils.dataset_loader import get_dataloaders
from models.vgg19_model import VGG19Model
from models.vit_model import VisionTransformer
from models.mlp_mixer import MLPMixerModel


def load_model(model_type, checkpoint_path, num_classes=2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_type == "vgg":
        model = VGG19Model(num_classes=num_classes, freeze_backbone=True)
    elif model_type == "vit":
        model = VisionTransformer(num_classes=num_classes, freeze_backbone=True)
    elif model_type == "mlp":
        model = MLPMixerModel(num_classes=num_classes, freeze_backbone=True)
    else:
        raise ValueError("Unknown model type")

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def evaluate(model_type, checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load validation dataloader
    data_root = r"E:\Research_seminar\small_mammal_classification\data"
    _, val_loader, num_classes, class_names = get_dataloaders(
        data_root=data_root,
        batch_size=32,
        num_workers=4
    )

    model = load_model(model_type, checkpoint_path, num_classes)

    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(val_loader):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(val_loader)
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    precision = precision_score(all_labels, all_preds, average="macro")
    recall = recall_score(all_labels, all_preds, average="macro")
    cm = confusion_matrix(all_labels, all_preds)

    print(f"\n=== {model_type.upper()} Evaluation ===")
    print(f"Loss: {avg_loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print("Confusion Matrix:")
    print(cm)


if __name__ == "__main__":
    evaluate("vgg", "checkpoints/vgg19_best.pth")
    evaluate("vit", "checkpoints/vit_best.pth")
    evaluate("mlp", "checkpoints/mlp_best.pth")
