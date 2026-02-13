import sys, os, json

# Allow imports from project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from models.vgg19_model import VGG19Model
from utils.dataset_loader import get_dataloaders
from utils.train_utils import train_model


if __name__ == "__main__":

    # --------------------------------------------------
    # DATASET
    # --------------------------------------------------
    data_root = r"E:\Research_seminar\small_mammal_classification\data_subclass"

    # Load dataloaders WITH augmentation (TRAIN ONLY)
    train_loader, val_loader, num_classes, class_names = get_dataloaders(
        data_root=data_root,
        batch_size=32,
        num_workers=4,
        use_augmentation=True
    )

    print(f"Detected classes: {class_names}")
    print(f"Training VGG19 (6-class) — AUGMENTED")

    # --------------------------------------------------
    # DEVICE
    # --------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --------------------------------------------------
    # MODEL
    # --------------------------------------------------
    model = VGG19Model(
        num_classes=num_classes,
        freeze_backbone=True   # keep identical to baseline
    )
    model.to(device)

    # --------------------------------------------------
    # OUTPUT PATHS
    # --------------------------------------------------
    os.makedirs("checkpoints_40epochs", exist_ok=True)

    checkpoint_path = "checkpoints_40epochs/vgg19_augmented.pth"
    history_path = "checkpoints_40epochs/vgg19_augmented_logs.json"

    # --------------------------------------------------
    # TRAINING
    # --------------------------------------------------
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_classes=num_classes,
        num_epochs=40,
        lr=1e-4,
        weight_decay=1e-4,
        checkpoint_path=checkpoint_path,
        use_amp=True
    )

    # --------------------------------------------------
    # SAVE HISTORY
    # --------------------------------------------------
    with open(history_path, "w") as f:
        json.dump(history, f, indent=4)

    print("\nTraining finished for VGG19 (6-class) — AUGMENTED")
    print(f"Checkpoint saved to : {checkpoint_path}")
    print(f"History saved to    : {history_path}")
