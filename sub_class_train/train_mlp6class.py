import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
from models.mlp_mixer import MLPMixerModel
from utils.dataset_loader import get_dataloaders
from utils.train_utils import train_model


if __name__ == "__main__":
    # NEW dataset path for 6-class training
    data_root = r"E:\Research_seminar\small_mammal_classification\data_subclass"

    # Load dataloaders (unified loader, detects classes automatically)
    train_loader, val_loader, num_classes, class_names = get_dataloaders(
        data_root=data_root,
        batch_size=32,
        num_workers=4,
    )

    print(f"Detected classes: {class_names}")
    print(f"Training MLP-Mixer with {num_classes} output classes")

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize MLP-Mixer model for 6-class output
    model = MLPMixerModel(
        num_classes=num_classes,
        freeze_backbone=True   # feature extraction mode (same as before)
    )
    model.to(device)

    # Train model
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_classes=num_classes,
        num_epochs=40,
        lr=1e-4,
        weight_decay=1e-4,
        checkpoint_path="checkpoints_40epochs/mlp_best.pth",  # NEW checkpoint name
        use_amp=True,   # mixed precision enabled
    )

    print("Training finished for MLP-Mixer (6-class version).")
