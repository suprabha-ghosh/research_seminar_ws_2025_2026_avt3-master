import torch
from models.mlp_mixer import MLPMixerModel
from utils.dataset_loader import get_dataloaders
from utils.train_utils import train_model
from pathlib import Path


if __name__ == "__main__":
    # Dataset path
    data_root = Path(__file__).resolve().parent

    # Load dataloaders
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

    # Initialize MLP-Mixer model
    model = MLPMixerModel(
        num_classes=num_classes,
        freeze_backbone=True   # feature extraction mode
    )
    model.to(device)

    # Train model
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_classes=num_classes,
        num_epochs=20,
        lr=1e-4,
        weight_decay=1e-4,
        checkpoint_path="checkpoints/mlp_best.pth",
        use_amp=True,   # mixed precision
    )

    print("Training finished for MLP-Mixer.")
