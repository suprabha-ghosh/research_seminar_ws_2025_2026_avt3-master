import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_dataloaders(
    data_root,
    image_size=224,
    batch_size=32,
    num_workers=4,
    pin_memory=True,
    use_augmentation=False
):
    """
    Loads train and val datasets using ImageFolder.
    Automatically detects class names and number of classes.
    """

    train_dir = os.path.join(data_root, "train")
    val_dir = os.path.join(data_root, "val")

    # === ImageNet normalization (VERY IMPORTANT for pretrained models) ===
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # ==== Transforms ====
    if use_augmentation:
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=12),
            transforms.ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.10,
                hue=0.02
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        # Baseline (no augmentation)
        transform_train = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    # Validation is ALWAYS deterministic
    transform_val = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # ==== Load Datasets ====
    train_dataset = datasets.ImageFolder(train_dir, transform=transform_train)
    val_dataset = datasets.ImageFolder(val_dir, transform=transform_val)

    # Get class names and num_classes
    class_names = train_dataset.classes
    num_classes = len(class_names)

    print(f"\n=== Dataset Loaded Successfully ===")
    print(f"Train directory: {train_dir}")
    print(f"Val directory:   {val_dir}")
    print(f"Classes: {class_names}")
    print(f"Number of classes: {num_classes}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples:   {len(val_dataset)}")
    print(f"Augmentation enabled: {use_augmentation}\n")

    # ==== DataLoaders ====
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_loader, val_loader, num_classes, class_names
