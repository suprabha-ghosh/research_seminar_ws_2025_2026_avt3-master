import os
import time
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def compute_metrics_from_confusion(conf_mat: torch.Tensor):
    num_classes = conf_mat.size(0)

    correct = conf_mat.diag().sum().item()
    total = conf_mat.sum().item()
    accuracy = correct / total if total > 0 else 0.0

    f1_per_class = []
    for c in range(num_classes):
        tp = conf_mat[c, c].item()
        fp = conf_mat[:, c].sum().item() - tp
        fn = conf_mat[c, :].sum().item() - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        f1_per_class.append(f1)

    macro_f1 = sum(f1_per_class) / num_classes if num_classes > 0 else 0.0
    return accuracy, macro_f1


def evaluate_model(model, data_loader, device, criterion, num_classes):
    model.eval()
    total_loss = 0.0
    total_samples = 0

    conf_mat = torch.zeros((num_classes, num_classes), dtype=torch.long, device=device)

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            preds = torch.argmax(outputs, dim=1)

            for t, p in zip(targets.view(-1), preds.view(-1)):
                conf_mat[t.long(), p.long()] += 1

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    acc, macro_f1 = compute_metrics_from_confusion(conf_mat)

    return avg_loss, acc, macro_f1


def train_model(
    model,
    train_loader,
    val_loader,
    device,
    num_classes,
    num_epochs=20,
    lr=1e-4,
    weight_decay=1e-4,
    checkpoint_path="checkpoints/model_best.pth",
    use_amp=True,
):
    # -----------------------------
    # Create checkpoint directory
    # -----------------------------
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    # -----------------------------
    # Initialize training components
    # -----------------------------
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and device.type == "cuda")

    best_f1 = 0.0

    # -----------------------------
    # Initialize log dictionary + timer
    # -----------------------------
    start_time = time.time()
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": [],
        "training_time_seconds": 0
    }

    # -----------------------------
    # Training Loop
    # -----------------------------
    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        total_train_samples = 0

        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            batch_size = targets.size(0)

            optimizer.zero_grad()

            if scaler.is_enabled():
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * batch_size
            total_train_samples += batch_size

        train_loss = running_loss / total_train_samples

        # -----------------------------
        # Validation
        # -----------------------------
        val_loss, val_acc, val_macro_f1 = evaluate_model(
            model, val_loader, device, criterion, num_classes
        )

        print(
            f"Epoch [{epoch}/{num_epochs}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Acc: {val_acc:.4f} | "
            f"Macro-F1: {val_macro_f1:.4f}"
        )

        # -----------------------------
        # Log current epoch
        # -----------------------------
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_macro_f1)

        # -----------------------------
        # Save best model
        # -----------------------------
        if val_macro_f1 > best_f1:
            best_f1 = val_macro_f1
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved best model at epoch {epoch} (F1 = {val_macro_f1:.4f})")

    # -----------------------------
    # End of training — save time & logs
    # -----------------------------
    end_time = time.time()
    history["training_time_seconds"] = end_time - start_time

    log_file = checkpoint_path.replace(".pth", "_logs.json")
    with open(log_file, "w") as f:
        json.dump(history, f, indent=4)

    print("Training complete.")
    print(f"Best validation Macro-F1: {best_f1:.4f}")

    return history
