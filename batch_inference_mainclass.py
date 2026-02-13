import os
import torch
import pandas as pd
from PIL import Image
from torchvision import transforms

from models.vgg19_model import VGG19Model
from models.vit_model import VisionTransformer
from models.mlp_mixer import MLPMixerModel


XLS_PATH = "labelled_image_3class.xlsx"
IMAGE_DIR = "manually_labelled"
IMAGE_SIZE = 224

OUTPUT_DIR = "mainclass_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CHECKPOINTS = {
    "VGG19": "checkpoints/vgg19_best.pth",
    "ViT": "checkpoints/vit_best.pth",
    "MLP_Mixer": "checkpoints/mlp_best.pth",
}

CLASS_NAMES = [
    "Animals",
    "Empty",
    "Insects",
]

class_to_idx = {c: i for i, c in enumerate(CLASS_NAMES)}
idx_to_class = {i: c for c, i in class_to_idx.items()}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

df = pd.read_excel(XLS_PATH)

def get_model(name):
    if name == "VGG19":
        return VGG19Model(num_classes=len(CLASS_NAMES))
    elif name == "ViT":
        return VisionTransformer(num_classes=len(CLASS_NAMES))
    elif name == "MLP_Mixer":
        return MLPMixerModel(num_classes=len(CLASS_NAMES))
    else:
        raise ValueError(f"Unknown model name: {name}")

for model_name, checkpoint_path in CHECKPOINTS.items():

    print(f"\nRunning manual inference for {model_name}...")

    model = get_model(model_name)
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    filenames, y_true, y_pred, correct_flags = [], [], [], []
    correct, total = 0, 0

    with torch.no_grad():
        for _, row in df.iterrows():
            gt_label = row["Category"]
            filename = row["File"]

            if gt_label not in class_to_idx:
                raise ValueError(f"Unknown label in Excel: '{gt_label}'")

            img_path = os.path.join(IMAGE_DIR, filename)
            image = Image.open(img_path).convert("RGB")
            image = transform(image).unsqueeze(0).to(DEVICE)

            outputs = model(image)
            pred_idx = torch.argmax(outputs, dim=1).item()
            pred_label = idx_to_class[pred_idx]

            is_correct = (gt_label == pred_label)

            filenames.append(filename)
            y_true.append(gt_label)
            y_pred.append(pred_label)
            correct_flags.append(is_correct)

            total += 1
            if is_correct:
                correct += 1

    results_df = pd.DataFrame({
        "filename": filenames,
        "ground_truth": y_true,
        "prediction": y_pred,
        "correct": correct_flags
    })

    output_csv = os.path.join(
        OUTPUT_DIR,
        f"manual_inference_results_{model_name}.csv"
    )
    results_df.to_csv(output_csv, index=False)

    accuracy = correct / total
    print(f"Images evaluated : {total}")
    print(f"Accuracy         : {accuracy:.4f}")
    print(f"Saved            : {output_csv}")
