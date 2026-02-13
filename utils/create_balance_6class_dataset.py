import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm

SOURCE = r"E:\Research_seminar\small_mammal_classification\data"
TARGET = r"E:\Research_seminar\small_mammal_classification\data_subclass"

NUM_IMAGES = 2100
TRAIN_RATIO = 0.8

ANIMAL_CLASSES = {
    "Big mammals": "Big_mammals",
    "Birds": "Birds",
    "Opossums": "Opossums",
    "Small mammals": "Small_mammals"
}

SIMPLE_CLASSES = {
    "Empty": "empty",
    "Insects": "insects"
}

def create_dirs():
    for split in ["train", "val"]:
        for cls in [*ANIMAL_CLASSES.values(), *SIMPLE_CLASSES.values()]:
            Path(f"{TARGET}/{split}/{cls}").mkdir(parents=True, exist_ok=True)

def process_class(src, dst_train, dst_val):
    images = [f for f in os.listdir(src)
              if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    selected = random.sample(images, NUM_IMAGES)
    random.shuffle(selected)

    train_count = int(NUM_IMAGES * TRAIN_RATIO)
    train_imgs = selected[:train_count]
    val_imgs = selected[train_count:]

    for img in tqdm(train_imgs, desc=f"Train -> {os.path.basename(src)}"):
        shutil.copy2(os.path.join(src, img), os.path.join(dst_train, img))

    for img in tqdm(val_imgs, desc=f"Val -> {os.path.basename(src)}"):
        shutil.copy2(os.path.join(src, img), os.path.join(dst_val, img))

def main():
    random.seed(42)
    create_dirs()

    # Animal subclasses
    for src_name, dst_name in ANIMAL_CLASSES.items():
        src = f"{SOURCE}/Animals/{src_name}"
        print(f"\nProcessing {src_name}")
        process_class(src, f"{TARGET}/train/{dst_name}", f"{TARGET}/val/{dst_name}")

    # Simple classes
    for src_name, dst_name in SIMPLE_CLASSES.items():
        src = f"{SOURCE}/{src_name}"
        print(f"\nProcessing {src_name}")
        process_class(src, f"{TARGET}/train/{dst_name}", f"{TARGET}/val/{dst_name}")

    print("\nBalanced 6-class dataset created successfully!")

if __name__ == "__main__":
    main()
