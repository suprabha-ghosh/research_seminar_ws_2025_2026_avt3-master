import os
import shutil
import random
from tqdm import tqdm

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def split_subfolder(src_folder, train_dst, val_dst, split_ratio=0.8):
    """
    Splits images in one Animals subfolder into train/val and copies them.
    """
    images = [f for f in os.listdir(src_folder)
              if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    random.shuffle(images)

    train_count = int(len(images) * split_ratio)
    train_imgs = images[:train_count]
    val_imgs = images[train_count:]

    for img in tqdm(train_imgs, desc=f"Train -> {os.path.basename(src_folder)}"):
        shutil.copy2(os.path.join(src_folder, img), os.path.join(train_dst, img))

    for img in tqdm(val_imgs, desc=f"Val -> {os.path.basename(src_folder)}"):
        shutil.copy2(os.path.join(src_folder, img), os.path.join(val_dst, img))


def balanced_animals_split(base_path):
    """
    Option B:
    ONLY split Animals subclasses for now.
    """
    animals_path = os.path.join(base_path, "Animals")

    subfolders = ["Big mammals", "Birds", "Opossums", "Small mammals"]

    train_animals = os.path.join(base_path, "train", "Animals")
    val_animals = os.path.join(base_path, "val", "Animals")

    create_dir(train_animals)
    create_dir(val_animals)

    for sub in subfolders:
        src = os.path.join(animals_path, sub)
        if not os.path.exists(src):
            print(f"Missing subfolder: {sub} — skipping.")
            continue

        print(f"Splitting Animals subclass: {sub}")
        split_subfolder(src, train_animals, val_animals)



def split_simple_class(base_path, class_name, split_ratio=0.8):
    """
    Splits simple classes like Empty and Insects.
    Automatically skips if folder is missing or empty.
    """
    src = os.path.join(base_path, class_name)

    if not os.path.exists(src) or len(os.listdir(src)) == 0:
        print(f"Skipping '{class_name}' — folder missing or empty.")
        return

    train_dst = os.path.join(base_path, "train", class_name)
    val_dst = os.path.join(base_path, "val", class_name)

    create_dir(train_dst)
    create_dir(val_dst)

    print(f"Splitting class: {class_name}")

    images = [f for f in os.listdir(src)
              if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    random.shuffle(images)

    train_count = int(len(images) * split_ratio)
    train_imgs = images[:train_count]
    val_imgs = images[train_count:]

    for img in tqdm(train_imgs, desc=f"Train -> {class_name}"):
        shutil.copy2(os.path.join(src, img), os.path.join(train_dst, img))

    for img in tqdm(val_imgs, desc=f"Val -> {class_name}"):
        shutil.copy2(os.path.join(src, img), os.path.join(val_dst, img))

if __name__ == "__main__":
    random.seed(42)
    BASE = r"E:\Research_seminar\small_mammal_classification\data"

    print("Starting dataset split (Animals only)...")

    # Balanced split for Animals ONLY
    # balanced_animals_split(BASE)
    split_simple_class(BASE, "Insects")

    print("Animals dataset split completed successfully.")
    print("Empty and Insects will be split later when available.")
