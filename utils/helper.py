from PIL import Image
import os

def resize_images(input_folder, output_folder, size=(224, 224)):
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(input_folder, filename)
            img = Image.open(img_path).convert("RGB")
            img = img.resize(size)
            img.save(os.path.join(output_folder, filename))
            print(f"Resized: {filename}")
    
    print("All images resized successfully to", size)