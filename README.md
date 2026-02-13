# Small Mammal Image Classification

This project compares CNN, Transformer, and MLP architectures for automated small mammal image classification. It includes pre-trained models (VGG19, Vision Transformer, and MLP-Mixer) for both main class (3-class) and subclass (6-class) classification tasks.

## Project Overview

The project provides batch inference scripts to evaluate images using pre-trained models:
- **Main Class Classification**: 3 classes (Animals, Empty, Insects)
- **Subclass Classification**: 6 classes (Big mammals, Birds, Opossum, Small mammals, Empty, Insects)

## Requirements

- Python 3.7+
- PyTorch 2.0+
- CUDA-capable GPU (optional, but recommended for faster inference)

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd small_mammal_classification
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   Note: You may also need to install `openpyxl` for Excel file reading:
   ```bash
   pip install openpyxl
   ```

3. **Download and extract the manually labelled images:**
   - Download `manually_labelled.zip` from: https://cloud.tu-ilmenau.de/s/KciXsYNnfrjXkFX?openfile=true
   - Extract the zip file to the project root directory (it should create a `manually_labelled/` folder)
   - This folder contains the images needed for running the inference scripts

4. **Ensure model checkpoints are available:**
   - For main class inference: Checkpoints should be in `checkpoints/` directory
   - For subclass inference: Checkpoints should be in `checkpoints_40epochs/` directory

## Usage

### Main Class Inference (3-class classification)

Run batch inference for main class classification:

```bash
python batch_inference_mainclass.py
```

**Requirements:**
- Excel file: `labelled_image_3class.xlsx` with columns:
  - `File`: Image filename
  - `Category`: Ground truth label (must be one of: "Animals", "Empty", "Insects")
- Image directory: `manually_labelled/` containing the images referenced in the Excel file

**Output:**
- Results saved in `mainclass_results/` directory
- CSV files: `manual_inference_results_{MODEL_NAME}.csv` for each model (VGG19, ViT, MLP_Mixer)
- Each CSV contains: `filename`, `ground_truth`, `prediction`, `correct`
- Accuracy metrics printed to console

**Models evaluated:**
- VGG19
- Vision Transformer (ViT)
- MLP-Mixer

### Subclass Inference (6-class classification)

Run batch inference for subclass classification:

```bash
python batch_inference_subclass.py
```

**Requirements:**
- Excel file: `labelled_imageCSV.xlsx` with columns:
  - `File`: Image filename
  - `Category`: Ground truth label (must be one of: "Big mammals", "Birds", "Opossum", "Small mammals", "Empty", "Insects")
- Image directory: `manually_labelled/` containing the images referenced in the Excel file

**Output:**
- CSV files: `test_inference{MODEL_NAME}.csv` for each model
- Each CSV contains: `filename`, `ground_truth`, `prediction`, `correct`
- Accuracy metrics printed to console

**Models evaluated:**
- VGG19
- Vision Transformer (ViT)
- MLP-Mixer

## Input Format

### Excel File Structure

Both scripts expect an Excel file (`.xlsx`) with the following columns:

| File | Category |
|------|----------|
| image1.jpg | Animals |
| image2.jpg | Empty |
| ... | ... |

**Important:** 
- The `Category` column values must exactly match the class names defined in the script
- Image filenames in the `File` column should match the actual image files in the image directory
- Supported image formats: JPG, JPEG, PNG

## Output Format

The inference scripts generate CSV files with the following columns:

- `filename`: Name of the image file
- `ground_truth`: True label from the Excel file
- `prediction`: Model's predicted label
- `correct`: Boolean indicating if prediction matches ground truth

Example output:
```csv
filename,ground_truth,prediction,correct
image1.jpg,Animals,Animals,True
image2.jpg,Empty,Insects,False
```

## Model Checkpoints

The scripts automatically load pre-trained checkpoints:

**Main Class Models:**
- `checkpoints/vgg19_best.pth`
- `checkpoints/vit_best.pth`
- `checkpoints/mlp_best.pth`

**Subclass Models:**
- `checkpoints_40epochs/vgg19_augmented.pth`
- `checkpoints_40epochs/vit_best_.pth`
- `checkpoints_40epochs/mlp_best.pth`

Ensure these checkpoint files exist before running inference.

## Class Labels

### Main Class (3-class)
- Animals
- Empty
- Insects

### Subclass (6-class)
- Big mammals
- Birds
- Opossum
- Small mammals
- Empty
- Insects

## Notes

- The scripts automatically detect and use GPU if available, otherwise fall back to CPU
- Images are automatically resized to 224x224 pixels and normalized using ImageNet statistics
- All three models are evaluated sequentially for each script
- The scripts will raise an error if any label in the Excel file doesn't match the expected class names

## Troubleshooting

1. **FileNotFoundError**: Ensure the Excel file and image directory paths are correct
2. **Unknown label error**: Verify that all labels in the Excel file exactly match the class names (case-sensitive)
3. **CUDA out of memory**: If using GPU, try processing fewer images at once or use CPU mode
4. **Missing checkpoint files**: Ensure all model checkpoint files are present in the specified directories
