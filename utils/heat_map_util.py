import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# =====================================================
# CONFIGURATION SWITCH
# =====================================================
# Set to 3 or 6
NUM_CLASSES = 3
# NUM_CLASSES = 3

# =====================================================
# CLASS DEFINITIONS
# =====================================================
CLASS_NAMES_3 = ["Animals", "Empty", "Insects"]

CLASS_NAMES_6 = [
    "Big mammals",
    "Small mammals",
    "Opossum",
    "Birds",
    "Empty",
    "Insects",
]

# =====================================================
# LABEL NORMALIZATION (robust to CSV variations)
# =====================================================
LABEL_MAP = {
    "Animals": "Animals",
    "Empty": "Empty",
    "Insects": "Insects",
    "Birds": "Birds",
    "Opossum": "Opossum",
    "Big mammals": "Big mammals",
    "Small mammals": "Small mammals",
    "Big_mammals": "Big mammals",
    "Small_mammals": "Small mammals",
}

# =====================================================
# FILE PATHS
# =====================================================
CSV_FILES_3 = {
    "VGG19": "mainclass_results/manual_inference_results_VGG19.csv",
    "ViT": "mainclass_results/manual_inference_results_ViT.csv",
    "MLP-Mixer": "mainclass_results/manual_inference_results_MLP_Mixer.csv",
}

CSV_FILES_6 = {
    "VGG19": "test_inferenceVGG19.csv",
    "ViT": "test_inferenceViT.csv",
    "MLP-Mixer": "test_inferenceMLP_Mixer.csv",
}

OUTPUT_FILES_3 = {
    "VGG19": "cm_vgg19_3class.svg",
    "ViT": "cm_vit_3class.svg",
    "MLP-Mixer": "cm_mlp_mixer_3class.svg",
}

OUTPUT_FILES_6 = {
    "VGG19": "cm_vgg19_6class.svg",
    "ViT": "cm_vit_6class.svg",
    "MLP-Mixer": "cm_mlp_mixer_6class.svg",
}

# =====================================================
# MODE SELECTION
# =====================================================
if NUM_CLASSES == 3:
    CLASS_NAMES = CLASS_NAMES_3
    CSV_FILES = CSV_FILES_3
    OUTPUT_FILES = OUTPUT_FILES_3
    FIG_SIZE = (3.6, 3.6)
elif NUM_CLASSES == 6:
    CLASS_NAMES = CLASS_NAMES_6
    CSV_FILES = CSV_FILES_6
    OUTPUT_FILES = OUTPUT_FILES_6
    FIG_SIZE = (4.8, 4.8)
else:
    raise ValueError("NUM_CLASSES must be 3 or 6")

# =====================================================
# PLOTTING FUNCTION
# =====================================================
def plot_single_confusion_matrix(csv_path, model_name, output_path):
    sns.set_style("white")

    df = pd.read_csv(csv_path)

    # Normalize labels
    df["ground_truth"] = df["ground_truth"].map(LABEL_MAP)
    df["prediction"] = df["prediction"].map(LABEL_MAP)

    # Safety checks
    assert set(df["ground_truth"].unique()).issubset(CLASS_NAMES), \
        "Ground truth labels do not match CLASS_NAMES"
    assert set(df["prediction"].unique()).issubset(CLASS_NAMES), \
        "Prediction labels do not match CLASS_NAMES"

    cm = confusion_matrix(
        df["ground_truth"],
        df["prediction"],
        labels=CLASS_NAMES
    )

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="viridis",
        square=True,
        linewidths=0.6,
        linecolor="gray",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        cbar=True,
        cbar_kws={
    "label": "Number of images",
    "shrink": 0.85,
    "fraction": 0.05,   # <<< controls width
    "pad": 0.04
}
,
        ax=ax
    )

    # Titles and axis labels
    ax.set_title(model_name, fontsize=11, pad=4)
    ax.set_xlabel("Predicted", fontsize=10, labelpad=2)
    ax.set_ylabel("True", fontsize=10, labelpad=2)

    # -------------------------------------------------
    # AXIS LABEL HANDLING (KEY FIX)
    # -------------------------------------------------
    if NUM_CLASSES == 3:
        # Vertical y-axis labels
        ax.set_yticklabels(CLASS_NAMES, rotation=0, ha="right", va="center")
        ax.tick_params(axis="x", labelsize=9, rotation=0)
    else:
        # Horizontal y-axis labels (no overlap)
        ax.set_yticklabels(CLASS_NAMES, rotation=0, ha="right")
        ax.tick_params(axis="x", labelsize=9, rotation=45)

    ax.tick_params(axis="y", labelsize=9, pad=6)

    plt.tight_layout(pad=0.6)
    plt.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")

# =====================================================
# MAIN
# =====================================================
if __name__ == "__main__":
    for model_name, csv_path in CSV_FILES.items():
        plot_single_confusion_matrix(
            csv_path=csv_path,
            model_name=model_name,
            output_path=OUTPUT_FILES[model_name]
        )
