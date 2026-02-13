import json
import matplotlib.pyplot as plt


# --------------------------------------------------
# PLOT SINGLE MODEL LOSS CURVE
# --------------------------------------------------
def plot_loss_curve(json_path, model_name, output_svg):
    with open(json_path, "r") as f:
        data = json.load(f)

    train_loss = data["train_loss"]
    val_loss = data["val_loss"]
    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(7, 4.5))

    plt.plot(epochs, train_loss, label="Train Loss", linewidth=2)
    plt.plot(epochs, val_loss, label="Validation Loss", linewidth=2)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("VGG19 After Augmentation")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_svg, format="svg")
    plt.close()

    print(f"Saved: {output_svg}")


# --------------------------------------------------
# MAIN
# --------------------------------------------------
if __name__ == "__main__":
    logs = {
        "VGG19": (
            r"E:\Research_seminar\small_mammal_classification\checkpoints_40epochs\vgg19_augmented_logs.json",
            "loss_cuve_vgg19_6cl_augmented.svg"
        ),
        # "ViT": (
        #     r"E:\Research_seminar\small_mammal_classification\checkpoints_40epochs\vit_best__logs.json",
        #     "loss_curve_vit_6cl.svg"
        # ),
        # "MLP-Mixer": (
        #     r"E:\Research_seminar\small_mammal_classification\checkpoints_40epochs\mlp_best_logs.json",
        #     "loss_curve_mlp_mixer_6cl.svg"
        # )
    }

    for model_name, (json_path, output_svg) in logs.items():
        plot_loss_curve(json_path, model_name, output_svg)
