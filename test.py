# test.py

import torch
import torchvision
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
from torchvision.utils import save_image
from model import EfficientUNet5Down
from train import ThinningDataset, multitask_loss, Config, saved_configs
from evaluate import (
    compute_test_loss,
    compute_distance_mse,
    compute_node_precision_recall,
    compute_iou_and_dice,
)

# --------------- Config ---------------
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

CONFIG = Config()
BATCH_SIZE = CONFIG.batch_size  # match your config system
DATA_DIR = "./data/test/thinning"

# Output folders
SAVE_DIR = os.environ.get("SAVE_DIR", "./outputs/test_predictions")
INPUT_SAVE_DIR = os.path.join(SAVE_DIR, "inputs")
SKELETON_SAVE_DIR = os.path.join(SAVE_DIR, "pred_skeletons")
DISTANCE_SAVE_DIR = os.path.join(SAVE_DIR, "pred_distances")

os.makedirs(INPUT_SAVE_DIR, exist_ok=True)
os.makedirs(SKELETON_SAVE_DIR, exist_ok=True)
os.makedirs(DISTANCE_SAVE_DIR, exist_ok=True)


# --------------- Test Loader ---------------
def get_test_loader(batch_size):
    test_dataset = ThinningDataset(
        root_dir=DATA_DIR,
        transform=torchvision.transforms.ToTensor(),
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    return test_loader


# --------------- Save Predictions ---------------
def save_test_predictions(model, dataloader, device):
    model.eval()

    with torch.no_grad():
        loop = tqdm(dataloader, desc="Saving Predictions", leave=True)
        for batch_idx, (inputs, skeleton_targets, distance_targets) in enumerate(loop):
            inputs = inputs.to(device)
            outputs = model(inputs)

            pred_skeleton = torch.sigmoid(outputs[:, 0:1, :, :])
            pred_distance = outputs[:, 1:2, :, :]

            pred_skeleton_binary = (pred_skeleton > 0.5).float()

            for i in range(inputs.size(0)):
                idx = batch_idx * dataloader.batch_size + i

                save_image(
                    inputs[i],
                    os.path.join(INPUT_SAVE_DIR, f"input_{idx:05d}.png"),
                    normalize=True,
                )
                save_image(
                    pred_skeleton_binary[i],
                    os.path.join(SKELETON_SAVE_DIR, f"pred_skeleton_{idx:05d}.png"),
                    normalize=True,
                )
                save_image(
                    pred_distance[i],
                    os.path.join(DISTANCE_SAVE_DIR, f"pred_distance_{idx:05d}.png"),
                    normalize=True,
                )


# --------------- Plots (Node PR + Full Metrics) ---------------
def plot_node_precision_recall(
    precision_dict,
    recall_dict,
    save_path=f"{SAVE_DIR}/node_pr_curve.png",
):
    valences = [1, 2, 3, 4]
    precision = [precision_dict[v] for v in valences]
    recall = [recall_dict[v] for v in valences]

    x = np.arange(len(valences))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 6))
    rects1 = ax.bar(x - width / 2, precision, width, label="Precision")
    rects2 = ax.bar(x + width / 2, recall, width, label="Recall")

    ax.set_ylabel("Score")
    ax.set_xlabel("Node Valence")
    ax.set_title("Precision and Recall by Node Valence")
    ax.set_xticks(x)
    ax.set_xticklabels(valences)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y")

    for rect in rects1 + rects2:
        height = rect.get_height()
        ax.annotate(
            f"{height:.2f}",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved Node PR plot to {save_path}")


def plot_full_test_metrics(
    precision,
    recall,
    iou,
    dice,
    test_loss,
    distance_mse,
    save_path=f"{SAVE_DIR}/full_test_metrics.png",
):
    valences = [1, 2, 3, 4]
    precision_vals = [precision[v] for v in valences]
    recall_vals = [recall[v] for v in valences]

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    x = np.arange(len(valences))
    width = 0.35

    rects1 = axs[0].bar(x - width / 2, precision_vals, width, label="Precision")
    rects2 = axs[0].bar(x + width / 2, recall_vals, width, label="Recall")

    axs[0].set_ylabel("Score")
    axs[0].set_xlabel("Node Valence")
    axs[0].set_title("Precision and Recall by Node Valence")
    axs[0].set_xticks(x)
    axs[0].set_xticklabels(valences)
    axs[0].set_ylim(0, 1.05)
    axs[0].legend()
    axs[0].grid(axis="y")

    for rect in rects1 + rects2:
        height = rect.get_height()
        axs[0].annotate(
            f"{height:.2f}",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )

    axs[1].axis("off")
    metrics_text = (
        f"Test Loss: {test_loss:.4f}\n"
        f"Distance MSE: {distance_mse:.4f}\n"
        f"IoU: {iou:.4f}\n"
        f"Dice: {dice:.4f}"
    )
    axs[1].text(
        0.5,
        0.5,
        metrics_text,
        ha="center",
        va="center",
        fontsize=16,
        fontfamily="monospace",
    )

    fig.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved Full Test Metrics plot to {save_path}")


# --------------- Main Testing ---------------
if __name__ == "__main__":
    # Load model
    config = saved_configs["long-run"]
    model = EfficientUNet5Down(
        in_channels=1, out_channels=2, base_filters=config.base_filters
    ).to(DEVICE)
    model.load_state_dict(torch.load(f"{SAVE_DIR}/model_best.pth", map_location=DEVICE))
    model.eval()

    # Load test data
    test_loader = get_test_loader(BATCH_SIZE)

    # Evaluate
    test_loss = compute_test_loss(model, test_loader, multitask_loss, DEVICE, CONFIG)
    distance_mse = compute_distance_mse(model, test_loader, DEVICE)
    precision, recall = compute_node_precision_recall(model, test_loader, DEVICE)
    iou, dice = compute_iou_and_dice(model, test_loader, DEVICE)

    # Make charts
    plot_node_precision_recall(precision, recall)
    plot_full_test_metrics(precision, recall, iou, dice, test_loss, distance_mse)

    # Print results
    print(f"--- Test Set Evaluation ---")
    print(f"Test Loss: {test_loss:.6f}")
    print(f"Distance MSE: {distance_mse:.6f}")
    for v in [1, 2, 3, 4]:
        print(f"{v}-valent Precision: {precision[v]:.3f} | Recall: {recall[v]:.3f}")
    print(f"IoU: {iou:.3f}")
    print(f"Dice Coefficient: {dice:.3f}")

    # Save predictions
    save_test_predictions(model, test_loader, DEVICE)

    print(f"Saved all test predictions to {SAVE_DIR}/")
