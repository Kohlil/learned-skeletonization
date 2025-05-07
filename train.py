# train.py

from functools import partial
import os
import glob
import argparse
from dataclasses import dataclass
from typing import Dict, Any

from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from PIL import Image
from tqdm import tqdm
from model import EfficientUNet5Down
from scipy import ndimage
from torchvision.utils import save_image
import pandas as pd

# --------------- Config ---------------
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

SAVE_DIR = "./outputs"
DATA_DIR = "./data/train/thinning"


# --------------- Data classes ---------------
@dataclass
class Config:
    name: str = "default"
    batch_size: int = 8
    learning_rate: float = 1e-4
    epochs: int = 100
    alpha: float = 1.0
    beta: float = 1.0
    gamma: float = 0.1
    base_filters: int = 32


def plot_losses(train_losses, val_losses):
    plt.figure(figsize=(8, 6))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(SAVE_DIR, "loss_curve.png"))
    plt.close()
    print(f"Loss curve saved to {SAVE_DIR}/loss_curve.png")


def save_predictions(inputs, outputs, epoch, save_dir=SAVE_DIR):
    os.makedirs(os.path.join(save_dir, f"epoch{epoch}"), exist_ok=True)

    pred_skeleton = torch.sigmoid(outputs[:, 0:1, :, :])  # (B,1,256,256)
    pred_distance = outputs[:, 1:2, :, :]  # (B,1,256,256)

    save_image(
        inputs[0], os.path.join(save_dir, f"epoch{epoch}", "input.png"), normalize=True
    )
    save_image(
        pred_skeleton[0],
        os.path.join(save_dir, f"epoch{epoch}", "pred_skeleton.png"),
        normalize=True,
    )
    save_image(
        pred_distance[0],
        os.path.join(save_dir, f"epoch{epoch}", "pred_distance.png"),
        normalize=True,
    )


# --------------- Dataset ---------------
class ThinningDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.image_paths = sorted(glob.glob(os.path.join(root_dir, "image_*.png")))
        self.target_paths = sorted(glob.glob(os.path.join(root_dir, "target_*.png")))
        self.transform = transform
        assert len(self.image_paths) == len(self.target_paths), "Mismatch!"

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("L")
        target = Image.open(self.target_paths[idx]).convert("L")

        if self.transform:
            image = self.transform(image)
            target = self.transform(target)

        skeleton_target = (target > 0.5).float()
        inverted = 1.0 - skeleton_target.squeeze(0).numpy()
        distance_map = torch.tensor(
            ndimage.distance_transform_edt(inverted), dtype=torch.float32
        )
        distance_map = distance_map / distance_map.max()

        return image, skeleton_target.squeeze(0), distance_map


# --------------- Dataloaders ---------------
def get_dataloaders(batch_size, train_ratio=0.8):
    dataset = ThinningDataset(root_dir=DATA_DIR, transform=T.ToTensor())
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    return train_loader, val_loader


# --------------- Loss ---------------
criterion_bce = nn.BCEWithLogitsLoss()
criterion_mse = nn.MSELoss()


def fast_thinning_loss(pred_skeleton_batch, threshold=0.5):
    kernel = torch.ones((1, 1, 3, 3), device=pred_skeleton_batch.device)
    pred_probs = torch.sigmoid(pred_skeleton_batch.unsqueeze(1))
    pred_binary = (pred_probs > threshold).float()
    neighbor_count = F.conv2d(pred_binary, kernel, padding=1)
    thick_pixels = (neighbor_count > 3).float()
    return thick_pixels.mean()


def multitask_loss(pred, skeleton_target, distance_target, alpha, beta, gamma):
    pred_skeleton = pred[:, 0, :, :]
    pred_distance = pred[:, 1, :, :]

    loss_skeleton = criterion_bce(pred_skeleton, skeleton_target)
    loss_distance = criterion_mse(pred_distance, distance_target)
    loss_thin = fast_thinning_loss(pred_skeleton)

    return alpha * loss_skeleton + beta * loss_distance + gamma * loss_thin


# --------------- Unified training function ---------------
def train(config: Config) -> float:
    train_losses = []
    val_losses = []

    train_loader, val_loader = get_dataloaders(config.batch_size)

    model = EfficientUNet5Down(
        in_channels=1, out_channels=2, base_filters=config.base_filters
    ).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    patience = 5
    is_hyperopt = os.environ.get("HYPEROPT", "0") == "1"

    for epoch in range(config.epochs):
        model.train()
        running_train_loss = 0.0
        loop = tqdm(train_loader, leave=False, disable=is_hyperopt)
        for inputs, skeleton_targets, distance_targets in loop:
            inputs = inputs.to(DEVICE)
            skeleton_targets = skeleton_targets.to(DEVICE)
            distance_targets = distance_targets.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = multitask_loss(
                outputs,
                skeleton_targets,
                distance_targets,
                alpha=config.alpha,
                beta=config.beta,
                gamma=config.gamma,
            )
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()
            avg_loss_so_far = running_train_loss / (loop.n + 1)
            loop.set_description(f"Epoch [{epoch+1}/{config.epochs}]")
            loop.set_postfix(train_loss=f"{avg_loss_so_far:.4f}")

        avg_train_loss = running_train_loss / len(train_loader)

        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            val_loop = tqdm(val_loader, leave=False, disable=is_hyperopt)
            for inputs, skeleton_targets, distance_targets in val_loop:
                inputs = inputs.to(DEVICE)
                skeleton_targets = skeleton_targets.to(DEVICE)
                distance_targets = distance_targets.to(DEVICE)

                outputs = model(inputs)
                val_loss = multitask_loss(
                    outputs,
                    skeleton_targets,
                    distance_targets,
                    alpha=config.alpha,
                    beta=config.beta,
                    gamma=config.gamma,
                )
                running_val_loss += val_loss.item()
                avg_val_loss_so_far = running_val_loss / (val_loop.n + 1)
                val_loop.set_description(f"Val [{epoch+1}/{config.epochs}]")
                val_loop.set_postfix(val_loss=f"{avg_val_loss_so_far:.4f}")

        avg_val_loss = running_val_loss / len(val_loader)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        if not is_hyperopt:
            save_predictions(inputs, outputs, epoch + 1)
            print(
                f"Epoch [{epoch+1}/{config.epochs}] - Train Loss: {avg_train_loss:.6f} - Val Loss: {avg_val_loss:.6f}"
            )

        # Save best model so far
        if not is_hyperopt and avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "model_best.pth"))
            print(
                f"New best model saved at epoch {epoch+1} with Val Loss: {avg_val_loss:.6f}"
            )
        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement} epochs.")

        # Early stopping
        if epochs_without_improvement >= patience:
            print(
                f"Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)"
            )
            break

    if not is_hyperopt:
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, "model_final.pth"))
        plt.figure(figsize=(8, 6))
        plt.plot(train_losses, label="Train Loss")
        plt.plot(val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training vs Validation Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(SAVE_DIR, "loss_curve.png"))
        plt.close()
        print(f"Saved training curve to {SAVE_DIR}/loss_curve.png")
    return best_val_loss


# --------------- Hyperopt ---------------
def objective(params: Dict[str, Any], trials_obj: Trials):
    trial_id = len(trials_obj.trials)
    config = Config(
        name=f"hyperopt_trial_{trial_id}",
        learning_rate=params["lr"],
        batch_size=int(params["batch_size"]),
        alpha=params["alpha"],
        beta=params["beta"],
        gamma=params["gamma"],
        base_filters=int(params["base_filters"]),
        epochs=10,  # or however many you want
    )
    val_loss = train(config)

    print(f"[Hyperopt Trial {trial_id}/{trials_obj.max_trials}] Loss = {val_loss:.6f}")

    return {"loss": val_loss, "status": STATUS_OK}


search_space = {
    "lr": hp.loguniform("lr", np.log(1e-5), np.log(1e-3)),
    "batch_size": hp.choice("batch_size", [4, 8, 16]),
    "alpha": hp.uniform("alpha", 0.5, 2.0),
    "beta": hp.uniform("beta", 0.5, 2.0),
    "gamma": hp.uniform("gamma", 0.05, 0.2),
    "base_filters": hp.choice("base_filters", [16, 32, 64]),
    "epochs": 10,
}


# Optional: decode choice indices back into real values
def decode_choices(df):
    if "batch_size" in df.columns:
        batch_map = {0: 4, 1: 8, 2: 16}
        df["batch_size"] = df["batch_size"].map(batch_map)

    if "base_filters" in df.columns:
        filters_map = {0: 16, 1: 32, 2: 64}
        df["base_filters"] = df["base_filters"].map(filters_map)

    return df


def print_summary(trials: Trials):
    results = []
    for trial in trials.trials:
        result = {
            "trial": trial["tid"],
            "loss": trial["result"]["loss"],
        }
        result.update(trial["misc"]["vals"])  # Add hyperparameter values
        results.append(result)

    # Turn into DataFrame
    df = pd.DataFrame(results)
    df = decode_choices(df)

    # Sort by best loss
    df = df.sort_values(by="loss")

    # Print table
    print("\nHyperopt Results Summary:")
    print(df)

    # Save table to CSV if you want
    df.to_csv(os.path.join(SAVE_DIR, "hyperopt_summary.csv"), index=False)
    print(f"Saved summary table to {SAVE_DIR}/hyperopt_summary.csv")


# --------------- Main ---------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ablation", action="store_true", help="Run hyperopt ablation study"
    )
    args = parser.parse_args()

    if args.ablation:
        os.environ["HYPEROPT"] = "1"  # used to prevent saving checkpoints during search
        trials = Trials()
        best = fmin(
            fn=partial(objective, trials_obj=trials),
            space=search_space,
            algo=tpe.suggest,
            max_evals=20,
            trials=trials,
        )
        print_summary(trials)
        print(f"Best hyperparameters found: {best}")

    else:
        config = Config()
        train(config)
