# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.transforms as T
import torch.nn.functional as F
from torchvision.utils import save_image
import os
from model import EfficientUNet5Down
from tqdm import tqdm
import matplotlib.pyplot as plt
import glob
from PIL import Image
from scipy import ndimage
import numpy as np
from skimage.morphology import skeletonize

# --------------- Config ---------------
if torch.backends.mps.is_available():
    DEVICE = "mps"  # Metal GPU on Mac
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

EPOCHS = 20
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
SAVE_DIR = "./outputs"
DATA_DIR = "./data/train/thinning"
INPUT_SAVE_DIR = os.path.join(SAVE_DIR, "inputs")
SKELETON_SAVE_DIR = os.path.join(SAVE_DIR, "pred_skeletons")
DISTANCE_SAVE_DIR = os.path.join(SAVE_DIR, "pred_distances")
os.makedirs(INPUT_SAVE_DIR, exist_ok=True)
os.makedirs(SKELETON_SAVE_DIR, exist_ok=True)
os.makedirs(DISTANCE_SAVE_DIR, exist_ok=True)


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


class ThinningDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        super().__init__()
        self.image_paths = sorted(glob.glob(os.path.join(root_dir, "image_*.png")))
        self.target_paths = sorted(glob.glob(os.path.join(root_dir, "target_*.png")))
        self.transform = transform

        assert len(self.image_paths) == len(
            self.target_paths
        ), "Number of images and targets do not match!"

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        target_path = self.target_paths[idx]

        image = Image.open(image_path).convert("L")  # grayscale
        target = Image.open(target_path).convert("L")  # grayscale

        if self.transform:
            image = self.transform(image)
            target = self.transform(target)

        # Binary skeleton
        skeleton_target = (target > 0.5).float()  # (B, 1, 256, 256)

        # Distance Transform
        skeleton_np = skeleton_target.squeeze(0).numpy()  # shape (256, 256)
        inverted = (
            1.0 - skeleton_np
        )  # so skeleton pixels (==1) become 0, background (==0) becomes 1
        distance_map = ndimage.distance_transform_edt(inverted)

        # Normalize distance map to [0,1] for stability
        distance_map = torch.tensor(distance_map, dtype=torch.float32)
        distance_map = distance_map / distance_map.max()

        return image, skeleton_target.squeeze(0), distance_map


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


# Loss function
criterion_bce = nn.BCEWithLogitsLoss()
criterion_mse = nn.MSELoss()


def fast_thinning_loss(pred_skeleton_batch, threshold=0.5):
    # 3x3 convolution kernel to count neighbors
    kernel = torch.ones((1, 1, 3, 3), device=pred_skeleton_batch.device)

    # Apply sigmoid to get probabilities
    pred_probs = torch.sigmoid(pred_skeleton_batch.unsqueeze(1))  # (B,1,H,W)

    # Threshold to binary
    pred_binary = (pred_probs > threshold).float()

    # Convolve to count how many neighbors each pixel has
    neighbor_count = F.conv2d(pred_binary, kernel, padding=1)

    # Ideal skeleton pixel has neighbor count ~2 or fewer (endpoints or thin lines)
    thick_pixels = (neighbor_count > 3).float()

    # We want thick_pixels to be 0 everywhere
    loss = thick_pixels.mean()

    return loss


def multitask_loss(
    pred, skeleton_target, distance_target, alpha=1.0, beta=1.0, gamma=0.1
):
    pred_skeleton = pred[:, 0, :, :]
    pred_distance = pred[:, 1, :, :]

    loss_skeleton = criterion_bce(pred_skeleton, skeleton_target)
    loss_distance = criterion_mse(pred_distance, distance_target)
    loss_thin = fast_thinning_loss(pred_skeleton)  # Use the fast version here

    return alpha * loss_skeleton + beta * loss_distance + gamma * loss_thin


# --------------- Main Training Loop ---------------
def train():
    # Loaders
    train_loader, val_loader = get_dataloaders(BATCH_SIZE)

    # Model
    model = EfficientUNet5Down(in_channels=1, out_channels=2).to(DEVICE)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Best validation loss tracker
    best_val_loss = float("inf")

    # Track losses for plotting
    train_losses = []
    val_losses = []

    # Training
    for epoch in range(EPOCHS):
        model.train()
        loop = tqdm(train_loader, total=len(train_loader))
        running_train_loss = 0.0

        for inputs, skeleton_targets, distance_targets in loop:
            inputs = inputs.to(DEVICE)
            skeleton_targets = skeleton_targets.to(DEVICE)
            distance_targets = distance_targets.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = multitask_loss(outputs, skeleton_targets, distance_targets)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()

            loop.set_description(f"Epoch [{epoch+1}/{EPOCHS}]")
            loop.set_postfix(loss=loss.item())

        avg_train_loss = running_train_loss / len(train_loader)

        # Validation
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for inputs, skeleton_targets, distance_targets in val_loader:
                inputs = inputs.to(DEVICE)
                skeleton_targets = skeleton_targets.to(DEVICE)
                distance_targets = distance_targets.to(DEVICE)

                outputs = model(inputs)
                val_loss = multitask_loss(outputs, skeleton_targets, distance_targets)
                running_val_loss += val_loss.item()

        avg_val_loss = running_val_loss / len(val_loader)

        # Record losses
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        print(
            f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}"
        )

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_path = os.path.join(SAVE_DIR, "model_best.pth")
            torch.save(model.state_dict(), best_model_path)
            print(
                f"New best model saved at epoch {epoch+1} with Val Loss: {best_val_loss:.6f}"
            )

        save_prediction(inputs, outputs, epoch)

    # Save final model
    final_model_path = os.path.join(SAVE_DIR, "model_final.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"Training finished. Final model saved to {final_model_path}")

    # After all epochs — Plot and save loss curves
    plot_losses(train_losses, val_losses)


def save_prediction(inputs, outputs, epoch):
    """Save input and prediction images for inspection"""
    pred_skeleton = torch.sigmoid(outputs[:, 0:1, :, :])  # (B,1,256,256)
    pred_distance = outputs[:, 1:2, :, :]  # (B,1,256,256)

    save_image(
        inputs[0],
        os.path.join(INPUT_SAVE_DIR, f"input_epoch{epoch+1}.png"),
        normalize=True,
    )
    save_image(
        pred_skeleton[0],
        os.path.join(SKELETON_SAVE_DIR, f"pred_skeleton_epoch{epoch+1}.png"),
        normalize=True,
    )
    save_image(
        pred_distance[0],
        os.path.join(DISTANCE_SAVE_DIR, f"pred_distance_epoch{epoch+1}.png"),
        normalize=True,
    )


if __name__ == "__main__":
    train()
