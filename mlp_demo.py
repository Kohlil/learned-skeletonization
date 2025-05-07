import os
from dataclasses import dataclass
import subprocess
from typing import Literal, Dict, Any
from functools import partial

import numpy as np  # conda install numpy
from sklearn.decomposition import PCA
import torch  # python -m pip install torch torchvision torchaudio
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris  # conda install scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter  # conda install tensorboard
from hyperopt import hp, tpe, fmin, Trials  # conda install hyperopt
from matplotlib import pyplot as plt

from tqdm import tqdm, trange  # conda install -c conda-forge tqdm


# Monkey patch print to use tqdm
def tqdm_print(*args, **kwargs):
    tqdm.write(" ".join(map(str, args)), **kwargs)


print = tqdm_print


DEBUG = os.environ.get("DEBUG", "0") == "1"
LOG_INTERVAL = int(os.environ.get("LOG_INTERVAL", 10))
CP_DIR = os.environ.get("CP_DIR", "./checkpoints")
LOG_DIR = os.environ.get("LOG_DIR", "./runs")
HYPEROPT = os.environ.get("HYPEROPT", "0") == "1"
CONFIG = os.environ.get("CONFIG", "ch05-mlp-baseline")
DEBUG_DATASET_SUBSAMPLE_SIZE = 32

# Advice: Use 0 workers for debugging to avoid issues with data loading in subprocesses
NUM_WORKERS = 0 if DEBUG else int(os.environ.get("NUM_WORKERS", "4"))

VISUALIZE = not HYPEROPT  # Do not visualize during hyperparameter search

os.makedirs(CP_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


# Advice: Bundle hyperparameters in a single class
@dataclass
class Config:
    name: str
    hidden_dims: list[int]
    activation: Literal["relu", "tanh"]
    loss_type: Literal["cross_entropy", "label_smoothing"]
    label_smoothing: float = 0.0
    learning_rate: float = 1e-2
    epochs: int = 100
    batch_size: int = 16
    scheduler_type: Literal["step", "none"] = "step"
    scheduler_step_size: int = 20
    scheduler_gamma: float = 0.5
    embed_layer: Literal["output"] = "output"


# Advice: Use named configurations for common setups
CONFIGS = {
    "ch05-mlp-baseline": Config(
        name="ch05-mlp-baseline",
        hidden_dims=[10, 10],
        activation="relu",
        loss_type="cross_entropy",
    ),
    "ch05-mlp-smooth": Config(
        name="ch05-mlp-smooth",
        hidden_dims=[10, 10],
        activation="relu",
        loss_type="label_smoothing",
        label_smoothing=0.1,
    ),
}


class IrisDataset(Dataset):
    INPUT_SIZE = 4
    NUM_CLASSES = 3

    # Advice: Include things like class names or colors for visualization in the dataset class
    # Advice: Often the dataset class is in its own file, with functions to visualize the data etc.

    def __init__(self, train: bool, scaler=None, transform=None):
        iris = load_iris()
        X = iris.data.astype(np.float32)
        y = iris.target.astype(np.int64)
        X = scaler.fit_transform(X) if train else scaler.transform(X)

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        self.X = X_train if train else X_val
        self.y = y_train if train else y_val

        if DEBUG and train:
            self.X = self.X[:DEBUG_DATASET_SUBSAMPLE_SIZE]
            self.y = self.y[:DEBUG_DATASET_SUBSAMPLE_SIZE]

        self.X = torch.tensor(self.X)
        self.y = torch.tensor(self.y)
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Often where data is loaded from disk to avoid keeping it all in memory
        if self.transform is not None:
            x, y = self.transform(x, y)
        return self.X[idx], self.y[idx]


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, activation):
        super().__init__()
        act_fn = nn.ReLU() if activation == "relu" else nn.Tanh()

        hidden_layers = []
        dim = input_dim
        for hidden_dim in hidden_dims[1:]:
            hidden_layers.append(nn.Linear(dim, hidden_dim))
            hidden_layers.append(act_fn)
            dim = hidden_dim

        self.hidden = nn.Sequential(*hidden_layers)
        self.output = nn.Linear(dim, output_dim)

    def forward(self, x):
        # Advice: Document tensor sizes in the code
        # x: (batch_size, input_dim)
        z = self.hidden(x)  # (batch_size, hidden_dim)
        z = self.output(z)  # (batch_size, output_dim)

        return z  # Logits (batch_size, output_dim)


def evaluate(model, loader, loss_fn):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in tqdm(loader, leave=False, desc="Evaluating Batches"):
            preds = model(xb)
            loss = loss_fn(preds, yb)
            total_loss += loss.item() * xb.size(0)
            correct += (preds.argmax(dim=1) == yb).sum().item()
            total += xb.size(0)
    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


# Advice: Visualize WHATEVER you can, you almost never really know what is going on
# Advice: Use type hints to get auto-completion in your IDE
def log_softmax_barycentric_projection(
    model: MLP, loader: DataLoader, writer: SummaryWriter, epoch: int
):
    """
    Visualizes class separation by projecting softmax outputs into 2D using barycentric coordinates.
    Each class is mapped to a fixed anchor on the unit circle, and softmax probabilities become weights.

    Saves both a TensorBoard figure and a PNG file to log_dir.
    """
    model.eval()
    logits_all, labels_all = [], []
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb)
            probs = torch.softmax(logits, dim=1)
            logits_all.append(probs.numpy())
            labels_all.append(yb.numpy())

    probs = np.concatenate(logits_all, axis=0)
    labels = np.concatenate(labels_all, axis=0)

    num_classes = probs.shape[1]
    # Place each class anchor equally spaced on the unit circle
    angles = np.linspace(0, 2 * np.pi, num_classes, endpoint=False)
    anchors = np.stack([np.cos(angles), np.sin(angles)], axis=1)  # shape: (C, 2)

    # Compute convex combination of anchors weighted by softmax probs
    points = probs @ anchors  # shape: (N, 2)

    # Plot
    fig, ax = plt.subplots(figsize=(5, 5))
    for c in range(num_classes):
        idxs = labels == c
        ax.scatter(points[idxs, 0], points[idxs, 1], alpha=0.6, label=f"Class {c}")
    ax.set_title(f"Softmax Barycentric Projection @ Epoch {epoch}")
    ax.legend()
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect("equal")

    writer.add_figure("embedding/barycentric_softmax", fig, global_step=epoch)

    # Save PNG for video rendering or inspection
    fig.savefig(
        os.path.join(writer.log_dir, f"barycentric_softmax_epoch_{epoch:04d}.png")
    )
    plt.close(fig)


# Ignore the complexity of the code below, focus on the training loop, refactoring obscures the details
def train(config: Config) -> float:
    scaler = StandardScaler()
    train_ds = IrisDataset(train=True, scaler=scaler)
    val_ds = IrisDataset(train=False, scaler=scaler)
    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True, num_workers=NUM_WORKERS
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.batch_size, num_workers=NUM_WORKERS
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = MLP(
        IrisDataset.INPUT_SIZE,
        config.hidden_dims,
        IrisDataset.NUM_CLASSES,
        config.activation,
    )
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    if config.scheduler_type == "step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.scheduler_step_size,
            gamma=config.scheduler_gamma,
        )
    else:
        scheduler = None

    loss_fn = (
        nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
        if config.loss_type == "label_smoothing"
        else nn.CrossEntropyLoss()
    )

    writer = SummaryWriter(log_dir=os.path.join(LOG_DIR, config.name))

    global_step = 0
    best_val_loss = float("inf")
    best_epoch = -1
    for epoch in trange(config.epochs, leave=False, desc=f"[{config.name}] Training"):
        model.train()
        for batch_idx, (xb, yb) in enumerate(
            tqdm(train_loader, leave=False, desc="Training Batches")
        ):
            preds = model(xb)
            loss = loss_fn(preds, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            writer.add_scalar("train/loss", loss.item(), global_step)
            if batch_idx % LOG_INTERVAL == 0:
                print(
                    f"[{config.name}] Epoch {epoch} Iter {batch_idx} Train Loss: {loss.item():.4f}"
                )
            global_step += 1

        val_loss, val_acc = evaluate(model, val_loader, loss_fn)
        writer.add_scalar("val/loss", val_loss, epoch)
        writer.add_scalar("val/acc", val_acc, epoch)
        print(
            f"[{config.name}] Epoch {epoch} Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2%}"
        )

        if scheduler:
            scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config,
                "val_loss": val_loss,
                "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            }
            torch.save(checkpoint, os.path.join(CP_DIR, f"{config.name}_best.pt"))

        if VISUALIZE:
            log_softmax_barycentric_projection(model, val_loader, writer, epoch)

    print(f"[{config.name}] Best Val Loss: {best_val_loss:.4f} @ Epoch {best_epoch}")

    # Log final hparams after training
    hparams = {
        k: (str(v) if isinstance(v, (list, tuple, dict)) else v)
        for k, v in config.__dict__.items()
    }
    writer.add_hparams(hparams, {"best_val_loss": best_val_loss})

    if VISUALIZE:
        # Convert the saved images to a video if ffmpeg is installed
        subprocess.run(
            f"ffmpeg -framerate 10 -i {writer.log_dir}/barycentric_softmax_epoch_%04d.png  {writer.log_dir}/barycentric_softmax.mp4",
            shell=True,
        )
        print(
            "Video saved to:", os.path.join(writer.log_dir, "barycentric_softmax.mp4")
        )

    return best_val_loss


def objective(params: Dict[str, Any], trials_obj: Trials):
    trial_id = f"{len(trials_obj.trials):04d}"
    config_name = f"hyperopt_{params['activation']}_{params['loss_type']}_T{trial_id}"
    trial_config = Config(
        name=config_name,
        hidden_dims=params["hidden_dims"],
        activation=params["activation"],
        loss_type=params["loss_type"],
        label_smoothing=(
            params["label_smoothing"]
            if params["loss_type"] == "label_smoothing"
            else 0.0
        ),
        learning_rate=params["learning_rate"],
        epochs=30,
        batch_size=16,
    )
    return train(trial_config)


search_space = {
    "hidden_dims": hp.choice(
        "hidden_dims",
        [
            [8],
            [16],
            [32],
            [16, 8],
            [32, 16],
            [64, 32],
        ],
    ),
    "activation": hp.choice("activation", ["relu", "tanh"]),
    "loss_type": hp.choice("loss_type", ["cross_entropy", "label_smoothing"]),
    "label_smoothing": hp.uniform("label_smoothing", 0.0, 0.2),
    "learning_rate": hp.loguniform("learning_rate", np.log(1e-4), np.log(1e-1)),
}


def main():
    if HYPEROPT:
        trials = Trials()
        best = fmin(
            fn=partial(objective, trials_obj=trials),
            space=search_space,
            algo=tpe.suggest,
            max_evals=10,
            trials=trials,
        )
        print("Best parameters found:", best)
    else:
        config = CONFIGS[CONFIG]
        train(config)


# Advice: Run without any arguments to make debbugging easier
# Advice: You can import this into another module to create a CLI if desired
if __name__ == "__main__":
    main()
