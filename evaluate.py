# evaluate.py

import torch
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import convolve
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm  # <-- Added tqdm
from train import Config

# --------------- Metrics Functions ---------------


def compute_test_loss(model, dataloader, loss_fn, device, config: Config):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        loop = tqdm(dataloader, desc="Computing Test Loss", leave=False)
        for inputs, skeleton_targets, distance_targets in loop:
            inputs = inputs.to(device)
            skeleton_targets = skeleton_targets.to(device)
            distance_targets = distance_targets.to(device)

            outputs = model(inputs)
            loss = loss_fn(
                outputs,
                skeleton_targets,
                distance_targets,
                config.alpha,
                config.beta,
                config.gamma,
            )
            running_loss += loss.item()

            avg_loss = running_loss / (loop.n + 1)
            loop.set_postfix(test_loss=f"{avg_loss:.6f}")

    avg_loss = running_loss / len(dataloader)
    return avg_loss


def compute_distance_mse(model, dataloader, device):
    model.eval()
    total_mse = 0.0
    with torch.no_grad():
        loop = tqdm(dataloader, desc="Computing Distance MSE", leave=False)
        for inputs, _, distance_targets in loop:
            inputs = inputs.to(device)
            distance_targets = distance_targets.to(device)

            outputs = model(inputs)
            pred_distance = outputs[:, 1, :, :]

            mse = F.mse_loss(pred_distance, distance_targets, reduction="mean")
            total_mse += mse.item()

            avg_mse = total_mse / (loop.n + 1)
            loop.set_postfix(distance_mse=f"{avg_mse:.6f}")

    avg_mse = total_mse / len(dataloader)
    return avg_mse


def detect_nodes(skeleton):
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    neighbor_count = convolve(skeleton, kernel, mode="constant", cval=0)

    nodes = []
    for y in range(skeleton.shape[0]):
        for x in range(skeleton.shape[1]):
            if skeleton[y, x] == 1:
                valence = neighbor_count[y, x]
                if valence in [1, 2, 3, 4]:
                    nodes.append((x, y, valence))
    return nodes


def match_nodes(pred_nodes, gt_nodes, valence, max_dist=3):
    pred_filtered = [(x, y) for (x, y, v) in pred_nodes if v == valence]
    gt_filtered = [(x, y) for (x, y, v) in gt_nodes if v == valence]

    if len(gt_filtered) == 0:
        return 1.0, 1.0
    if len(pred_filtered) == 0:
        return 0.0, 0.0

    cost_matrix = np.zeros((len(pred_filtered), len(gt_filtered)))
    for i, (px, py) in enumerate(pred_filtered):
        for j, (gx, gy) in enumerate(gt_filtered):
            cost_matrix[i, j] = np.sqrt((px - gx) ** 2 + (py - gy) ** 2)

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matches = sum(cost_matrix[r, c] <= max_dist for r, c in zip(row_ind, col_ind))

    precision = matches / len(pred_filtered)
    recall = matches / len(gt_filtered)

    return precision, recall


def compute_node_precision_recall(model, dataloader, device):
    model.eval()
    total_precision = {1: [], 2: [], 3: [], 4: []}
    total_recall = {1: [], 2: [], 3: [], 4: []}

    with torch.no_grad():
        loop = tqdm(dataloader, desc="Computing Node Precision/Recall", leave=False)
        for inputs, skeleton_targets, _ in loop:
            inputs = inputs.to(device)
            outputs = model(inputs)
            pred_skeleton = torch.sigmoid(outputs[:, 0, :, :])
            pred_binary = (pred_skeleton > 0.5).float()

            for i in range(inputs.size(0)):
                pred_np = pred_binary[i].cpu().numpy()
                gt_np = skeleton_targets[i].cpu().numpy()

                pred_nodes = detect_nodes(pred_np)
                gt_nodes = detect_nodes(gt_np)

                for valence in [1, 2, 3, 4]:
                    precision, recall = match_nodes(pred_nodes, gt_nodes, valence)
                    total_precision[valence].append(precision)
                    total_recall[valence].append(recall)

    avg_precision = {v: np.mean(total_precision[v]) for v in total_precision}
    avg_recall = {v: np.mean(total_recall[v]) for v in total_recall}

    return avg_precision, avg_recall


def compute_iou_and_dice(model, dataloader, device):
    model.eval()
    total_iou = 0.0
    total_dice = 0.0

    with torch.no_grad():
        loop = tqdm(dataloader, desc="Computing IoU/Dice", leave=False)
        for inputs, skeleton_targets, _ in loop:
            inputs = inputs.to(device)
            skeleton_targets = skeleton_targets.to(device)

            outputs = model(inputs)
            pred_skeleton = torch.sigmoid(outputs[:, 0, :, :])
            pred_binary = (pred_skeleton > 0.5).float()

            intersection = (pred_binary * skeleton_targets).sum(dim=(1, 2))
            union = (
                pred_binary + skeleton_targets - pred_binary * skeleton_targets
            ).sum(dim=(1, 2))
            dice = (2 * intersection) / (
                pred_binary.sum(dim=(1, 2)) + skeleton_targets.sum(dim=(1, 2)) + 1e-8
            )

            batch_iou = (intersection / (union + 1e-8)).mean()
            batch_dice = dice.mean()

            total_iou += batch_iou.item()
            total_dice += batch_dice.item()

    avg_iou = total_iou / len(dataloader)
    avg_dice = total_dice / len(dataloader)

    return avg_iou, avg_dice
