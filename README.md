# Road Skeletonization with Efficient U-Net

This project implements a deep learning pipeline for **road skeletonization** from aerial imagery, using a customized **Efficient U-Net** model. The system includes data preparation, training, evaluation, and inference scripts.

---

## Table of Contents

- [Road Skeletonization with Efficient U-Net](#road-skeletonization-with-efficient-u-net)
  - [Table of Contents](#table-of-contents)
  - [Project Overview](#project-overview)
  - [Installation](#installation)
  - [Usage](#usage)
    - [Prepare Data](#prepare-data)
    - [Train Model](#train-model)
    - [Test Model](#test-model)
  - [Model Architecture](#model-architecture)
  - [File Descriptions](#file-descriptions)
  - [License](#license)
  - [Notes](#notes)

---

## Project Overview

This project aims to generate 1-pixel-wide **road skeleton maps** from input aerial images. It uses a 5-down Efficient U-Net structure with **Leaky ReLU** activations and **Batch Normalization** layers.

---

## Installation

You can install the required dependencies using the provided installation script:

```bash
bash install.sh
```

Alternatively, install manually:

```bash
pip install -r requirements.txt
```

---

## Usage

### Prepare Data

First, prepare your dataset using:

```bash
python make_data.py
```

This script preprocesses raw data into training, validation, and testing folders.

### Train Model

To train the model:

```bash
python train.py
```

Training outputs model checkpoints to the `saved_models/` directory.

### Test Model

To test the model on unseen data:

```bash
python test.py
```

The predictions will be saved under a `results/` directory.

---

## Model Architecture

The model is based on a simplified U-Net design with the following structure:

- Double Convolution blocks (Conv2D -> BatchNorm -> LeakyReLU -> Conv2D -> BatchNorm -> LeakyReLU)
- 5 levels of downsampling and upsampling
- Skip connections between matching resolution levels

Implemented in `model.py`.

---

## File Descriptions

| File           | Description                                   |
| -------------- | --------------------------------------------- |
| `install.sh`   | Installs the required Python packages.        |
| `make_data.py` | Prepares and preprocesses the dataset.        |
| `train.py`     | Trains the Efficient U-Net model.             |
| `test.py`      | Runs inference on the test dataset.           |
| `evaluate.py`  | Evaluates model outputs against ground truth. |
| `model.py`     | Defines the Efficient U-Net architecture.     |

---

## License

This project is released under the MIT License.

---

## Notes

- Make sure your dataset is structured correctly (input images and corresponding masks).
- CUDA-enabled GPU is recommended for faster training.
- The project assumes input and output images are 256x256 pixels.
