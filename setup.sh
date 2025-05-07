#!/bin/sh

# Install dependencies
echo "\n##############    Installing Dependencies    ##############\n"
python -m pip install torch torchvision torchaudio numpy scikit-learn hyperopt tensorboard matplotlib geopandas osmnx rasterio torch-tb-profiler

# Download train / test data
echo "\n\n##############    Downloading Dataset    ##############\n"
python make_data.py