#!/bin/sh

# Install dependencies
echo "\n##############    Installing Dependencies    ##############\n"
conda install -y numpy scikit-learn hyperopt tensorboard matplotlib geopandas osmnx rasterio torch-tb-profiler
python -m pip install torch torchvision torchaudio

# Download train / test data
echo "\n\n##############    Downloading Dataset    ##############\n"
python make_data.py