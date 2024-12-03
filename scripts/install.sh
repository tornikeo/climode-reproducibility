#!/bin/bash
## NOTE:
# most of the heaviest packages are already available on
# docker pytorch devel
# You can use for example: 
# docker pull pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel
# Which has most of what you will need already installed.

set -e

## General utils
sudo apt-get install tree -y
sudo apt-get install unzip -y 
sudo apt-get install build-essential -y

## Pytorch geom stuff
conda install pytorch-cluster -c pyg -c pytorch -c nvidia -y

# Numba
conda install -c conda-forge cuda-nvcc cuda-nvrtc "cuda-version>=12.0" -y
conda install numba -y
pip install git+https://github.com/patrick-kidger/torchcubicspline.git

# weather prediction items
conda install -c conda-forge xarray dask netCDF4 bottleneck -y

# python utilities
pip install pooch matplotlib numpy ipykernel \
    tqdm pandas datasets transformers \
    scipy properscoring torchdiffeq
