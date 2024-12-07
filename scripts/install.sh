#!/bin/bash
## NOTE:
# most of the heaviest packages are already available on
# docker pytorch devel
# You can use for example: 
# docker pull pytorch/pytorch:2.2.1-cuda12.1-cudnn8-devel
# Which has most of what you will need already installed.

set -e

python -c "import torch; assert torch.cuda.is_available()"

## General utils
sudo apt-get install tree -y
sudo apt-get install unzip -y 
sudo apt-get install build-essential -y

## Pytorch geom stuff
conda install pyg -c pyg -y

python -c "import torch; assert torch.cuda.is_available()"

# Numba
# conda install -c conda-forge cuda-nvcc cuda-nvrtc "cuda-version>=12.0" -y
conda install numba -y
pip install git+https://github.com/patrick-kidger/torchcubicspline.git

# weather prediction items
conda install xarray dask netCDF4 bottleneck -c conda-forge -y

# python utilities
pip install pooch matplotlib numpy ipykernel \
    tqdm pandas datasets transformers \
    scipy properscoring torchdiffeq

code --install-extension ms-python.python
code --install-extension ms-toolsai.jupyter
code --install-extensions ms-python.black-formatter