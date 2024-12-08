# ClimODE reproducibility

This repository attempts to replicate the ClimODE model's results. 

The base repository is at https://github.com/Aalto-QuML/ClimODE. 

# Quickstart

Hardware: 1xRTX4090, 50GB Disk, 32GB RAM.
Environment: `pytorch/pytorch:2.2.1-cuda12.1-cudnn8-devel`

Clone locally
```sh
git clone https://github.com/tornikeo/climode-reproducibility.git
```

Move to cloned directory, then install and download:

```sh
cd climode-reproducibility
bash scripts/install.sh
python scripts/download.py
```

Install needs 16GB, and dataset needs 5GB.

Run training:

```sh
python scripts/train.py
```

# 


Please refer to the [documentation](./docs/on_vastai.md) on running. 

Experiments were done on hardware rented on [vast.ai](https://vast.ai/). 

At minimum, you will need one RTX4090 GPU with 24GBs VRAM and at least 32GB of RAM. The dataset, the packages and the experiments only used up to 50GB of disk space in total. 

# Installing

You need to run `bash scripts/install.sh`. For details see [install.sh](./scripts/install.sh)