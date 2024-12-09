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

Run training (global model, batch size 6, odeint euler, all else default):

First-time run is slower. It will create suppring `.npy` files, which are reused in later runs.

```sh
python scripts/train.py
```

This shall create several checkpoints under `checkpoints/`. Select one for evaluation with for example:

```sh
python scripts/evaluate.py --checkpoint_path checkpoints/ClimODE_global_euler_0_model_10_-438.79186260700226.pt
```

# No RTX4090?

Please refer to the [documentation](./docs/on_vastai.md) on running. RTX4090 costs ~$0.45/hr to rent.

Experiments were done on hardware rented on [vast.ai](https://vast.ai/). 

At minimum, you will need one RTX4090 GPU with 24GBs VRAM and at least 32GB of RAM. The dataset, the packages and the experiments only used up to 50GB of disk space in total. 

# Testing

A simple testcase for quickly testing train/eval scripts have been added. Run it with:

```sh
pytest -sx
```
