# Start an RTX4090 instance on vast.ai

1. Sign up for vast.ai, and use credit card to buy credits ($10 will be enough)
1. Search for an europe-based instance with 256 GB of Disk space, with 1x RTX4090, using search bar in vast.ai (eu, because lower latency)

![alt text](vastai.png)

1. "Edit Image" -> Select pytorch:latest docker image. 

![alt text](pytorch.png)

# SSH into the instance

1. First, grab the ssh connection info:

![alt text](ssh.png)

1. Paste it into terminal, and say "yes". Check you have "nvidia-smi" available. This means GPU is online.

1. Open VSCode, and install "remote development extension pack", from extensions.

1. Press ctrl+p, and type "add new ssh host", and paste in the full "ssh connection info", like "ssh -p 51167 root@85.167.26.137 -L 8080:localhost:8080"

1. Connect.

# install

1. Clone repo with 
```sh
git clone https://github.com/tornikeo/climode-reproducibility.git --recursive
```

1. Run pip install -r requirements.txt
1. You will need to have a command `unzip` available, so install it by running scripts `commands.sh`

# download ERA5

1. run `python scripts/download.py` this will make `era5_data` directory and download your data. Downloaded files will look like this:

```
$ tree era5_data

era5_data/
├── 10m_u_component_of_wind
│   └── 10m_u_component_of_wind
├── 10m_v_component_of_wind
│   └── 10m_v_component_of_wind
├── 2m_temperature
│   └── 2m_temperature
├── constants
│   └── constants
├── geopotential_500
│   └── geopotential_500
└── temperature_850
    └── temperature_850
6 directories, 6 files
```
Each of 6 files is a zipfile here, actually. We need to extract each.

1. run `python scripts/upzip.py` This will both extract files and move them to `climode/ClimODE`
1. Check that all files are under . You must have a lot of .nc files in each directory.

```
$ tree -du -h data/
[root      182]  data/
├── [root     4.0K]  10m_u_component_of_wind
│   └── [root       58]  10m_u_component_of_wind
├── [root     4.0K]  10m_v_component_of_wind
│   └── [root       58]  10m_v_component_of_wind
├── [root     4.0K]  2m_temperature
│   └── [root       49]  2m_temperature
├── [root       31]  constants
│   └── [root       43]  constants
├── [root     4.0K]  geopotential_500
│   └── [root       51]  geopotential_500
└── [root     4.0K]  temperature_850
    └── [root       50]  temperature_850
```

**NOTE** `climode/ClimODE/era5_data` directory will now be 26GB total at this point.

# Install climode-specific requirements

1. First, install pyg, using `conda install pytorch-cluster -c pyg -y`
1. Then, run `pip install -r requirements.txt` to install the rest of the packages.
1. Run `conda install -c conda-forge xarray dask netCDF4 bottleneck -y` which will install `xarray` package and data I/O libraries.

All of above will take a 30 mins or so, so be patient. This is tested on the following vast.ai template:
"https://cloud.vast.ai/templates/edit?templateHashId=4a5b7f1e0aba3527f1f75cfb3bfc75b5"

Locally, you could also use either a:

`docker pull pytorch/pytorch:2.2.1-cuda12.1-cudnn8-devel` docker image (9GB)

or 

`docker pull pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime` docker image (3.5GB), but lacks build tools.

# Run and fix

1. Run eval first. Make sure to be in `~/climode-reproducibility/climode/ClimODE$` directory
1. Run `python evaluation_global.py --spectral 0 --scale 0 --batch_size 8` and 
    You will get a `FileNotFoundError: [Errno 2] No such file or directory: b'/root/climode-reproducibility/climode/ClimODE/era5_data/constants/constants_5.625deg.nc'` error. 
1. Fix this by moving the constants file above:
    `cp era5_data/constants/constants/constants_5.625deg.nc era5_data/constants/`

1. Re-run that darn script.

1. You WILL get another error of `FileNotFoundError: [Errno 2] No such file or directory: '### Test velocity here'`. This is where developer messed up. You can see this as an issue on their [repo](https://github.com/Aalto-QuML/ClimODE/issues/7), but it seems devs don't care. We have to fix it.

