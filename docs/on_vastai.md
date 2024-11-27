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

1. Clone repo locally
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

1. run `python scripts/upzip.py`
1. Check that all files are under data. You must have a lot of .nc files in each directory.

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

**NOTE** `data/` directory will be 26GB total at this point.

