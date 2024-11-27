# https://dataserv.ub.tum.de/s/m1524895/download?path=%2F5.625deg&files=2m_temperature&downloadStartSecret=12lsaq32pafa
# https://dataserv.ub.tum.de/s/m1524895/download?path=%2F5.625deg&files=constants&downloadStartSecret=3zfris9dr71
# https://dataserv.ub.tum.de/s/m1524895/download?path=%2F5.625deg&files=temperature_850&downloadStartSecret=kmnengl2089
# https://dataserv.ub.tum.de/s/m1524895/download?path=%2F5.625deg&files=geopotential_500&downloadStartSecret=gdz07op16a9
# https://dataserv.ub.tum.de/s/m1524895/download?path=%2F5.625deg&files=10m_u_component_of_wind&downloadStartSecret=189bpp6gakj
# https://dataserv.ub.tum.de/s/m1524895/download?path=%2F5.625deg&files=10m_v_component_of_wind&downloadStartSecret=4hyltlocgzs

import pooch
import os

# Define the base directory for the dataset
data_dir = "era5_data"

# Define the mapping of file names to URLs
files = {
    "2m_temperature": "https://dataserv.ub.tum.de/s/m1524895/download?path=%2F5.625deg&files=2m_temperature&downloadStartSecret=12lsaq32pafa",
    "constants": "https://dataserv.ub.tum.de/s/m1524895/download?path=%2F5.625deg&files=constants&downloadStartSecret=3zfris9dr71",
    "temperature_850": "https://dataserv.ub.tum.de/s/m1524895/download?path=%2F5.625deg&files=temperature_850&downloadStartSecret=kmnengl2089",
    "geopotential_500": "https://dataserv.ub.tum.de/s/m1524895/download?path=%2F5.625deg&files=geopotential_500&downloadStartSecret=gdz07op16a9",
    "10m_u_component_of_wind": "https://dataserv.ub.tum.de/s/m1524895/download?path=%2F5.625deg&files=10m_u_component_of_wind&downloadStartSecret=189bpp6gakj",
    "10m_v_component_of_wind": "https://dataserv.ub.tum.de/s/m1524895/download?path=%2F5.625deg&files=10m_v_component_of_wind&downloadStartSecret=4hyltlocgzs",
}

# Ensure the data directory exists
os.makedirs(data_dir, exist_ok=True)

# Use pooch to download each file
for name, url in files.items():
    # Define the target path for the file
    target_dir = os.path.join(data_dir, name)
    os.makedirs(target_dir, exist_ok=True)

    # Define the file path
    file_path = os.path.join(target_dir, name)

    # Download the file using pooch
    print(f"Downloading {name}...")
    pooch.retrieve(
        url=url,
        fname=name,
        path=target_dir,
        progressbar=True,
        known_hash=None,
    )

print("All files downloaded and organized in the era5_data directory.")
