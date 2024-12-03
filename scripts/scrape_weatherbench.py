# https://dataserv.ub.tum.de/s/m1524895/download?path=%2F5.625deg&files=2m_temperature&downloadStartSecret=12lsaq32pafa
# https://dataserv.ub.tum.de/s/m1524895/download?path=%2F5.625deg&files=constants&downloadStartSecret=3zfris9dr71
# https://dataserv.ub.tum.de/s/m1524895/download?path=%2F5.625deg&files=temperature_850&downloadStartSecret=kmnengl2089
# https://dataserv.ub.tum.de/s/m1524895/download?path=%2F5.625deg&files=geopotential_500&downloadStartSecret=gdz07op16a9
# https://dataserv.ub.tum.de/s/m1524895/download?path=%2F5.625deg&files=10m_u_component_of_wind&downloadStartSecret=189bpp6gakj
# https://dataserv.ub.tum.de/s/m1524895/download?path=%2F5.625deg&files=10m_v_component_of_wind&downloadStartSecret=4hyltlocgzs

import pooch
import os
import os
import zipfile

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

# Base directories
data_dir = "era5_data"
out_dir = "climode/ClimODE/era5_data"

# Ensure the output directory exists
os.makedirs(out_dir, exist_ok=True)

# List of subdirectories to process
subdirs = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_temperature",
    "constants",
    "geopotential_500",
    "temperature_850",
]

# Process each subdirectory
for subdir in subdirs:
    print(f"Processing {subdir}...")

    # Path to the first-level zip file
    first_zip_path = os.path.join(data_dir, subdir, subdir)

    # Define the output path for the first unzip
    first_unzip_dir = os.path.join(out_dir, subdir)
    os.makedirs(first_unzip_dir, exist_ok=True)

    # Unzip the first-level zip file
    with zipfile.ZipFile(first_zip_path, 'r') as first_zip:
        print(f"Unzipping {first_zip_path} to {first_unzip_dir}...")
        first_zip.extractall(first_unzip_dir)

    # Locate the second-level zip file
    second_zip_path = None
    for root, dirs, files in os.walk(first_unzip_dir):
        for file in files:
            if file.endswith(".zip"):
                second_zip_path = os.path.join(root, file)
                break
        if second_zip_path:
            break

    if not second_zip_path:
        print(f"No second-level zip file found in {first_unzip_dir}")
        continue

    # Define the output path for the second unzip
    second_unzip_dir = os.path.join(first_unzip_dir)

    # Unzip the second-level zip file
    with zipfile.ZipFile(second_zip_path, 'r') as second_zip:
        print(f"Unzipping {second_zip_path} to {second_unzip_dir}...")
        second_zip.extractall(second_unzip_dir)

print("All files have been unzipped.")