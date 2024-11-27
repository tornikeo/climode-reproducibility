import os
import zipfile

# Base directories
data_dir = "era5_data"
out_dir = "data"

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