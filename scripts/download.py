from huggingface_hub import snapshot_download
import shutil
if __name__ == "__main__":
    snapshot_download(
        repo_id="TornikeO/era5-5.625deg",  # dataset is ready at our hub on https://huggingface.co/datasets/TornikeO/era5-5.625deg
        repo_type="dataset",
        local_dir="era5_data",
        allow_patterns="*",
    )
    print("\n"*4)
    print("Done!")
