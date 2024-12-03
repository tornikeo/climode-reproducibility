from huggingface_hub import snapshot_download

if __name__ == '__main__':
    snapshot_download(repo_id="TornikeO/era5-5.625deg",  # dataset is ready at our hub on https://huggingface.co/datasets/TornikeO/era5-5.625deg
                    repo_type='dataset',
                    local_dir='climode/ClimODE/era5_data', 
                    allow_patterns="*")