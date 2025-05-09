#%%
import json
from pathlib import Path
import sys

sys.path.insert(0, '..')
from project_settings import DATA_FOLDER

with open(Path(DATA_FOLDER) / Path("configs/MUSE_data_config.json"), "r") as f:
    folder_data = json.load(f)

MUSE_DATASET_FOLDER = folder_data["MUSE_dataset_folder"]
MUSE_FITTING_FOLDER = folder_data["MUSE_fitting_folder"]
MUSE_CUBES_FOLDER   = folder_data["MUSE_cubes_folder"]
MUSE_DATA_FOLDER    = folder_data["MUSE_data_folder"]
MUSE_RAW_FOLDER     = folder_data["MUSE_raw_folder"]
LIFT_PATH           = folder_data["LIFT_path"]
