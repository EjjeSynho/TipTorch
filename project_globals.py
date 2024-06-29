import json
import torch
import pathlib
from pathlib import Path

path = pathlib.Path(__file__).parent.resolve()

with open(path / Path("data/global_config.json"), "r") as f:
    folder_data = json.load(f)

SPHERE_OOPAO_FITTING_FOLDER = folder_data["SPHERE_OOPAO_fitting_folder"]
SPHERE_FITTING_FOLDER = folder_data["SPHERE_fitting_folder"]
SPHERE_DATASET_FOLDER = folder_data["SPHERE_dataset_folder"]
SPHERE_DATA_FOLDER = folder_data["SPHERE_data_folder"]
MUSE_CUBES_FOLDER = folder_data["MUSE_cubes_folder"]
MUSE_DATA_FOLDER = folder_data["MUSE_data_folder"]
MUSE_RAW_FOLDER = folder_data["MUSE_raw_folder"]
WEIGHTS_FOLDER = path / Path("data/weights")
DATA_FOLDER = path / Path("data/")
MAX_NDIT = folder_data["Max NDITs"]
DEVICE = folder_data["device"]

device = torch.device(DEVICE) if torch.cuda.is_available else torch.device('cpu')
