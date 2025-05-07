import json
from pathlib import Path
import sys

from project_settings import DATA_FOLDER
sys.path.insert(0, '..')

with open(Path(DATA_FOLDER) / Path("configs/SPHERE_data_config.json"), "r") as f:
    folder_data = json.load(f)

SPHERE_OOPAO_FITTING_FOLDER = folder_data["SPHERE_OOPAO_fitting_folder"]
SPHERE_FITTING_FOLDER = folder_data["SPHERE_fitting_folder"]
SPHERE_DATASET_FOLDER = folder_data["SPHERE_dataset_folder"]
SPHERE_DATA_FOLDER = folder_data["SPHERE_data_folder"]
MAX_NDIT = folder_data["Max NDITs"]
