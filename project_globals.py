import json
import torch
import pathlib
from pathlib import Path

path = pathlib.Path(__file__).parent.resolve()

with open(path / Path("data/global_config.json"), "r") as f:
    folder_data = json.load(f)

SPHERE_FITTING_FOLDER = folder_data["SPHERE_fitting_folder"]
SPHERE_DATA_FOLDER = folder_data["SPHERE_data_folder"]
WEIGHTS_FOLDER = path / Path("data/weights")
DATA_FOLDER = path / Path("data/")
MAX_NDIT = folder_data["Max NDITs"]
DEVICE = folder_data["device"]

device = torch.device(DEVICE) if torch.cuda.is_available else torch.device('cpu')
# device = torch.device('cpu')

# # Print the folder paths to check if they are assigned correctly
# if __name__ == "__main__":
#     print("Data folder:", DATA_FOLDER)
#     print("Output folder:", OUTPUT_FOLDER)
