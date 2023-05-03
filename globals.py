import torch
import pathlib
from pathlib import Path
import json

path = pathlib.Path(__file__).parent.resolve()

with open(path/Path("data/global_config.json"), "r") as f:
    folder_data = json.load(f)

SPHERE_DATA_FOLDER = folder_data["SPHERE_data_folder"]
SPHERE_FITTING_FOLDER = folder_data["SPHERE_fitting_folder"]

MAX_NDIT = folder_data["Max NDITs"]

device = torch.device('cuda') if torch.cuda.is_available else torch.device('cpu')

# # Print the folder paths to check if they are assigned correctly
# if __name__ == "__main__":
#     print("Data folder:", DATA_FOLDER)
#     print("Output folder:", OUTPUT_FOLDER)
