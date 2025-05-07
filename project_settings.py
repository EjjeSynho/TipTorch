#%%
import json
import torch
from pathlib import Path

path = Path(__file__).parent.resolve()
PROJECT_PATH = path

with open(path / Path("project_config.json"), "r") as f:
    project_settings = json.load(f)

WEIGHTS_FOLDER = PROJECT_PATH / Path(project_settings["model_weights_folder"])
DATA_FOLDER    = PROJECT_PATH / Path(project_settings["project_data_folder"])
DEVICE         = project_settings["device"]

device = torch.device(DEVICE) if torch.cuda.is_available else torch.device('cpu')
