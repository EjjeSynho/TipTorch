#%%
from logging import warning
import os
import json
import torch
from pathlib import Path
from tools.utils import DownloadFromRemote
import warnings
# warnings.filterwarnings("ignore", category=UserWarning)

PROJECT_PATH = Path(__file__).parent.resolve()

# Downloading default project settings if they are not found locally
if not os.path.exists(path_to_config := Path(PROJECT_PATH) / Path("project_config.json")):
    DownloadFromRemote(
        share_url   = 'https://drive.google.com/file/d/1VJbqGtxISYzRlirHfe-dS4Urx3u7wYO2/view?usp=sharing',
        output_path = path_to_config,
        overwrite   = False,
        verbose     = False
    )

with open(PROJECT_PATH / Path("project_config.json"), "r") as f:
    project_settings = json.load(f)

# Load project-wide folders settings
WEIGHTS_FOLDER = PROJECT_PATH / Path(project_settings["model_weights_folder"])
DATA_FOLDER    = PROJECT_PATH / Path(project_settings["project_data_folder"])

# Load instrument specific folders settings
MUSE_DATA_FOLDER   = Path(project_settings["MUSE_data_folder"])
SPHERE_DATA_FOLDER = Path(project_settings["SPHERE_data_folder"])
LIFT_PATH          = Path(project_settings["LIFT_path"])

# Set up the device used by PyTorch in the project
DEVICE = project_settings["device"]
device = torch.device(DEVICE) if torch.cuda.is_available else torch.device('cpu')

# Check if GPU is available and has sufficient VRAM to use CuPy
#TODO: do the same for PyTorch
try:
    import cupy as xp
    # Check if GPU is available and has sufficient VRAM
    device_info = xp.cuda.runtime.getDeviceProperties(0) # Get device information
    total_memory = device_info["totalGlobalMem"] / (1024**3) # Total memory in bytes, convert to GB
    # mempool = xp.get_default_memory_pool() 
    # used_memory = mempool.used_bytes() / (1024**3) # Get current memory usage
    available_memory = total_memory #- used_memory   # Calculate available memory

    use_cupy = available_memory > 2.0  # Set GPU flag based on available memory

    if not use_cupy:
        warnings.warn(f"GPU detected but only {available_memory:.2f}GB VRAM available (< 2GB required). Using CPU backend instead.")
        xp = np

# If cupy is not available, use numpy instead
except ImportError:
    import numpy as np
    xp = np
    use_cupy = False
    warnings.warn("No GPU or CuPy installation detected. Using CPU/NumPy instead.")

# Handle other errors that might occur during GPU detection
except Exception as e:
    import numpy as np
    xp = np
    use_cupy = False
    warnings.warn(f"Error during CuPy/GPU detection or initialization:\n {str(e)}. Using NumPy instead.")
