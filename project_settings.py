#%%
from logging import warning
import os, sys
import json
import torch
from pathlib import Path
from tools.utils import DownloadFromRemote
import warnings
import platform
# warnings.filterwarnings("ignore", category=UserWarning)

PROJECT_PATH = Path(__file__).parent.resolve()

# Adding project folders to the system path so that the modules can be imported
sys.path.insert(0, str(PROJECT_PATH) )
sys.path.insert(0, str(PROJECT_PATH / 'tools/') )
sys.path.insert(0, str(PROJECT_PATH / 'managers/') )
sys.path.insert(0, str(PROJECT_PATH / 'data_processing/') )

# Downloading default project settings if they are not found locally
if not os.path.exists(path_to_config := Path(PROJECT_PATH) / Path("project_config.json")):
    DownloadFromRemote(
        share_url   = 'https://drive.google.com/file/d/1VJbqGtxISYzRlirHfe-dS4Urx3u7wYO2/view?usp=sharing',
        output_path = os.path.normpath(path_to_config).replace('\\','/'),
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


def resolve_device(preferred: str) -> torch.device:
    """
    Resolve a torch.device by:
      1. Trying the user-preferred device string.
      2. Falling back to CUDA if available.
      3. On macOS, falling back to MPS if available.
      4. Otherwise, using CPU.
    """
    preferred = preferred.lower()
    # 1) Try the config value if it makes sense
    if preferred != "cpu":
        # a) CUDA case
        if "cuda" in preferred and torch.cuda.is_available():
            return torch.device(preferred)
        # b) MPS case (macOS)
        if preferred in ("mps", "metal") and platform.system() == "Darwin":
            if getattr(torch.backends.mps, "is_available", lambda: False)():
                return torch.device("mps")

    # 2) Config choice didn’t work → pick best on this machine
    if torch.cuda.is_available():
        return torch.device("cuda")
    
    if platform.system() == "Darwin" and getattr(torch.backends.mps, "is_available", lambda: False)():
        return torch.device("mps")

    # 3) Give up and fall back to CPU
    return torch.device("cpu")

device = resolve_device(project_settings["device"])

# Check if GPU is available and has sufficient VRAM to use CuPy
#TODO: do the same memory check for PyTorch
try:
    import cupy as xp
    # Check if GPU is available and has sufficient VRAM
    device_info = xp.cuda.runtime.getDeviceProperties(0) # Get device information
    total_memory = device_info["totalGlobalMem"] / (1024**3) # Total memory in bytes, convert to GB
    # mempool = xp.get_default_memory_pool() 
    # used_memory = mempool.used_bytes() / (1024**3) # Get current memory usage
    available_memory = total_memory #- used_memory   # Calculate available memory TODO: fix it

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
