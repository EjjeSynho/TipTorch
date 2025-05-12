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
        print(f"GPU detected but only {available_memory:.2f}GB VRAM available (< 2GB required). Using CPU instead.")
        xp = np

# If cupy is not available, use numpy instead
except ImportError:
    import numpy as np
    xp = np
    use_cupy = False
    print("No GPU or CuPy detected. Using CPU and NumPy instead.")

# Handle other errors that might occur during GPU detection
except Exception as e:
    import numpy as np
    xp = np
    use_cupy = False
    print(f"Error during Cupy/GPU detection or initialization: {str(e)}. Using NumPy instead.")
