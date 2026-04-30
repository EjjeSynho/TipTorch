#%%
import os, sys
import json
from turtle import up
import torch
import numpy as np
from pathlib import Path

import warnings
import platform

SOURCE_PATH = Path(__file__).parent.resolve()

# Adding project folders to the system path so that the modules can be imported
for p in ['.', 'tools', 'managers', 'data_processing']:
    sys.path.insert(0, str(SOURCE_PATH / p))

#%%
# =============== Manage the project cache folder ================
# Resolve the user cache / config directory from env var, or default to ~/.tiptorch
_cache_env = os.environ.get("TIPTORCH_CACHE")
if _cache_env:
    CACHE_PATH = Path(_cache_env).expanduser().resolve()
else:
    CACHE_PATH = Path("~/.tiptorch").expanduser().resolve()

# Create the project directory if it doesn't exist
if not CACHE_PATH.exists():
    warnings.warn(f"TipTorch cache folder not found. Creating a new one at: {CACHE_PATH}. If you want to specify another folder, please create a TIPTORCH_CACHE environment variable pointing to your desired location.")
    CACHE_PATH.mkdir(parents=True, exist_ok=True)

# Generate a default project config if not found
path_to_config = CACHE_PATH / "project_config.json"

if not path_to_config.exists():
    default_config = {
        "device":            "cuda:0",
        "data":              "./",
        "model_weights":     "./weights/",
        "calibrations":      "./calibrations/",
        "reduced_telemetry": "./reduced_telemetry/",
        "parameter_files":   "./parameter_files/",
        "resource_packs":    "./resource_packs/",
        "temp_folder":       "./temp/",
        "registry_url":      "https://drive.google.com/file/d/15xUS4pSqS1Ppm8Gp22sRue4NGaNf1Br4/view?usp=drive_link"
    }
    with open(path_to_config, "w") as f:
        json.dump(default_config, f, indent=4)
    
    warnings.warn(f"Config with default settings created at: {path_to_config}.")


# Load project-wide folders settings
with open(path_to_config, "r") as f:
    project_settings = json.load(f)


def update_config(updates: dict) -> None:
    """
    Merge *updates* into the on-disk project config and reload ``project_settings``.

    Only the keys present in *updates* are changed; all other keys are preserved.
    Raises ``KeyError`` if any key in *updates* is not already in the config.
    """
    with open(path_to_config, "r") as f:
        config = json.load(f)

    unknown = set(updates) - set(config)
    if unknown:
        raise KeyError(f"Unknown config key(s): {unknown}. Valid keys: {set(config)}")

    config.update(updates)

    with open(path_to_config, "w") as f:
        json.dump(config, f, indent=4)

    project_settings.update(config)


# =========================== Project-wide settings and paths ===========================
# Make sure paths are absolute when they're relative
def _resolve_path(raw: str, base: Path) -> Path:
    p = Path(raw)
    return (base / p).resolve() if not p.is_absolute() else p.resolve()

# Model weights folder
WEIGHTS_FOLDER  = _resolve_path(project_settings["model_weights"], CACHE_PATH)
# Where data is stored
DATA_FOLDER     = _resolve_path(project_settings["data"],  CACHE_PATH)
# Folder where all reduced telemetry necessary for model calibration is stored
TELEMETRY_CACHE = _resolve_path(project_settings["reduced_telemetry"], CACHE_PATH)
# Folder to store instrument-specific calibrations
CALIBRATIONS_PATH = _resolve_path(project_settings["calibrations"], CACHE_PATH)
# Folder where parameter files for different science instruments are stored
PARAMETER_FILES_PATH = _resolve_path(project_settings["parameter_files"], CACHE_PATH)
# Folder where resource packs are stored. Resource packs tell which files to udownload from remote and where to put them
RESOURCE_PACKS_DIR = _resolve_path(project_settings["resource_packs"], CACHE_PATH)

TEMP_DIR = _resolve_path(project_settings["temp_folder"], CACHE_PATH)

# GDrive URL or file ID for the remote registry.json
REGISTRY_URL = project_settings.get("registry_url", "")

# =============== Device and default floating-point type settings ================
# Default floating-point data type used across the project
default_torch_type = torch.float32

def resolve_device(preferred: str) -> torch.device:
    global default_torch_type
    preferred = preferred.lower()

    if "cuda" in preferred:
        if torch.cuda.is_available():
            index = torch.device(preferred).index or 0
            if index < torch.cuda.device_count():
                return torch.device('cuda:0')
            warnings.warn(f"CUDA device '{preferred}' not found ({torch.cuda.device_count()} available). Falling back.")
        if torch.cuda.is_available():
            return torch.device("cuda")

    # TODO: implement MPS support; for now fall through to CPU
    if platform.system() == "Darwin" and getattr(torch.backends.mps, "is_available", lambda: False)():
        torch.set_default_dtype(torch.float32)
        default_torch_type = torch.float32
        warnings.warn("MPS (Apple Silicon GPU) support is not yet implemented. Using CPU.")

    return torch.device("cpu")


default_device = resolve_device(project_settings["device"])
project_settings["device"] = str(default_device) # Update the config with the resolved device for consistency

# =============== GPU/Backend detection and setup ================
# Check if GPU is available and has sufficient VRAM to use CuPy
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
        xp = np  # np is imported at the top of the file

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

# Update the project settings with the resolved device and backend information
update_config(project_settings)

# =============== Download resource packs if necessary, synchronize them with remote registry ================
# Late import to avoid circular dependency
from managers.resources_manager import sync_resource_packs
sync_resource_packs()

# %%
