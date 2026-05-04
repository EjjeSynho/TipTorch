# TipTorch

> **⚠️ Work in Progress:** *This project is under active development. APIs, examples, and features may change without notice.*

A PyTorch-based adaptive optics (AO) Point Spread Function (PSF) simulator for astronomical instruments.


## Features

- Differentiable PSF simulation
- Parallel multi-wavelength and multi-source AO modeling
- GPU-accelerated (CUDA) with optional CuPy integration
- Fast PSF fitting to observational data
- ML-based calibration from integrated AO telemetry

## Currently supported AO regimes

- **SCAO**
- **LTAO**
- **MCAO**

So far, the code was tested on SPHERE/IRDIS (VLT) and MUSE Narrow-Field Mode (VLT) instruments.


## Installation

### Prerequisites

- Python 3.10+
- (Recommended) NVIDIA GPU with CUDA

### Option 1 — Full development environment (recommended)

The setup scripts create a Conda environment, detect your hardware (CPU vendor, CUDA availability), install all dependencies, and install `tiptorch` itself in editable mode.

**Windows (PowerShell):**
```powershell
.\setup_env_Windows.ps1 -EnvName "TipTorch" -PythonVersion "3.12"
```

**Linux / macOS / WSL:**
```bash
./setup_env_Linux_MacOS_WSL.sh --env-name TipTorch
```

### Option 2 — Install the package only

If you already have a Python environment with PyTorch, you can install `tiptorch` directly:

```bash
# Editable (development) install — changes to src/ are picked up immediately
pip install -e .

# Or a regular install
pip install .
```


## Project Structure

```
src/tiptorch/            # Installable package (core library)
├── PSF_models/          #   PSF engine and instrument wrappers (TipTorch, IRDIS, MUSE NFM)
├── managers/            #   Configuration parsing, input management, resource sync
├── tools/               #   Utilities, normalizers, Zernike/static-phase bases, cubic splines
├── _config.py           #   Project-wide settings, device selection, paths
└── _resources/          #   Bundled defaults (config template, registry, required fields)

fitting/                 # PSF parameter fitting pipelines (MUSE, SPHERE)
machine_learning/        # ML calibration and training scripts
data_processing/         # Telemetry and dataset preparation utilities
tools/                   # Non-distributed utilities (plotting, multi-source helpers)
tests/                   # Examples, unit tests, and profiling scripts
```


## Configuration

### Cache folder (`~/.tiptorch`)

On first import, TipTorch creates a cache directory at `~/.tiptorch` (or the path set by the `TIPTORCH_CACHE` environment variable). This folder stores:

- **`project_config.json`** — your local configuration (auto-generated from a bundled template on first run)
- **Resource packs** — instrument calibrations, parameter files, model weights, etc., fetched from a remote registry

The cache location can be overridden:

```bash
# Linux / macOS
export TIPTORCH_CACHE="/path/to/my/cache"

# Windows PowerShell
$env:TIPTORCH_CACHE = "C:\path\to\my\cache"
```

### Project config (`project_config.json`)

The config file controls device selection and folder layout inside the cache. Default contents:

```json
{
    "device":            "cuda:0",
    "data":              "./",
    "model_weights":     "./weights/",
    "calibrations":      "./calibrations/",
    "reduced_telemetry": "./reduced_telemetry/",
    "parameter_files":   "./parameter_files/",
    "resource_packs":    "./resource_packs/",
    "temp_folder":       "./temp/",
    "registry_url":      "https://drive.google.com/file/d/..."
}
```

All relative paths are resolved against the cache folder. You can edit the file directly, or update it programmatically:

```python
from tiptorch._config import update_config

update_config({"device": "cpu"})
```

### Resource synchronization

Required data files (instrument calibrations, PSF parameter files, model weights, etc.) are **not** bundled with the package. Instead, they are downloaded lazily from a remote registry the first time they are needed.

- `ensure_resources()` triggers the sync explicitly.
- It is also called automatically on the first model instantiation, so no manual action is normally required.
- Downloaded files are cached locally in `~/.tiptorch/resource_packs/` and are only re-downloaded when the remote registry indicates an update.

### Instrument / telescope parameters

The basic instrument and telescope parameters are defined in `.ini` files under `parameter_files/` inside the cache. These configs follow the same structure as the ones used in [astro-TipTop](https://github.com/astro-tiptop/TIPTOP).


## Quick Start

```python
import tiptorch
from tiptorch.PSF_models.IRDIS_wrapper import PSFModelIRDIS
from tiptorch.managers.config_manager import ConfigManager
```

> ⚠️ *Full usage examples are still under construction.*


## License

This project is licensed under the [MIT License](LICENSE).