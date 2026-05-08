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

- Python 3.11+
- (Recommended) NVIDIA GPU with CUDA 12.x

### Option 1 — Install the package with pip

If you already have a Python environment with PyTorch installed, you can add `tiptorch` directly:

```bash
pip install .
```

> **Note:** PyTorch must be installed *before* `tiptorch`. For CUDA support, install PyTorch from the appropriate index first:
> ```bash
> pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
> ```

### Option 2 — Editable (development) install

For development or to modify the source code, install in editable mode so changes under `src/` are picked up immediately:

```bash
pip install -e .
```

### Option 3 — Full managed environment (recommended for first-time setup)

The setup scripts automatically create a Conda environment, detect your hardware (CPU vendor, CUDA availability), install all dependencies with optimized BLAS/LAPACK backends, and install `tiptorch` in editable mode.

**Linux / macOS / WSL:**
```bash
./setup_env_Linux_MacOS_WSL.sh
```

**Windows (PowerShell):**
```powershell
.\setup_env_Windows.ps1
```

Both scripts accept options to customize the environment:

| Flag | Description |
|------|-------------|
| `--env-name` / `-EnvName` | Environment name (default: `TipTorch`) |
| `--python-version` / `-PythonVersion` | Python version (default: `3.12`) |
| `--development` / `-Development` | Include Jupyter, profiling, and ML extras |
| `--no-mamba` | Use plain conda instead of mamba |

Example:
```bash
./setup_env_Linux_MacOS_WSL.sh --env-name TipTorch --python-version 3.12 --development
```

### Option 4 — Build and install as a conda package

```bash
conda build conda/
conda install --use-local tiptorch
```

You can set the version at build time via the `TIPTORCH_VERSION` environment variable:
```bash
TIPTORCH_VERSION=0.1.0 conda build conda/
```

### AMD AOCL acceleration (advanced)

On AMD Linux systems, the setup script can link NumPy/SciPy against AMD's optimized AOCL BLAS/LAPACK libraries. This is controlled by the `--amd-aocl` flag (Linux only):

| Mode | Behavior |
|------|----------|
| `auto` (default) | Uses conda-forge AOCL packages automatically on AMD Linux |
| `conda` | Force conda-forge AOCL BLAS/LAPACK |
| `amd-wheels` | Install official AMD AOCL Python wheels from a local cache |
| `off` | Disable AMD-specific packages entirely |

For the `amd-wheels` mode, you must first manually download the wheels from AMD (they are EULA-gated) and place them in `./amd_aocl_wheels/`:

```bash
./setup_env_Linux_MacOS_WSL.sh --amd-aocl amd-wheels --amd-wheel-cache ./amd_aocl_wheels
```

Required wheels (for Python 3.12):
- `numpy-2.1.3-cp312-cp312-linux_x86_64.whl`
- `scipy-1.14.1-cp312-cp312-linux_x86_64.whl`

Optional:
- `numexpr-2.11.0-cp312-cp312-linux_x86_64.whl`
- `aocl_sparse-0.1.0-cp312-cp312-linux_x86_64.whl`


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