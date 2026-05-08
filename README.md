# TipTorch

> **⚠️ Work in Progress:** *This project is under active development. APIs, examples, and features may change without notice.*

A PyTorch-based adaptive optics (AO) Point Spread Function (PSF) simulator for astronomical instruments.


### Features

- Differentiable PSF simulation
- Parallel multi-wavelength and multi-source AO modeling
- GPU-accelerated (CUDA) with optional CuPy integration
- Fast PSF fitting to observational data
- ML-based calibration from integrated AO telemetry
- Currently supports SCAO, LTAO, MCAO

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

> **Note:** For development purposes, install in editable mode: `pip install -e .`

### Option 2 — Install the code (recommended)

The setup scripts automatically creates an optimal Conda environment, which provides the best performance and compatibility. To do so:

**Linux / macOS / WSL:**
```bash
./setup_env_Linux_MacOS_WSL.sh
```

**Windows (PowerShell):**
```powershell
.\setup_env_Windows.ps1
```

<details>
<summary><strong>(Advanced) Customizable setup options</strong></summary>

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

</details>

<details>
<summary><strong>(Advanced) Acceleration for AMD CPUs (Linux only)</strong></summary>

On AMD Linux systems, use the `--amd-aocl` flag to link NumPy/SciPy against AMD's optimized AOCL BLAS/LAPACK:

```bash
./setup_env_Linux_MacOS_WSL.sh --amd-aocl auto
```

For more control, manually [download wheels from AMD](https://www.amd.com/en/developer/zen-software-studio/applications/python-libraries-with-aocl.html), place in `./amd_aocl_wheels/`, and then run:

```bash
./setup_env_Linux_MacOS_WSL.sh --amd-aocl amd-wheels --amd-wheel-cache ./amd_aocl_wheels
```

</details>


## Configuration

On first import, TipTorch creates a directory at `~/.tiptorch`, which stores:

- **`project_config.json`** — global TipTorch configuration file
- **Resource packs** — parameter files, instrument calibrations, model weights, etc.


To override the cache location, set the `TIPTORCH_CACHE` environment variable:

```bash
# Linux / macOS
export TIPTORCH_CACHE="/path/to/my/cache"

# Windows PowerShell
$env:TIPTORCH_CACHE = "C:\path\to\my\cache"
```

The `project_config.json` file controls preferred computation device (e.g., `'cpu'`, `'cuda:0'`, `'cuda:1'`) and folder layout. Use this file to define any custom project-wide configurations. By default:

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

> **Note:** All relative paths are resolved against the cache folder

### Resource synchronization

Required data files (instrument calibrations, PSF parameter files, model weights, etc.) are downloaded automatically from a remote registry the first time they are needed. Downloaded files are cached locally in `~/.tiptorch/resource_packs/` and are only re-downloaded when the remote registry indicates an update.

### Instrument / telescope parameters

The basic instrument and telescope parameters are defined in `.ini` files under `parameter_files/` inside the cache. These configs follow the same structure as the ones used in [astro-TipTop](https://github.com/astro-tiptop/TIPTOP).


<!--
## Quick Start

```python
import tiptorch
from tiptorch.PSF_models.IRDIS_wrapper import PSFModelIRDIS
from tiptorch.managers.config_manager import ConfigManager
```
-->

> ⚠️ *Full usage examples are still under construction.*

<!--
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
-->

## License

This project is licensed under the [MIT License](LICENSE).