# TipTorch

> **⚠️ Work in Progress:** *This project is under active development. APIs, examples, and features may change without notice.*

A PyTorch-based adaptive optics (AO) Point Spread Function (PSF) simulator for astronomical instruments.


## Features

- Differentiable PSF simulation
- Parallel multi-wavelength and multi-source AO modeling
<!-- - Shack-Hartmann and Pyramid wavefront sensor support -->
- GPU-accelerated (CUDA) with optional CuPy integration
- Fast PSF fitting to observational data
- ML-based calibration from integrated AO telemetry

## Currenty supported AO regimes

- **SCAO**
- **LTAO**
- **MCAO**

So far, the code was tested on SPHERE/IRDIS (VLT) and MUSE Narrow-Field Mode (VLT) intruments.

## Installation

### Prerequisites

- Python 3.12+
- (Recommended) NVIDIA GPU with CUDA

### Setup

**Windows (PowerShell):**
```powershell
.\setup_env_Windows.ps1 -EnvName "TipTorch" -PythonVersion "3.12"
```

**Linux / macOS / WSL:**
```bash
./setup_env_Linux_MacOS_WSL.sh --env-name TipTorch
```

The setup scripts create a Conda environment, detect your hardware (CPU vendor, CUDA availability), and install all dependencies.

<!-- ## Project Structure

| Directory | Description |
|---|---|
| `PSF_models/` | Core TipTorch PSF model and instrument-specific wrappers |
| `managers/` | Configuration parsing and input management |
| `fitting/` | PSF parameter fitting pipelines (MUSE, SPHERE) |
| `machine_learning/` | ML calibration and training scripts |
| `data_processing/` | Telemetry and dataset preparation utilities |
| `tools/` | Shared utilities (plotting, normalization, Zernike basis, etc.) |
| `data/parameter_files/` | Instrument configuration files (`.ini`) |
| `tests/` | Examples, unit tests, and profiling scripts | -->

## Quick Start


1. Configure `project_config.json` with your data paths and device:
   ```json
   {
       "device": "cuda:0",
       "project_data_folder": "./data/"
   }
   ```

2. Run an example script:
> ⚠️ *Example usage is still under construction.*

   <!-- ```bash
   python tests/MUSE_onsky.py
   ``` -->


## License

This project is licensed under the [MIT License](LICENSE).

## Configuration

The basic instrument and telescope parameters are defined in `.ini` files under `data/parameter_files/`. The configs follow exactly the same structure as ones used in [astro-TipTop](https://github.com/astro-tiptop/TIPTOP).