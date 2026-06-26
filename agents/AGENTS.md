# Repository Guidelines

## Project Structure & Module Organization

TipTorch is an installable Python package under `src/tiptorch/`. Core PSF models live in `src/tiptorch/PSF_models/`, resource/config managers in `src/tiptorch/managers/`, shared numerical helpers in `src/tiptorch/tools/`, and bundled defaults in `src/tiptorch/_resources/`. Research and pipeline scripts are kept outside the package: `data_processing/`, `fitting/`, `machine_learning/`, `datasets/`, and `development/`. `tests/` currently contains exploratory validation and profiling scripts rather than a fully isolated test suite. The `data` path points to the local TipTorch cache and should not be committed. 

## Build, Test, and Development Commands

- `pip install -e .` installs the package in editable mode after PyTorch is available.
- `pip install -e ".[dev]"` adds plotting, ML, image-processing, and build tools used by local scripts.
- `.\setup_env_Windows.ps1 -Development` creates a Conda development environment on Windows.
- `./setup_env_Linux_MacOS_WSL.sh --development` creates the equivalent environment on Linux, macOS, or WSL.
- `python -m build` builds source and wheel artifacts into `dist/`.
- `python tests/TipTorch_profiler.py` runs the profiler script when required data and GPU settings are available.

## Coding Style & Naming Conventions

Use Python 3.11+ and follow the existing scientific-code style: 4-space indentation, descriptive snake_case for functions and variables, PascalCase for model/manager classes, and uppercase names for constants. Keep tensor device and dtype explicit when touching PyTorch paths, usually via `default_device` and `default_torch_type`. Prefer `pathlib.Path` for file paths and keep package code importable from `src/tiptorch` without relying on notebook state.

## Testing Guidelines

Add focused tests or validation scripts near `tests/` for changes to PSF math, config parsing, or resource management. Name new runnable checks with a clear subject, for example `tests/test_config_manager.py` or `tests/PSD_comparison_with_P3.py`. Document required cache files, CUDA assumptions, and expected runtime at the top of heavyweight scripts. Before opening a PR, at minimum import the package and run the relevant script on the target device.

## Commit & Pull Request Guidelines

Recent history uses short, direct summaries such as `Fixed unstacking mapping issue...` or `New validation plots...`. Keep commits imperative or past-tense, scoped to one change, and mention affected subsystem when useful. Pull requests should describe the scientific or engineering change, list commands/scripts run, note data or cache requirements, and include plots or screenshots for visual validation changes.

## Security & Configuration Tips

Do not commit local caches, generated resource packs, model weights, or `project_config.json`. Use `TIPTORCH_CACHE` to redirect local data. Keep credentials and private data out of scripts and notebooks.


# ExecPlans

Use an ExecPlan for multi-step work, multi-file changes, new features, refactors, or tasks likely to take more than about an hour.

Use `.agents/PLANS.md` as the template and rules.

Store concrete ExecPlans in `.agents/execplans/` using filenames like:

    <feature-slug>-execplan.md

Keep Progress, Surprises & Discoveries, Decision Log, and Outcomes & Retrospective updated while working.