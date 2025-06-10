#!/usr/bin/env bash
set -euo pipefail

# ─── Usage / Argument Parsing ───────────────────────────────────────────────
USE_MAMBA=true

function usage() {
  echo "Usage: $0 [--no-mamba] [-h|--help]"
  echo
  echo "Options:"
  echo "  --no-mamba    Skip installing and using mamba; force use of conda"
  echo "  -h, --help    Show this help message and exit"
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-mamba)
      USE_MAMBA=false
      shift
      ;;
    -h|--help)
      usage
      ;;
    *)
      echo "Unknown option: $1"
      usage
      ;;
  esac
done

echo "→ USE_MAMBA set to: $USE_MAMBA"

# ─── Abort on Windows ────────────────────────────────────────────────────────
UNAME="$(uname | tr '[:upper:]' '[:lower:]')"
if [[ "$UNAME" == msys* || "$UNAME" == mingw* || "$UNAME" == cygwin* ]]; then
  echo "⚠️  This setup script does not support Windows. Please use the PowerShell installer instead."
  exit 1
fi

# ─── Install mamba if missing (unless skipped) ────────────────────────────────
if [[ "$USE_MAMBA" == "true" ]]; then
  if ! command -v mamba &>/dev/null; then
    echo "→ mamba not found in PATH. Installing into base…"
    conda install -n base mamba -c conda-forge -y
    if [[ $? -eq 0 ]]; then
      echo "→ mamba installed successfully."
    else
      echo "⚠️  Failed to install mamba. Continuing with conda."
    fi
  else
    echo "→ mamba is already installed."
  fi
else
  echo "→ Skipping mamba installation. If you want mamba, re-run with --no-mamba omitted."
fi

# ─── Pick the solver ─────────────────────────────────────────────────────────
if [[ "$USE_MAMBA" == "true" && $(command -v mamba &>/dev/null; echo $?) -eq 0 ]]; then
  SOLVER="mamba"
else
  SOLVER="conda"
fi
echo "→ Using solver: $SOLVER"

# ─── 1) Platform detection ───────────────────────────────────────────────────
# We already know it's not Windows; keep PLATFORM for record.
if [[ "$UNAME" == linux* ]]; then
  PLATFORM="linux"
elif [[ "$UNAME" == darwin* ]]; then
  PLATFORM="osx"
else
  PLATFORM="linux"
fi

# ─── 2) CUDA availability ───────────────────────────────────────────────────
CUDA_AVAILABLE=false
if command -v nvidia-smi &>/dev/null; then
  if nvidia-smi &>/dev/null; then
    CUDA_AVAILABLE=true
  else
    echo "→ nvidia-smi found but failed to execute properly."
  fi
fi
echo "→ Detected: PLATFORM=$PLATFORM, CUDA_AVAILABLE=$CUDA_AVAILABLE"

# ─── 3) Build environment.yml ───────────────────────────────────────────────
ENV_NAME="TipTorch_test"
ENV_YML="environment.yml"
cat > "$ENV_YML" <<EOF
name: $ENV_NAME
channels:
  - nodefaults
  - conda-forge
  - astropy
  - pytorch
  - nvidia

dependencies:
  - python=3.11
  - numpy>=1.26
  - scipy
  - scikit-learn
  - pillow
  - pandas
  - matplotlib
  - seaborn
  - scikit-image
  - tabulate
  - tqdm
  - gdown
  - astropy
  - astroquery
  - photutils
  - pip
EOF

if [[ "$CUDA_AVAILABLE" == "true" ]]; then
  cat >> "$ENV_YML" <<EOF
  - pytorch=2.4.1
  - torchvision
  - torchaudio=2.4.1
  - pytorch-cuda=12.1
EOF
else
  cat >> "$ENV_YML" <<EOF
  - pytorch-cpu=2.4.1
  - torchvision
  - torchaudio=2.4.1
EOF
fi

echo "→ Generated $ENV_YML"

# ─── 4) Create the environment ───────────────────────────────────────────────
echo
echo "▶ Creating environment '$ENV_NAME' with $SOLVER (this might take a while)…"
$SOLVER env create -f "$ENV_YML" -v
echo "✅ Environment '$ENV_NAME' created."
echo "   To start using it, run: conda activate $ENV_NAME"

# ─── 5) Install pytorch-minimize, prefer PyPI then fallback to GitHub ────────
echo
echo "→ Installing 'pytorch-minimize' into '$ENV_NAME'…"
# Attempt to install from PyPI
conda run -n "$ENV_NAME" pip install pytorch-minimize

if [[ $? -eq 0 ]]; then
  echo "✅ Successfully installed 'pytorch-minimize' from PyPI."
else
  echo "⚠️  'pytorch-minimize' not found on PyPI or installation failed. Falling back to GitHub…"

  if [[ ! -d "./pytorch-minimize" ]]; then
    git clone https://github.com/rfeinman/pytorch-minimize.git
  fi

  echo "→ Installing 'pytorch-minimize' from local Git repository…"
  conda run -n "$ENV_NAME" pip install -e ./pytorch-minimize

  if [[ $? -eq 0 ]]; then
    echo "✅ Successfully installed 'pytorch-minimize' from GitHub."
  else
    echo "❌ Failed to install 'pytorch-minimize' from GitHub."
    exit 1
  fi
fi
