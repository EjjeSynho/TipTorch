#!/usr/bin/env bash
set -euo pipefail

# ─── Configuration ──────────────────────────────────────────────────────────
USE_MAMBA=true
ENV_NAME="TipTorch"
PYTHON_VERSION="3.12"
DEVELOPMENT=false

function usage() {
  echo "Usage: $0 [--no-mamba] [--env-name NAME] [--python-version VERSION] [--development] [-h|--help]"
  echo
  echo "Options:"
  echo "  --no-mamba             Skip installing and using mamba; force use of conda"
  echo "  --env-name NAME        Set environment name (default: TipTorch)"
  echo "  --python-version VER   Set Python version (default: 3.12)"
  echo "  --development          Include development packages (Jupyter, etc.)"
  echo "  -h, --help             Show this help message and exit"
  echo
  echo "Example: $0 --env-name TipTorch_dev --python-version 3.11 --development"
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-mamba)
      USE_MAMBA=false
      shift
      ;;
    --env-name)
      ENV_NAME="$2"
      shift 2
      ;;
    --python-version)
      PYTHON_VERSION="$2"
      shift 2
      ;;
    --development)
      DEVELOPMENT=true
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

echo "→ Configuration: ENV_NAME=$ENV_NAME, PYTHON_VERSION=$PYTHON_VERSION, DEVELOPMENT=$DEVELOPMENT, USE_MAMBA=$USE_MAMBA"

# ─── Abort on Windows ────────────────────────────────────────────────────────
UNAME="$(uname | tr '[:upper:]' '[:lower:]')"
if [[ "$UNAME" == msys* || "$UNAME" == mingw* || "$UNAME" == cygwin* ]]; then
  echo "⚠️  This setup script does not support Windows. Please use the PowerShell installer instead."
  exit 1
fi

# ─── Install mamba if not present (recommended for speed) ──────────────────
if ! command -v mamba &>/dev/null; then
  if [[ "$USE_MAMBA" == "true" ]]; then
    echo "→ mamba not found. Installing into base for faster solving…"
    conda install -n base mamba -c conda-forge -y
    if [[ $? -eq 0 ]]; then
      echo "→ mamba installed successfully."
    else
      echo "⚠️  Failed to install mamba. Continuing with conda."
    fi
  else
    echo "→ mamba not found, but installation skipped per user request."
  fi
else
  echo "→ mamba is already installed."
fi

# ─── Pick the solver ─────────────────────────────────────────────────────────
if [[ "$USE_MAMBA" == "true" && $(command -v mamba &>/dev/null; echo $?) -eq 0 ]]; then
  SOLVER="mamba"
else
  SOLVER="conda"
fi
echo "→ Using solver: $SOLVER"

# ─── 1) Detect CPU vendor & platform ───────────────────────────────────────
if [[ "$UNAME" == linux* ]]; then
  PLATFORM="linux"
elif [[ "$UNAME" == darwin* ]]; then
  PLATFORM="osx"
else
  PLATFORM="linux"
fi

# CPU vendor detection for Intel optimizations
INTEL_CPU=false
if [[ -f /proc/cpuinfo ]]; then
  if grep -q "vendor_id.*GenuineIntel" /proc/cpuinfo; then
    INTEL_CPU=true
  fi
elif [[ "$PLATFORM" == "osx" ]]; then
  CPU_BRAND=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "")
  if [[ "$CPU_BRAND" == *"Intel"* ]]; then
    INTEL_CPU=true
  fi
fi

# ─── 2) Enhanced CUDA detection ─────────────────────────────────────────────
CUDA_AVAILABLE=false
CUDA_VERSION="Unknown"
if command -v nvidia-smi &>/dev/null; then
  NVIDIA_SMI_OUTPUT=$(nvidia-smi --query-gpu=driver_version,cuda_version --format=csv,noheader,nounits 2>/dev/null || echo "")
  if [[ -n "$NVIDIA_SMI_OUTPUT" && "$NVIDIA_SMI_OUTPUT" != *"No devices were found"* ]]; then
    CUDA_AVAILABLE=true
    CUDA_VERSION=$(echo "$NVIDIA_SMI_OUTPUT" | cut -d',' -f2 | tr -d ' ')
    echo "→ CUDA detected: version $CUDA_VERSION"
  else
    echo "→ nvidia-smi found but no devices detected or command failed."
  fi
fi

# Additional check for CUDA toolkit
if [[ "$CUDA_AVAILABLE" == "false" && -f "/usr/local/cuda/bin/nvcc" ]]; then
  CUDA_AVAILABLE=true
  echo "→ CUDA toolkit found via /usr/local/cuda"
fi

echo "→ Detected: PLATFORM=$PLATFORM, INTEL_CPU=$INTEL_CPU, CUDA_AVAILABLE=$CUDA_AVAILABLE"

# ─── 3) Build environment.yml ───────────────────────────────────────────────
ENV_YML="environment.yml"
cat > "$ENV_YML" <<EOF
name: $ENV_NAME
channels:
  - nodefaults
EOF

# Add Intel channel first (higher priority) if Intel CPU is detected
if [[ "$INTEL_CPU" == "true" ]]; then
  cat >> "$ENV_YML" <<EOF
  - https://software.repos.intel.com/python/conda
EOF
  echo "→ Intel CPU detected, adding Intel Python channel with priority"
fi

cat >> "$ENV_YML" <<EOF
  - conda-forge

dependencies:
  - python=$PYTHON_VERSION
  - numpy
  - scipy
  - scikit-learn
EOF

# Add Intel MKL packages explicitly for Intel systems to avoid pip conflicts
if [[ "$INTEL_CPU" == "true" ]]; then
  cat >> "$ENV_YML" <<EOF
  - mkl
  - mkl-service
  - mkl_fft
  - mkl_random
EOF
  echo "→ Adding explicit Intel MKL packages to prevent pip conflicts"
fi

# Common packages
cat >> "$ENV_YML" <<EOF
  - pandas
  - matplotlib
  - seaborn
  - plotly
  - scikit-image
  - tabulate
  - tqdm
  - gdown
  - pooch
  - astropy
  - astroquery
  - photutils
  - imageio
  - gwcs
  - asdf
  - asdf-astropy
  - mpl-scatter-density
  - ipykernel
EOF

# Development packages (only if Development flag is set)
if [[ "$DEVELOPMENT" == "true" ]]; then
  cat >> "$ENV_YML" <<EOF
  - jupyterlab
  - torchmetrics
  - pytorch-model-summary
  - sympy
  - imbalanced-learn
  - shap
  - networkx
  - statsmodels
  - invoke
  - nbconvert
EOF
  echo "→ Development packages will be included"
else
  echo "→ Skipping development packages. Use --development flag to include them."
fi

# Pip packages
cat >> "$ENV_YML" <<EOF
  - pip
  - pip:
    - pillow
    - xgboost
EOF

if [[ "$DEVELOPMENT" == "true" ]]; then
  cat >> "$ENV_YML" <<EOF
    - elasticsearch==6.8.2
    - elasticsearch-dsl==6.4.0
EOF
fi

if [[ "$INTEL_CPU" == "true" ]]; then
  cat >> "$ENV_YML" <<EOF
    - mkl
EOF
  echo "→ Adding Intel MKL via pip to satisfy pip dependency resolver"
fi

echo "→ Generated $ENV_YML"

if [[ "$INTEL_CPU" == "false" ]]; then
  echo "→ Non-Intel CPU detected, skipping Intel channel"
fi

# ─── 4) Create the environment ───────────────────────────────────────────────
echo
echo "▶ Creating environment '$ENV_NAME' with $SOLVER (this may take a while)…"
$SOLVER env create -f "$ENV_YML" -v
echo "✅ Environment '$ENV_NAME' created."

# ─── 5) Install PyTorch via pip with CUDA support ──────────────────────────
if [[ "$CUDA_AVAILABLE" == "true" ]]; then
  echo
  echo "→ Installing PyTorch with CUDA 12.8 support…"
  echo "→ Using CUDA index: https://download.pytorch.org/whl/cu128"
  conda run -n "$ENV_NAME" pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
  
  # Verify CUDA installation
  echo "→ Verifying PyTorch CUDA installation…"
  CUDA_CHECK=$(conda run -n "$ENV_NAME" python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'Device count: {torch.cuda.device_count()}')" 2>&1)
  if echo "$CUDA_CHECK" | grep -q "CUDA available: True"; then
    echo "✅ PyTorch CUDA verification successful."
    echo "   $CUDA_CHECK"
  else
    echo "⚠️  PyTorch installed but CUDA not available. Output:"
    echo "   $CUDA_CHECK"
  fi
else
  echo
  echo "→ Installing PyTorch CPU-only version…"
  conda run -n "$ENV_NAME" pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

if [[ $? -eq 0 ]]; then
  echo "✅ Successfully installed PyTorch."
else
  echo "❌ Failed to install PyTorch."
  exit 1
fi

# ─── 5.5) Install CUDA development toolkit and CuPy for CUDA systems ──────
echo
if [[ "$CUDA_AVAILABLE" == "true" ]]; then
  echo "→ Installing CUDA development headers for CuPy…"
  conda run -n "$ENV_NAME" pip install cuda-python
  
  echo "→ Installing CuPy for CUDA 12.x…"
  conda run -n "$ENV_NAME" pip install cupy-cuda12x
  
  if [[ $? -eq 0 ]]; then
    echo "✅ Successfully installed CuPy."
  else
    echo "⚠️ Failed to install CuPy, but continuing…"
  fi
else
  echo "→ CUDA not available, skipping CuPy installation"
fi

# ─── 6) Install pytorch-minimize, prefer PyPI then fallback to GitHub ────────
echo
echo "→ Attempting to install 'pytorch-minimize' from PyPI…"
conda run -n "$ENV_NAME" pip install pytorch-minimize

if [[ $? -eq 0 ]]; then
  echo "✅ Successfully installed 'pytorch-minimize' from PyPI."
else
  echo "⚠️  'pytorch-minimize' not found on PyPI or installation failed. Falling back to GitHub…"

  if [[ ! -d "../pytorch-minimize" ]]; then
    echo "→ Cloning pytorch-minimize to ../pytorch-minimize…"
    git clone https://github.com/rfeinman/pytorch-minimize.git ../pytorch-minimize
  else
    echo "→ Found existing pytorch-minimize at ../pytorch-minimize"
  fi
  
  echo "→ Installing 'pytorch-minimize' from local Git repository at ../pytorch-minimize…"
  conda run -n "$ENV_NAME" pip install -e ../pytorch-minimize

  if [[ $? -eq 0 ]]; then
    echo "✅ Successfully installed 'pytorch-minimize' from GitHub."
  else
    echo "❌ Failed to install 'pytorch-minimize' from GitHub."
    exit 1
  fi
fi

# ─── 7) Install tiptorch package in editable mode ────────────────────────────
echo
echo "→ Installing tiptorch package in editable (development) mode…"
conda run -n "$ENV_NAME" pip install -e .

if [[ $? -eq 0 ]]; then
  echo "✅ Successfully installed tiptorch in editable mode."
else
  echo "❌ Failed to install tiptorch."
  exit 1
fi

echo
echo "🎉 Environment setup complete!"
echo "   To start using it, run: conda activate $ENV_NAME"
