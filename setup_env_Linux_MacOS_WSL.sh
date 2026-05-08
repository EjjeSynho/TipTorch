#!/usr/bin/env bash
set -euo pipefail

# ─── Configuration ──────────────────────────────────────────────────────────
USE_MAMBA=true
ENV_NAME="TipTorch"
PYTHON_VERSION="3.12"
DEVELOPMENT=false

# AMD AOCL mode:
#   auto       = on AMD Linux, use conda-forge AOCL BLAS/LAPACK automatically
#   off        = disable AMD-specific packages
#   conda      = force conda-forge AOCL BLAS/LAPACK on AMD Linux
#   amd-wheels = install official AMD AOCL Python wheels, but only if they are
#                already present in AMD_WHEEL_CACHE. The script does NOT try
#                to bypass AMD's browser/EULA download flow.
AMD_AOCL_MODE="auto"        # auto|off|conda|amd-wheels
AMD_WHEEL_CACHE="${PWD}/amd_aocl_wheels"

function usage() {
  echo "Usage: $0 [--no-mamba] [--env-name NAME] [--python-version VERSION] [--development]"
  echo "          [--amd-aocl auto|off|conda|amd-wheels] [--amd-wheel-cache DIR]"
  echo "          [-h|--help]"
  echo
  echo "Options:"
  echo "  --no-mamba             Skip installing and using mamba; force use of conda"
  echo "  --env-name NAME        Set environment name (default: TipTorch)"
  echo "  --python-version VER   Set Python version (default: 3.12)"
  echo "  --development          Include development packages (Jupyter, etc.)"
  echo
  echo "AMD options:"
  echo "  --amd-aocl MODE        AMD AOCL mode: auto, off, conda, amd-wheels (default: auto)"
  echo "  --amd-wheel-cache DIR  Directory containing manually downloaded AMD AOCL wheels"
  echo "                          default: ./amd_aocl_wheels"
  echo
  echo "Examples:"
  echo "  $0 --env-name TipTorch_amd --python-version 3.12 --amd-aocl conda"
  echo "  $0 --env-name TipTorch_amd_wheels --python-version 3.12 --amd-aocl amd-wheels --amd-wheel-cache ./amd_aocl_wheels"
  echo
  echo "For --amd-aocl amd-wheels with Python 3.12, manually download these wheels from AMD first:"
  echo "  numpy-2.1.3-cp312-cp312-linux_x86_64.whl"
  echo "  scipy-1.14.1-cp312-cp312-linux_x86_64.whl"
  echo "Optional:"
  echo "  numexpr-2.11.0-cp312-cp312-linux_x86_64.whl"
  echo "  aocl_sparse-0.1.0-cp312-cp312-linux_x86_64.whl"
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
    --amd-aocl)
      AMD_AOCL_MODE="$2"
      shift 2
      ;;
    --amd-wheel-cache)
      AMD_WHEEL_CACHE="$2"
      shift 2
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

case "$AMD_AOCL_MODE" in
  auto|off|conda|amd-wheels) ;;
  *)
    echo "❌ Invalid --amd-aocl mode: $AMD_AOCL_MODE"
    echo "   Valid modes: auto, off, conda, amd-wheels"
    exit 1
    ;;
esac

# ─── Abort on Windows ───────────────────────────────────────────────────────
UNAME="$(uname | tr '[:upper:]' '[:lower:]')"
if [[ "$UNAME" == msys* || "$UNAME" == mingw* || "$UNAME" == cygwin* ]]; then
  echo "⚠️  This setup script does not support Windows. Please use the PowerShell installer instead."
  exit 1
fi

# ─── Install mamba if not present ───────────────────────────────────────────
if ! command -v mamba &>/dev/null; then
  if [[ "$USE_MAMBA" == "true" ]]; then
    echo "→ mamba not found. Installing into base for faster solving…"
    conda install -n base mamba -c conda-forge -y || {
      echo "⚠️  Failed to install mamba. Continuing with conda."
    }
  else
    echo "→ mamba not found, but installation skipped per user request."
  fi
else
  echo "→ mamba is already installed."
fi

# ─── Pick solver ────────────────────────────────────────────────────────────
if [[ "$USE_MAMBA" == "true" && $(command -v mamba &>/dev/null; echo $?) -eq 0 ]]; then
  SOLVER="mamba"
else
  SOLVER="conda"
fi

echo "→ Using solver: $SOLVER"

# ─── Detect platform / CPU vendor ───────────────────────────────────────────
if [[ "$UNAME" == linux* ]]; then
  PLATFORM="linux"
elif [[ "$UNAME" == darwin* ]]; then
  PLATFORM="osx"
else
  PLATFORM="linux"
fi

INTEL_CPU=false
AMD_CPU=false

if [[ -f /proc/cpuinfo ]]; then
  if grep -q "vendor_id.*GenuineIntel" /proc/cpuinfo; then
    INTEL_CPU=true
  fi
  if grep -q "vendor_id.*AuthenticAMD" /proc/cpuinfo; then
    AMD_CPU=true
  fi
elif [[ "$PLATFORM" == "osx" ]]; then
  CPU_BRAND=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "")
  if [[ "$CPU_BRAND" == *"Intel"* ]]; then
    INTEL_CPU=true
  fi
  if [[ "$CPU_BRAND" == *"AMD"* || "$CPU_BRAND" == *"Ryzen"* || "$CPU_BRAND" == *"EPYC"* ]]; then
    AMD_CPU=true
  fi
fi

AMD_AOCL_ELIGIBLE=false
if [[ "$AMD_CPU" == "true" && "$PLATFORM" == "linux" ]]; then
  AMD_AOCL_ELIGIBLE=true
fi

USE_AMD_WHEELS=false
USE_AMD_CONDA_AOCL=false

function configure_amd_aocl_mode() {
  USE_AMD_CONDA_AOCL=false
  USE_AMD_WHEELS=false

  if [[ "$AMD_AOCL_MODE" == "off" ]]; then
    return
  fi

  if [[ "$AMD_AOCL_ELIGIBLE" != "true" ]]; then
    echo "→ AMD AOCL requested/auto, but this is not AMD Linux. Skipping AOCL."
    return
  fi

  case "$AMD_AOCL_MODE" in
    auto)
      # Safer, fully automatic default.
      USE_AMD_CONDA_AOCL=true
      ;;
    conda)
      USE_AMD_CONDA_AOCL=true
      ;;
    amd-wheels)
      USE_AMD_WHEELS=true
      ;;
  esac
}

configure_amd_aocl_mode

# ─── CUDA detection ─────────────────────────────────────────────────────────
CUDA_AVAILABLE=false
CUDA_VERSION="Unknown"

if command -v nvidia-smi &>/dev/null; then
  NVIDIA_SMI_OUTPUT=$(nvidia-smi --query-gpu=driver_version,cuda_version --format=csv,noheader,nounits 2>/dev/null || echo "")
  if [[ -n "$NVIDIA_SMI_OUTPUT" && "$NVIDIA_SMI_OUTPUT" != *"No devices were found"* ]]; then
    CUDA_AVAILABLE=true
    CUDA_VERSION=$(echo "$NVIDIA_SMI_OUTPUT" | head -n1 | cut -d',' -f2 | tr -d ' ')
    echo "→ CUDA detected via nvidia-smi: version $CUDA_VERSION"
  else
    echo "→ nvidia-smi found but no devices detected or command failed."
  fi
fi

if [[ "$CUDA_AVAILABLE" == "false" && -x "/usr/local/cuda/bin/nvcc" ]]; then
  CUDA_AVAILABLE=true
  CUDA_VERSION=$(/usr/local/cuda/bin/nvcc --version | sed -n 's/.*release \([0-9.]*\).*/\1/p' | head -n1 || echo "Unknown")
  echo "→ CUDA toolkit found via /usr/local/cuda, version ${CUDA_VERSION:-Unknown}"
fi

# ─── Helper: Python ABI tag ─────────────────────────────────────────────────
PY_TAG="cp${PYTHON_VERSION/./}"

case "$PYTHON_VERSION" in
  3.11|3.12|3.13) ;;
  *)
    if [[ "$USE_AMD_WHEELS" == "true" ]]; then
      echo "❌ AMD AOCL 5.2 Python wheels support Python 3.11, 3.12, and 3.13."
      echo "   Requested Python: $PYTHON_VERSION"
      exit 1
    fi
    ;;
esac

echo "→ Configuration:"
echo "   ENV_NAME=$ENV_NAME"
echo "   PYTHON_VERSION=$PYTHON_VERSION"
echo "   DEVELOPMENT=$DEVELOPMENT"
echo "   USE_MAMBA=$USE_MAMBA"
echo "   AMD_AOCL_MODE=$AMD_AOCL_MODE"
echo "   AMD_WHEEL_CACHE=$AMD_WHEEL_CACHE"
echo "   PLATFORM=$PLATFORM"
echo "   INTEL_CPU=$INTEL_CPU"
echo "   AMD_CPU=$AMD_CPU"
echo "   AMD_AOCL_ELIGIBLE=$AMD_AOCL_ELIGIBLE"
echo "   USE_AMD_WHEELS=$USE_AMD_WHEELS"
echo "   USE_AMD_CONDA_AOCL=$USE_AMD_CONDA_AOCL"
echo "   CUDA_AVAILABLE=$CUDA_AVAILABLE"
echo "   CUDA_VERSION=$CUDA_VERSION"

# ─── Build environment.yml ──────────────────────────────────────────────────
ENV_YML="environment.yml"
cat > "$ENV_YML" <<EOF
name: $ENV_NAME
channels:
  - nodefaults
EOF

if [[ "$INTEL_CPU" == "true" && "$USE_AMD_WHEELS" != "true" && "$USE_AMD_CONDA_AOCL" != "true" ]]; then
  cat >> "$ENV_YML" <<EOF
  - https://software.repos.intel.com/python/conda
EOF
  echo "→ Intel CPU detected, adding Intel Python channel with priority"
fi

cat >> "$ENV_YML" <<EOF
  - conda-forge

dependencies:
  - python=$PYTHON_VERSION
EOF

# Numerical stack.
# If using AMD wheels, install NumPy/SciPy/NumExpr after env creation.
if [[ "$USE_AMD_WHEELS" == "true" ]]; then
  cat >> "$ENV_YML" <<EOF
  # NumPy/SciPy/NumExpr will be installed later from official AMD AOCL wheels in the local cache.
EOF
elif [[ "$USE_AMD_CONDA_AOCL" == "true" ]]; then
  cat >> "$ENV_YML" <<EOF
  - numpy>=2.0
  - scipy
  - scikit-learn
  - numexpr
  - aocl-blas
  - aocl-lapack
EOF
  echo "→ Adding conda-forge AOCL BLAS/LAPACK packages"
else
  cat >> "$ENV_YML" <<EOF
  - numpy>=2.0
  - scipy
  - scikit-learn
EOF
fi

# Intel MKL packages for Intel systems only, and only when not using AMD mode.
if [[ "$INTEL_CPU" == "true" && "$USE_AMD_WHEELS" != "true" && "$USE_AMD_CONDA_AOCL" != "true" ]]; then
  cat >> "$ENV_YML" <<EOF
  - mkl
  - mkl-service
  - mkl_fft
  - mkl_random
EOF
  echo "→ Adding explicit Intel MKL packages"
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
  - astropy>=7.0
  - astroquery
  - photutils>=2.0
  - imageio
  - gwcs
  - asdf
  - asdf-astropy
  - mpl-scatter-density
  - ipykernel
EOF

if [[ "$DEVELOPMENT" == "true" ]]; then
  cat >> "$ENV_YML" <<EOF
  - jupyterlab
  - torchmetrics
  - pytorch-model-summary
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

echo "→ Generated $ENV_YML"

# ─── Create environment ─────────────────────────────────────────────────────
echo
echo "▶ Creating environment '$ENV_NAME' with $SOLVER…"
$SOLVER env create -f "$ENV_YML" -v
echo "✅ Environment '$ENV_NAME' created."

# ─── Fix runtime library lookup inside the Conda environment ────────────────
echo
echo "→ Adding Conda activation hook to prioritize environment libraries…"

ENV_PREFIX=$(conda run -n "$ENV_NAME" python -c "import sys; print(sys.prefix)")

mkdir -p "$ENV_PREFIX/etc/conda/activate.d"
cat > "$ENV_PREFIX/etc/conda/activate.d/fix_libstdcxx.sh" <<'EOF'
#!/usr/bin/env bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
EOF
chmod +x "$ENV_PREFIX/etc/conda/activate.d/fix_libstdcxx.sh"

# Also apply it to the current installer process, because conda run may not
# evaluate activation hooks exactly like an interactive conda activate.
export LD_LIBRARY_PATH="$ENV_PREFIX/lib:${LD_LIBRARY_PATH:-}"

echo "✅ Added activation hook:"
echo "   $ENV_PREFIX/etc/conda/activate.d/fix_libstdcxx.sh"

# ─── Official AMD AOCL Python wheels: local-cache-only path ─────────────────
function amd_eula_page_for_package() {
  local package="$1"

  case "$package" in
    numpy)
      echo "https://www.amd.com/en/developer/zen-software-studio/applications/python-libraries-with-aocl/numpy-5-2-eula-gcc.html"
      ;;
    scipy)
      echo "https://www.amd.com/en/developer/zen-software-studio/applications/python-libraries-with-aocl/scipy-5-2-eula-gcc.html"
      ;;
    numexpr)
      echo "https://www.amd.com/en/developer/zen-software-studio/applications/python-libraries-with-aocl/numexpr-5-2-eula-gcc.html"
      ;;
    aocl_sparse)
      echo "https://www.amd.com/en/developer/zen-software-studio/applications/python-libraries-with-aocl/aocl-sparse-5-2-eula-gcc.html"
      ;;
    *)
      return 1
      ;;
  esac
}

function amd_wheel_filename() {
  local package="$1"
  local py_tag="$2"

  case "$package" in
    numpy)
      echo "numpy-2.1.3-${py_tag}-${py_tag}-linux_x86_64.whl"
      ;;
    scipy)
      echo "scipy-1.14.1-${py_tag}-${py_tag}-linux_x86_64.whl"
      ;;
    numexpr)
      echo "numexpr-2.11.0-${py_tag}-${py_tag}-linux_x86_64.whl"
      ;;
    aocl_sparse)
      echo "aocl_sparse-0.1.0-${py_tag}-${py_tag}-linux_x86_64.whl"
      ;;
    *)
      return 1
      ;;
  esac
}

function require_cached_amd_aocl_wheel() {
  local package="$1"
  local required="$2"
  local py_tag="cp${PYTHON_VERSION/./}"
  local fname
  local wheel
  local eula_page

  fname="$(amd_wheel_filename "$package" "$py_tag")"
  wheel="$AMD_WHEEL_CACHE/$fname"
  eula_page="$(amd_eula_page_for_package "$package")"

  if [[ -f "$wheel" ]]; then
    echo "→ Found cached AMD AOCL wheel: $wheel"
    echo "$wheel"
    return 0
  fi

  echo "⚠️  AMD AOCL wheel not found in cache:"
  echo "   $wheel"
  echo
  echo "   AMD's official AOCL Python wheels are browser/EULA-gated."
  echo "   This safer installer does not try to curl those pages."
  echo "   Download this wheel manually in a browser and place it at the expected path:"
  echo
  echo "   ${eula_page}?filename=${fname}"
  echo
  echo "   Expected file:"
  echo "   $wheel"

  if [[ "$required" == "true" ]]; then
    exit 1
  else
    return 1
  fi
}

function install_cached_amd_aocl_wheel() {
  local package="$1"
  local required="$2"
  local wheel

  if ! wheel="$(require_cached_amd_aocl_wheel "$package" "$required")"; then
    return 1
  fi

  # require_cached_amd_aocl_wheel prints status lines as well as the wheel path;
  # the actual path is the last line.
  wheel="$(echo "$wheel" | tail -n 1)"

  echo "→ Installing AMD AOCL wheel: $wheel"
  conda run -n "$ENV_NAME" pip install --force-reinstall --no-deps "$wheel"
}

if [[ "$USE_AMD_WHEELS" == "true" ]]; then
  echo
  echo "→ Official AMD AOCL wheel mode enabled."
  echo "→ Wheel cache: $AMD_WHEEL_CACHE"
  echo "→ This mode is local-cache-only. It will not try to download from AMD automatically."

  mkdir -p "$AMD_WHEEL_CACHE"

  install_cached_amd_aocl_wheel "numpy" "true"
  install_cached_amd_aocl_wheel "scipy" "true"
  install_cached_amd_aocl_wheel "numexpr" "false"

  if [[ "$PYTHON_VERSION" == "3.12" || "$PYTHON_VERSION" == "3.13" ]]; then
    install_cached_amd_aocl_wheel "aocl_sparse" "false"
  fi

  echo "→ Installing scikit-learn after AMD AOCL NumPy/SciPy…"
  conda run -n "$ENV_NAME" pip install scikit-learn
fi

# ─── Install PyTorch: always prefer CUDA PyTorch on CUDA machines ───────────
echo
if [[ "$CUDA_AVAILABLE" == "true" ]]; then
  echo "→ Installing PyTorch with CUDA 12.8 support…"
  echo "→ Using CUDA index: https://download.pytorch.org/whl/cu128"
  conda run -n "$ENV_NAME" pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

  echo "→ Verifying PyTorch CUDA installation…"
  CUDA_CHECK=$(conda run -n "$ENV_NAME" python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'Device count: {torch.cuda.device_count()}')" 2>&1)
  if echo "$CUDA_CHECK" | grep -q "CUDA available: True"; then
    echo "✅ PyTorch CUDA verification successful."
    echo "$CUDA_CHECK"
  else
    echo "❌ PyTorch was installed, but CUDA is not available. Output:"
    echo "$CUDA_CHECK"
    echo "   Refusing to silently continue with CPU PyTorch on a CUDA machine."
    exit 1
  fi
else
  echo "→ CUDA was not detected. Installing official PyTorch CPU-only version…"
  conda run -n "$ENV_NAME" pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

echo "✅ PyTorch installation step complete."

# ─── Install CUDA development toolkit and CuPy for CUDA systems ─────────────
echo
if [[ "$CUDA_AVAILABLE" == "true" ]]; then
  echo "→ Installing CUDA Python bindings…"
  conda run -n "$ENV_NAME" pip install cuda-python

  echo "→ Installing CuPy for CUDA 12.x…"
  conda run -n "$ENV_NAME" pip install cupy-cuda12x || {
    echo "⚠️  Failed to install CuPy, but continuing…"
  }
else
  echo "→ CUDA not available, skipping CuPy installation"
fi

# ─── Install pytorch-minimize ───────────────────────────────────────────────
echo
echo "→ Attempting to install 'pytorch-minimize' from PyPI…"
if conda run -n "$ENV_NAME" pip install pytorch-minimize; then
  echo "✅ Successfully installed 'pytorch-minimize' from PyPI."
else
  echo "⚠️  'pytorch-minimize' not found on PyPI or installation failed. Falling back to GitHub…"

  if [[ ! -d "../pytorch-minimize" ]]; then
    echo "→ Cloning pytorch-minimize to ../pytorch-minimize…"
    git clone https://github.com/rfeinman/pytorch-minimize.git ../pytorch-minimize
  else
    echo "→ Found existing pytorch-minimize at ../pytorch-minimize"
  fi

  echo "→ Installing 'pytorch-minimize' from local Git repository…"
  conda run -n "$ENV_NAME" pip install -e ../pytorch-minimize
fi

# ─── Install tiptorch package in editable mode ──────────────────────────────
echo
echo "→ Installing tiptorch package in editable mode…"
conda run -n "$ENV_NAME" pip install -e .

# ─── Verification ───────────────────────────────────────────────────────────
echo
echo "→ Verifying numerical libraries…"
conda run -n "$ENV_NAME" python - <<'PY'
import sys
print("Python:", sys.version)

try:
    import numpy as np
    print("NumPy:", np.__version__)
    try:
        np.show_config()
    except Exception as exc:
        print("NumPy show_config failed:", exc)
except Exception as exc:
    print("NumPy import failed:", exc)

try:
    import scipy
    print("SciPy:", scipy.__version__)
except Exception as exc:
    print("SciPy import failed:", exc)

try:
    import numexpr
    print("NumExpr:", numexpr.__version__)
except Exception as exc:
    print("NumExpr import failed:", exc)

try:
    import sklearn
    print("scikit-learn:", sklearn.__version__)
except Exception as exc:
    print("scikit-learn import failed:", exc)

try:
    import torch
    print("PyTorch:", torch.__version__)
    print("Torch CUDA available:", torch.cuda.is_available())
    print("Torch CUDA version:", torch.version.cuda)
    print("Torch CUDA device count:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("Torch CUDA device 0:", torch.cuda.get_device_name(0))
except Exception as exc:
    print("PyTorch import failed:", exc)
PY

echo
echo "🎉 Environment setup complete!"
echo "   To start using it, run: conda activate $ENV_NAME"
