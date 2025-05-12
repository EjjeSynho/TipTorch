#!/usr/bin/env bash
set -e

# ─── Abort on Windows ────────────────────────────────────────────────────────
UNAME="$(uname | tr '[:upper:]' '[:lower:]')"
if [[ "$UNAME" == msys* || "$UNAME" == mingw* || "$UNAME" == cygwin* ]]; then
  echo "⚠️  This setup script does not support Windows. Please use the PowerShell installer instead."
  exit 1
fi

# ─── Install mamba if missing ───────────────────────────────────────────────
if ! command -v mamba &>/dev/null; then
  echo "→ mamba not found in PATH. Installing into base…"
  # ensure conda is initialized in this shell
  # source "$(conda info --base)/etc/profile.d/conda.sh"
  conda install -n base mamba -c conda-forge -y
  echo "→ mamba installed."
fi

# ─── Pick the solver ─────────────────────────────────────────────────────────
if command -v mamba &>/dev/null; then
  SOLVER="mamba"
else
  SOLVER="conda"
fi
echo "→ Using solver: $SOLVER"

# ─── 1) Platform detection ───────────────────────────────────────────────────
case "$UNAME" in
  linux*)  PLATFORM=linux ;;
  darwin*) PLATFORM=osx   ;;
  *)       PLATFORM=linux ;;  # safe fallback
esac

# ─── 2) CPU‐vendor detection ─────────────────────────────────────────────────
INTEL_CPU=false
if [[ "$PLATFORM" == "linux" ]]; then
  VENDOR_ID=$(grep -m1 vendor_id /proc/cpuinfo 2>/dev/null | awk '{print $3}')
  [[ "$VENDOR_ID" == "GenuineIntel" ]] && INTEL_CPU=true
elif [[ "$PLATFORM" == "osx" ]]; then
  VENDOR_ID=$(sysctl -n machdep.cpu.vendor)
  [[ "$VENDOR_ID" == "Intel" ]] && INTEL_CPU=true
fi

# ─── 3) CUDA availability ───────────────────────────────────────────────────
CUDA_AVAILABLE=false
if command -v nvidia-smi &>/dev/null; then
  CUDA_AVAILABLE=true
fi

echo "→ Detected: PLATFORM=$PLATFORM, INTEL_CPU=$INTEL_CPU, CUDA_AVAILABLE=$CUDA_AVAILABLE"

# ─── 4) Build environment.yml ───────────────────────────────────────────────
ENV_YML=environment.yml
cat > "$ENV_YML" <<EOF
name: TipTorch
channels:
  - nodefaults
EOF

# Intel channel only on Linux + Intel CPU
if [[ "$PLATFORM" == "linux" && "$INTEL_CPU" == "true" ]]; then
  cat >> "$ENV_YML" <<EOF
  - https://software.repos.intel.com/python/conda/
EOF
fi

cat >> "$ENV_YML" <<EOF
  - conda-forge
  - astropy

dependencies:
  - python=3.12
EOF

# Pull in intelpython3_full (which bundles Intel-optimized NumPy/SciPy/etc.)
if [[ "$PLATFORM" == "linux" && "$INTEL_CPU" == "true" ]]; then
  cat >> "$ENV_YML" <<EOF
  - intelpython3_full
EOF
else
  # generic builds from conda-forge
  cat >> "$ENV_YML" <<EOF
  - numpy>=1.26
  - scipy
  - scikit-learn
EOF
fi

# common conda packages
cat >> "$ENV_YML" <<'EOF'
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

  - pip:
EOF

# pip/torch index selection
if [[ "$CUDA_AVAILABLE" == "true" && "$PLATFORM" == "linux" ]]; then
  cat >> "$ENV_YML" <<EOF
    - --index-url https://download.pytorch.org/whl/cu128
EOF
fi

# pip-only packages
cat >> "$ENV_YML" <<'EOF'
    - torch
    - torchvision
    - torchaudio
    - pytorch-minimize
EOF

echo "→ Generated $ENV_YML"

# ─── 5) Create the environment ───────────────────────────────────────────────
echo
echo "▶ Creating environment 'TipTorch' with $SOLVER (this might take a while)…"
eval "$SOLVER env create -f $ENV_YML -v"
echo "✅ Environment 'TipTorch' created."
echo "   To start using it, run: conda activate TipTorch"
