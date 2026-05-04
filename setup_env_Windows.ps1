<#
.SYNOPSIS
  Create the Conda environment for running TipTorch on Windows with optional CUDA support.

.DESCRIPTION
  • Installs mamba in base if requested (for faster, low-memory solves).  
  • Detects Intel CPU (no Intel channel on Windows).  
  • Detects NVIDIA GPU via nvidia-smi → if present, adds PyTorch CUDA index.  
  • Generates an environment.yml with:
      – nodefaults, conda-forge, astropy channels  
      – python + generic NumPy/SciPy/Sklearn  
      – common libs + pip requirements (torch*, pytorch-minimize, photutils, etc.)  
  • Invokes mamba (or conda) to build the env verbosely.
  • Attempts to install pytorch-minimize from PyPI first; if that fails, falls back to GitHub.
  
.PARAMETER InstallMamba
  If specified, attempts to install mamba into the base environment if it's not already present.  
  If not specified, skips any automatic mamba installation and uses conda if mamba is missing.
#>

# ─── 0) Configuration ───────────────────────────────────────────────────────
param (
    [switch]$InstallMamba,
    [string]$EnvName = 'TipTorch',
    [string]$PythonVersion = '3.12',
    [switch]$Development
)
# EXAMPLE USAGE: .\setup_env_Windows.ps1 -InstallMamba -EnvName "TipTorch" -PythonVersion "3.12" -Development

# ─── 1) Ensure Windows PowerShell ───────────────────────────────────────────
if ($env:OS -ne 'Windows_NT') {
    Write-Host "⚠️  This script only supports Windows PowerShell." -ForegroundColor Yellow
    exit 1
}

# ─── 1) Install mamba if not present (recommended for speed) ──────────────────
if (-not (Get-Command mamba -ErrorAction SilentlyContinue)) {
    if ($InstallMamba -or (-not $InstallMamba.IsPresent)) {
        Write-Host "→ mamba not found. Installing into base for faster solving…" -ForegroundColor Cyan
        conda install -n base mamba -c conda-forge -y
        if ($LASTEXITCODE -eq 0) {
            Write-Host "→ mamba installed successfully." -ForegroundColor Green
        } else {
            Write-Host "⚠️  Failed to install mamba. Continuing with conda." -ForegroundColor Yellow
        }
    } else {
        Write-Host "→ mamba not found, but installation skipped per user request." -ForegroundColor Yellow
    }
} else {
    Write-Host "→ mamba is already installed." -ForegroundColor Green
}

# ─── 2) Pick solver ─────────────────────────────────────────────────────────
$Solver = if (Get-Command mamba -ErrorAction SilentlyContinue) { 'mamba' } else { 'conda' }
Write-Host "→ Using solver: $Solver" -ForegroundColor Cyan

# ─── 3) Detect CPU vendor & CUDA ────────────────────────────────────────────
$CpuVendor = (Get-CimInstance Win32_Processor).Manufacturer
$IntelCPU  = $CpuVendor -match 'Intel'

# Enhanced CUDA detection - checks nvidia-smi AND CUDA runtime
$CudaPresent = $false
$CudaVersion = "Unknown"
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    try {
        $NvidiaSmiOutput = & nvidia-smi --query-gpu=driver_version,cuda_version --format=csv,noheader,nounits 2>&1
        if ($LASTEXITCODE -eq 0 -and $NvidiaSmiOutput -notmatch "No devices were found") {
            $CudaPresent = $true
            $CudaVersion = ($NvidiaSmiOutput -split ',')[1].Trim()
            Write-Host "→ CUDA detected: version $CudaVersion" -ForegroundColor Green
        }
    }
    catch {
        Write-Host "→ nvidia-smi found but failed to execute properly." -ForegroundColor Yellow
    }
}

# Additional check for CUDA runtime libraries
if (-not $CudaPresent -and (Test-Path "$env:CUDA_PATH\bin\nvcc.exe")) {
    $CudaPresent = $true
    Write-Host "→ CUDA toolkit found via CUDA_PATH environment variable" -ForegroundColor Green
}

Write-Host "→ Detected: PLATFORM=windows, INTEL_CPU=$IntelCPU, CUDA_AVAILABLE=$CudaPresent" -ForegroundColor Cyan

# ─── 4) Build environment.yml ──────────────────────────────────────────────
$Yml = @()
$Yml += "name: $EnvName"
$Yml += 'channels:'
$Yml += '  - nodefaults'

# Add Intel channel first (higher priority) if Intel CPU is detected
if ($IntelCPU) {
    $Yml += '  - https://software.repos.intel.com/python/conda'
    Write-Host "→ Intel CPU detected, adding Intel Python channel with priority" -ForegroundColor Green
}

$Yml += '  - conda-forge'

if (-not $IntelCPU) {
    Write-Host "→ Non-Intel CPU detected, skipping Intel channel" -ForegroundColor Yellow
}
$Yml += ''
$Yml += 'dependencies:'
$Yml += "  - python=$PythonVersion"

# Core scientific packages with Intel MKL support if Intel CPU detected
$Yml += '  - numpy>=2.0'
$Yml += '  - scipy'
$Yml += '  - scikit-learn'

# Add Intel MKL packages explicitly for Intel systems to avoid pip conflicts
if ($IntelCPU) {
    $Yml += '  - mkl'
    $Yml += '  - mkl-service'
    $Yml += '  - mkl_fft'
    $Yml += '  - mkl_random'
    Write-Host "→ Adding explicit Intel MKL packages to prevent pip conflicts" -ForegroundColor Green
}

# Common Conda packages
$Yml += '  - pandas'
$Yml += '  - matplotlib'
$Yml += '  - seaborn'
$Yml += '  - plotly'
$Yml += '  - scikit-image'
$Yml += '  - tabulate'
$Yml += '  - tqdm'
$Yml += '  - gdown'
$Yml += '  - pooch'
$Yml += '  - astropy'
$Yml += '  - astroquery'
$Yml += '  - photutils'
$Yml += '  - imageio'
$Yml += '  - gwcs'
$Yml += '  - asdf'
$Yml += '  - asdf-astropy'
$Yml += '  - mpl-scatter-density'
$Yml += '  - ipykernel'

# Development packages (only if Development flag is set)
if ($Development) {
    $Yml += '  - jupyterlab'
    $Yml += '  - torchmetrics'
    $Yml += '  - pytorch-model-summary'
    $Yml += '  - sympy'
    $Yml += '  - imbalanced-learn'
    $Yml += '  - shap'
    $Yml += '  - networkx'
    $Yml += '  - statsmodels'
    $Yml += '  - invoke'
    $Yml += '  - nbconvert'
} else {
    Write-Host "→ Skipping development packages. Use -Development flag to include them." -ForegroundColor Yellow
}

$Yml += '  - pip'

# PyTorch will be installed via pip after environment creation
# (conda packages removed to use pip with CUDA 13.0 index)

# Add Intel MKL via pip for Intel systems to prevent dependency conflicts
if ($IntelCPU) {
    $PipPackages += 'mkl'
    Write-Host "→ Adding Intel MKL via pip to satisfy pip dependency resolver" -ForegroundColor Green
}

# Add pip packages
$PipPackages = @(
    'pillow'
)

# Add development pip packages if Development flag is set
if ($Development) {
    # Add any development-specific pip packages here if needed
    $PipPackages += 'xgboost'
    $PipPackages += 'elasticsearch==6.8.2'
    $PipPackages += 'elasticsearch-dsl==6.4.0'
}

if ($PipPackages.Count -gt 0) {
    $Yml += '  - pip:'
    foreach ($pkg in $PipPackages) {
        $Yml += "    - $pkg"
    }
}

# Write out environment.yml
$Yml | Set-Content -Path environment.yml -Encoding UTF8
Write-Host "→ Generated environment.yml" -ForegroundColor Green

# ─── 5) Create the environment ───────────────────────────────────────────────
Write-Host "`n▶ Creating environment '$EnvName' with $Solver (this may take a while)…" -ForegroundColor Cyan
& $Solver env create -f environment.yml -v

Write-Host "`n✅ Environment '$EnvName' created." -ForegroundColor Green

# ─── 6) Install PyTorch via pip with CUDA support ──────────────────────────
if ($CudaPresent) {
    Write-Host "`n→ Installing PyTorch with CUDA 12.8 support…" -ForegroundColor Cyan
    Write-Host "→ Using CUDA index: https://download.pytorch.org/whl/cu128" -ForegroundColor Cyan
    & conda run -n $EnvName pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
    
    # Verify CUDA installation
    Write-Host "→ Verifying PyTorch CUDA installation…" -ForegroundColor Cyan
    $CudaCheck = & conda run -n $EnvName python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'Device count: {torch.cuda.device_count()}')" 2>&1
    if ($CudaCheck -match "CUDA available: True") {
        Write-Host "✅ PyTorch CUDA verification successful." -ForegroundColor Green
        Write-Host "   $CudaCheck" -ForegroundColor Gray
    } else {
        Write-Host "⚠️  PyTorch installed but CUDA not available. Output:" -ForegroundColor Yellow
        Write-Host "   $CudaCheck" -ForegroundColor Gray
    }
} else {
    Write-Host "`n→ Installing PyTorch CPU-only version…" -ForegroundColor Cyan
    & conda run -n $EnvName pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
}

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Successfully installed PyTorch." -ForegroundColor Green
} else {
    Write-Host "❌ Failed to install PyTorch." -ForegroundColor Red
    exit 1
}

# ─── 6.5) Install CUDA development toolkit and CuPy via conda ──────────────
Write-Host "`n" -NoNewline
if ($CudaPresent) {
    Write-Host "→ Installing CUDA development headers for CuPy…" -ForegroundColor Cyan
    & conda run -n $EnvName pip install cuda-python
    
    Write-Host "→ Installing CuPy for CUDA 12.x…" -ForegroundColor Cyan
    & conda run -n $EnvName pip install cupy-cuda12x
} else {
    Write-Host "→ CUDA not available, skipping CuPy installation" -ForegroundColor Yellow
}

if ($LASTEXITCODE -eq 0 -and $CudaPresent) {
    Write-Host "✅ Successfully installed CuPy." -ForegroundColor Green
} elseif ($CudaPresent) {
    Write-Host "⚠️ Failed to install CuPy, but continuing…" -ForegroundColor Yellow
}

# ─── 7) Install pytorch-minimize, prefer pip then fallback to GitHub ────────
if (-not (conda info --envs | Select-String "$EnvName")) {
    Write-Host "❌ Environment $EnvName not found. Something went wrong." -ForegroundColor Red
    exit 1
}

Write-Host "`n→ Attempting to install 'pytorch-minimize' from PyPI…" -ForegroundColor Cyan
& conda run -n $EnvName pip install pytorch-minimize

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Successfully installed 'pytorch-minimize' from PyPI." -ForegroundColor Green
} else {
    Write-Host "⚠️  'pytorch-minimize' not found on PyPI or installation failed. Falling back to GitHub…" -ForegroundColor Yellow

    if (-not (Test-Path "../pytorch-minimize")) {
        Write-Host "→ Cloning pytorch-minimize to ../pytorch-minimize…" -ForegroundColor Cyan
        git clone https://github.com/rfeinman/pytorch-minimize.git ../pytorch-minimize
    } else {
        Write-Host "→ Found existing pytorch-minimize at ../pytorch-minimize" -ForegroundColor Green
    }
    Write-Host "→ Installing 'pytorch-minimize' from local Git repository at ../pytorch-minimize…" -ForegroundColor Cyan
    & conda run -n $EnvName pip install -e ../pytorch-minimize

    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Successfully installed 'pytorch-minimize' from GitHub." -ForegroundColor Green
    } else {
        Write-Host "❌ Failed to install 'pytorch-minimize' from GitHub." -ForegroundColor Red
        exit 1
    }
}

# ─── 8) Install tiptorch package in editable mode ────────────────────────────
Write-Host "`n→ Installing tiptorch package in editable (development) mode…" -ForegroundColor Cyan
& conda run -n $EnvName pip install -e .

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Successfully installed tiptorch in editable mode." -ForegroundColor Green
} else {
    Write-Host "❌ Failed to install tiptorch." -ForegroundColor Red
    exit 1
}

Write-Host "`n🎉 Environment setup complete!" -ForegroundColor Green
Write-Host "   To start using it, run: `n     conda activate $EnvName" -ForegroundColor Green