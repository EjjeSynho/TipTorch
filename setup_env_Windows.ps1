<#
.SYNOPSIS
  Create the 'TipTorch' Conda environment on Windows with optional CUDA support.

.DESCRIPTION
  • Installs mamba in base if requested (for faster, low-memory solves).  
  • Detects Intel CPU (no Intel channel on Windows).  
  • Detects NVIDIA GPU via nvidia-smi → if present, adds PyTorch CUDA index.  
  • Generates an environment.yml with:
      – nodefaults, conda-forge, astropy channels  
      – python=3.11 + generic NumPy/SciPy/Sklearn  
      – common libs + pip requirements (torch*, pytorch-minimize, photutils)  
  • Invokes mamba (or conda) to build the env verbosely.
  • Attempts to install pytorch-minimize from PyPI first; if that fails, falls back to GitHub.
  
.PARAMETER InstallMamba
  If specified, attempts to install mamba into the base environment if it's not already present.  
  If not specified, skips any automatic mamba installation and uses conda if mamba is missing.
#>

param (
    [switch]$InstallMamba
)

# ─── 0) Ensure Windows PowerShell ───────────────────────────────────────────
if ($env:OS -ne 'Windows_NT') {
    Write-Host "⚠️  This script only supports Windows PowerShell." -ForegroundColor Yellow
    exit 1
}

# ─── 1) (Optional) Install mamba in base if requested ────────────────────────
if ($InstallMamba) {
    if (-not (Get-Command mamba -ErrorAction SilentlyContinue)) {
        Write-Host "→ mamba not found. Installing into base…" -ForegroundColor Cyan
        conda install -n base mamba -c conda-forge -y
        if ($LASTEXITCODE -eq 0) {
            Write-Host "→ mamba installed successfully." -ForegroundColor Green
        } else {
            Write-Host "⚠️  Failed to install mamba. Continuing with conda." -ForegroundColor Yellow
        }
    } else {
        Write-Host "→ mamba is already installed." -ForegroundColor Green
    }
} else {
    Write-Host "→ Skipping mamba installation. If you want mamba, re-run with -InstallMamba." -ForegroundColor Yellow
}

# ─── 2) Pick solver ─────────────────────────────────────────────────────────
$Solver = if (Get-Command mamba -ErrorAction SilentlyContinue) { 'mamba' } else { 'conda' }
Write-Host "→ Using solver: $Solver" -ForegroundColor Cyan

# ─── 3) Detect CPU vendor & CUDA ────────────────────────────────────────────
$CpuVendor    = (Get-CimInstance Win32_Processor).Manufacturer
$IntelCPU     = $CpuVendor -match 'Intel'

# CUDA detection - checks if nvidia-smi exists AND runs successfully
$CudaPresent = $false
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    try {
        $NvidiaSmiOutput = & nvidia-smi 2>&1
        if ($LASTEXITCODE -eq 0) {
            $CudaPresent = $true
        }
    }
    catch {
        Write-Host "→ nvidia-smi found but failed to execute properly." -ForegroundColor Yellow
    }
}
Write-Host "→ Detected: PLATFORM=windows, INTEL_CPU=$IntelCPU, CUDA_AVAILABLE=$CudaPresent" -ForegroundColor Cyan

# ─── 4) Build environment.yml ──────────────────────────────────────────────
$Yml = @()
$Yml += 'name: TipTorch_test'
$Yml += 'channels:'
$Yml += '  - nodefaults'
$Yml += '  - conda-forge'
$Yml += '  - astropy'
$Yml += '  - pytorch'
$Yml += '  - nvidia'
$Yml += ''
$Yml += 'dependencies:'
$Yml += '  - python=3.11'

# On Windows we do _not_ inject intelpython3_full (Linux only in Bash wrapper)
$Yml += '  - numpy>=1.26'
$Yml += '  - scipy'
$Yml += '  - scikit-learn'

# Common Conda packages
$Yml += '  - pillow'
$Yml += '  - pandas'
$Yml += '  - matplotlib'
$Yml += '  - seaborn'
$Yml += '  - scikit-image'
$Yml += '  - tabulate'
$Yml += '  - tqdm'
$Yml += '  - gdown'
$Yml += '  - astropy'
$Yml += '  - astroquery'
$Yml += '  - photutils'
$Yml += '  - pip'

# Add PyTorch and CUDA packages here for conda to handle
if ($CudaPresent) {
    # PyTorch 2.4.1 + CUDA 12.1 stack
    $Yml += '  - pytorch=2.4.1'
    $Yml += '  - torchvision'
    $Yml += '  - torchaudio=2.4.1'
    $Yml += '  - pytorch-cuda=12.1'
} else {
    # CPU-only PyTorch 2.4.1
    $Yml += '  - pytorch-cpu=2.4.1'
    $Yml += '  - torchvision'
    $Yml += '  - torchaudio=2.4.1'
}

# Keep other pip packages if needed (excluding torch-related packages)
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
Write-Host "`n▶ Creating environment 'TipTorch_test' with $Solver (this may take a while)…" -ForegroundColor Cyan
& $Solver env create -f environment.yml -v

Write-Host "`n✅ Environment 'TipTorch_test' created." -ForegroundColor Green
Write-Host "   To start using it, run: `n     conda activate TipTorch_test" -ForegroundColor Green

# ─── 6) Install pytorch-minimize, prefer pip then fallback to GitHub ────────
if (-not (conda info --envs | Select-String "TipTorch_test")) {
    Write-Host "❌ Environment TipTorch_test not found. Something went wrong." -ForegroundColor Red
    exit 1
}

Write-Host "`n→ Attempting to install 'pytorch-minimize' from PyPI…" -ForegroundColor Cyan
& conda run -n TipTorch_test pip install pytorch-minimize

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Successfully installed 'pytorch-minimize' from PyPI." -ForegroundColor Green
} else {
    Write-Host "⚠️  'pytorch-minimize' not found on PyPI or installation failed. Falling back to GitHub…" -ForegroundColor Yellow

    if (-not (Test-Path "./pytorch-minimize")) {
        git clone https://github.com/rfeinman/pytorch-minimize.git
    }
    Write-Host "→ Installing 'pytorch-minimize' from local Git repository…" -ForegroundColor Cyan
    & conda run -n TipTorch_test pip install -e ./pytorch-minimize

    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Successfully installed 'pytorch-minimize' from GitHub." -ForegroundColor Green
    } else {
        Write-Host "❌ Failed to install 'pytorch-minimize' from GitHub." -ForegroundColor Red
        exit 1
    }
}