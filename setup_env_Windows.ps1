<#
.SYNOPSIS
  Create the 'TipTorch' Conda environment on Windows with optional CUDA support.

.DESCRIPTION
  • Installs mamba in base if missing (for faster, low-memory solves).  
  • Detects Intel CPU (no Intel channel on Windows).  
  • Detects NVIDIA GPU via nvidia-smi → if present, adds PyTorch CUDA index.  
  • Generates an environment.yml with:
      – nodefaults, conda-forge, astropy channels  
      – python=3.12 + either generic NumPy/SciPy/Sklearn or Intel stack (Linux only)  
      – common libs + pip requirements (torch*, pytorch-minimize, photutils)  
  • Invokes mamba (or conda) to build the env verbosely.
#>

# ─── 0) Ensure Windows PowerShell ───────────────────────────────────────────
if ($env:OS -ne 'Windows_NT') {
  Write-Host "⚠️  This script only supports Windows PowerShell." -ForegroundColor Yellow
  exit 1
}

# ─── 1) Install mamba in base if missing ────────────────────────────────────
if (-not (Get-Command mamba -ErrorAction SilentlyContinue)) {
  Write-Host "→ mamba not found. Installing into base…" -ForegroundColor Cyan
  conda install -n base mamba -c conda-forge -y
  Write-Host "→ mamba installed." -ForegroundColor Green
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
        # Only mark CUDA as present if nvidia-smi executed without errors
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
$Yml += 'name: TipTorch'
$Yml += 'channels:'
$Yml += '  - nodefaults'
$Yml += '  - conda-forge'
$Yml += '  - astropy'
$Yml += ''
$Yml += 'dependencies:'
$Yml += '  - python=3.12'

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
$Yml += '  - astropy'
$Yml += '  - astroquery'
$Yml += '  - photutils'
$Yml += '  - pip'
$Yml += ''
$Yml += '  - pip:'

# If an NVIDIA GPU is present, point pip at the CUDA-128 wheels
if ($CudaPresent) {
  $Yml += '    - --index-url https://download.pytorch.org/whl/cu128'
}

# Pip-only packages
$Yml += '    - torch'
$Yml += '    - torchvision'
$Yml += '    - torchaudio'
$Yml += '    - pytorch-minimize'

# Write out environment.yml
$Yml | Set-Content -Path environment.yml -Encoding UTF8
Write-Host "→ Generated environment.yml" -ForegroundColor Green

# ─── 5) Create the environment ───────────────────────────────────────────────
Write-Host "`n▶ Creating environment 'TipTorch' with $Solver (this may take a while)…" -ForegroundColor Cyan
& $Solver env create -f environment.yml -v

Write-Host "`n✅ Environment 'TipTorch' created." -ForegroundColor Green
Write-Host "   To start using it, run: `n     conda activate TipTorch" -ForegroundColor Green
