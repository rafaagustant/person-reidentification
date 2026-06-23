$ErrorActionPreference = "Stop"

Write-Host "Person Re-ID CUDA setup for Windows + NVIDIA GPU" -ForegroundColor Cyan
Write-Host "This script keeps the existing .venv. It will not delete .venv automatically." -ForegroundColor Yellow
Write-Host "It installs non-PyTorch dependencies first, then installs PyTorch CUDA cu126." -ForegroundColor Yellow

$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RepoRoot

$VenvDir = Join-Path $RepoRoot ".venv"
$PythonExe = Join-Path $VenvDir "Scripts\python.exe"
$ActivateScript = Join-Path $VenvDir "Scripts\Activate.ps1"

if (-not (Test-Path -LiteralPath $PythonExe)) {
    Write-Host "Creating .venv..." -ForegroundColor Cyan
    python -m venv .venv
}

if (-not (Test-Path -LiteralPath $PythonExe)) {
    throw "Failed to create .venv. Make sure Python is installed and available as 'python'."
}

Write-Host "Activating .venv..." -ForegroundColor Cyan
. $ActivateScript

Write-Host "Using Python:" -ForegroundColor Cyan
& $PythonExe -c "import sys; print(sys.executable)"

Write-Host "Upgrading pip/setuptools/wheel..." -ForegroundColor Cyan
& $PythonExe -m pip install --upgrade pip setuptools wheel

Write-Host "Installing non-PyTorch dependencies from requirements.txt..." -ForegroundColor Cyan
& $PythonExe -m pip install --prefer-binary -r requirements.txt

Write-Host "Checking currently installed PyTorch..." -ForegroundColor Cyan
$TorchProbe = & $PythonExe -c "import importlib.util; import sys; spec=importlib.util.find_spec('torch'); print('installed' if spec else 'missing')"
if ($TorchProbe -eq "installed") {
    $TorchVersion = & $PythonExe -c "import torch; print(torch.__version__)"
    Write-Host "Current torch version: $TorchVersion"
    if ($TorchVersion.ToLower().Contains("+cpu")) {
        Write-Host "CPU-only torch detected. Uninstalling torch/torchvision/torchaudio from this .venv before CUDA install..." -ForegroundColor Yellow
        & $PythonExe -m pip uninstall -y torch torchvision torchaudio
    }
}

Write-Host "Installing PyTorch CUDA cu126. This wheel is large; timeout/retry is intentionally high." -ForegroundColor Cyan
& $PythonExe -m pip install --timeout 1000 --retries 20 --prefer-binary -r requirements-cuda.txt

Write-Host "Validating CUDA..." -ForegroundColor Cyan
& $PythonExe check_cuda.py
$ExitCode = $LASTEXITCODE

if ($ExitCode -eq 0) {
    Write-Host "CUDA setup succeeded. Run the app with:" -ForegroundColor Green
    Write-Host "python -m streamlit run app.py" -ForegroundColor Green
    exit 0
}

Write-Host "CUDA setup did not succeed." -ForegroundColor Red
Write-Host "If torch still shows '+cpu', run:" -ForegroundColor Yellow
Write-Host "python -m pip uninstall -y torch torchvision torchaudio" -ForegroundColor Yellow
Write-Host "python -m pip install --timeout 1000 --retries 20 --prefer-binary -r requirements-cuda.txt" -ForegroundColor Yellow
Write-Host "python check_cuda.py" -ForegroundColor Yellow
exit 1
