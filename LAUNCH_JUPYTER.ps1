# 🚀 NEUROFIRE JUPYTER LAUNCHER - ANACONDA PYTHON
# Run this in PowerShell to launch Jupyter notebook

$AnacondaPath = "C:\ProgramData\anaconda3.1"
$PythonExe = Join-Path $AnacondaPath "python.exe"
$PipExe = Join-Path $AnacondaPath "Scripts\pip.exe"

Write-Host ""
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "  NEUROFIRE JUPYTER NOTEBOOK - ANACONDA PYTHON" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host ""

# Check if Anaconda exists
if (-not (Test-Path $PythonExe)) {
    Write-Host "❌ ERROR: Anaconda Python not found at:" -ForegroundColor Red
    Write-Host "   $AnacondaPath" -ForegroundColor Red
    Write-Host ""
    Write-Host "Solution:" -ForegroundColor Yellow
    Write-Host "  1. Install Anaconda from anaconda.com" -ForegroundColor Yellow
    Write-Host "  2. Or update the AnacondaPath variable" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "✅ Found Python: $PythonExe" -ForegroundColor Green
& $PythonExe --version
Write-Host ""

# Install Jupyter
Write-Host "Installing Jupyter and dependencies..." -ForegroundColor Cyan
& $PipExe install -q jupyter notebook ipython 2>$null

if ($LASTEXITCODE -ne 0) {
    Write-Host "⚠️  Using alternative install method..." -ForegroundColor Yellow
    & $PythonExe -m pip install -q jupyter notebook ipython
}

Write-Host "✅ Jupyter installed!" -ForegroundColor Green
Write-Host ""

# Change to notebook directory
$NotebookDir = Get-Location
$NotebookFile = "RL_Algorithm_Comparison_NeuroFire.ipynb"

if (-not (Test-Path $NotebookFile)) {
    Write-Host "❌ ERROR: Notebook file not found!" -ForegroundColor Red
    Write-Host "   Looking for: $NotebookFile" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "  LAUNCHING JUPYTER NOTEBOOK" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Notebook: $NotebookFile" -ForegroundColor Cyan
Write-Host "Browser: http://localhost:8888" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press CTRL+C to stop the server" -ForegroundColor Yellow
Write-Host ""

# Launch Jupyter
& $PythonExe -m jupyter notebook $NotebookFile

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "❌ ERROR: Could not start Jupyter" -ForegroundColor Red
    Write-Host ""
    Write-Host "Troubleshooting:" -ForegroundColor Yellow
    Write-Host "  1. Check Anaconda installation" -ForegroundColor Yellow
    Write-Host "  2. Try manual command:" -ForegroundColor Yellow
    Write-Host "     $PythonExe -m jupyter notebook" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
}
