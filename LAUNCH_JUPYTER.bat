@echo off
REM 🚀 NEUROFIRE JUPYTER SETUP - Using Anaconda Python
REM This script installs Jupyter and runs the notebook

echo.
echo ================================================================================
echo   SETTING UP JUPYTER WITH ANACONDA PYTHON
echo ================================================================================
echo.

REM Set Anaconda path
set ANACONDA_PATH=C:\ProgramData\anaconda3.1
set PYTHON_EXE=%ANACONDA_PATH%\python.exe
set PIP_EXE=%ANACONDA_PATH%\Scripts\pip.exe

REM Check if Anaconda exists
if not exist "%PYTHON_EXE%" (
    echo ❌ ERROR: Anaconda Python not found at:
    echo    %ANACONDA_PATH%
    echo.
    echo Solution:
    echo   1. Install Anaconda from anaconda.com
    echo   2. Or update the ANACONDA_PATH in this script
    pause
    exit /b 1
)

echo ✅ Found Python: %PYTHON_EXE%
"%PYTHON_EXE%" --version
echo.

REM Install required packages
echo Installing required packages...
echo.

"%PIP_EXE%" install -q jupyter notebook ipython 2>nul

if %ERRORLEVEL% NEQ 0 (
    echo ⚠️  Trying alternative install method...
    "%PYTHON_EXE%" -m pip install -q jupyter notebook ipython
)

echo.
echo ✅ Installation complete!
echo.
echo ================================================================================
echo   LAUNCHING JUPYTER NOTEBOOK
echo ================================================================================
echo.
echo Opening: RL_Algorithm_Comparison_NeuroFire.ipynb
echo URL: http://localhost:8888
echo.
echo Press CTRL+C to stop the server
echo.

REM Launch notebook
"%PYTHON_EXE%" -m jupyter notebook RL_Algorithm_Comparison_NeuroFire.ipynb

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ❌ ERROR: Could not start Jupyter
    echo.
    echo Troubleshooting:
    echo   1. Check Anaconda installation
    echo   2. Try manual command:
    echo      "%PYTHON_EXE%" -m jupyter notebook
    echo   3. Or use the Python runner:
    echo      "%PYTHON_EXE%" RL_Algorithms_Comparison.py
    pause
)
