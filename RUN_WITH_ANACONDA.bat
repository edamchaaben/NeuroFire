@echo off
REM 🚀 NEUROFIRE COMPARISON - Using Anaconda Python
REM This script runs the RL comparison without needing Jupyter

setlocal enabledelayedexpansion

set ANACONDA_PATH=C:\ProgramData\anaconda3.1
set PYTHON_EXE=%ANACONDA_PATH%\python.exe
set PIP_EXE=%ANACONDA_PATH%\Scripts\pip.exe

echo.
echo ================================================================================
echo   NEUROFIRE RL ALGORITHM COMPARISON - ANACONDA PYTHON
echo ================================================================================
echo.

REM Check if Anaconda exists
if not exist "%PYTHON_EXE%" (
    echo ❌ ERROR: Anaconda Python not found at:
    echo    %ANACONDA_PATH%
    echo.
    echo Please install Anaconda or update the path in this script.
    pause
    exit /b 1
)

echo ✅ Using Python: %PYTHON_EXE%
"%PYTHON_EXE%" --version
echo.

REM Install dependencies
echo Installing dependencies...
"%PIP_EXE%" install -q torch numpy matplotlib seaborn pandas 2>nul

if %ERRORLEVEL% NEQ 0 (
    echo Trying alternative install...
    "%PYTHON_EXE%" -m pip install -q torch numpy matplotlib seaborn pandas
)

echo ✅ Dependencies ready
echo.

REM Run the comparison
echo ================================================================================
echo Starting RL Algorithm Comparison (3-5 minutes)...
echo ================================================================================
echo.

"%PYTHON_EXE%" RL_Algorithms_Comparison.py

echo.
echo ================================================================================
echo   ✅ EXECUTION COMPLETE!
echo ================================================================================
echo.
echo Check for output files:
echo   • neurofire_rl_comparison.png (main visualization)
echo   • comparison_results.png (additional analysis)
echo.
pause
