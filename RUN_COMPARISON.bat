@echo off
REM 🚀 FASTEST WAY TO RUN NEUROFIRE COMPARISON
REM This batch file executes the complete RL algorithm comparison

echo.
echo ================================================================================
echo   NEUROFIRE RL ALGORITHM COMPARISON - QUICK START
echo ================================================================================
echo.
echo Starting full comparison (3-5 minutes for complete results)...
echo.

cd /d "%~dp0"

REM Install dependencies if needed
echo Installing dependencies...
python -m pip install -q torch numpy matplotlib seaborn pandas 2>nul

REM Run the comparison
echo.
echo Executing RL_Algorithms_Comparison.py...
python RL_Algorithms_Comparison.py

echo.
echo ================================================================================
echo   EXECUTION COMPLETE!
echo ================================================================================
echo.
echo Results available:
echo   • neurofire_rl_comparison.png - Main visualization
echo   • comparison_results.png - Additional analysis
echo   • Console output above shows all metrics
echo.
echo Documents to read:
echo   • INDEX.md - Navigation guide
echo   • QUICK_START.md - Setup and usage
echo   • README_ENHANCED.md - Complete documentation
echo.
pause
