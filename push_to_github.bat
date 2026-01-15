@echo off
echo ========================================
echo Checking Git Status and Pushing to GitHub
echo ========================================
echo.

cd /d "C:\Users\Edam\Downloads\RL\NeuroFire"

echo Step 1: Checking git status...
git status
echo.

echo Step 2: Checking last commit...
git log --oneline -n 1
echo.

echo Step 3: Checking remote configuration...
git remote -v
echo.

echo Step 4: Attempting to push to GitHub...
git push -u origin main
echo.

echo ========================================
echo Push Complete!
echo ========================================
echo.

echo Please check: https://github.com/edamchaaben/NeuroFire
pause
