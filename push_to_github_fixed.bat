@echo off
echo ========================================
echo Pushing to GitHub with Correct URL
echo ========================================
echo.

cd /d "C:\Users\Edam\Downloads\RL\NeuroFire"

echo Remote URL is now: https://github.com/edamchaaben/NeuroFire.git
echo.

echo Pushing to GitHub...
git push -u origin main
echo.

if %errorlevel% equ 0 (
    echo ========================================
    echo SUCCESS! Push completed successfully!
    echo ========================================
    echo.
    echo Your repository is live at:
    echo https://github.com/edamchaaben/NeuroFire
) else (
    echo ========================================
    echo Push failed. Please check the error above.
    echo ========================================
)

echo.
pause
