@echo off
REM ---------------------------------------------------------------------------
REM  UnFakeable - full training + evaluation run.
REM  Usage:  scripts\run_training.bat                  (configs\default.yaml)
REM          scripts\run_training.bat configs\large.yaml
REM  Every stage caches its output, so if it stops you can simply run it again.
REM ---------------------------------------------------------------------------
setlocal
cd /d "%~dp0\.."
call ".venv\Scripts\activate.bat" || (echo Run scripts\setup_windows.bat first & pause & exit /b 1)
set CFG=%~1
if "%CFG%"=="" set CFG=configs\default.yaml

echo === Checking the dataset folders ===
python -m unfakeable.cli check-data --config "%CFG%"
echo.
echo If the counts above look wrong, press Ctrl+C now and fix the paths in %CFG%.
pause

python -m unfakeable.cli all --config "%CFG%"
if errorlevel 1 (echo [ERROR] a stage failed - see artifacts\*\logs\ & pause & exit /b 1)
echo.
echo Finished. Results summary: artifacts\^<run_name^>\REPORT.md
echo Start the dashboard with:  scripts\start_dashboard.bat %CFG%
pause
