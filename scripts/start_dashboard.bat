@echo off
REM  Starts the backend inference service + dashboard for a finished run, then opens the browser.
REM  Usage:  scripts\start_dashboard.bat [configs\default.yaml]
setlocal
cd /d "%~dp0\.."
call ".venv\Scripts\activate.bat" || (echo Run scripts\setup_windows.bat first & pause & exit /b 1)
set CFG=%~1
if "%CFG%"=="" set CFG=configs\default.yaml
start "" http://127.0.0.1:8000
python -m unfakeable.cli serve --config "%CFG%"
