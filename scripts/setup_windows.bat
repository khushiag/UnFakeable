@echo off
REM ---------------------------------------------------------------------------
REM  UnFakeable - one-time setup on Windows (RTX 3050 laptop).
REM  Double-click, or run from the project folder:   scripts\setup_windows.bat
REM  Creates .venv, installs the CUDA build of PyTorch, all other packages,
REM  checks the GPU and runs the fast unit tests.
REM ---------------------------------------------------------------------------
setlocal
cd /d "%~dp0\.."
echo Project folder: %CD%

REM ---- 1. virtual environment (Python 3.12 preferred) -----------------------
if not exist ".venv\Scripts\python.exe" (
    py -3.12 -m venv .venv 2>nul || python -m venv .venv
)
if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Could not create a virtual environment. Install Python 3.12 from python.org
    echo         and tick "Add python.exe to PATH" during installation.
    pause & exit /b 1
)
call ".venv\Scripts\activate.bat"
python -m pip install --upgrade pip

REM ---- 2. PyTorch with CUDA: try the newest CUDA build first -----------------
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>nul && goto torch_ok
for %%C in (cu130 cu128 cu126 cu124) do (
    echo Trying PyTorch CUDA build %%C ...
    pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/%%C
    python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" && goto torch_ok
)
echo.
echo [WARNING] PyTorch is installed but cannot see the GPU.
echo           Update the NVIDIA driver (NVIDIA App / GeForce Experience), reboot, run this script again.
echo           Training still works on the CPU, just slower.
:torch_ok

REM ---- 3. everything else ----------------------------------------------------
pip install -r requirements.txt || (echo [ERROR] pip install failed & pause & exit /b 1)
pip install --no-deps facenet-pytorch
pip install -e .
pip install -r requirements-optional.txt || echo [info] MediaPipe skipped (optional)

REM ---- 4. checks -------------------------------------------------------------
python -m unfakeable.cli gpu
python -m pytest -q -m "not slow"
echo.
echo Setup finished. Next: put the datasets in data\ and run   scripts\run_training.bat
pause
