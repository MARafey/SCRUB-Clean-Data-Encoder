@echo off
REM Clean Data Encoder launcher (Windows)
REM
REM Runs through uv, which creates and syncs the virtualenv from pyproject.toml
REM on first use. No pretrained backbone, so no HuggingFace / legacy-Keras /
REM torch setup is needed - plain TensorFlow only.
REM
REM   run.bat run_ablation.py --dataset poi_library
REM   run.bat train.py --run myrun query_instruction=noise_domain --eval-after
REM   run.bat evaluate.py --run myrun --best
REM   run.bat -m tensorboard.main --logdir runs
REM
REM Install uv once:  powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

setlocal
set TF_CPP_MIN_LOG_LEVEL=2

where uv >nul 2>nul
if errorlevel 1 (
    echo uv is not on PATH. Install it with:
    echo     powershell -c "irm https://astral.sh/uv/install.ps1 ^| iex"
    echo then reopen this terminal.
    exit /b 1
)

if "%~1"=="" (
    echo Usage: run.bat ^<script.py^> [args...]
    echo.
    echo   run.bat run_ablation.py --dataset poi_library
    echo   run.bat train.py --run myrun --eval-after
    echo   run.bat evaluate.py --run myrun --best
    echo   run.bat -m tensorboard.main --logdir runs
    exit /b 1
)

uv run python %*
endlocal
