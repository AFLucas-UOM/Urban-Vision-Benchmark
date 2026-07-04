@echo off
rem ============================================================
rem Creates (or updates) the project's Conda environments (cmd).
rem
rem Usage:
rem   setup_conda_env.cmd MDWD ^| mtsd-attrcls ^| mtsd-base ^| mtsd-la ^| all  [cpu]
rem
rem Pass "cpu" as the second argument to install CPU-only PyTorch
rem wheels (machines without an NVIDIA GPU).
rem
rem Examples:
rem   setup_conda_env.cmd mtsd-base
rem   setup_conda_env.cmd all cpu
rem ============================================================
setlocal EnableDelayedExpansion
set "HERE=%~dp0"

if "%~1"=="" goto :usage

if /I "%~1"=="all" (
    call "%~f0" MDWD %2         || exit /b 1
    call "%~f0" mtsd-attrcls %2 || exit /b 1
    call "%~f0" mtsd-base %2    || exit /b 1
    call "%~f0" mtsd-la %2      || exit /b 1
    exit /b 0
)

rem ---- Per-environment definitions (tested versions, 2026-07) ----
set "ENVNAME="
if /I "%~1"=="MDWD"         set "ENVNAME=MDWD"         & set "YAML=environment-mdwd.yml"         & set "TORCH=torch==2.11.0 torchvision==0.26.0" & set "INDEX=https://download.pytorch.org/whl/cu130"
if /I "%~1"=="mtsd-attrcls" set "ENVNAME=mtsd-attrcls" & set "YAML=environment-mtsd-attrcls.yml" & set "TORCH=torch==2.11.0 torchvision==0.26.0" & set "INDEX=https://download.pytorch.org/whl/cu128"
if /I "%~1"=="mtsd-base"    set "ENVNAME=mtsd-base"    & set "YAML=environment-mtsd-base.yml"    & set "TORCH=torch==2.10.0 torchvision==0.25.0" & set "INDEX=https://download.pytorch.org/whl/cu128"
if /I "%~1"=="mtsd-la"      set "ENVNAME=mtsd-la"      & set "YAML=environment-mtsd-la.yml"      & set "TORCH=torch==2.11.0 torchvision==0.26.0" & set "INDEX=https://download.pytorch.org/whl/cu128"
if not defined ENVNAME goto :usage

if /I "%~2"=="cpu" set "INDEX=https://download.pytorch.org/whl/cpu"

rem ---- Locate conda ----
set "CONDA=conda"
where conda >nul 2>nul
if errorlevel 1 (
    if exist "%USERPROFILE%\anaconda3\Scripts\conda.exe"  set "CONDA=%USERPROFILE%\anaconda3\Scripts\conda.exe"
    if exist "%USERPROFILE%\miniconda3\Scripts\conda.exe" set "CONDA=%USERPROFILE%\miniconda3\Scripts\conda.exe"
)
"%CONDA%" --version >nul 2>nul || (echo conda was not found on PATH or in the default install locations.& exit /b 1)

echo.
echo === [%ENVNAME%] ===
"%CONDA%" env list | findstr /R /C:"^%ENVNAME% " /C:"^%ENVNAME%	" >nul
if errorlevel 1 (
    echo [%ENVNAME%] creating from %YAML%
    "%CONDA%" env create -f "%HERE%%YAML%" || exit /b 1
) else (
    echo [%ENVNAME%] exists - updating from %YAML% ^(--prune^)
    "%CONDA%" env update -n %ENVNAME% -f "%HERE%%YAML%" --prune || exit /b 1
)

echo [%ENVNAME%] installing PyTorch (%TORCH%) from %INDEX%
"%CONDA%" run -n %ENVNAME% python -m pip install %TORCH% --index-url %INDEX% || exit /b 1

if /I "%ENVNAME%"=="mtsd-base" (
    echo [mtsd-base] installing Meta sam3 package ^(SAM 3 / SAM 3.1^)
    "%CONDA%" run -n mtsd-base python -m pip install "git+https://github.com/facebookresearch/sam3.git" || exit /b 1
)

echo [%ENVNAME%] done. Activate with:  conda activate %ENVNAME%
echo Note: mtsd-base / mtsd-la / MDWD download gated Hugging Face weights on first use - run "hf auth login" inside the env once (see Documents\PromptDetect.md).
exit /b 0

:usage
echo Usage: %~nx0 MDWD ^| mtsd-attrcls ^| mtsd-base ^| mtsd-la ^| all  [cpu]
exit /b 1
