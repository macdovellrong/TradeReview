@echo off
setlocal EnableExtensions
chcp 65001 >nul

set "ROOT=%~dp0"
set "PYTHON_EXE=%ROOT%.venv\Scripts\python.exe"
set "SCRIPT=%ROOT%tools\preprocess_qdm_tick_csv.py"
set "OUTPUT_DIR=%ROOT%data"

pushd "%ROOT%" >nul || exit /b 1

if not exist "%PYTHON_EXE%" (
    echo [ERROR] Missing virtualenv python: "%PYTHON_EXE%"
    popd >nul
    call :maybe_pause
    exit /b 1
)

if not exist "%SCRIPT%" (
    echo [ERROR] Missing script: "%SCRIPT%"
    popd >nul
    call :maybe_pause
    exit /b 1
)

if "%~1"=="" (
    set /p "CSV_FILE=Enter QDM CSV path (or drag a CSV onto this BAT): "
    if not defined CSV_FILE (
        echo [ERROR] No CSV path provided.
        popd >nul
        call :maybe_pause
        exit /b 1
    )
    if not exist "%CSV_FILE%" (
        echo [ERROR] CSV file not found: "%CSV_FILE%"
        popd >nul
        call :maybe_pause
        exit /b 1
    )
    echo [INFO] Output directory: "%OUTPUT_DIR%"
    "%PYTHON_EXE%" "%SCRIPT%" "%CSV_FILE%" --output-dir "%OUTPUT_DIR%"
) else (
    if not exist "%~1" (
        echo [ERROR] CSV file not found: "%~1"
        popd >nul
        call :maybe_pause
        exit /b 1
    )
    echo [INFO] Output directory: "%OUTPUT_DIR%"
    "%PYTHON_EXE%" "%SCRIPT%" %* --output-dir "%OUTPUT_DIR%"
)

set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" (
    echo [ERROR] Conversion failed with exit code %EXIT_CODE%.
    popd >nul
    call :maybe_pause
    exit /b %EXIT_CODE%
)

echo [INFO] Conversion finished.
popd >nul
call :maybe_pause
exit /b 0

:maybe_pause
if /I "%NO_PAUSE%"=="1" exit /b 0
pause
exit /b 0
