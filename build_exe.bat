@echo off
setlocal enabledelayedexpansion

REM use uv 
where uv >nul 2>nul
if %errorlevel%==0 (
    uv run python build_exe.py %*
) else (
    python build_exe.py %*
)

if errorlevel 1 (
    echo.
    echo [build] FAILED. See messages above.
    pause
    exit /b 1
)

echo.
echo [build] DONE.
pause
