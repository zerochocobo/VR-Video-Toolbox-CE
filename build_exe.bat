@echo off
setlocal enabledelayedexpansion
REM ============================================================
REM  VR Video Toolbox - onedir 打包入口
REM  实际构建逻辑在 build_exe.py：PyInstaller 主程序 + 独立 DLNA exe，
REM  并把 CUDA/cuDNN/nvrtc/头文件全部捆绑进 dist，脱离系统 CUDA Toolkit 也能运行。
REM ============================================================

REM 优先用 uv 启动；如无 uv 则用当前 python
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
