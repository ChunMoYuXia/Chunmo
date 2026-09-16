@echo off
chcp 65001 >nul
rem QuantAI 一键启动脚本
rem 优先使用 conda 环境 Xm_env；若不存在则用当前 python
title QuantAI 量化研报
cd /d "%~dp0"

echo [QuantAI] 正在启动...

rem 尝试激活 conda 环境
if exist "C:\Users\chunmo\.conda\envs\Xm_env\python.exe" (
    set "PYTHON=C:\Users\chunmo\.conda\envs\Xm_env\python.exe"
) else if exist "D:\Miniconda3\envs\Xm_env\python.exe" (
    set "PYTHON=D:\Miniconda3\envs\Xm_env\python.exe"
) else (
    set "PYTHON=python"
)

echo [QuantAI] 使用 Python: %PYTHON%
"%PYTHON%" run.py

if errorlevel 1 (
    echo.
    echo [QuantAI] 启动失败，请检查依赖是否安装（pip install -r requirements.txt）
    pause
)
