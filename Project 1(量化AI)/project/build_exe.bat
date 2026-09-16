@echo off
chcp 65001 >nul
rem QuantAI exe 打包脚本
rem 使用 PyInstaller 将 Streamlit 应用打包为单目录 exe
title QuantAI 打包
cd /d "%~dp0"

echo [QuantAI] 开始打包 exe...
echo [QuantAI] 注意：打包可能需要 5-10 分钟，请耐心等待

pyinstaller quantai.spec --noconfirm --clean

if errorlevel 1 (
    echo.
    echo [QuantAI] 打包失败！请检查错误信息。
    echo [QuantAI] 可能需要：pip install -r requirements.txt
    pause
    exit /b 1
)

echo.
echo [QuantAI] 打包完成！
echo [QuantAI] 可执行文件位于：dist\QuantAI\QuantAI.exe
echo [QuantAI] 双击 QuantAI.exe 即可启动
pause
