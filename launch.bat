@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul
cd /d "%~dp0"

title SolidForge 3D - Studio Launcher

echo ========================================================
echo   ⚡ SolidForge 3D (ソリッドフォージ 3D) 起動中...
echo ========================================================
echo.

:: 1. Python 実行可能ファイルの検出 (優先順位付け)
set "PYTHON_EXE="

if exist "C:\Users\Gisa_M3\AppData\Local\Programs\Python\Python310\python.exe" (
    set "PYTHON_EXE=C:\Users\Gisa_M3\AppData\Local\Programs\Python\Python310\python.exe"
)

if "%PYTHON_EXE%"=="" (
    where py >nul 2>&1
    if !errorlevel! equ 0 (
        set "PYTHON_EXE=py -3.10"
    )
)

if "%PYTHON_EXE%"=="" (
    where python >nul 2>&1
    if !errorlevel! equ 0 (
        set "PYTHON_EXE=python"
    )
)

if "%PYTHON_EXE%"=="" (
    echo [ERROR] 適切な Python 実行環境が見つかりませんでした。
    echo Python 3.10+ がインストールされているかご確認ください。
    echo.
    pause
    exit /b 1
)

echo [INFO] 使用する Python: %PYTHON_EXE%
echo [INFO] アプリケーションを起動しています...
echo.

:: 2. アプリケーションの実行 (エラー発生時は画面を保持)
%PYTHON_EXE% main.py
if !errorlevel! neq 0 (
    echo.
    echo ========================================================
    echo [ERROR] アプリケーションの実行が異常終了しました (コード: !errorlevel!)。
    echo 上記のエラーメッセージをご確認ください。
    echo ========================================================
    echo.
    pause
)
