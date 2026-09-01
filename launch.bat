@echo off
chcp 65001 >nul
cd /d "%~dp0"
title SolidForge 3D

echo ========================================================
echo   ⚡ SolidForge 3D (ソリッドフォージ 3D) 起動中...
echo ========================================================
echo.

:: 1. ユーザー環境の Python 3.10 を直接実行
if exist "%LOCALAPPDATA%\Programs\Python\Python310\python.exe" (
    echo [INFO] Python 3.10 を検出しました。起動します...
    "%LOCALAPPDATA%\Programs\Python\Python310\python.exe" "%~dp0main.py"
    if errorlevel 1 goto :error
    goto :end
)

:: 2. py ランチャーによる実行
where py >nul 2>&1
if %errorlevel% equ 0 (
    echo [INFO] py ランチャーで起動します...
    py -3.10 "%~dp0main.py"
    if errorlevel 1 goto :error
    goto :end
)

:: 3. システム PATH の python による実行
where python >nul 2>&1
if %errorlevel% equ 0 (
    echo [INFO] システム python で起動します...
    python "%~dp0main.py"
    if errorlevel 1 goto :error
    goto :end
)

:error
echo.
echo ========================================================
echo [ERROR] アプリケーションの起動に失敗しました。
echo ========================================================
echo.
pause

:end
