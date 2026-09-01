# SolidForge 3D - PowerShell Launcher
$ErrorActionPreference = "Continue"
Set-Location -Path $PSScriptRoot

Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "  ⚡ SolidForge 3D (ソリッドフォージ 3D) 起動中..." -ForegroundColor Green
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host ""

$pythonPath = "C:\Users\Gisa_M3\AppData\Local\Programs\Python\Python310\python.exe"
if (-not (Test-Path $pythonPath)) {
    $pythonPath = (Get-Command python -ErrorAction SilentlyContinue).Source
}

if (-not $pythonPath) {
    Write-Host "[ERROR] Python 実行環境が見つかりませんでした。" -ForegroundColor Red
    Read-Host "Enterキーを押して終了してください..."
    exit 1
}

Write-Host "[INFO] 使用する Python: $pythonPath" -ForegroundColor DarkGray
Write-Host "[INFO] アプリケーションを起動しています..." -ForegroundColor Yellow

& $pythonPath "$PSScriptRoot\main.py"
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "[ERROR] アプリケーションがエラーで終了しました (ExitCode: $LASTEXITCODE)" -ForegroundColor Red
    Read-Host "Enterキーを押して終了してください..."
}
