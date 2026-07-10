#!/usr/bin/env pwsh
# Descarga `ggml-base.bin` (modelo multilingüe por defecto) desde
# huggingface.co/ggerganov/whisper.cpp al directorio de modelos de oido.
#
# Uso:
#   .\scripts\download_model.ps1                  # default: ggml-base.bin
#   .\scripts\download_model.ps1 -Model ggml-small.bin
#   .\scripts\download_model.ps1 -TargetDir D:\models
#
# Variables de entorno:
#   OIDO_MODELS_DIR  → si está, se usa en vez del default de la app.

[CmdletBinding()]
param(
    [string]$Model = "ggml-base.bin",
    [string]$TargetDir = ""
)

$ErrorActionPreference = "Stop"

if (-not $TargetDir) {
    if ($env:OIDO_MODELS_DIR) {
        $TargetDir = $env:OIDO_MODELS_DIR
    } else {
        $dataDir = if ($env:APPDATA) { $env:APPDATA } else { Join-Path $HOME ".local/share" }
        $TargetDir = Join-Path $dataDir "oido\models"
    }
}

if (-not (Test-Path $TargetDir)) {
    New-Item -ItemType Directory -Path $TargetDir -Force | Out-Null
}

$dest = Join-Path $TargetDir $Model
$url  = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/$Model"

if (Test-Path $dest) {
    Write-Host "[ok] ya existe: $dest" -ForegroundColor Green
    exit 0
}

Write-Host "[..] bajando $Model" -ForegroundColor Cyan
Write-Host "    desde: $url"
Write-Host "    hacia: $dest"

try {
    Invoke-WebRequest -Uri $url -OutFile $dest -UseBasicParsing
} catch {
    Write-Host "[err] falló la descarga: $_" -ForegroundColor Red
    if (Test-Path $dest) { Remove-Item $dest -Force }
    exit 1
}

Write-Host "[ok] modelo guardado en $dest" -ForegroundColor Green
Write-Host ""
Write-Host "Próximo paso: cargo run --release -p oido" -ForegroundColor Yellow