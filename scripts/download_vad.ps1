#!/usr/bin/env pwsh
# Descarga el modelo Silero-VAD en formato GGML desde HuggingFace al
# directorio de modelos de oido. Lo usa whisper.cpp internamente para
# recortar silencios antes del encoder (reduce latencia en audios 10-30s
# con pausas).
#
# Uso (invocado por `oido` automáticamente al boot si el modelo falta):
#   .\scripts\download_vad.ps1 <URL> <Destino>
#
# Uso manual:
#   .\scripts\download_vad.ps1 `
#     https://huggingface.co/ggml-org/whisper-vad/resolve/main/ggml-silero-v5.1.2.bin `
#     $env:APPDATA\oido\models\ggml-silero-v5.1.2.bin

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$Url,
    [Parameter(Mandatory = $true, Position = 1)]
    [string]$Dest
)

$ErrorActionPreference = "Stop"

if (Test-Path $Dest) {
    Write-Host "[ok] ya existe: $Dest" -ForegroundColor Green
    exit 0
}

$destDir = Split-Path -Parent $Dest
if ($destDir -and -not (Test-Path $destDir)) {
    New-Item -ItemType Directory -Path $destDir -Force | Out-Null
}

Write-Host "[..] bajando modelo VAD" -ForegroundColor Cyan
Write-Host "    desde: $Url"
Write-Host "    hacia: $Dest"

try {
    Invoke-WebRequest -Uri $Url -OutFile $Dest -UseBasicParsing
} catch {
    Write-Host "[err] falló la descarga: $_" -ForegroundColor Red
    if (Test-Path $Dest) { Remove-Item $Dest -Force }
    exit 1
}

Write-Host "[ok] modelo VAD guardado en $Dest" -ForegroundColor Green