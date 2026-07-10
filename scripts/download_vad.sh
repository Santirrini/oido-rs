#!/usr/bin/env bash
# Descarga el modelo Silero-VAD en formato GGML desde HuggingFace al
# directorio de modelos de oido. Lo usa whisper.cpp internamente para
# recortar silencios antes del encoder (reduce latencia en audios 10-30s
# con pausas).
#
# Uso (invocado por `oido` automáticamente al boot si el modelo falta):
#   ./scripts/download_vad.sh <URL> <Destino>
#
# Uso manual:
#   ./scripts/download_vad.sh \
#     https://huggingface.co/ggml-org/whisper-vad/resolve/main/ggml-silero-v5.1.2.bin \
#     "$HOME/.local/share/oido/models/ggml-silero-v5.1.2.bin"

set -euo pipefail

URL="${1:-}"
DEST="${2:-}"

if [[ -z "$URL" || -z "$DEST" ]]; then
    echo "[err] uso: $0 <URL> <Destino>" >&2
    exit 1
fi

if [[ -f "$DEST" ]]; then
    echo "[ok] ya existe: $DEST"
    exit 0
fi

DEST_DIR="$(dirname "$DEST")"
mkdir -p "$DEST_DIR"

echo "[..] bajando modelo VAD"
echo "    desde: $URL"
echo "    hacia: $DEST"

if command -v curl >/dev/null 2>&1; then
    curl -fL --retry 3 -o "$DEST" "$URL"
elif command -v wget >/dev/null 2>&1; then
    wget -O "$DEST" "$URL"
else
    echo "[err] ni curl ni wget disponibles; instalá uno de los dos." >&2
    exit 1
fi

echo "[ok] modelo VAD guardado en $DEST"