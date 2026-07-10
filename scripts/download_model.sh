#!/usr/bin/env bash
# Descarga `ggml-base.bin` (modelo multilingüe por defecto) desde
# huggingface.co/ggerganov/whisper.cpp al directorio de modelos de oido.
#
# Uso:
#   ./scripts/download_model.sh                  # default: ggml-base.bin
#   ./scripts/download_model.sh ggml-small.bin   # otro modelo
#   OIDO_MODELS_DIR=/tmp/models ./scripts/download_model.sh
#
# Salida estándar en Linux: $XDG_DATA_HOME/oido/models o
# $HOME/.local/share/oido/models.
# En macOS: $HOME/Library/Application Support/oido/models.

set -euo pipefail

MODEL="${1:-ggml-base.bin}"

if [[ -n "${OIDO_MODELS_DIR:-}" ]]; then
    TARGET_DIR="$OIDO_MODELS_DIR"
elif [[ "$(uname -s)" == "Darwin" ]]; then
    TARGET_DIR="$HOME/Library/Application Support/oido/models"
else
    TARGET_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/oido/models"
fi

mkdir -p "$TARGET_DIR"

DEST="$TARGET_DIR/$MODEL"
URL="https://huggingface.co/ggerganov/whisper.cpp/resolve/main/$MODEL"

if [[ -f "$DEST" ]]; then
    echo "[ok] ya existe: $DEST"
    exit 0
fi

echo "[..] bajando $MODEL"
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

echo "[ok] modelo guardado en $DEST"
echo ""
echo "Próximo paso: cargo run --release -p oido"