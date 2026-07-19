#!/bin/bash
# Script de recovery: baixa e extrai o ZIP 32 do Portinari (obras 1001-1200)
# que falha por rate limit do Google Drive quando baixado em sequencia.
# Uso: bash scripts/download_portinari_zip32.sh

set -e

ROOT="/sonic_home/larissa.gomide/sourcecode_refactor"
DATA_DIR="$ROOT/data/portinari"
VENV_DOWNLOAD="$ROOT/venv_download"
ZIP_ID="1-3Q3xH-O-yuIwy504PR17xB8mdk-mqkI"
ZIP_NAME="obras_de_1001_a_1200.zip"
TMP_ZIP="/tmp/$ZIP_NAME"
TMP_EXTRACT="/tmp/portinari_extract_32"

source "$VENV_DOWNLOAD/bin/activate"

echo "Baixando ZIP 32: $ZIP_ID"
gdown "$ZIP_ID" --output "$TMP_ZIP"

echo "Extraindo para $DATA_DIR/Imagens/ ..."
rm -rf "$TMP_EXTRACT"
mkdir -p "$TMP_EXTRACT" "$DATA_DIR/Imagens"
unzip -q "$TMP_ZIP" -d "$TMP_EXTRACT"

count=0
for f in "$TMP_EXTRACT"/**/*.jpg "$TMP_EXTRACT"/**/*.jpeg "$TMP_EXTRACT"/**/*.png; do
    [ -f "$f" ] || continue
    dest="$DATA_DIR/Imagens/$(basename "$f")"
    if [ ! -f "$dest" ]; then
        cp "$f" "$dest"
        count=$((count + 1))
    fi
done

rm -rf "$TMP_EXTRACT" "$TMP_ZIP"

echo "$ZIP_ID" >> "$DATA_DIR/.downloaded_ids"
sort -u "$DATA_DIR/.downloaded_ids" -o "$DATA_DIR/.downloaded_ids"

echo "Pronto: $count imagens extraidas do ZIP 32."
