#!/bin/bash
# Baixa as imagens do APDDv2 via sparse clone do repositório ICCC
# Uso: bash scripts/download_apddv2_images.sh [--out /caminho/destino]

set -e

OUT_DIR="/snfs1/speed/larissa.gomide/data/apddv2"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --out) OUT_DIR="$2"; shift ;;
    esac
    shift
done

IMAGES_DIR="$OUT_DIR/APDDv2images"
CLONE_DIR="$(mktemp -d)/ICCC"

echo "=== Baixando APDDv2images para $IMAGES_DIR ==="

mkdir -p "$IMAGES_DIR"

echo "--- Clonando repositório (sparse) ---"
git clone --filter=blob:none --sparse https://github.com/LarissaDG/ICCC.git "$CLONE_DIR"
cd "$CLONE_DIR"
git sparse-checkout set APDDv2images

echo "--- Movendo imagens para $IMAGES_DIR ---"
mv "$CLONE_DIR/APDDv2images/"* "$IMAGES_DIR/"

echo "--- Limpando clone temporário ---"
rm -rf "$CLONE_DIR"

echo "=== Concluído: $(ls "$IMAGES_DIR" | wc -l) imagens em $IMAGES_DIR ==="
