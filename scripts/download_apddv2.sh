#!/bin/bash
# Baixa imagens e CSVs do APDDv2
# Uso: bash scripts/download_apddv2.sh [--out /caminho/destino]

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

mkdir -p "$OUT_DIR" "$IMAGES_DIR"

echo "=== [1/3] Baixando CSVs do APDDv2 ==="
curl -L "https://raw.githubusercontent.com/BestiVictory/APDDv2/main/APDDv2-10023.csv" \
    -o "$OUT_DIR/APDDv2-10023.csv"
echo "  APDDv2-10023.csv: $(wc -l < "$OUT_DIR/APDDv2-10023.csv") linhas"

curl -L "https://raw.githubusercontent.com/BestiVictory/APDDv2/main/filesource.csv" \
    -o "$OUT_DIR/filesource.csv"
echo "  filesource.csv: $(wc -l < "$OUT_DIR/filesource.csv") linhas"

echo "=== [2/3] Clonando imagens do ICCC (sparse clone) ==="
git clone --filter=blob:none --sparse https://github.com/LarissaDG/ICCC.git "$CLONE_DIR"
cd "$CLONE_DIR"
git sparse-checkout set APDDv2images

echo "=== [3/3] Movendo imagens para $IMAGES_DIR ==="
mv "$CLONE_DIR/APDDv2images/"* "$IMAGES_DIR/"
rm -rf "$CLONE_DIR"

echo "=== Concluído ==="
echo "  Imagens: $(ls "$IMAGES_DIR" | wc -l)"
echo "  CSVs:    $(ls "$OUT_DIR"/*.csv)"
