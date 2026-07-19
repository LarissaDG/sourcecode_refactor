#!/bin/bash
#SBATCH --job-name=portinari_zip32
#SBATCH --time=01:00:00
#SBATCH -N 1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=larissa.gomide@dcc.ufmg.br

# Recovery script: baixa o ZIP 32 do Portinari (obras 1001-1200)
# que falha por rate limit do Google Drive quando baixado em sequencia.

set -x

ROOT="/sonic_home/larissa.gomide/sourcecode_refactor"
VENV_DOWNLOAD="$ROOT/venv_download"
DATA_DIR="$ROOT/data/portinari"
ZIP_ID="1-3Q3xH-O-yuIwy504PR17xB8mdk-mqkI"
TMP_ZIP="/tmp/obras_de_1001_a_1200.zip"
TMP_EXTRACT="/tmp/portinari_extract_32"

export HF_HOME="/snfs1/larissa.gomide/hf_cache"
export TRANSFORMERS_CACHE="/snfs1/larissa.gomide/hf_cache"
export XDG_CACHE_HOME="/sonic_home/larissa.gomide/minha_home/.cache"
export MPLCONFIGDIR="/sonic_home/larissa.gomide/minha_home/.matplotlib"\n\ncd "$ROOT"

notify() {
    local code=$?
    if [ $code -eq 0 ]; then
        source "$VENV_DOWNLOAD/bin/activate"
        python3 scripts/manda_email.py \
            "ZIP 32 Portinari concluido - Phocus4" \
            "obras 1001-1200 extraidas para $DATA_DIR/Imagens/"
    else
        source "$VENV_DOWNLOAD/bin/activate"
        python3 scripts/manda_email.py \
            "ZIP 32 Portinari FALHOU - Phocus4" \
            "Erro (codigo $code). Verifique: $ROOT/slurm-${SLURM_JOB_ID}.out"
    fi
}
trap notify EXIT

source "$VENV_DOWNLOAD/bin/activate"

echo "Baixando ZIP 32: $ZIP_ID"
gdown "$ZIP_ID" --output "$TMP_ZIP" || { echo "ERRO: gdown falhou"; exit 1; }

echo "Extraindo para $DATA_DIR/Imagens/ ..."
rm -rf "$TMP_EXTRACT"
mkdir -p "$TMP_EXTRACT" "$DATA_DIR/Imagens"
unzip -q "$TMP_ZIP" -d "$TMP_EXTRACT"

count=0
for f in "$TMP_EXTRACT"/*.jpg "$TMP_EXTRACT"/*.jpeg "$TMP_EXTRACT"/*.png \
          "$TMP_EXTRACT"/**/*.jpg "$TMP_EXTRACT"/**/*.jpeg "$TMP_EXTRACT"/**/*.png; do
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
echo "=== FINALIZADO ===" && hostname
