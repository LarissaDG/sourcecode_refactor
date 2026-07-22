#!/bin/bash
#SBATCH --job-name=unzip_apddv2
#SBATCH --time=02:00:00
#SBATCH -N 1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=larissa.gomide@dcc.ufmg.br

ROOT="/sonic_home/larissa.gomide/sourcecode_refactor"
VENV="/sonic_home/larissa.gomide/venv"
APDDV2_DIR="/snfs1/speed/larissa.gomide/data/apddv2"
IMG_DIR="$APDDV2_DIR/APDDv2images"

notify() {
    local code=$?
    source "$VENV/bin/activate"
    if [ $code -eq 0 ]; then
        python3 "$ROOT/scripts/manda_email.py" \
            "✅ unzip APDDv2 concluído" \
            "Total de imagens: $(ls "$IMG_DIR" | wc -l)"
    else
        python3 "$ROOT/scripts/manda_email.py" \
            "❌ unzip APDDv2 FALHOU" \
            "Erro (código $code). Verifique: $ROOT/slurm-${SLURM_JOB_ID}.out"
    fi
}
trap notify EXIT

set -euo pipefail

mkdir -p "$IMG_DIR"

for PART in part1 part2; do
    ZIP="$APDDV2_DIR/APDDv2images_${PART}.zip"
    echo "Descomprimindo $ZIP..."
    unzip -o "$ZIP" -d "$IMG_DIR"
    echo "  $(ls "$IMG_DIR" | wc -l) imagens até agora."
done

N=$(ls "$IMG_DIR" | wc -l)
echo "Total final de imagens: $N"
