#!/bin/bash
#SBATCH --job-name=unzip_apddv2
#SBATCH --time=01:00:00
#SBATCH -N 1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=larissa.gomide@dcc.ufmg.br

ROOT="/sonic_home/larissa.gomide/sourcecode_refactor"
VENV="/sonic_home/larissa.gomide/venv"
APDDV2_DIR="/snfs1/speed/larissa.gomide/data/apddv2"

notify() {
    local code=$?
    source "$VENV/bin/activate"
    if [ $code -eq 0 ]; then
        python3 "$ROOT/scripts/manda_email.py" \
            "✅ unzip APDDv2 concluído" \
            "Total de imagens: $(ls "$APDDV2_DIR/APDDv2images/APDDv2images" | wc -l)"
    else
        python3 "$ROOT/scripts/manda_email.py" \
            "❌ unzip APDDv2 FALHOU" \
            "Erro (código $code). Verifique: $ROOT/slurm-${SLURM_JOB_ID}.out"
    fi
}
trap notify EXIT

set -euo pipefail

echo "Descomprimindo APDDv2images_full.zip..."
unzip -o "$APDDV2_DIR/APDDv2images_full.zip" -d "$APDDV2_DIR/APDDv2images/"

N=$(ls "$APDDV2_DIR/APDDv2images/APDDv2images" | wc -l)
echo "Total de imagens apos unzip: $N"
