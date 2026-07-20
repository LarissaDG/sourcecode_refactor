#!/bin/bash
#SBATCH --job-name=unzip_apddv2
#SBATCH --time=01:00:00
#SBATCH -N 1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=larissa.gomide@dcc.ufmg.br

set -euo pipefail

APDDV2_DIR="/snfs1/speed/larissa.gomide/data/apddv2"

echo "Descomprimindo APDDv2images.zip..."
unzip -o "$APDDV2_DIR/APDDv2images.zip" -d "$APDDV2_DIR/APDDv2images/"

N=$(ls "$APDDV2_DIR/APDDv2images" | wc -l)
echo "Total de imagens apÃ³s unzip: $N"
