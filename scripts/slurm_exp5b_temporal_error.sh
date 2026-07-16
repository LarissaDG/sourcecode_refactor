#!/bin/bash
#SBATCH --job-name=exp5b_temporal_err
#SBATCH --time=06:00:00
#SBATCH -N 1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=larissa.gomide@dcc.ufmg.br

set -x

ROOT="/sonic_home/larissa.gomide/sourcecode_refactor"
VENV_APDDV2="/sonic_home/larissa.gomide/apddv2"

module load python3.12.1
module load cuda/11.8.0

export HOME="/sonic_home/larissa.gomide/minha_home"
export HF_HOME="$HOME/.cache/huggingface"
export TRANSFORMERS_CACHE="$HOME/.cache/huggingface"
export XDG_CACHE_HOME="$HOME/.cache"
export MPLCONFIGDIR="$HOME/.matplotlib"

cd "$ROOT"

echo "--- sampling + scoring (ArtCLIP) ---"
source "$VENV_APDDV2/bin/activate"
python3 run.py --config configs/exp5b_temporal_error.yaml --steps sampling,scoring \
    || { echo "ERRO"; deactivate; exit 1; }

python3 scripts/manda_email.py "exp5b_temporal_error concluÃ­do â€” Phocus4" "Resultados em: $ROOT/outputs/exp5b_temporal_error/"
deactivate

echo "=== FINALIZADO ===" && hostname
