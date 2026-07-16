#!/bin/bash
#SBATCH --job-name=exp1_apdd
#SBATCH --time=16:00:00
#SBATCH -N 1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=larissa.gomide@dcc.ufmg.br

set -x

ROOT="/sonic_home/larissa.gomide/sourcecode_refactor"
VENV="/sonic_home/larissa.gomide/venv"
VENV_APDDV2="/sonic_home/larissa.gomide/apddv2"

module load python3.12.1
module load cuda/11.8.0

export HOME="/sonic_home/larissa.gomide/minha_home"
export HF_HOME="$HOME/.cache/huggingface"
export TRANSFORMERS_CACHE="$HOME/.cache/huggingface"
export XDG_CACHE_HOME="$HOME/.cache"
export MPLCONFIGDIR="$HOME/.matplotlib"

cd "$ROOT"

echo "--- Fase 1: sampling + captioning + generation (Janus) ---"
source "$VENV/bin/activate"
python3 run.py --config configs/exp1_apdd.yaml --steps sampling,captioning,generation \
    || { echo "ERRO fase Janus"; deactivate; exit 1; }
deactivate

echo "--- Fase 2: scoring (ArtCLIP) ---"
source "$VENV_APDDV2/bin/activate"
python3 run.py --config configs/exp1_apdd.yaml --steps scoring \
    || { echo "ERRO fase ArtCLIP"; deactivate; exit 1; }
deactivate

source "$VENV/bin/activate"
python3 scripts/manda_email.py "exp1_apdd concluÃ­do â€” Phocus4" "Resultados em: $ROOT/outputs/exp1_apdd/"
deactivate

echo "=== FINALIZADO ===" && hostname
