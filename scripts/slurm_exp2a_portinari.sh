#!/bin/bash
#SBATCH --job-name=exp2a_portinari
#SBATCH --time=16:00:00
#SBATCH -N 1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=larissa.gomide@dcc.ufmg.br

set -x

ROOT="/sonic_home/larissa.gomide/sourcecode_refactor"
VENV="/sonic_home/larissa.gomide/venv"
VENV_APDDV2="/sonic_home/larissa.gomide/apddv2"

module load cuda/11.8.0

export HOME="/sonic_home/larissa.gomide/minha_home"
export HF_HOME="$HOME/.cache/huggingface"
export TRANSFORMERS_CACHE="$HOME/.cache/huggingface"
export XDG_CACHE_HOME="$HOME/.cache"
export MPLCONFIGDIR="$HOME/.matplotlib"

cd "$ROOT"

notify() {
    local code=$?
    if [ $code -eq 0 ]; then
        source "$VENV/bin/activate"
        python3 scripts/manda_email.py \
            "✅ exp2a_portinari CONCLUÍDO — Phocus4" \
            "Job finalizado com sucesso. Resultados em: $ROOT/outputs/exp2a_portinari/"
    else
        source "$VENV/bin/activate"
        python3 scripts/manda_email.py \
            "❌ exp2a_portinari FALHOU — Phocus4" \
            "Job abortou com erro (código $code). Verifique o log: $ROOT/slurm-${SLURM_JOB_ID}.out"
    fi
}
trap notify EXIT

echo "--- Fase 1: sampling + captioning + generation (Janus) ---"
source "$VENV/bin/activate"
python3 run.py --config configs/exp2a_portinari.yaml --steps sampling,captioning,generation \
    || { echo "ERRO fase Janus"; exit 1; }
deactivate

echo "--- Fase 2: scoring (ArtCLIP) ---"
source "$VENV_APDDV2/bin/activate"
python3 run.py --config configs/exp2a_portinari.yaml --steps scoring \
    || { echo "ERRO fase ArtCLIP"; exit 1; }
deactivate

echo "=== FINALIZADO ===" && hostname
