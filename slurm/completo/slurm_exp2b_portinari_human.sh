#!/bin/bash
#SBATCH --job-name=exp2b_portinari
#SBATCH --time=12:00:00
#SBATCH -N 1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=larissa.gomide@dcc.ufmg.br

set -x

ROOT="/sonic_home/larissa.gomide/sourcecode_refactor"
VENV="/sonic_home/larissa.gomide/venv"
VENV_APDDV2="/sonic_home/larissa.gomide/apddv2"

module load cuda/11.8.0

export HOME="/sonic_home/larissa.gomide/casa"
export HF_HOME="/snfs1/speed/larissa.gomide/hf_cache"
export TRANSFORMERS_CACHE="/snfs1/speed/larissa.gomide/hf_cache"
export CLIP_CACHE="/snfs1/speed/larissa.gomide/hf_cache"
export XDG_CACHE_HOME="/sonic_home/larissa.gomide/casa/.cache"
export MPLCONFIGDIR="/sonic_home/larissa.gomide/casa/.matplotlib"




notify() {
    local code=$?
    if [ $code -eq 0 ]; then
        source "$VENV/bin/activate"
        python3 scripts/manda_email.py \
            "✅ exp2b_portinari_human CONCLUÍDO — Phocus4" \
            "Job finalizado com sucesso. Resultados em: $ROOT/outputs/exp2b_portinari_human/"
    else
        source "$VENV/bin/activate"
        python3 scripts/manda_email.py \
            "❌ exp2b_portinari_human FALHOU — Phocus4" \
            "Job abortou com erro (código $code). Verifique o log: $ROOT/slurm-${SLURM_JOB_ID}.out"
    fi
}
trap notify EXIT

echo "--- Fase 1: sampling + generation (Janus, sem captioning) ---"
source "$VENV/bin/activate"
python3 run.py --config configs/exp2b_portinari_human.yaml --steps sampling,generation \
    || { echo "ERRO fase Janus"; exit 1; }
deactivate

echo "--- Fase 2: scoring (ArtCLIP) ---"
source "$VENV_APDDV2/bin/activate"
python3 run.py --config configs/exp2b_portinari_human.yaml --steps scoring \
    || { echo "ERRO fase ArtCLIP"; exit 1; }
deactivate

echo "=== FINALIZADO ===" && hostname
