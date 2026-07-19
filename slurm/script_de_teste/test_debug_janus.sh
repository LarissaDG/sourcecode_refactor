#!/bin/bash
#SBATCH --job-name=debug_janus
#SBATCH --time=02:00:00
#SBATCH -N 1

set -x

ROOT="/sonic_home/larissa.gomide/sourcecode_refactor"
VENV="/sonic_home/larissa.gomide/venv"

module load cuda/11.8.0

export HOME="/sonic_home/larissa.gomide/casa"
export HF_HOME="/snfs1/speed/larissa.gomide/hf_cache"
export TRANSFORMERS_CACHE="/snfs1/speed/larissa.gomide/hf_cache"
export CLIP_CACHE="/snfs1/speed/larissa.gomide/hf_cache"
export XDG_CACHE_HOME="/sonic_home/larissa.gomide/casa/.cache"
export MPLCONFIGDIR="/sonic_home/larissa.gomide/casa/.matplotlib"

cd "$ROOT"

notify() {
    local code=$?
    source "$VENV/bin/activate"
    if [ $code -eq 0 ]; then
        python3 scripts/manda_email.py \
            "✅ debug_janus CONCLUÍDO — Phocus4" \
            "Janus-Pro-1B rodou com sucesso. Verifique o log: $ROOT/slurm-${SLURM_JOB_ID}.out"
    else
        python3 scripts/manda_email.py \
            "❌ debug_janus FALHOU — Phocus4" \
            "Job abortou com erro (código $code). Verifique o log: $ROOT/slurm-${SLURM_JOB_ID}.out"
    fi
}
trap notify EXIT

source "$VENV/bin/activate"
python3 scripts/debug_janus.py

echo "=== FINALIZADO ===" && hostname
