#!/bin/bash
#SBATCH --job-name=exp4_noise
#SBATCH --time=08:00:00
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
            "✅ exp4_noise CONCLUÍDO — Phocus4" \
            "Job finalizado com sucesso. Resultados em: /snfs1/speed/larissa.gomide/outputs/exp4_noise/"
    else
        source "$VENV/bin/activate"
        python3 scripts/manda_email.py \
            "❌ exp4_noise FALHOU — Phocus4" \
            "Job abortou com erro (código $code). Verifique o log: $ROOT/slurm-${SLURM_JOB_ID}.out"
    fi
}
trap notify EXIT

echo "--- sampling + scoring (ArtCLIP) ---"
source "$VENV_APDDV2/bin/activate"
python3 run.py --config configs/exp4_noise.yaml --steps sampling,scoring \
    || { echo "ERRO"; exit 1; }
deactivate

echo "=== FINALIZADO ===" && hostname
