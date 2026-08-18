#!/bin/bash
#SBATCH --job-name=exp5d_temporal_micro
#SBATCH --time=06:00:00
#SBATCH -N 1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=larissa.gomide@dcc.ufmg.br

# Roda DEPOIS de slurm_extract_exp5d_frames.sh (precisa dos frames micro já
# extraídos em /snfs1/speed/larissa.gomide/data/temporal_micro/frames/).

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
            "✅ exp5d_temporal_micro CONCLUÍDO — Phocus4" \
            "Job finalizado com sucesso. Resultados em: /snfs1/speed/larissa.gomide/outputs/exp5d_temporal_micro/"
    else
        source "$VENV/bin/activate"
        python3 scripts/manda_email.py \
            "❌ exp5d_temporal_micro FALHOU — Phocus4" \
            "Job abortou com erro (código $code). Verifique o log: $ROOT/slurm-${SLURM_JOB_ID}.out"
    fi
}
trap notify EXIT

echo "--- sampling + scoring (ArtCLIP) ---"
source "$VENV_APDDV2/bin/activate"
python3 run.py --config configs/exp5d_temporal_micro.yaml --steps sampling,samples,scoring \
    || { echo "ERRO"; exit 1; }
deactivate

echo "=== FINALIZADO ===" && hostname
