#!/bin/bash
#SBATCH --job-name=test_exp1_apdd
#SBATCH --time=02:00:00
#SBATCH -N 1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=larissa.gomide@dcc.ufmg.br

set -x

ROOT="/sonic_home/larissa.gomide/sourcecode_refactor"
VENV="/sonic_home/larissa.gomide/venv"
VENV_APDDV2="/sonic_home/larissa.gomide/apddv2"

module load cuda/11.8.0

export HOME="/sonic_home/larissa.gomide/minha_home"
export HF_HOME="/snfs1/speed/larissa.gomide/hf_cache"
export TRANSFORMERS_CACHE="/snfs1/speed/larissa.gomide/hf_cache"
export CLIP_CACHE="/snfs1/speed/larissa.gomide/hf_cache"
export XDG_CACHE_HOME="/sonic_home/larissa.gomide/minha_home/.cache"
export MPLCONFIGDIR="/sonic_home/larissa.gomide/minha_home/.matplotlib"




notify() {
    local code=$?
    if [ $code -eq 0 ]; then
        source "$VENV/bin/activate"
        python3 scripts/manda_email.py \
            "✅ [TESTE] exp1_apdd CONCLUÍDO — Phocus4" \
            "Job de teste finalizado com sucesso. Resultados em: $ROOT/outputs/test_exp1_apdd/"
    else
        source "$VENV/bin/activate"
        python3 scripts/manda_email.py \
            "❌ [TESTE] exp1_apdd FALHOU — Phocus4" \
            "Job de teste abortou com erro (código $code). Verifique o log: $ROOT/slurm-${SLURM_JOB_ID}.out"
    fi
}
trap notify EXIT

echo "--- Fase 1: sampling + captioning + generation (Janus) [MODO TESTE: 5 amostras] ---"
source "$VENV/bin/activate"
python3 run.py --config configs/exp1_apdd.yaml --steps sampling,captioning,generation --test \
    || { echo "ERRO fase Janus"; exit 1; }
deactivate

echo "--- Fase 2: scoring (ArtCLIP) ---"
source "$VENV_APDDV2/bin/activate"
python3 run.py --config configs/exp1_apdd.yaml --steps scoring --test \
    || { echo "ERRO fase ArtCLIP"; exit 1; }
deactivate

echo "=== FINALIZADO ===" && hostname
