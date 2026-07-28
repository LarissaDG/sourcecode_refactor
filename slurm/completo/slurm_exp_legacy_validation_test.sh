#!/bin/bash
#SBATCH --job-name=legacy_val_test
#SBATCH --time=01:00:00
#SBATCH -N 1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=larissa.gomide@dcc.ufmg.br

set -x

ROOT="/sonic_home/larissa.gomide/sourcecode_refactor"
VENV="/sonic_home/larissa.gomide/venv"
VENV_APDDV2="/sonic_home/larissa.gomide/apddv2"
DATA="/snfs1/speed/larissa.gomide/data"

module load cuda/11.8.0

export HOME="/sonic_home/larissa.gomide/casa"
export HF_HOME="/snfs1/speed/larissa.gomide/hf_cache"
export TRANSFORMERS_CACHE="/snfs1/speed/larissa.gomide/hf_cache"
export CLIP_CACHE="/snfs1/speed/larissa.gomide/hf_cache"
export XDG_CACHE_HOME="/sonic_home/larissa.gomide/casa/.cache"
export MPLCONFIGDIR="/sonic_home/larissa.gomide/casa/.matplotlib"

LEGACY_SMALL="$DATA/legacy_iccc/sampled_SMALL_with_gen_scored.csv"
LEGACY_BIG="$DATA/legacy_iccc/sampled_BIG_with_gen_scored.csv"
APDDV2_DIR="$DATA/apddv2/APDDv2images/"
OUT_DIR="/snfs1/speed/larissa.gomide/outputs/test_legacy_validation"

notify() {
    local code=$?
    if [ $code -eq 0 ]; then
        source "$VENV/bin/activate"
        python3 scripts/manda_email.py \
            "✅ legacy_val_test CONCLUÍDO — Phocus4" \
            "Teste finalizado. Resultados em: $OUT_DIR/figures_validation/"
    else
        source "$VENV/bin/activate"
        python3 scripts/manda_email.py \
            "❌ legacy_val_test FALHOU — Phocus4" \
            "Job abortou com erro (código $code). Verifique o log: $ROOT/slurm-${SLURM_JOB_ID}.out"
    fi
}
trap notify EXIT

echo "--- Fase 1: build pipeline_data (1 imagem) ---"
source "$VENV/bin/activate"
python3 scripts/run_legacy_validation.py \
    --config       configs/exp_legacy_validation.yaml \
    --legacy-small "$LEGACY_SMALL" \
    --legacy-big   "$LEGACY_BIG" \
    --apddv2-dir   "$APDDV2_DIR" \
    --out-dir      "$OUT_DIR" \
    --build-only   --test 1 \
    || { echo "ERRO build pipeline_data"; exit 1; }

python3 run.py --config configs/exp_legacy_validation.yaml --steps generation \
    || { echo "ERRO fase generation"; exit 1; }
deactivate

echo "--- Fase 2: scoring (ArtCLIP) ---"
source "$VENV_APDDV2/bin/activate"
python3 run.py --config configs/exp_legacy_validation.yaml --steps scoring \
    || { echo "ERRO fase scoring"; exit 1; }
deactivate

echo "--- Fase 3: comparação ---"
source "$VENV_APDDV2/bin/activate"
python3 scripts/run_legacy_validation.py \
    --config       configs/exp_legacy_validation.yaml \
    --legacy-small "$LEGACY_SMALL" \
    --legacy-big   "$LEGACY_BIG" \
    --apddv2-dir   "$APDDV2_DIR" \
    --out-dir      "$OUT_DIR" \
    --skip-run     --test 1 \
    || { echo "ERRO fase comparação"; exit 1; }
deactivate

echo "=== FINALIZADO ===" && hostname
