#!/bin/bash
#SBATCH --job-name=exp_legacy_validation
#SBATCH --time=16:00:00
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
APDDV2_DIR="$DATA/apddv2/"
OUT_DIR="/snfs1/speed/larissa.gomide/outputs/exp_legacy_validation"

notify() {
    local code=$?
    if [ $code -eq 0 ]; then
        source "$VENV/bin/activate"
        python3 scripts/manda_email.py \
            "✅ exp_legacy_validation CONCLUÍDO — Phocus4" \
            "Validação finalizada. Resultados em: $OUT_DIR/figures_validation/"
    else
        source "$VENV/bin/activate"
        python3 scripts/manda_email.py \
            "❌ exp_legacy_validation FALHOU — Phocus4" \
            "Job abortou com erro (código $code). Verifique o log: $ROOT/slurm-${SLURM_JOB_ID}.out"
    fi
}
trap notify EXIT

# ── Fase 1: generation (Janus-1B + Janus-7B) com as 502 imagens legadas ──
echo "--- Fase 1: build pipeline_data + generation (Janus) ---"
source "$VENV/bin/activate"
python3 scripts/run_legacy_validation.py \
    --config       configs/exp_legacy_validation.yaml \
    --legacy-small "$LEGACY_SMALL" \
    --legacy-big   "$LEGACY_BIG" \
    --apddv2-dir   "$APDDV2_DIR" \
    --out-dir      "$OUT_DIR" \
    --skip-run \
    || { echo "ERRO build pipeline_data"; exit 1; }

python3 run.py --config configs/exp_legacy_validation.yaml --steps generation \
    || { echo "ERRO fase generation"; exit 1; }
deactivate

# ── Fase 2: scoring (ArtCLIP) ─────────────────────────────────────────────
echo "--- Fase 2: scoring (ArtCLIP) ---"
source "$VENV_APDDV2/bin/activate"
python3 run.py --config configs/exp_legacy_validation.yaml --steps scoring \
    || { echo "ERRO fase scoring"; exit 1; }
deactivate

# ── Fase 3: comparação e geração de gráficos ──────────────────────────────
echo "--- Fase 3: comparação legacy vs novo pipeline ---"
source "$VENV/bin/activate"
python3 scripts/run_legacy_validation.py \
    --config       configs/exp_legacy_validation.yaml \
    --legacy-small "$LEGACY_SMALL" \
    --legacy-big   "$LEGACY_BIG" \
    --apddv2-dir   "$APDDV2_DIR" \
    --out-dir      "$OUT_DIR" \
    --skip-run \
    || { echo "ERRO fase comparação"; exit 1; }
deactivate

echo "=== FINALIZADO ===" && hostname
