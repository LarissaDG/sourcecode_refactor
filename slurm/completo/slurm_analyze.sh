#!/bin/bash
#SBATCH --job-name=analyze_samples
#SBATCH --time=00:30:00
#SBATCH -N 1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=larissa.gomide@dcc.ufmg.br

set -x

ROOT="/sonic_home/larissa.gomide/sourcecode_refactor"
VENV_APDDV2="/sonic_home/larissa.gomide/apddv2"

export HOME="/sonic_home/larissa.gomide/casa"
export MPLCONFIGDIR="/sonic_home/larissa.gomide/casa/.matplotlib"
export XDG_CACHE_HOME="/sonic_home/larissa.gomide/casa/.cache"

notify() {
    local code=$?
    source "$VENV_APDDV2/bin/activate"
    if [ $code -eq 0 ]; then
        python3 scripts/manda_email.py \
            "✅ analyze_samples CONCLUÍDO — Phocus4" \
            "Samples gerados em: /snfs1/speed/larissa.gomide/reports/samples/"
    else
        python3 scripts/manda_email.py \
            "❌ analyze_samples FALHOU — Phocus4" \
            "Erro (código $code). Log: $ROOT/slurm-${SLURM_JOB_ID}.out"
    fi
}
trap notify EXIT

cd "$ROOT"
source "$VENV_APDDV2/bin/activate"

echo "--- Gerando samples ---"
python3 scripts/analyze.py --config configs/analysis.yaml --skip-analysis \
    || { echo "ERRO samples"; exit 1; }

echo "--- Zipando samples ---"
cd /snfs1/speed/larissa.gomide
zip -r samples.zip reports/samples/ \
    || echo "AVISO: zip falhou, samples em reports/samples/"

echo "=== FINALIZADO ===" && hostname
