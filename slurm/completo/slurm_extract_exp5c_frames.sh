#!/bin/bash
#SBATCH --job-name=extract_exp5c_frames
#SBATCH --time=02:00:00
#SBATCH -N 1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=larissa.gomide@dcc.ufmg.br

# Prepara os dados do Exp5c: mede a duração/fps REAIS dos 26 vídeos já
# baixados por download_timecraft.py (sem baixar nada de novo) e extrai
# 1 frame/segundo, truncado ao mínimo real detectado entre eles, pra uma
# análise de consistência temporal em nível macro (o vídeo inteiro, não só
# os ~24 primeiros frames do Exp5a/5b). Ver scripts/extract_exp5c_frames.py.
#
# O e-mail final já inclui o resumo de duração (mín/máx/média) direto do
# duration_report.csv -- confira esses números antes de rodar o Exp5c em si.
#
# Job leve, roda em CPU (não precisa de módulo CUDA nem da venv com Janus).

set -x

ROOT="/sonic_home/larissa.gomide/sourcecode_refactor"
VENV_DOWNLOAD="$ROOT/venv_download"
DATA_DIR="/snfs1/speed/larissa.gomide/data"
VIDEOS_DIR="$DATA_DIR/temporal/videos"
OUT_DIR="$DATA_DIR/temporal_1fps"

export HOME="/sonic_home/larissa.gomide/casa/"
export XDG_CACHE_HOME="/sonic_home/larissa.gomide/casa/.cache"

notify() {
    local code=$?
    local summary=""
    if [ -f "$OUT_DIR/duration_report.csv" ]; then
        summary=$(python3 - <<PYEOF
import pandas as pd
df = pd.read_csv("$OUT_DIR/duration_report.csv")
d = df[df["duration_sec"] > 0]["duration_sec"]
print(f"min={d.min():.1f}s max={d.max():.1f}s media={d.mean():.1f}s mediana={d.median():.1f}s (n={len(d)})")
PYEOF
)
    fi
    if [ $code -eq 0 ]; then
        source "$VENV_DOWNLOAD/bin/activate"
        python3 scripts/manda_email.py \
            "✅ Extração Exp5c concluída — Phocus4" \
            "Duração real dos vídeos: $summary. Frames 1fps extraídos em: $OUT_DIR/frames/. Confira $OUT_DIR/duration_report.csv antes de rodar o Exp5c."
    else
        source "$VENV_DOWNLOAD/bin/activate"
        python3 scripts/manda_email.py \
            "❌ Extração Exp5c FALHOU/PAROU — Phocus4" \
            "Job saiu com código $code. Log: $ROOT/slurm-${SLURM_JOB_ID}.out. Reenviar o mesmo sbatch retoma (pula vídeos já extraídos)."
    fi
}
trap notify EXIT

if [ ! -d "$VENV_DOWNLOAD" ]; then
    python3 -m venv "$VENV_DOWNLOAD"
fi
source "$VENV_DOWNLOAD/bin/activate"

python3 -m ensurepip --upgrade 2>/dev/null || true
pip install --quiet --upgrade pip
pip install --quiet --no-cache-dir imageio imageio-ffmpeg pandas Pillow numpy

cd "$ROOT"

python3 scripts/extract_exp5c_frames.py --videos-dir "$VIDEOS_DIR" --out "$OUT_DIR" \
    || { echo "ERRO: extract_exp5c_frames.py falhou"; exit 1; }

echo "=== FINALIZADO ===" && hostname
