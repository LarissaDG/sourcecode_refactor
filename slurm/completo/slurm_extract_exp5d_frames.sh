#!/bin/bash
#SBATCH --job-name=extract_exp5d_frames
#SBATCH --time=02:00:00
#SBATCH -N 1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=larissa.gomide@dcc.ufmg.br

# Prepara os dados do Exp5d: mede a duração/fps REAIS dos 26 vídeos já
# baixados por download_timecraft.py (sem baixar nada de novo, e sem depender
# do resultado de slurm_extract_exp5c_frames.sh -- mede por conta própria) e
# extrai até 10 sub-amostras de 3s por vídeo, escolhidas aleatoriamente ao
# longo de TODO o vídeo mas SEM embaralhar a ordem (índice do frame = segundo
# real no vídeo original, com lacunas entre as janelas não-adjacentes), pra
# uma análise de consistência temporal em nível micro. Vídeos com menos de 3s
# são pulados e registrados; vídeos que não cabem as 10 janelas completas
# usam quantas couberem (ver scripts/extract_exp5d_frames.py).
#
# O e-mail final já inclui o resumo de duração (mín/máx/média) direto do
# duration_report.csv -- confira esses números antes de rodar o Exp5d em si.
#
# Job leve, roda em CPU (não precisa de módulo CUDA nem da venv com Janus).

set -x

ROOT="/sonic_home/larissa.gomide/sourcecode_refactor"
VENV_DOWNLOAD="$ROOT/venv_download"
DATA_DIR="/snfs1/speed/larissa.gomide/data"
VIDEOS_DIR="$DATA_DIR/temporal/videos"
OUT_DIR="$DATA_DIR/temporal_micro"

export HOME="/sonic_home/larissa.gomide/casa/"
export XDG_CACHE_HOME="/sonic_home/larissa.gomide/casa/.cache"

notify() {
    local code=$?
    local summary=""
    if [ -f "$OUT_DIR/metadata.csv" ]; then
        summary=$(python3 - <<PYEOF
import pandas as pd
df = pd.read_csv("$OUT_DIR/metadata.csv")
ok = df[df["status"].isin(["ok", "ok_ja_existia"])]
full = (df["n_blocks_usados"] == 10).sum()
curtos = (df["status"] == "video_curto_demais").sum()
print(f"{len(ok)}/{len(df)} videos com frames extraidos, {full} com as 10 janelas completas, {curtos} curtos demais (0 janelas)")
PYEOF
)
    fi
    if [ $code -eq 0 ]; then
        source "$VENV_DOWNLOAD/bin/activate"
        python3 scripts/manda_email.py \
            "✅ Extração Exp5d concluída — Phocus4" \
            "Resumo: $summary. Frames extraídos em: $OUT_DIR/frames/. Confira $OUT_DIR/metadata.csv (blocos usados por vídeo) antes de rodar o Exp5d."
    else
        source "$VENV_DOWNLOAD/bin/activate"
        python3 scripts/manda_email.py \
            "❌ Extração Exp5d FALHOU/PAROU — Phocus4" \
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

python3 scripts/extract_exp5d_frames.py --videos-dir "$VIDEOS_DIR" --out "$OUT_DIR" \
    || { echo "ERRO: extract_exp5d_frames.py falhou"; exit 1; }

echo "=== FINALIZADO ===" && hostname
