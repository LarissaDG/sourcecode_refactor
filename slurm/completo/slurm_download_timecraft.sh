#!/bin/bash
#SBATCH --job-name=download_timecraft
#SBATCH --time=04:00:00
#SBATCH -N 1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=larissa.gomide@dcc.ufmg.br

# Baixa só o dataset temporal (TimeCraft — Digital Paintings), separado do
# job combinado slurm_download_data.sh. Útil pra re-rodar so o Exp5 sem
# repetir APDDv2/Portinari/MNIST.
#
# Retomável: se o job estourar o walltime (--time acima) ou for cancelado no
# meio, é só reenviar o mesmo sbatch de novo — download_timecraft.py pula
# vídeos já baixados e frames já extraídos, então o job não recomeça do zero.
#
# ANTES de rodar pela primeira vez, apague o dataset antigo (@ArtsyLolaCo)
# pra não misturar os dois:
#   rm -rf /snfs1/speed/larissa.gomide/data/temporal

set -x

ROOT="/sonic_home/larissa.gomide/sourcecode_refactor"
VENV_DOWNLOAD="$ROOT/venv_download"
DATA_DIR="/snfs1/speed/larissa.gomide/data"

export HOME="/sonic_home/larissa.gomide/casa/"
export XDG_CACHE_HOME="/sonic_home/larissa.gomide/casa/.cache"

mkdir -p "$DATA_DIR"

notify() {
    local code=$?
    if [ $code -eq 0 ]; then
        source "$VENV_DOWNLOAD/bin/activate"
        python3 scripts/manda_email.py \
            "✅ Download TimeCraft concluído — Phocus4" \
            "Vídeos temporais baixados. Em: $DATA_DIR/temporal — confira metadata.csv pra ver quantos vídeos deram certo."
    else
        source "$VENV_DOWNLOAD/bin/activate"
        python3 scripts/manda_email.py \
            "❌ Download TimeCraft FALHOU/PAROU — Phocus4" \
            "Job saiu com código $code (pode ser erro real ou walltime estourado). Log: $ROOT/slurm-${SLURM_JOB_ID}.out. Reenviar o mesmo sbatch retoma de onde parou."
    fi
}
trap notify EXIT

if [ ! -d "$VENV_DOWNLOAD" ]; then
    python3 -m venv "$VENV_DOWNLOAD"
fi
source "$VENV_DOWNLOAD/bin/activate"

python3 -m ensurepip --upgrade 2>/dev/null || true
pip install --quiet --upgrade pip
pip install --quiet --no-cache-dir \
    yt-dlp imageio imageio-ffmpeg pandas tqdm Pillow numpy

command -v git >/dev/null 2>&1 || { echo "ERRO: git não encontrado no PATH (necessário para clonar o repositório TimeCraft)"; exit 1; }

cd "$ROOT"

python3 scripts/download_timecraft.py --out "$DATA_DIR/temporal" \
    || { echo "ERRO: download_timecraft.py falhou"; exit 1; }

echo "=== FINALIZADO ===" && hostname
