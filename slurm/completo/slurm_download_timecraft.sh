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
# Apaga o dataset antigo (@ArtsyLolaCo) automaticamente, mas só na PRIMEIRA
# vez: usa um marcador ($TEMPORAL_DIR/.timecraft_download) pra saber se já
# limpou antes. Sem isso, um reenvio do job depois de estourar o walltime
# apagaria os vídeos/frames já baixados nessa mesma rodada -- o que
# destruiria justamente a retomada que o download_timecraft.py foi feito pra
# suportar (pula vídeo já baixado, pula frame já extraído). Se algum dia
# você quiser forçar uma limpeza total de novo (ex: pra pegar vídeos que
# ficaram disponíveis de novo no YouTube), apague o marcador ou a pasta
# inteira na mão antes de reenviar:
#   rm -rf /snfs1/speed/larissa.gomide/data/temporal

set -x

ROOT="/sonic_home/larissa.gomide/sourcecode_refactor"
VENV_DOWNLOAD="$ROOT/venv_download"
DATA_DIR="/snfs1/speed/larissa.gomide/data"
TEMPORAL_DIR="$DATA_DIR/temporal"
MARKER="$TEMPORAL_DIR/.timecraft_download"

export HOME="/sonic_home/larissa.gomide/casa/"
export XDG_CACHE_HOME="/sonic_home/larissa.gomide/casa/.cache"

mkdir -p "$DATA_DIR"

if [ ! -f "$MARKER" ]; then
    echo "Marcador não encontrado em $TEMPORAL_DIR -- assumindo dataset antigo (@ArtsyLolaCo) ou pasta vazia. Apagando..."
    rm -rf "$TEMPORAL_DIR"
    mkdir -p "$TEMPORAL_DIR"
    touch "$MARKER"
else
    echo "Marcador encontrado em $TEMPORAL_DIR -- retomando download já em andamento (nada é apagado)."
fi

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
