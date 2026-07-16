#!/bin/bash
#SBATCH --job-name=download_data
#SBATCH --time=08:00:00
#SBATCH -N 1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=larissa.gomide@dcc.ufmg.br

set -x

ROOT="/sonic_home/larissa.gomide/sourcecode_refactor"
VENV_DOWNLOAD="$ROOT/venv_download"
DATA_DIR="/sonic_home/larissa.gomide/sourcecode_refactor/data"

export HOME="/sonic_home/larissa.gomide/minha_home"
export HF_HOME="$HOME/.cache/huggingface"
export TRANSFORMERS_CACHE="$HOME/.cache/huggingface"
export XDG_CACHE_HOME="$HOME/.cache"
export MPLCONFIGDIR="$HOME/.matplotlib"
mkdir -p "$HF_HOME" "$MPLCONFIGDIR"

notify() {
    local code=$?
    if [ $code -eq 0 ]; then
        source "$VENV_DOWNLOAD/bin/activate"
        python3 scripts/manda_email.py \
            "Download concluido - Phocus4" \
            "download_data job finalizado com sucesso. Datasets em: $DATA_DIR"
    else
        source "$VENV_DOWNLOAD/bin/activate"
        python3 scripts/manda_email.py \
            "Download FALHOU - Phocus4" \
            "Job abortou com erro (codigo $code). Verifique o log: $ROOT/slurm-${SLURM_JOB_ID}.out"
    fi
}
trap notify EXIT

if [ ! -d "$VENV_DOWNLOAD" ]; then
    python3 -m venv "$VENV_DOWNLOAD"
fi
source "$VENV_DOWNLOAD/bin/activate"

pip install --quiet --upgrade pip
pip install --quiet --no-cache-dir \
    gdown requests pandas tqdm \
    yt-dlp imageio imageio-ffmpeg \
    transformers sentencepiece sacremoses \
    Pillow

cd "$ROOT"

echo "=== [1/3] Portinari ==="
python3 scripts/download_portinari.py --out "$DATA_DIR/portinari" \
    || echo "AVISO: download_portinari.py falhou (continue com MNIST e videos)"

echo "=== Traducao Portinari ==="
python3 scripts/portinari_translate.py \
    --csv "$DATA_DIR/portinari/acervoPortinari.csv" \
    --out "$DATA_DIR/portinari/MiniBasePortinari_Translated.csv" \
    --n 500 --seed 42 \
    || echo "AVISO: portinari_translate.py falhou (pode rodar separado depois)"

echo "=== [2/3] MNIST ==="
python3 scripts/download_mnist.py --out "$DATA_DIR/mnist" \
    || { echo "ERRO: download_mnist.py falhou"; exit 1; }

echo "=== [3/3] Videos temporais ==="
python3 scripts/download_temporal.py --out "$DATA_DIR/temporal" \
    || { echo "ERRO: download_temporal.py falhou"; exit 1; }

echo "=== FINALIZADO ===" && hostname
