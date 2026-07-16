#!/bin/bash
#SBATCH --job-name=download_data
#SBATCH --time=08:00:00
#SBATCH -N 1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=larissa.gomide@dcc.ufmg.br

set -x

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT="/sonic_home/larissa.gomide/sourcecode_refactor"
VENV_DOWNLOAD="$ROOT/venv_download"
DATA_DIR="/sonic_home/larissa.gomide/data"

# ── Módulos ───────────────────────────────────────────────────────────────────
module load python3.12.1

# ── Variáveis de cache (evita escrever em $HOME do cluster) ──────────────────
export HOME="/sonic_home/larissa.gomide/minha_home"
export HF_HOME="$HOME/.cache/huggingface"
export TRANSFORMERS_CACHE="$HOME/.cache/huggingface"
export XDG_CACHE_HOME="$HOME/.cache"
export MPLCONFIGDIR="$HOME/.matplotlib"
mkdir -p "$HF_HOME" "$MPLCONFIGDIR"

# ── Cria venv de download (só na primeira vez) ────────────────────────────────
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

# ── Download dos datasets ─────────────────────────────────────────────────────
cd "$ROOT"

echo "=== [1/3] Portinari ==="
python3 scripts/download_portinari.py --out "$DATA_DIR/portinari" \
    || { echo "ERRO: download_portinari.py falhou"; exit 1; }

echo "=== Tradução Portinari (Description_en) ==="
python3 scripts/portinari_translate.py \
    --csv "$DATA_DIR/portinari/acervoPortinari.csv" \
    --out "$DATA_DIR/portinari/MiniBasePortinari_Translated.csv" \
    --n 500 --seed 42 \
    || echo "AVISO: portinari_translate.py falhou (pode rodar separado depois)"

echo "=== [2/3] MNIST ==="
python3 scripts/download_mnist.py --out "$DATA_DIR/mnist" \
    || { echo "ERRO: download_mnist.py falhou"; exit 1; }

echo "=== [3/3] Vídeos temporais (@ArtsyLolaCo) ==="
python3 scripts/download_temporal.py --out "$DATA_DIR/temporal" \
    || { echo "ERRO: download_temporal.py falhou"; exit 1; }

# ── Notificação ───────────────────────────────────────────────────────────────
python3 scripts/manda_email.py \
    "Download concluído — Phocus4" \
    "download_data job finalizado. Datasets em: $DATA_DIR"

echo "=== FINALIZADO ==="
hostname
