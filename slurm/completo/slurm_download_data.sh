#!/bin/bash
#SBATCH --job-name=download_data
#SBATCH --time=08:00:00
#SBATCH -N 1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=larissa.gomide@dcc.ufmg.br

set -x

ROOT="/sonic_home/larissa.gomide/sourcecode_refactor"
VENV_DOWNLOAD="$ROOT/venv_download"
DATA_DIR="/snfs1/speed/larissa.gomide/data"

export HOME="/sonic_home/larissa.gomide/casa/"
export TRANSFORMERS_CACHE="/sonic_home/larissa.gomide/casa/.cache/huggingface"
export CLIP_CACHE="/sonic_home/larissa.gomide/casa/.cache/clip"
export HF_HOME="/sonic_home/larissa.gomide/casa/.cache/huggingface"
export XDG_CACHE_HOME="/sonic_home/larissa.gomide/casa/.cache"
export MPLCONFIGDIR="/sonic_home/larissa.gomide/casa/.matplotlib"

mkdir -p "$DATA_DIR" "$HF_HOME" "$MPLCONFIGDIR"

notify() {
    local code=$?
    if [ $code -eq 0 ]; then
        source "$VENV_DOWNLOAD/bin/activate"
        python3 scripts/manda_email.py \
            "✅ Download concluído — Phocus4" \
            "Todos os datasets baixados com sucesso. Em: $DATA_DIR"
    else
        source "$VENV_DOWNLOAD/bin/activate"
        python3 scripts/manda_email.py \
            "❌ Download FALHOU — Phocus4" \
            "Job abortou com erro (código $code). Verifique o log: $ROOT/slurm-${SLURM_JOB_ID}.out"
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
    gdown requests pandas tqdm \
    yt-dlp imageio imageio-ffmpeg \
    deep-translator \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    Pillow

cd "$ROOT"

echo "=== [1/4] APDDv2 (imagens + CSVs) ==="
APDDV2_DIR="$DATA_DIR/apddv2"
mkdir -p "$APDDV2_DIR/APDDv2images"

# TODO: subir dataset completo para o Kaggle e atualizar threshold para 10023
N_APDDV2=$(ls "$APDDV2_DIR/APDDv2images" 2>/dev/null | wc -l)
if [ "$N_APDDV2" -ge 986 ]; then  # 10023 quando dataset completo estiver disponível
    echo "  APDDv2 já completo ($N_APDDV2 imagens). Pulando."
else
    echo "  APDDv2 incompleto ($N_APDDV2 imagens). Baixando CSVs e imagens do ICCC..."
    curl -L "https://raw.githubusercontent.com/BestiVictory/APDDv2/main/APDDv2-10023.csv" \
        -o "$APDDV2_DIR/APDDv2-10023.csv"
    curl -L "https://raw.githubusercontent.com/BestiVictory/APDDv2/main/filesource.csv" \
        -o "$APDDV2_DIR/filesource.csv"
    CLONE_DIR="$(mktemp -d)/ICCC"
    git clone --filter=blob:none --sparse https://github.com/LarissaDG/ICCC.git "$CLONE_DIR"
    cd "$CLONE_DIR"
    git sparse-checkout set APDDv2images
    mv "$CLONE_DIR/APDDv2images/"* "$APDDV2_DIR/APDDv2images/"
    rm -rf "$CLONE_DIR"
    cd "$ROOT"
    echo "  Imagens: $(ls "$APDDV2_DIR/APDDv2images" | wc -l)"
fi

echo "=== [2/4] Portinari ==="
if [ -f "$DATA_DIR/portinari/MiniBasePortinari_Translated.csv" ]; then
    echo "  Portinari já baixado e traduzido. Pulando."
else
    if [ ! -f "$DATA_DIR/portinari/acervoPortinari.csv" ]; then
        python3 scripts/download_portinari.py --out "$DATA_DIR/portinari" \
            || echo "AVISO: download_portinari.py falhou (continue com MNIST e videos)"
    else
        echo "  Portinari já baixado. Pulando download."
    fi
    echo "=== Tradução Portinari ==="
    python3 scripts/portinari_translate.py \
        --csv "$DATA_DIR/portinari/acervoPortinari.csv" \
        --out "$DATA_DIR/portinari/MiniBasePortinari_Translated.csv" \
        --n 500 --seed 42 \
        || echo "AVISO: portinari_translate.py falhou (pode rodar separado depois)"
fi

echo "=== [3/4] MNIST ==="
if [ -d "$DATA_DIR/mnist" ] && [ "$(ls "$DATA_DIR/mnist" 2>/dev/null | wc -l)" -gt 0 ]; then
    echo "  MNIST já baixado. Pulando."
else
    python3 scripts/download_mnist.py --out "$DATA_DIR/mnist" \
        || { echo "ERRO: download_mnist.py falhou"; exit 1; }
fi

echo "=== [4/4] Vídeos temporais ==="
if [ -d "$DATA_DIR/temporal/frames" ] && [ "$(ls "$DATA_DIR/temporal/frames" 2>/dev/null | wc -l)" -gt 0 ]; then
    echo "  Vídeos temporais já baixados. Pulando."
else
    python3 scripts/download_temporal.py --out "$DATA_DIR/temporal" \
        || { echo "ERRO: download_temporal.py falhou"; exit 1; }
fi

echo "=== FINALIZADO ===" && hostname
