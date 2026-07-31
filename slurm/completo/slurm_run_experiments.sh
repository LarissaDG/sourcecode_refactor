#!/bin/bash
#SBATCH --job-name=run_experiments
#SBATCH --time=20:00:00
#SBATCH -N 1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=larissa.gomide@dcc.ufmg.br

set -x

# â”€â”€ Paths â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
ROOT="/sonic_home/larissa.gomide/sourcecode_refactor"
VENV="/sonic_home/larissa.gomide/venv"
VENV_APDDV2="/sonic_home/larissa.gomide/apddv2"

# â”€â”€ MÃ³dulos â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
module load python3.12.1
module load cuda/11.8.0

export HOME="/sonic_home/larissa.gomide/casa"
export HF_HOME="/snfs1/speed/larissa.gomide/hf_cache"
export TRANSFORMERS_CACHE="/snfs1/speed/larissa.gomide/hf_cache"
export CLIP_CACHE="/snfs1/speed/larissa.gomide/hf_cache"
export XDG_CACHE_HOME="/sonic_home/larissa.gomide/casa/.cache"
export MPLCONFIGDIR="/sonic_home/larissa.gomide/casa/.matplotlib"



# â”€â”€ VariÃ¡veis de cache â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


# â”€â”€ FunÃ§Ã£o auxiliar â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
run_exp() {
    local CONFIG="$1"
    local STEPS_JANUS="$2"   # steps que rodam no venv (Janus)
    local STEPS_SCORE="$3"   # steps que rodam no apddv2 (ArtCLIP)
    local TEST_FLAG="$4"     # "--test" ou ""

    echo ""
    echo "========================================================"
    echo "  EXPERIMENTO: $CONFIG"
    echo "========================================================"

    if [ -n "$STEPS_JANUS" ]; then
        echo "--- Fase 1: Janus ($STEPS_JANUS) ---"
        source "$VENV/bin/activate"
        python3 run.py --config "$CONFIG" --steps "$STEPS_JANUS" $TEST_FLAG \
            || { echo "ERRO na fase Janus de $CONFIG"; deactivate; return 1; }
        deactivate
    fi

    echo "--- Fase 2: ArtCLIP ($STEPS_SCORE) ---"
    source "$VENV_APDDV2/bin/activate"
    python3 run.py --config "$CONFIG" --steps "$STEPS_SCORE" $TEST_FLAG \
        || { echo "ERRO na fase ArtCLIP de $CONFIG"; deactivate; return 1; }
    deactivate
}

# â”€â”€ Experimentos â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Altere "--test" para "" quando quiser rodar completo
TEST="--test"

# Exp 0 â€” ICCC (proportional_stratified, 502 imagens): todas as etapas
run_exp "configs/exp0_iccc.yaml" \
        "sampling,captioning,generation,samples" \
        "scoring" \
        "$TEST"

# Exp 1 â€” APDDv2: todas as etapas
run_exp "configs/exp1_apdd.yaml" \
        "sampling,captioning,generation,samples" \
        "scoring" \
        "$TEST"

# Exp 2a â€” Portinari (AI captions): todas as etapas
run_exp "configs/exp2a_portinari.yaml" \
        "sampling,captioning,generation,samples" \
        "scoring" \
        "$TEST"

# Exp 2b â€” Portinari (human captions): sem captioning
# IMPORTANTE: reusa as imagens do Exp2a (reuse_from) — só rode depois que
# outputs/exp2a_portinari/pipeline_data.json já existir (Exp2a concluído).
run_exp "configs/exp2b_portinari_human.yaml" \
        "sampling,generation,samples" \
        "scoring" \
        "$TEST"

# Exp 3 â€” MNIST: só scoring
run_exp "configs/exp3_mnist.yaml" \
        "" \
        "sampling,samples,scoring" \
        "$TEST"

# Exp 4 â€” Ruído: só scoring
run_exp "configs/exp4_noise.yaml" \
        "" \
        "sampling,samples,scoring" \
        "$TEST"

# Exp 5a â€” Temporal sequencial: só scoring
run_exp "configs/exp5a_temporal.yaml" \
        "" \
        "sampling,samples,scoring" \
        "$TEST"

# Exp 5b â€” Temporal com erro: só scoring
run_exp "configs/exp5b_temporal_error.yaml" \
        "" \
        "sampling,samples,scoring" \
        "$TEST"

# â”€â”€ NotificaÃ§Ã£o â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
source "$VENV/bin/activate"
python3 scripts/manda_email.py \
    "Experimentos concluÃ­dos â€” Phocus4" \
    "slurm_run_experiments.sh finalizado. Resultados em: $ROOT/outputs/"
deactivate

echo "=== FINALIZADO ==="
hostname
