# Automatic Aesthetic Evaluation and Prompt Controllability in Generative Image Models

Code for the experiments in *Automatic Aesthetic Evaluation and Prompt Controllability in Generative Image Models*, submitted to ICCC 2025.

## Overview

This pipeline evaluates the aesthetic quality of AI-generated paintings using ArtCLIP, a CLIP-based model trained on human aesthetic annotations. It covers five experiments across four datasets (APDDv2, Portinari, MNIST, and temporal video frames), organized as a 4-stage pipeline: **Sampling → Captioning → Generation → Scoring**.

---

## Repository Structure

```
configs/          YAML configs for each experiment (one per exp)
datasets/         Dataset loaders (APDDv2, Portinari, MNIST, video frames, noise)
pipeline/         Pipeline stages: sampling, captioning, generation, scoring
scripts/          Data download and utility scripts
tests/            Unit tests
run.py            Entry point
base.yaml         Default pipeline config (overridden per experiment)
requirements.txt  Python dependencies for the pipeline venv
```

---

## Environment Setup

Two virtual environments are required, one per pipeline stage group:

### venv — Captioning + Generation (Janus)

```bash
python3 -m venv venv
source venv/bin/activate
pip install --no-cache-dir -r requirements.txt
pip install -e path/to/Janus   # see https://github.com/deepseek-ai/Janus
```

> **Note:** On Python 3.10 + torchvision, install `pip install "numpy<2.0"` to avoid a runtime error.

### apddv2 — Scoring (ArtCLIP)

Follow the setup instructions at https://github.com/BestiVictory/APDDv2 — the repo ships its own `requirements.txt`.

```bash
python3 -m venv apddv2
source apddv2/bin/activate
pip install --no-cache-dir -r path/to/APDDv2/requirements.txt
```

### Environment variables (SLURM cluster)

Set these before running on the cluster to avoid writing to the shared `$HOME`:

```bash
export HOME="/sonic_home/larissa.gomide/minha_home"
export HF_HOME="$HOME/.cache/huggingface"
export TRANSFORMERS_CACHE="$HOME/.cache/huggingface"
export XDG_CACHE_HOME="$HOME/.cache"
export MPLCONFIGDIR="$HOME/.matplotlib"
```

---

## Datasets

### Download (automated)

All datasets except APDDv2 can be downloaded with a single command:

```bash
python3 scripts/download_all.py --out /sonic_home/larissa.gomide/data
```

Or individually:

```bash
python3 scripts/download_all.py --out data/ --only portinari
python3 scripts/download_all.py --out data/ --only mnist
python3 scripts/download_all.py --out data/ --only temporal   # requires yt-dlp
```

On the SLURM cluster, submit the dedicated download job:

```bash
sbatch scripts/slurm_download_data.sh
```

This job creates a throwaway `venv_download`, installs only the download dependencies, downloads all three datasets, runs the Portinari translation, and sends an e-mail notification when done.

### APDDv2

The public image download link is no longer available. Use an existing local copy or contact the authors.

- Paper: https://arxiv.org/abs/2411.08545
- Repository: https://github.com/BestiVictory/APDDv2

Download the pre-trained ArtCLIP weights:

```bash
gdown --folder "1AOVKmSqZCW09J_Ypr7KzSYfRxQre-w_m" -O model_weights/
```

> Updated weights are also available at [Baidu Pan](https://pan.baidu.com/s/1HA8c9nnCRdBOR_zHNC781A?pwd=miwi) (requires Baidu account). Model 6 (*The sense of order*) has a known bug and is excluded from evaluation.

Expected structure:
```
apddv2/
├── APDDv2images/      (or images/)
├── model_weights/
└── APDDv2-10023.csv
```

### Portinari

Downloaded automatically by `download_portinari.py` (32 ZIP archives from Google Drive + CSV from Google Sheets).

After download, generate English translations for Exp 2b:

```bash
python3 scripts/portinari_translate.py \
    --csv data/portinari/acervoPortinari.csv \
    --out data/portinari/MiniBasePortinari_Translated.csv \
    --n 500 --seed 42
```

Three variants used across experiments:

| Variant | Description | Used in |
|---|---|---|
| Full dataset | All ~5000 images + `acervoPortinari.csv` | Reference |
| 500-image sample | `n=500, seed=42` (sampled at runtime) | Exp 2a |
| Translated CSV | `MiniBasePortinari_Translated.csv` with `Description_en` | Exp 2b |

### MNIST

Downloaded automatically by `download_mnist.py`. Samples 500 digits (balanced across 10 classes, seed=42).

### Temporal (video frames)

Downloaded automatically by `download_temporal.py` from `@ArtsyLolaCo` YouTube Shorts. Downloads up to 500 videos, extracts 1 frame/second into per-video subfolders.

---

## Running Experiments

### Test mode (5 samples — fast sanity check)

```bash
python3 run.py --config configs/exp1_apdd.yaml --test
python3 run.py --config configs/exp2a_portinari.yaml --test
```

### Full run

```bash
python3 run.py --config configs/exp1_apdd.yaml
python3 run.py --config configs/exp2a_portinari.yaml
python3 run.py --config configs/exp2b_portinari_human.yaml
python3 run.py --config configs/exp3_mnist.yaml
python3 run.py --config configs/exp4_noise.yaml
python3 run.py --config configs/exp5a_temporal.yaml
python3 run.py --config configs/exp5b_temporal_error.yaml
```

### Experiment overview

| Config | Description | Pipeline steps |
|---|---|---|
| `exp1_apdd.yaml` | APDDv2 → caption (Janus-7B) → generate (Janus-1B + 7B) → score | all |
| `exp2a_portinari.yaml` | Portinari → caption → generate → score | all |
| `exp2b_portinari_human.yaml` | Portinari with human descriptions → generate → score | skip captioning |
| `exp3_mnist.yaml` | MNIST digits → score only (ArtCLIP baseline) | skip caption + gen |
| `exp4_noise.yaml` | APDDv2 × 3 noise types × 10 levels → score | skip caption + gen |
| `exp5a_temporal.yaml` | Video frames sequential → score | skip caption + gen |
| `exp5b_temporal_error.yaml` | Video frames + persistent error from frame 12 → score | skip caption + gen |

### Running on SLURM (Phocus4)

Submit jobs in this order. Each job sends an e-mail on success or failure.

```bash
# 1. Download all datasets (Portinari, MNIST, temporal videos)
sbatch scripts/slurm_download_data.sh

# 2. Download Portinari ZIP 32 separately (rate-limited by Google Drive)
#    Run after slurm_download_data.sh finishes, or in parallel
sbatch scripts/link_32.sh

# 3. Run experiments (exp1, exp3, exp4 can start right after download)
sbatch scripts/slurm_exp1_apdd.sh
sbatch scripts/slurm_exp2a_portinari.sh
sbatch scripts/slurm_exp2b_portinari_human.sh
sbatch scripts/slurm_exp3_mnist.sh
sbatch scripts/slurm_exp4_noise.sh
sbatch scripts/slurm_exp5a_temporal.sh
sbatch scripts/slurm_exp5b_temporal_error.sh
```

> **Note on ZIP 32:** Google Drive rate-limits downloads when many files are fetched in sequence.
> `link_32.sh` downloads `obras_de_1001_a_1200.zip` (images 1001–1200) separately.
> If it also fails, wait a few hours and resubmit.

> **Note on git:** the cluster branch is `slurm`. To pull updates from `main`:
> ```bash
> git pull origin main
> ```

### Clean outputs

```bash
python3 scripts/clean_outputs.py                    # limpa tudo
python3 scripts/clean_outputs.py --exp exp1_apdd    # limpa só um experimento
python3 scripts/clean_outputs.py --dry-run          # prévia sem deletar
```

---

## Citation

```bibtex
@inproceedings{gomide2025iccc,
  title={Automatic Aesthetic Evaluation and Prompt Controllability in Generative Image Models},
  author={Larissa Gomide and Lucas Nascimento Ferreira and Wagner Meira Jr.},
  booktitle={Proceedings of the ICCC 2025},
  year={2025}
}
```

## Licenses

| Content | License |
|---|---|
| Software | MIT — see [LICENSE](./LICENSE) |
| Dataset | CC BY 4.0 |

## Contact

📧 laladg18@gmail.com
