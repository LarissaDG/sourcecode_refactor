# Avaliação Estética Automática de Imagens Artísticas

[![Tests](https://github.com/LarissaDG/sourcecode_refactor/actions/workflows/tests.yml/badge.svg)](https://github.com/LarissaDG/sourcecode_refactor/actions/workflows/tests.yml)
[![GitHub Pages](https://github.com/LarissaDG/sourcecode_refactor/actions/workflows/pages/pages-build-deployment/badge.svg)](https://larissadg.github.io/sourcecode_refactor)

> **Dissertação de Mestrado** — Larissa Dolabella Gomide  
> Orientadores: Lucas Nascimento Ferreira · Wagner Meira Jr. · UFMG  
> Publicado no **ICCC 2025** (Campinas, Brasil) · Early Career Symposium **ICCC 2026** (Coimbra, Portugal)

Este repositório contém o código dos experimentos de *Avaliação Estética Automática de Imagens Artísticas: Uma abordagem heurística para apoiar processos criativos em artes visuais*. O pipeline avalia a qualidade estética de pinturas geradas por IA usando o modelo **ArtCLIP** (CLIP treinado com anotações de especialistas em arte), cobrindo cinco experimentos em quatro datasets. 

O [artigo](https://arxiv.org/pdf/2411.08545) que propôs o conjunto de dados APDDv2 e o modelo ArtClip, foi a principal fonte e referência desse trabalho. Seu respectivo [repositório](https://github.com/BestiVictory/APDDv2).

---

## Índice

- [Visão Geral](#visão-geral)
- [Estrutura do Repositório](#estrutura-do-repositório)
- [Ambientes Virtuais](#ambientes-virtuais)
  - [Cluster (Phocus4/Gorgona)](#cluster-phocus4gorgona)
  - [Local (Windows)](#local-windows)
- [Datasets](#datasets)
- [Executando os Experimentos](#executando-os-experimentos)
  - [Modo teste](#modo-teste)
  - [Execução completa](#execução-completa)
  - [No cluster (SLURM)](#no-cluster-slurm)
- [Análise e Visualizações](#análise-e-visualizações)
  - [Local (Windows)](#análise-local-windows)
  - [No cluster (samples)](#samples-no-cluster)
  - [Relatórios HTML](#relatórios-html)
- [Citação](#citação)
- [Contato](#contato)

---

## Visão Geral

Pipeline de 4 estágios: **Sampling → Captioning → Generation → Scoring**

| Experimento | Dataset | Objetivo |
|---|---|---|
| Exp 1 — APDDv2 | 448 pinturas amostradas | Baseline: Human GT vs Janus-1B/7B |
| Exp 2a — Portinari | 500 pinturas, captions por IA | Impacto de captions automáticas |
| Exp 2b — Portinari | 498 pinturas, captions humanas | Impacto de captions humanas |
| Exp 3 — MNIST | Dígitos manuscritos | Arte vs. Não-Arte |
| Exp 4 — Ruído | APDDv2 + ruído sintético | Robustez estética |
| Exp 5 — Temporal | Frames de vídeo | Consistência e degradação temporal |

---

## Estrutura do Repositório

```
configs/            Configs YAML por experimento + análise
datasets/           Loaders de dataset (APDDv2, Portinari, MNIST, vídeo, ruído)
pipeline/           Estágios: sampling, captioning, generation, scoring
scripts/
  analyze.py        Visualizações e amostras dos experimentos
  analyze_iccc.py   Análise fiel à metodologia do paper ICCC
  generate_html_reports.py  Gera relatórios HTML para GitHub Pages
slurm/              Scripts SLURM para o cluster
tests/              Testes unitários
run.py              Entry point principal
requirements.txt    Dependências do pipeline (venv do cluster)
```

---

## Ambientes Virtuais

### Cluster (Phocus4/Gorgona)

Dois ambientes são necessários no cluster:

#### `venv` — Captioning + Generation (Janus)

```bash
python3 -m venv venv
source venv/bin/activate
pip install --no-cache-dir -r requirements.txt
pip install -e path/to/Janus   # https://github.com/deepseek-ai/Janus
```

> **Atenção:** Em Python 3.10 + torchvision, instale `pip install "numpy<2.0"` para evitar conflito de runtime.

#### `apddv2` — Scoring (ArtCLIP)

```bash
python3 -m venv apddv2
source apddv2/bin/activate
pip install --no-cache-dir -r path/to/APDDv2/requirements.txt
```

#### Variáveis de ambiente (cluster)

```bash
export HOME="/sonic_home/larissa.gomide/minha_home"
export HF_HOME="$HOME/.cache/huggingface"
export TRANSFORMERS_CACHE="$HOME/.cache/huggingface"
export XDG_CACHE_HOME="$HOME/.cache"
export MPLCONFIGDIR="$HOME/.matplotlib"
```

---

### Local (Windows)

Para rodar os scripts de análise e geração de relatórios localmente (sem GPU):

#### Criar o ambiente

```powershell
python -m venv venv_local
```

#### Ativar o ambiente

```powershell
venv_local\Scripts\activate
```

#### Instalar dependências de análise

```powershell
pip install matplotlib seaborn pandas scipy scikit-learn imageio pyyaml pillow numpy
```

#### Configurar paths locais

Edite `configs/analysis_local.yaml` com os caminhos da sua máquina:

```yaml
paths:
  outputs: "C:\\Users\\jggom\\Downloads\\Execucao dia 28\\outputs"
  reports: "C:\\Users\\jggom\\Downloads\\Execucao dia 28\\reports"
  apddv2_csv: "C:\\...\\APDDv2\\APDDv2-10023.csv"
```

---

## Datasets

### Download automatizado

```bash
python3 scripts/download_all.py --out /sonic_home/larissa.gomide/data
```

Ou por dataset:

```bash
python3 scripts/download_all.py --out data/ --only portinari
python3 scripts/download_all.py --out data/ --only mnist
python3 scripts/download_all.py --out data/ --only temporal   # requer yt-dlp
```

No cluster:

```bash
sbatch scripts/slurm_download_data.sh
```

### APDDv2

O link público de download não está mais disponível. Use uma cópia local ou contate os autores.

- Paper: https://arxiv.org/abs/2411.08545
- Repositório: https://github.com/BestiVictory/APDDv2

Pesos do ArtCLIP:

```bash
gdown --folder "1AOVKmSqZCW09J_Ypr7KzSYfRxQre-w_m" -O model_weights/
```

> Também disponível no [Baidu Pan](https://pan.baidu.com/s/1HA8c9nnCRdBOR_zHNC781A?pwd=miwi). 
> O modelo 6 (*The sense of order*) tem bug conhecido e é excluído da avaliação. Os dados do Baidu se encontram atualizados e com o modelo 6 funcionando, pelo que os autores informam, mas baixar esses dados se mostrou mais desafiados que se pensava.

Estrutura esperada:
```
apddv2/
├── APDDv2images/
├── model_weights/
└── APDDv2-10023.csv
```

### Portinari

Baixado automaticamente (32 ZIPs do Google Drive + CSV do Google Sheets). Gerar traduções para Exp 2b:

```bash
python3 scripts/portinari_translate.py \
    --csv data/portinari/acervoPortinari.csv \
    --out data/portinari/MiniBasePortinari_Translated.csv \
    --n 500 --seed 42
```

### MNIST

Baixado automaticamente. Amostra 500 dígitos (50 por classe, seed=42).

### Temporal (frames de vídeo)

Baixado de `@ArtsyLolaCo` (YouTube Shorts). Até 500 vídeos, 1 frame/segundo.

---

## Executando os Experimentos

### Modo teste (5 amostras — rápido)

```bash
python3 run.py --config configs/exp1_apdd.yaml --test
python3 run.py --config configs/exp2a_portinari.yaml --test
```

### Execução completa

```bash
python3 run.py --config configs/exp1_apdd.yaml
python3 run.py --config configs/exp2a_portinari.yaml
python3 run.py --config configs/exp2b_portinari_human.yaml
python3 run.py --config configs/exp3_mnist.yaml
python3 run.py --config configs/exp4_noise.yaml
python3 run.py --config configs/exp5a_temporal.yaml
python3 run.py --config configs/exp5b_temporal_error.yaml
```

| Config | Estágios executados |
|---|---|
| `exp1_apdd.yaml` | sampling → captioning → generation → scoring |
| `exp2a_portinari.yaml` | sampling → captioning → generation → scoring |
| `exp2b_portinari_human.yaml` | sampling → generation → scoring (pula captioning) |
| `exp3_mnist.yaml` | scoring apenas |
| `exp4_noise.yaml` | scoring apenas (ruído aplicado na leitura) |
| `exp5a_temporal.yaml` | scoring apenas |
| `exp5b_temporal_error.yaml` | scoring apenas |

### No cluster (SLURM)

Submeter na ordem abaixo. Cada job envia e-mail ao terminar:

```bash
# 1. Download de datasets
sbatch scripts/slurm_download_data.sh
sbatch scripts/link_32.sh          # ZIP 32 do Portinari (rate-limited pelo Google Drive)

# 2. Experimentos
sbatch slurm/completo/slurm_exp1_apdd.sh
sbatch slurm/completo/slurm_exp2a_portinari.sh
sbatch slurm/completo/slurm_exp2b_portinari_human.sh
sbatch slurm/completo/slurm_exp3_mnist.sh
sbatch slurm/completo/slurm_exp4_noise.sh
sbatch slurm/completo/slurm_exp5a_temporal.sh
sbatch slurm/completo/slurm_exp5b_temporal_error.sh
```

#### Upload do APDDv2 (Windows → cluster)

```powershell
python zip_and_upload.py   # gera APDDv2images_part1.zip e part2.zip e faz upload
```

```bash
git pull
sbatch slurm/completo/slurm_unzip_apddv2.sh
```

#### Limpar outputs

```bash
python3 scripts/clean_outputs.py                    # limpa tudo
python3 scripts/clean_outputs.py --exp exp1_apdd    # limpa só um experimento
python3 scripts/clean_outputs.py --dry-run          # prévia sem deletar
```

---

## Análise e Visualizações

### Análise local (Windows)

Com o `venv_local` ativado:

#### Gráficos estatísticos (metodologia do paper ICCC)

```powershell
python scripts/analyze_iccc.py --config configs/analysis_local.yaml
```

Com todas as visualizações estendidas dos notebooks:

```powershell
python scripts/analyze_iccc.py --config configs/analysis_local.yaml --all-viz
```

#### Gráficos + amostras de todos os experimentos

```powershell
python scripts/analyze.py --config configs/analysis_local.yaml
```

Só gráficos (sem montar painéis de imagens):

```powershell
python scripts/analyze.py --config configs/analysis_local.yaml --skip-samples
```

Só amostras (sem gráficos estatísticos):

```powershell
python scripts/analyze.py --config configs/analysis_local.yaml --skip-analysis
```

#### Outputs gerados

| Pasta | Conteúdo |
|---|---|
| `reports/figures_iccc/` | Gráficos da metodologia ICCC (t-test, Mann-Whitney, ANOVA, radar) |
| `reports/figures/` | Gráficos por experimento (Friedman, Wilcoxon, CLD, clusters, noise, temporal) |
| `reports/samples/` | Painéis visuais de amostras por experimento + GIFs (exp5) |

---

### Samples no cluster

Para gerar os painéis de amostras usando as imagens oficiais do cluster:

```bash
git pull
sbatch slurm/completo/slurm_analyze.sh
```

O job roda `analyze.py --skip-analysis` e compacta o resultado:

```bash
# Após receber e-mail de conclusão, baixar localmente:
scp phocus4:/snfs1/speed/larissa.gomide/samples.zip "C:\Users\jggom\Downloads\samples.zip"
```

```powershell
# Extrair e substituir pasta local
Expand-Archive -Path "C:\Users\jggom\Downloads\samples.zip" `
               -DestinationPath "C:\Users\jggom\Downloads\samples_cluster" -Force

Copy-Item "C:\Users\jggom\Downloads\samples_cluster\reports\samples\*" `
          "C:\Users\jggom\Downloads\Execucao dia 28\reports\samples\" -Force
```

---

### Relatórios HTML

Gera as páginas do GitHub Pages com todas as figuras embutidas (base64):

```powershell
python scripts/generate_html_reports.py --config configs/analysis_local.yaml
```

Arquivos gerados em `reports/`:

| Arquivo | Conteúdo |
|---|---|
| `index.html` | Página inicial com resumo e navegação |
| `Paper_iccc.html` | Metodologia original ICCC 2025 (legacy) |
| `exp1_apdd.html` | Experimento 1 — APDDv2 |
| `exp2a_portinari.html` | Experimento 2a — Portinari (AI captions) |
| `exp2b_portinari_human.html` | Experimento 2b — Portinari (human captions) |
| `exp3_mnist.html` | Experimento 3 — MNIST |
| `exp4_noise.html` | Experimento 4 — Ruído |
| `exp5_temporal.html` | Experimento 5 — Temporal |

---

## Citação

```bibtex
@inproceedings{gomide2025iccc,
  title     = {Automatic Aesthetic Evaluation and Prompt Controllability in Generative Image Models},
  author    = {Larissa Gomide and Lucas Nascimento Ferreira and Wagner Meira Jr.},
  booktitle = {Proceedings of the 16th International Conference on Computational Creativity (ICCC)},
  year      = {2025}
}
```

---

## Licenças

| Conteúdo | Licença |
|---|---|
| Código | MIT — ver [LICENSE](./LICENSE) |
| Dataset | CC BY 4.0 |

---

## Contato

Larissa Dolabella Gomide · laladg18@gmail.com
