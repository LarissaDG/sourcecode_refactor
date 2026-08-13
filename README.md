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
- [Amostras Visuais](#amostras-visuais)
- [Validação e Depuração](#validação-e-depuração)
- [Análise e Visualizações](#análise-e-visualizações)
  - [Local (Windows)](#análise-local-windows)
  - [Relatórios HTML](#relatórios-html)
- [Citação](#citação)
- [Contato](#contato)

---

## Visão Geral

Pipeline de 5 estágios: **Sampling → Captioning → Generation → Samples → Scoring**

| Experimento | Dataset | Objetivo |
|---|---|---|
| Exp 0 — ICCC | 502 pinturas, 2 amostragens (uniform_bins + amostra original via legacy_csv) | Replica a metodologia original do paper ICCC 2025 |
| Exp 1 — APDDv2 | ~500 pinturas, 2 amostragens (uniform_bins + proportional_stratified) | Baseline: Human GT vs Janus-1B/7B |
| Exp 2a — Portinari | 500 pinturas, captions por IA | Impacto de captions automáticas |
| Exp 2b — Portinari | 498 pinturas, captions humanas | Impacto de captions humanas |
| Exp 3 — MNIST | Dígitos manuscritos | Arte vs. Não-Arte |
| Exp 4 — Ruído | APDDv2 + ruído sintético (gaussian/blur/shapes) | Robustez estética |
| Exp 5 — Temporal | Frames de vídeo (TimeCraft — Digital Paintings) + ruído sintético | Consistência e degradação temporal |

---

## Estrutura do Repositório

```
configs/            Configs YAML por experimento + análise
datasets/           Loaders de dataset (APDDv2, Portinari, MNIST, vídeo)
                    + noise.py (ruído sintético compartilhado: gaussian/blur/shapes)
pipeline/           Estágios: sampling, captioning, generation, samples, scoring
scripts/
  analyze.py                Visualizações e amostras dos experimentos
  analyze_iccc.py            Análise fiel à metodologia do paper ICCC
  run_legacy_validation.py   Validação de reprodutibilidade vs. resultados legados do ICCC
  debug_janus.py             Utilitário de debug: testa a geração de imagem do Janus isoladamente
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
sbatch slurm/completo/slurm_download_data.sh
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

Baixado do dataset "Digital Paintings" do [TimeCraft](https://github.com/xamyzhao/timecraft)
(CVPR 2020 — *Painting Many Pasts*). `download_timecraft.py` clona o repositório, lê os 83
`.pkl` publicados em `dataset/digital_vid_caches_minimal.zip` (um por vídeo/peça, cada um com
o ID do YouTube e a lista de frames do vídeo original), baixa cada vídeo via `yt-dlp` e extrai
24 frames a partir do offset 24 dentro dessa lista (pula os 24 primeiros frames — alguns
vídeos abrem com a pintura já finalizada antes de reiniciar do zero). Nem todo vídeo de 2019
ainda está disponível hoje; os que falharem são pulados e listados no final.

---

## Executando os Experimentos

### Modo teste (5 amostras — rápido)

```bash
python3 run.py --config configs/exp0_iccc.yaml --test
python3 run.py --config configs/exp1_apdd.yaml --test
python3 run.py --config configs/exp2a_portinari.yaml --test
```

### Execução completa

```bash
python3 run.py --config configs/exp0_iccc.yaml
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
| `exp0_iccc.yaml` | sampling (uniform_bins **e** proportional_stratified) → captioning → generation → scoring |
| `exp1_apdd.yaml` | sampling (uniform_bins **e** proportional_stratified) → captioning → generation → scoring |
| `exp2a_portinari.yaml` | sampling → captioning → generation → scoring |
| `exp2b_portinari_human.yaml` | sampling → generation → scoring (pula captioning) |
| `exp3_mnist.yaml` | scoring apenas |
| `exp4_noise.yaml` | scoring apenas (ruído aplicado na leitura) |
| `exp5a_temporal.yaml` | scoring apenas |
| `exp5b_temporal_error.yaml` | scoring apenas |

> **`exp1_apdd.yaml` e `exp0_iccc.yaml` rodam duas amostragens em uma única chamada**
> (`sampling.strategies` no YAML): o pipeline completo é executado uma vez por estratégia,
> cada uma em sua própria pasta:
> - `outputs/exp1_apdd_uniform_bins/` / `outputs/exp0_iccc_uniform_bins/`
> - `outputs/exp1_apdd_proportional_stratified/` / `outputs/exp0_iccc_proportional_stratified/`
>
> Em `exp1_apdd.yaml`, `proportional_stratified` calcula uma amostra nova: os N imagens são
> alocadas entre bins de score estético proporcionalmente ao tamanho de cada bin (preserva a
> distribuição original do dataset, ao contrário do `uniform_bins`, que amostra igualmente por
> bin). Ao final, grava `sampling_bin_distribution.txt` com a distribuição por bin.
>
> Em **`exp0_iccc.yaml`**, `proportional_stratified` **não recalcula** — via
> `sampling.legacy_csv`, reusa exatamente as mesmas 502 imagens do experimento ICCC original
> (`/snfs1/speed/larissa.gomide/data/legacy_iccc/sampled_dataset.csv`), casando por nome de
> arquivo (stem, ignorando extensão). Itens do CSV legado sem correspondência no dataset atual
> são ignorados com aviso. `sampling_bin_distribution.txt` ainda é gerado, mas só como relatório
> da distribuição por bin dessa amostra fixa — não influencia a seleção. Já o `uniform_bins`
> do `exp0_iccc.yaml` ignora `legacy_csv` e amostra normalmente, igual ao `exp1_apdd.yaml`.

### No cluster (SLURM)

Submeter na ordem abaixo. Cada job envia e-mail ao terminar:

```bash
# 1. Download de datasets
sbatch slurm/completo/slurm_download_data.sh
sbatch scripts/link_32.sh          # ZIP 32 do Portinari (rate-limited pelo Google Drive)
# Para baixar/re-baixar só o dataset temporal (TimeCraft), sem repetir os
# outros 3 datasets — retomável se o walltime estourar, é só reenviar:
# sbatch slurm/completo/slurm_download_timecraft.sh

# 2. Experimentos
sbatch slurm/completo/slurm_exp0_iccc.sh
sbatch slurm/completo/slurm_exp1_apdd.sh
sbatch slurm/completo/slurm_exp2a_portinari.sh
# exp2b reusa as imagens do exp2a (reuse_from) — só submeta depois que o
# exp2a_portinari acima já tiver terminado (outputs/exp2a_portinari/pipeline_data.json
# precisa existir). Não são independentes, apesar de aparecerem em sequência aqui.
sbatch slurm/completo/slurm_exp2b_portinari_human.sh
sbatch slurm/completo/slurm_exp3_mnist.sh
sbatch slurm/completo/slurm_exp4_noise.sh
sbatch slurm/completo/slurm_exp5a_temporal.sh
sbatch slurm/completo/slurm_exp5b_temporal_error.sh
```

> Todos os jobs acima já incluem a etapa `samples` (amostras visuais, ver
> [Amostras Visuais](#amostras-visuais)). Ela roda na mesma venv que já usa naquela fase
> (`venv` do Janus para exp0/1/2a/2b, `apddv2` para exp3/4/5a/5b) — confirme que
> `matplotlib`/`pillow` estão instalados na venv `apddv2` antes de rodar
> (`source apddv2/bin/activate && python -c "import matplotlib, PIL"`), já que ela usa um
> `requirements.txt` próprio do ArtCLIP, separado do deste repositório.

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

## Amostras Visuais

Cada experimento gera automaticamente exemplos reais de entrada/saída (3 instâncias,
escolhidas de forma determinística — as 3 primeiras após ordenar por `filename`/`video_id`,
sem sorteio) para permitir inspeção visual sem precisar baixar as bases completas. Isso roda
dentro do próprio `run.py`, no cluster (`pipeline.steps.samples: true` no YAML, ativado por
padrão em todos os experimentos), logo após a etapa de generation — não depende de imagens
locais nem duplica as bases: só salva os painéis/grids/GIFs já compostos, em
`outputs/<experimento>/samples/`.

Como as instâncias são escolhidas por ordenação (não por sorteio), o Exp2b — que reusa
exatamente as mesmas 500 imagens do Exp2a via `reuse_from` — mostra automaticamente as
mesmas 3 imagens do Exp2a, só trocando a coluna de descrição (IA → humana).

| Experimento | Arquivo(s) | Conteúdo |
|---|---|---|
| Exp 0 (ICCC) / Exp 1 | `sample_panel.png` | 3 linhas × [Original \| Descrição (Janus-7B) \| Gerada Janus-1B \| Gerada Janus-7B] |
| Exp 2a | `sample_panel.png` | idem, descrição gerada pelo Janus-7B |
| Exp 2b | `sample_panel.png` | idem, descrição humana (mesmas 3 imagens do Exp2a) |
| Exp 3 (MNIST) | `sample_panel.png` | 3 dígitos com o respectivo label |
| Exp 4 (Ruído) | `noise_grid_01.png` .. `_03.png` | 1 grid por instância: linhas = Blur/Gaussian/Shapes, colunas = 10%-100% |
| Exp 5a (Temporal) | `sequence_<video_id>.gif` (×3) + `frame_grid_last6.png` | GIF da sequência amostrada sem ruído por vídeo + grid dos últimos 6 frames (3 vídeos), com número do frame |
| Exp 5b (Temporal) | `degradation_<video_id>.gif` (×3) + `frame_grid_uniform6.png` | GIF da degradação progressiva por vídeo + grid de 6 frames uniformemente distribuídos (3 vídeos), com número do frame e % de degradação |

> **Exp5b usa `gaussian` como tipo de ruído representativo** no GIF e no grid: cada frame do
> Exp5b tem 3 variantes (gaussian/blur/shapes) no mesmo nível de degradação, mas para manter
> 1 GIF/1 grid por vídeo (em vez de 3), o tipo gaussian é usado como exemplo — os outros dois
> seguem a mesma curva de degradação por frame, só muda a textura do ruído.

---

## Validação e Depuração

### Validação de reprodutibilidade (legado ICCC)

`scripts/run_legacy_validation.py` re-roda o pipeline com as mesmas 502 imagens e captions
do experimento original do ICCC 2025 e compara os scores resultantes com os scores legados
(`sampled_SMALL`/`sampled_BIG`), atributo a atributo — usado para confirmar que o pipeline
refatorado reproduz os resultados publicados.

Fluxo do script:
1. Lê `sampled_dataset.csv` (502 imagens + ground truth humano do APDDv2)
2. Lê os CSVs legados `sampled_SMALL_with_gen_scored.csv` (Janus-1B) e
   `sampled_BIG_with_gen_scored.csv` (Janus-7B)
3. Constrói `pipeline_data.json` com as mesmas imagens e captions (coluna `Description`)
4. Roda `run.py --steps generation,scoring` sobre `configs/exp_legacy_validation.yaml`
5. Compara os novos scores com os legados e gera relatório + gráfico de comparação

```bash
python3 scripts/run_legacy_validation.py \
    --config configs/exp_legacy_validation.yaml \
    --legacy-small /path/to/sampled_SMALL_with_gen_scored.csv \
    --legacy-big   /path/to/sampled_BIG_with_gen_scored.csv \
    --apddv2-dir   /snfs1/speed/larissa.gomide/data/apddv2/ \
    --out-dir      /snfs1/speed/larissa.gomide/outputs/exp_legacy_validation
```

`configs/exp_legacy_validation.yaml` pula sampling e captioning (usa o `pipeline_data.json`
pré-construído pelo script acima) e roda só generation → scoring.

### Debug do Janus

`scripts/debug_janus.py` é um utilitário isolado (sem depender do pipeline) para testar a
geração de imagem do Janus-Pro-1B no cluster: carrega o modelo, gera uma imagem a partir de
uma caption fixa de teste e salva em `debug_janus_output.png`. Útil para isolar problemas de
geração (ex: versão do Janus, CUDA, prompt/template) sem rodar o pipeline inteiro.

```bash
python3 scripts/debug_janus.py
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

#### Gráficos estatísticos (metodologia nova: Friedman + Wilcoxon + CLD)

```powershell
python scripts/analyze.py --config configs/analysis_local.yaml
```

#### Outputs gerados

| Pasta | Conteúdo |
|---|---|
| `reports/figures_iccc/` | Gráficos da metodologia ICCC (t-test, Mann-Whitney, ANOVA, radar) |
| `reports/figures/` | Gráficos por experimento (Friedman, Wilcoxon, CLD, clusters, noise, temporal) + diagnósticos avançados |

#### Diagnósticos avançados (sem outro modelo pra comparar — só o ArtCLIP)

Quatro análises que validam o ArtCLIP contra si mesmo, já que não há outro modelo de
referência disponível para comparação direta:

| Análise | Arquivos | O que mede |
|---|---|---|
| Monotonicidade | `monotonicity_table.png` | Spearman ρ / Kendall τ entre nível de ruído e score (Exp4) — o score deve cair conforme o ruído aumenta |
| Validade discriminante | `discriminative_validity_density.png`, `_table.png` | KS test + densidade sobreposta: Humano (APDDv2) vs. Sintético (Janus-1B/7B) vs. MNIST |
| Viés cultural | `cultural_bias_boxplot.png`, `_table.png` | Score dentro da base de treino (APDDv2) vs. fora (Portinari) — desvio de calibração cultural |
| Grupos de dificuldade | `difficulty_groups_density.png`, `_means.png`, `_table.png` | Regra de monotonicidade Fácil (humano) > Médio (sintético limpo) > Difícil (sintético + ruído estrutural no nível máximo); tabela sinaliza % de imagens do grupo Difícil acima da mediana do Fácil ("erro de calibração") |

Todas usam `outputs/exp1_apdd_uniform_bins/` como o Exp1 "principal" (já que `exp1_apdd`
agora tem 2 pastas — ver nota sobre `sampling.strategies` mais acima).

As **amostras visuais** (imagens de exemplo de entrada/saída de cada experimento) não são
geradas por esses scripts — elas rodam automaticamente dentro do `run.py`, no cluster, junto
com cada experimento (ver [Amostras visuais](#amostras-visuais) abaixo), e já chegam prontas
em `outputs/<experimento>/samples/` junto com o resto do download — sem precisar de um job
separado nem das imagens originais localmente.

---

### Relatórios HTML

`Paper_iccc.html` e `exp1_apdd.html` contam uma narrativa própria (EDA do dataset, valores
ausentes, amostragem antes/depois com toggle uniform_bins/proportional_stratified, amostras
de exemplo, e as perguntas sobre impacto do tamanho do modelo — Friedman+Wilcoxon, diferença
de score, consistência) — gerada por `scripts/analyze_paper.py`, separado do
`scripts/analyze.py`/`analyze_iccc.py` genéricos. Rode antes de gerar o HTML:

```powershell
python scripts/analyze_paper.py --config configs/analysis_local.yaml
```

Toda tabela sai em 2 formatos em `reports/figures_paper/`: `<nome>.png` (imagem, embutida no
HTML) e `<nome>.tex.txt` (LaTeX, pra colar direto na dissertação/paper).

Depois, gera as páginas do GitHub Pages com todas as figuras embutidas (base64):

```powershell
python scripts/generate_html_reports.py --config configs/analysis_local.yaml
```

Arquivos gerados em `reports/`:

| Arquivo | Conteúdo |
|---|---|
| `index.html` | Página inicial com resumo e navegação |
| `Paper_iccc.html` | Amostra original do ICCC 2025 (`exp0_iccc`, via `legacy_csv`) — EDA, amostragem, perguntas 1-3, validação de reprodutibilidade |
| `exp1_apdd.html` | Experimento 1 — mesma narrativa, `exp1_apdd_*`, com toggle uniform_bins/proportional_stratified + comparação entre as duas estratégias |
| `exp2a_portinari.html` | Experimento 2a — Portinari (AI captions) |
| `exp2b_portinari_human.html` | Experimento 2b — Portinari (human captions) |
| `exp3_mnist.html` | Experimento 3 — MNIST |
| `exp4_noise.html` | Experimento 4 — Ruído |
| `exp5_temporal.html` | Experimento 5 — Temporal |

> `exp2a`/`exp2b`/`exp3`/`exp4`/`exp5` ainda usam o gerador antigo (figuras de
> `scripts/analyze.py`/`analyze_iccc.py` + `reports/samples/`, que não é mais preenchido desde
> que as amostras passaram a ser geradas por `pipeline/samples.py` em `outputs/<exp>/samples/`)
> — atualização pendente, fora do escopo desta rodada.

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
