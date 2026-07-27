"""
Validação de reprodutibilidade: re-roda o pipeline com os mesmos dados do ICCC legado
e compara os scores resultantes com os do experimento original.

Fluxo:
  1. Lê sampled_dataset.csv (502 imagens + human GT do APDDv2)
  2. Lê sampled_SMALL (Janus-1B legado) e sampled_BIG (Janus-7B legado)
  3. Constrói pipeline_data.json com as mesmas imagens e captions (Description)
  4. Roda run.py --steps generation,scoring
  5. Compara os novos scores com os legados atributo a atributo
  6. Gera relatório + gráfico de comparação

Uso (local ou no cluster):
    python scripts/run_legacy_validation.py \\
        --config configs/exp_legacy_validation.yaml \\
        --legacy-small /path/to/sampled_SMALL_with_gen_scored.csv \\
        --legacy-big   /path/to/sampled_BIG_with_gen_scored.csv \\
        --apddv2-dir   /snfs1/speed/larissa.gomide/data/apddv2/ \\
        --out-dir      /snfs1/speed/larissa.gomide/outputs/exp_legacy_validation
"""

import argparse
import json
import os
import subprocess
import sys
import warnings

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

SCORE_ATTRS = [
    "Total aesthetic score", "Theme and logic", "Creativity",
    "Layout and composition", "Space and perspective",
    "Light and shadow", "Color", "Details and texture",
    "The overall", "Mood",
]


# ── helpers ───────────────────────────────────────────────────────────────────

def _stem(path):
    return os.path.splitext(os.path.basename(str(path)))[0]


def load_cfg(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_legacy(path):
    return pd.read_csv(path, encoding="latin1")


def load_new_scores(out_dir, model_key):
    """Lê scores do pipeline novo. model_key: 'Janus-Pro-1B' ou 'Janus-Pro-7B'."""
    scores_dir = os.path.join(out_dir, "scores")
    path = os.path.join(scores_dir, f"scores_{model_key}.csv")
    if not os.path.exists(path):
        print(f"[validate] scores não encontrados: {path}")
        return None
    return pd.read_csv(path)


# ── Step 1: construir pipeline_data.json ──────────────────────────────────────

def build_pipeline_data(legacy_small, apddv2_dir, out_dir):
    """
    Monta pipeline_data.json com os 502 itens do experimento legado.
    Cada item: filename (original APDDv2), path, caption (Description legada).
    """
    os.makedirs(out_dir, exist_ok=True)
    pipeline_path = os.path.join(out_dir, "pipeline_data.json")

    if os.path.exists(pipeline_path):
        print(f"[build] pipeline_data.json já existe em {pipeline_path} — pulando.")
        return pipeline_path

    data = []
    missing = []
    for _, row in legacy_small.iterrows():
        fname = str(row["filename"])
        stem  = _stem(fname)
        # tenta extensões comuns
        img_path = None
        for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
            candidate = os.path.join(apddv2_dir, stem + ext)
            if os.path.exists(candidate):
                img_path = candidate
                break
        if img_path is None:
            missing.append(stem)
            img_path = os.path.join(apddv2_dir, fname)  # tenta mesmo assim

        caption = str(row.get("Description", "")).strip()
        data.append({
            "filename": fname,
            "path":     img_path,
            "caption":  caption,
        })

    if missing:
        print(f"[build] {len(missing)} imagens não encontradas em {apddv2_dir}")
        print(f"         Primeiros 5: {missing[:5]}")

    with open(pipeline_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[build] pipeline_data.json criado: {len(data)} itens → {pipeline_path}")
    return pipeline_path


# ── Step 2: rodar pipeline ────────────────────────────────────────────────────

def run_pipeline(config_path, out_dir):
    """Roda run.py --steps generation,scoring."""
    cmd = [
        sys.executable, os.path.join(ROOT, "run.py"),
        "--config", config_path,
        "--steps", "generation,scoring",
    ]
    print(f"\n[run] {' '.join(cmd)}\n")
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        print(f"[run] FALHOU com código {result.returncode}")
        sys.exit(result.returncode)
    print("[run] Pipeline concluído.")


# ── Step 3: comparar scores ───────────────────────────────────────────────────

def _align(df_legacy, df_new, attr):
    """
    Alinha por stem do filename original.
    df_legacy tem 'filename' = stem da imagem APDDv2.
    df_new tem 'original_filename' (se existir) ou 'filename' (UUID gerada).
    Retorna (series_legacy, series_new) alinhadas.
    """
    legacy_stem = df_legacy["filename"].apply(_stem)

    if "original_filename" in df_new.columns:
        new_stem = df_new["original_filename"].apply(_stem)
    else:
        # fallback: ordinal (mesma ordem de geração)
        print(f"[align] 'original_filename' ausente — usando alinhamento posicional para {attr}")
        n = min(len(df_legacy), len(df_new))
        return df_legacy[attr].iloc[:n].reset_index(drop=True), \
               df_new[attr].iloc[:n].reset_index(drop=True)

    ldf = df_legacy[["filename", attr]].copy()
    ldf["stem"] = legacy_stem
    ndf = df_new[["original_filename", attr]].copy()
    ndf["stem"] = new_stem

    merged = ldf.merge(ndf, on="stem", suffixes=("_legacy", "_new"))
    return merged[f"{attr}_legacy"], merged[f"{attr}_new"]


def compare_scores(df_legacy, df_new, label, attrs):
    """Retorna dict attr → {mean_legacy, mean_new, diff, corr}."""
    results = {}
    for attr in attrs:
        if attr not in df_legacy.columns or attr not in df_new.columns:
            continue
        s_leg, s_new = _align(df_legacy, df_new, attr)
        s_leg = pd.to_numeric(s_leg, errors="coerce").dropna()
        s_new = pd.to_numeric(s_new, errors="coerce").dropna()
        n = min(len(s_leg), len(s_new))
        if n == 0:
            continue
        s_leg, s_new = s_leg.iloc[:n], s_new.iloc[:n]
        corr = s_leg.corr(s_new)
        results[attr] = {
            "mean_legacy": float(s_leg.mean()),
            "mean_new":    float(s_new.mean()),
            "diff":        float(s_new.mean() - s_leg.mean()),
            "corr":        float(corr) if not np.isnan(corr) else 0.0,
            "n":           n,
        }
    return results


# ── Step 4: gerar gráfico e relatório ─────────────────────────────────────────

def plot_comparison(results_1b, results_2b, out_path):
    attrs = [a for a in SCORE_ATTRS if a in results_1b or a in results_2b]
    x = np.arange(len(attrs))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(max(14, len(attrs) * 1.5), 6))

    for ax, results, title in [
        (axes[0], results_1b, "Janus-Pro-1B"),
        (axes[1], results_2b, "Janus-Pro-7B"),
    ]:
        means_leg = [results.get(a, {}).get("mean_legacy", np.nan) for a in attrs]
        means_new = [results.get(a, {}).get("mean_new",    np.nan) for a in attrs]

        ax.bar(x - width/2, means_leg, width, label="Legacy ICCC",
               color="#F2A007", hatch="xxx", edgecolor="black", alpha=0.85)
        ax.bar(x + width/2, means_new, width, label="Novo pipeline",
               color="#448FF2", hatch="", edgecolor="black", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(attrs, rotation=40, ha="right", fontsize=9)
        ax.set_ylabel("Score médio (ArtCLIP)")
        ax.set_title(f"{title}\nLegacy vs Novo pipeline (mesmas imagens + captions)")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle("Validação de Reprodutibilidade — Legacy ICCC vs Novo Pipeline",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[validate] gráfico salvo: {out_path}")


def plot_diff(results_1b, results_2b, out_path):
    """Gráfico das diferenças: novo − legacy por atributo."""
    attrs = [a for a in SCORE_ATTRS if a in results_1b or a in results_2b]
    x = np.arange(len(attrs))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(12, len(attrs) * 1.4), 5))
    diffs_1b = [results_1b.get(a, {}).get("diff", np.nan) for a in attrs]
    diffs_2b = [results_2b.get(a, {}).get("diff", np.nan) for a in attrs]

    ax.bar(x - width/2, diffs_1b, width, label="Janus-1B (novo − legacy)",
           color="#33A650", hatch="///", edgecolor="black", alpha=0.85)
    ax.bar(x + width/2, diffs_2b, width, label="Janus-7B (novo − legacy)",
           color="#448FF2", hatch="", edgecolor="black", alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(attrs, rotation=40, ha="right", fontsize=9)
    ax.set_ylabel("Diferença de score (Novo − Legacy)")
    ax.set_title("Diferença entre novo pipeline e legacy ICCC\n(valores próximos de 0 = boa reprodutibilidade)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[validate] gráfico de diff salvo: {out_path}")


def write_report(results_1b, results_2b, out_path):
    lines = [
        "RELATÓRIO DE VALIDAÇÃO — Legacy ICCC vs Novo Pipeline",
        "=" * 60,
        "Metodologia: mesmas 502 imagens e captions do ICCC original",
        "           geração e scoring re-executados pelo novo pipeline",
        "",
        f"{'Atributo':<30} {'Mean Legacy':>12} {'Mean Novo':>10} {'Diff':>8} {'Corr':>6} {'N':>5}",
        "-" * 75,
    ]
    for label, results in [("Janus-Pro-1B", results_1b), ("Janus-Pro-7B", results_2b)]:
        lines.append(f"\n── {label} ──")
        for attr in SCORE_ATTRS:
            if attr not in results:
                continue
            r = results[attr]
            lines.append(
                f"  {attr:<28} {r['mean_legacy']:>12.4f} {r['mean_new']:>10.4f} "
                f"{r['diff']:>+8.4f} {r['corr']:>6.3f} {r['n']:>5}"
            )
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[validate] relatório salvo: {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Re-roda pipeline com dados legados do ICCC e valida reprodutibilidade."
    )
    parser.add_argument("--config",       required=True,
                        help="YAML do experimento (ex: configs/exp_legacy_validation.yaml)")
    parser.add_argument("--legacy-small", required=True,
                        help="sampled_SMALL_with_gen_scored.csv (Janus-1B legado)")
    parser.add_argument("--legacy-big",   required=True,
                        help="sampled_BIG_with_gen_scored.csv (Janus-7B legado)")
    parser.add_argument("--apddv2-dir",   required=True,
                        help="Pasta com as imagens APDDv2 (ex: /snfs1/.../data/apddv2/)")
    parser.add_argument("--out-dir",      default=None,
                        help="Pasta de saída (padrão: lida do config)")
    parser.add_argument("--skip-run",     action="store_true",
                        help="Pula a execução do pipeline (usa scores já gerados)")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    out_dir = args.out_dir or os.path.join(
        "/snfs1/speed/larissa.gomide/outputs",
        cfg["experiment"]["name"]
    )
    figures_dir = os.path.join(out_dir, "figures_validation")
    os.makedirs(figures_dir, exist_ok=True)

    print(f"[validate] out_dir: {out_dir}")
    print(f"[validate] figures: {figures_dir}")

    # ── 1. Carrega legado ──────────────────────────────────────────────────
    print("\n[validate] Carregando CSVs legados...")
    df_small = load_legacy(args.legacy_small)
    df_big   = load_legacy(args.legacy_big)
    print(f"  SMALL: {len(df_small)} rows | BIG: {len(df_big)} rows")

    # ── 2. Constrói pipeline_data.json ────────────────────────────────────
    print("\n[validate] Construindo pipeline_data.json...")
    build_pipeline_data(df_small, args.apddv2_dir, out_dir)

    # ── 3. Roda pipeline (generation + scoring) ───────────────────────────
    if not args.skip_run:
        run_pipeline(args.config, out_dir)
    else:
        print("[validate] --skip-run: pulando pipeline.")

    # ── 4. Lê novos scores ────────────────────────────────────────────────
    print("\n[validate] Lendo scores do novo pipeline...")
    df_new_1b = load_new_scores(out_dir, "Janus-Pro-1B")
    df_new_7b = load_new_scores(out_dir, "Janus-Pro-7B")

    if df_new_1b is None and df_new_7b is None:
        print("[validate] Nenhum score novo encontrado. Abortando comparação.")
        sys.exit(1)

    # ── 5. Compara ────────────────────────────────────────────────────────
    print("\n[validate] Comparando scores...")
    results_1b = compare_scores(df_small, df_new_1b, "Janus-Pro-1B", SCORE_ATTRS) \
                 if df_new_1b is not None else {}
    results_7b = compare_scores(df_big,   df_new_7b, "Janus-Pro-7B", SCORE_ATTRS) \
                 if df_new_7b is not None else {}

    # ── 6. Gera outputs ───────────────────────────────────────────────────
    plot_comparison(results_1b, results_7b,
                    os.path.join(figures_dir, "validation_means.png"))
    plot_diff(results_1b, results_7b,
              os.path.join(figures_dir, "validation_diff.png"))
    write_report(results_1b, results_7b,
                 os.path.join(out_dir, "validation_report.txt"))

    print("\n✓ Validação concluída.")
    print(f"  {figures_dir}/validation_means.png  — scores lado a lado")
    print(f"  {figures_dir}/validation_diff.png   — diferença novo − legacy")
    print(f"  {out_dir}/validation_report.txt     — tabela numérica")


if __name__ == "__main__":
    main()
