"""
Análise e visualização dos experimentos.

Uso:
    python3 scripts/analyze.py --config configs/analysis.yaml

Gera todos os gráficos e relatório estatístico em:
    <paths.reports>/figures/
    <paths.reports>/stats_report.txt
"""

import argparse
import json
import os
import random
import sys

# Garante que o raiz do projeto está no path (necessário quando rodado como script)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import textwrap
import warnings
from pathlib import Path

import imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import yaml
from PIL import Image
from scipy.stats import (f_oneway, mannwhitneyu, pearsonr, spearmanr,
                         ttest_ind)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════════════════════
# Configuração
# ═══════════════════════════════════════════════════════════════════════════════

def load_cfg(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def L(cfg: dict, *keys):
    """Acessa rótulo no idioma configurado."""
    node = cfg["labels"][cfg["lang"]]
    for k in keys:
        node = node[k]
    return node


def attr_label(cfg: dict, attr: str) -> str:
    return cfg["labels"][cfg["lang"]]["attributes"].get(attr, attr)


def ds_label(cfg: dict, key: str) -> str:
    return cfg["labels"][cfg["lang"]]["datasets"].get(key, key)


# ═══════════════════════════════════════════════════════════════════════════════
# Carregamento de dados
# ═══════════════════════════════════════════════════════════════════════════════

def load_scores(exp_dir: str, source: str) -> pd.DataFrame | None:
    path = os.path.join(exp_dir, "scores", f"scores_{source}.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def load_pipeline_data(exp_dir: str) -> list:
    path = os.path.join(exp_dir, "pipeline_data.json")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)


def available_sources(exp_dir: str) -> list[str]:
    scores_dir = os.path.join(exp_dir, "scores")
    if not os.path.isdir(scores_dir):
        return []
    sources = []
    for f in os.listdir(scores_dir):
        if f.startswith("scores_") and f.endswith(".csv"):
            sources.append(f.replace("scores_", "").replace(".csv", ""))
    return sources


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers visuais
# ═══════════════════════════════════════════════════════════════════════════════

def style_key(name: str) -> str:
    """Mapeia nome de fonte/dataset para chave de estilo."""
    n = name.lower()
    if "1b" in n:   return "janus_1b"
    if "7b" in n:   return "janus_7b"
    if "mnist" in n: return "mnist"
    return "original"


def get_color(cfg, key: str) -> str:
    return cfg["palette"].get(key, cfg["palette"]["original"])


def get_hatch(cfg, key: str) -> str:
    return cfg["hatches"].get(key, "")


def get_ls(cfg, key: str) -> str:
    return cfg["linestyles"].get(key, "solid")


def get_marker(cfg, key: str) -> str:
    return cfg["markers"].get(key, "o")


def fig_base(cfg, square=False):
    size = cfg["figures"]["figsize_sq"] if square else cfg["figures"]["figsize"]
    fig, ax = plt.subplots(figsize=size)
    plt.rcParams.update({"font.size": cfg["figures"]["font_size"]})
    return fig, ax


def save(fig, path: str, cfg: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=cfg["figures"]["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path}")


def sig_marker(p, alpha=0.05) -> str:
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < alpha: return "*"
    return "ns"


# ═══════════════════════════════════════════════════════════════════════════════
# Testes estatísticos
# ═══════════════════════════════════════════════════════════════════════════════

def run_stats(cfg, groups: dict[str, pd.Series], attr: str, report_lines: list):
    """
    groups: dict {label -> Series of scores}
    """
    alpha = cfg["stats"]["alpha"]
    report_lines.append(f"\n  Attribute: {attr_label(cfg, attr)}")

    keys  = list(groups.keys())
    vals  = [groups[k].dropna() for k in keys]

    # ANOVA (se ≥ 3 grupos)
    if "anova" in cfg["stats"]["tests"] and len(vals) >= 3:
        stat, p = f_oneway(*vals)
        report_lines.append(f"    ANOVA: F={stat:.3f}, p={p:.4f} {sig_marker(p, alpha)}")

    # Pairwise t-test + Mann-Whitney
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = vals[i], vals[j]
            if len(a) < 2 or len(b) < 2:
                continue
            if "ttest" in cfg["stats"]["tests"]:
                t, p_t = ttest_ind(a, b, equal_var=False, nan_policy="omit")
                report_lines.append(
                    f"    t-test [{keys[i]} vs {keys[j]}]: t={t:.3f}, p={p_t:.4f} {sig_marker(p_t, alpha)}"
                )
            if "mannwhitney" in cfg["stats"]["tests"]:
                u, p_u = mannwhitneyu(a, b, alternative="two-sided")
                report_lines.append(
                    f"    Mann-Whitney [{keys[i]} vs {keys[j]}]: U={u:.1f}, p={p_u:.4f} {sig_marker(p_u, alpha)}"
                )

    # Correlação Pearson / Spearman entre os dois primeiros grupos
    if len(keys) >= 2 and len(vals[0]) == len(vals[1]):
        combined = pd.concat([vals[0].reset_index(drop=True),
                               vals[1].reset_index(drop=True)], axis=1).dropna()
        if len(combined) >= 3:
            if "pearson" in cfg["stats"]["tests"]:
                r, p = pearsonr(*combined.T.values)
                report_lines.append(f"    Pearson r={r:.3f}, p={p:.4f} {sig_marker(p, alpha)}")
            if "spearman" in cfg["stats"]["tests"]:
                r, p = spearmanr(*combined.T.values)
                report_lines.append(f"    Spearman ρ={r:.3f}, p={p:.4f} {sig_marker(p, alpha)}")


# ═══════════════════════════════════════════════════════════════════════════════
# EXP 1 — Diagnóstico de amostragem (antes vs. depois)
# ═══════════════════════════════════════════════════════════════════════════════

def exp1_sampling_diagnostic(cfg, out_dir: str):
    """
    Compara distribuição do score total no dataset completo (imagens disponíveis)
    vs. conjunto amostrado (pipeline_data.json).
    Também exibe distribuição por bin para evidenciar uniformidade.
    """
    total   = "Total aesthetic score"
    exp_dir = os.path.join(cfg["paths"]["outputs"], "exp1_apdd")
    meta_csv = cfg["paths"]["apddv2_csv"]

    # ── Dataset completo (filtrado pelas imagens que existem em disco) ─────────
    if not os.path.exists(meta_csv):
        print("[diag/exp1] CSV do APDDv2 não encontrado — pulando diagnóstico.")
        return

    df_full = pd.read_csv(meta_csv, encoding="ISO-8859-1")

    # Tenta inferir a coluna de score total
    score_col = next((c for c in df_full.columns
                      if "total" in c.lower() and "aesthetic" in c.lower()), None)
    if score_col is None:
        score_col = next((c for c in df_full.columns if "score" in c.lower()), None)
    if score_col is None:
        print("[diag/exp1] Coluna de score não encontrada no CSV — pulando diagnóstico.")
        return

    fn_col   = next((c for c in df_full.columns if "filename" in c.lower()), None)

    # Filtra imagens disponíveis no exp_dir
    img_root = cfg["paths"].get("apddv2_images",
                os.path.join(os.path.dirname(meta_csv), "APDDv2images", "APDDv2images"))
    if os.path.isdir(img_root):
        available = set(os.listdir(img_root))
        if fn_col:
            df_full = df_full[df_full[fn_col].apply(
                lambda f: os.path.basename(str(f).strip()) in available)]

    scores_full = df_full[score_col].dropna()

    # ── Conjunto amostrado: tenta pipeline_data.json, depois scores CSV ──────
    pipeline = load_pipeline_data(exp_dir)
    scores_sampled = pd.Series(
        [s.get("score") for s in pipeline if s.get("score") is not None],
        dtype=float,
    ).dropna()

    # Fallback: usa scores_original.csv (campo "Total aesthetic score") quando
    # o pipeline_data não tem scores ainda
    if len(scores_sampled) == 0:
        df_scores = load_scores(exp_dir, "original")
        if df_scores is not None and total in df_scores.columns:
            scores_sampled = df_scores[total].dropna()
        # Fallback 2: usa o score do CSV completo filtrado pelos filenames amostrados
        if len(scores_sampled) == 0 and pipeline and fn_col:
            sampled_fns = {os.path.basename(str(s.get("filename", ""))) for s in pipeline}
            scores_sampled = df_full[df_full[fn_col].apply(
                lambda f: os.path.basename(str(f).strip()) in sampled_fns
            )][score_col].dropna()

    n_full    = len(scores_full)
    n_sampled = len(scores_sampled)

    # ── Gráfico 1: KDE antes vs. depois ───────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"EXP 1 — Diagnóstico de Amostragem\n"
                 f"Disponível: {n_full} imagens  →  Amostrado: {n_sampled} imagens",
                 fontsize=12)

    for ax, scores, label, sk in [
        (axes[0], scores_full,    f"Antes (N={n_full})",    "original"),
        (axes[1], scores_sampled, f"Depois (N={n_sampled})", "janus_1b"),
    ]:
        if len(scores) == 0:
            ax.text(0.5, 0.5, "Sem dados", ha="center", va="center",
                    transform=ax.transAxes, fontsize=12)
            ax.set_title(label)
            continue
        ax.hist(scores, bins=30,
                color=get_color(cfg, sk), hatch=get_hatch(cfg, sk),
                edgecolor="black", alpha=0.7, label=label, density=True)
        scores.plot.kde(ax=ax, color="black", linewidth=1.5, linestyle="--")
        ax.axvline(scores.mean(), color=get_color(cfg, "highlight"),
                   linestyle="solid", linewidth=1.5,
                   label=f"Média = {scores.mean():.2f}")
        ax.set_xlabel(L(cfg, "axes", "score"))
        ax.set_ylabel("Densidade")
        ax.set_title(label)
        ax.legend(fontsize=9)
        if cfg["figures"]["grid"]: ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save(fig, os.path.join(out_dir, "exp1_sampling_before_after.png"), cfg)

    # ── Gráfico 2: contagem por bin (uniformidade) ────────────────────────────
    n_bins = 30
    bin_edges = np.linspace(
        min(scores_full.min(), scores_sampled.min()),
        max(scores_full.max(), scores_sampled.max()),
        n_bins + 1,
    )

    counts_full,    _ = np.histogram(scores_full,    bins=bin_edges)
    counts_sampled, _ = np.histogram(scores_sampled, bins=bin_edges)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    width = (bin_edges[1] - bin_edges[0]) * 0.4

    fig, ax = fig_base(cfg)
    ax.bar(bin_centers - width / 2, counts_full, width=width,
           color=get_color(cfg, "original"), hatch=get_hatch(cfg, "original"),
           edgecolor="black", alpha=0.7, label=f"Disponível (N={n_full})")
    ax.bar(bin_centers + width / 2, counts_sampled, width=width,
           color=get_color(cfg, "janus_1b"), hatch=get_hatch(cfg, "janus_1b"),
           edgecolor="black", alpha=0.7, label=f"Amostrado (N={n_sampled})")
    ax.set_xlabel(L(cfg, "axes", "score"))
    ax.set_ylabel("Contagem")
    ax.set_title("EXP 1 — Uniformidade da Amostragem por Bin de Score")
    ax.legend()
    if cfg["figures"]["grid"]: ax.grid(True, axis="y", alpha=0.3)
    save(fig, os.path.join(out_dir, "exp1_sampling_uniformity.png"), cfg)

    print(f"  Disponível: {n_full} imagens | Amostrado: {n_sampled} imagens")
    print(f"  Antes  — média={scores_full.mean():.3f}, std={scores_full.std():.3f}")
    print(f"  Depois — média={scores_sampled.mean():.3f}, std={scores_sampled.std():.3f}")


# ═══════════════════════════════════════════════════════════════════════════════
# EXP 1 — APDDv2 baseline
# ═══════════════════════════════════════════════════════════════════════════════

def exp1_analysis(cfg, out_dir: str, report: list):
    exp_dir = os.path.join(cfg["paths"]["outputs"], "exp1_apdd")
    df = load_scores(exp_dir, "original")
    if df is None:
        print("[exp1] scores_original.csv não encontrado — pulando.")
        return

    attrs  = [a for a in cfg["score_attributes"] if a in df.columns]
    total  = "Total aesthetic score"

    report.append("\n" + "═"*60)
    report.append("EXP 1 — APDDv2 Baseline")
    report.append("═"*60)
    report.append(f"  N = {len(df)}")
    if total in df.columns:
        report.append(f"  {attr_label(cfg, total)}: mean={df[total].mean():.3f}, std={df[total].std():.3f}")

    # ── 1a. Distribuição de scores (KDE + histograma) ─────────────────────────
    fig, ax = fig_base(cfg)
    for attr in attrs[:6]:  # máximo 6 para não poluir a legenda
        key = style_key(attr)
        df[attr].plot.kde(ax=ax, label=attr_label(cfg, attr),
                          color=get_color(cfg, "original"), alpha=0.7)
    ax.set_xlabel(L(cfg, "axes", "score"))
    ax.set_title(L(cfg, "titles", "dist_scores"))
    if cfg["figures"]["grid"]: ax.grid(True, alpha=0.3)

    # Histograma do score total com hachura
    fig2, ax2 = fig_base(cfg)
    ax2.hist(df[total].dropna(), bins=30,
             color=get_color(cfg, "original"),
             hatch=get_hatch(cfg, "original"),
             edgecolor="black", alpha=0.7,
             label=ds_label(cfg, "original"))
    ax2.set_xlabel(L(cfg, "axes", "score"))
    ax2.set_ylabel("Count")
    ax2.set_title(L(cfg, "titles", "dist_scores"))
    if cfg["figures"]["grid"]: ax2.grid(True, alpha=0.3)
    ax2.legend()
    save(fig2, os.path.join(out_dir, "exp1_dist_scores.png"), cfg)
    plt.close(fig)

    # ── 1b. Score por categoria artística ────────────────────────────────────
    meta_csv = cfg["paths"]["apddv2_csv"]
    if os.path.exists(meta_csv) and total in df.columns:
        meta = pd.read_csv(meta_csv, encoding="ISO-8859-1")
        cat_col = next((c for c in ["Artistic Categories", "category"] if c in meta.columns), None)
        if cat_col:
            merged = df.merge(meta[["filename", cat_col]], on="filename", how="left")
            cats = merged[cat_col].dropna().unique()
            fig, ax = fig_base(cfg)
            data_by_cat = [merged.loc[merged[cat_col] == c, total].dropna() for c in cats]
            bp = ax.boxplot(data_by_cat, patch_artist=True, labels=cats)
            colors = list(cfg["palette"].values())
            for patch, color in zip(bp["boxes"], colors * len(cats)):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            ax.set_xlabel(L(cfg, "axes", "category"))
            ax.set_ylabel(L(cfg, "axes", "score"))
            ax.set_title(L(cfg, "titles", "by_category"))
            plt.xticks(rotation=30, ha="right")
            if cfg["figures"]["grid"]: ax.grid(True, axis="y", alpha=0.3)
            save(fig, os.path.join(out_dir, "exp1_by_category.png"), cfg)

            # Stats por categoria
            report.append("\n  Score por categoria:")
            for c, d in zip(cats, data_by_cat):
                report.append(f"    {c}: n={len(d)}, mean={d.mean():.3f}, std={d.std():.3f}")
            if len(data_by_cat) >= 3:
                _, p = f_oneway(*[d for d in data_by_cat if len(d) > 1])
                report.append(f"  ANOVA entre categorias: p={p:.4f} {sig_marker(p)}")

    # ── 1c. Radar — médias por atributo ───────────────────────────────────────
    radar_attrs = [a for a in cfg["radar_attributes"] if a in df.columns]
    if len(radar_attrs) >= 3:
        means = [df[a].mean() for a in radar_attrs]
        labels_r = [attr_label(cfg, a) for a in radar_attrs]
        N = len(radar_attrs)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        means_plot = means + means[:1]
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=cfg["figures"]["figsize_sq"],
                               subplot_kw=dict(polar=True))
        ax.fill(angles, means_plot, color=get_color(cfg, "original"), alpha=0.3,
                hatch=get_hatch(cfg, "original"))
        ax.plot(angles, means_plot, color=get_color(cfg, "original"),
                linewidth=2, marker=get_marker(cfg, "original"),
                label=ds_label(cfg, "original"))
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([textwrap.fill(l, 10) for l in labels_r], fontsize=9)
        ax.set_title(L(cfg, "titles", "radar"))
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
        save(fig, os.path.join(out_dir, "exp1_radar.png"), cfg)

    # ── 1d. Clustering — melhores e piores imagens ────────────────────────────
    feat_cols = [a for a in cfg["radar_attributes"] if a in df.columns]
    if len(feat_cols) >= 2 and len(df) >= cfg["clustering"]["n_clusters"]:
        X = df[feat_cols].dropna()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        km = KMeans(n_clusters=cfg["clustering"]["n_clusters"],
                    random_state=42, n_init=10)
        labels_k = km.fit_predict(X_scaled)

        fig, ax = fig_base(cfg, square=True)
        palette_list = list(cfg["palette"].values())
        hatches_list = list(cfg["hatches"].values())
        for k in range(cfg["clustering"]["n_clusters"]):
            mask = labels_k == k
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                       color=palette_list[k % len(palette_list)],
                       marker=list(cfg["markers"].values())[k % 5],
                       label=f"Cluster {k+1}", alpha=0.6, s=30)

        # Destaca top-N e bottom-N
        n_top = cfg["clustering"]["n_top"]
        if total in df.columns:
            scores_aligned = df.loc[X.index, total]
            top_idx  = scores_aligned.nlargest(n_top).index
            bot_idx  = scores_aligned.nsmallest(n_top).index
            top_mask = np.isin(X.index, top_idx)
            bot_mask = np.isin(X.index, bot_idx)
            ax.scatter(X_pca[top_mask, 0], X_pca[top_mask, 1],
                       color=get_color(cfg, "highlight"), marker="*",
                       s=120, label=f"Top-{n_top}", zorder=5)
            ax.scatter(X_pca[bot_mask, 0], X_pca[bot_mask, 1],
                       color=get_color(cfg, "mnist"), marker="v",
                       s=80, label=f"Bottom-{n_top}", zorder=5)

        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
        ax.set_title(f"Cluster Analysis — {L(cfg, 'titles', 'dist_scores')}")
        ax.legend()
        if cfg["figures"]["grid"]: ax.grid(True, alpha=0.3)
        save(fig, os.path.join(out_dir, "exp1_cluster.png"), cfg)

        report.append(f"\n  PCA variance explained: PC1={pca.explained_variance_ratio_[0]*100:.1f}%, "
                      f"PC2={pca.explained_variance_ratio_[1]*100:.1f}%")


# ═══════════════════════════════════════════════════════════════════════════════
# EXP 2 — Portinari (original vs geradas, 2a vs 2b)
# ═══════════════════════════════════════════════════════════════════════════════

def exp2_analysis(cfg, out_dir: str, report: list):
    total = "Total aesthetic score"
    dfs   = {}

    for exp_name, label_key in [("exp2a_portinari", "exp2a"),
                                  ("exp2b_portinari_human", "exp2b")]:
        exp_dir = os.path.join(cfg["paths"]["outputs"], exp_name)
        for src in available_sources(exp_dir):
            df = load_scores(exp_dir, src)
            if df is not None and total in df.columns:
                key = f"{label_key}_{src}"
                dfs[key] = df

    if not dfs:
        print("[exp2] Nenhum CSV de scores encontrado — pulando.")
        return

    report.append("\n" + "═"*60)
    report.append("EXP 2 — Portinari: Original vs. Geradas")
    report.append("═"*60)

    # ── 2a. Boxplot comparativo ────────────────────────────────────────────────
    fig, ax = fig_base(cfg)
    positions  = range(len(dfs))
    labels_box = []
    for pos, (key, df) in zip(positions, dfs.items()):
        sk = style_key(key)
        data = df[total].dropna()
        bp = ax.boxplot(data, positions=[pos], patch_artist=True, widths=0.6)
        bp["boxes"][0].set_facecolor(get_color(cfg, sk))
        bp["boxes"][0].set_hatch(get_hatch(cfg, sk))
        bp["boxes"][0].set_alpha(0.7)
        bp["boxes"][0].set_edgecolor("black")
        short = key.replace("exp2a_", "2a/").replace("exp2b_", "2b/")
        labels_box.append(short)
        report.append(f"  {key}: n={len(data)}, mean={data.mean():.3f}, std={data.std():.3f}")

    ax.set_xticks(list(positions))
    ax.set_xticklabels(labels_box, rotation=20, ha="right")
    ax.set_ylabel(L(cfg, "axes", "score"))
    ax.set_title(L(cfg, "titles", "portinari_compare"))
    if cfg["figures"]["grid"]: ax.grid(True, axis="y", alpha=0.3)
    save(fig, os.path.join(out_dir, "exp2_boxplot.png"), cfg)

    # ── 2b. Linha: original vs 1B vs 7B (ordenado pelo original) ─────────────
    df_orig = dfs.get("exp2a_original") if dfs.get("exp2a_original") is not None else dfs.get("exp2b_original")
    df_1b   = dfs.get("exp2a_Janus-Pro-1B") if dfs.get("exp2a_Janus-Pro-1B") is not None else dfs.get("exp2b_Janus-Pro-1B")
    df_7b   = dfs.get("exp2a_Janus-Pro-7B") if dfs.get("exp2a_Janus-Pro-7B") is not None else dfs.get("exp2b_Janus-Pro-7B")

    if df_orig is not None and total in df_orig.columns:
        orig_sorted = df_orig[total].dropna().sort_values().reset_index(drop=True)
        fig, ax = fig_base(cfg)
        x = range(len(orig_sorted))
        ax.plot(x, orig_sorted.values,
                color=get_color(cfg, "original"),
                linestyle=get_ls(cfg, "original"),
                label=ds_label(cfg, "original"), linewidth=1.5)

        for df_gen, sk, lbl_key in [(df_1b, "janus_1b", "janus_1b"),
                                     (df_7b, "janus_7b", "janus_7b")]:
            if df_gen is not None and total in df_gen.columns:
                gen_vals = df_gen[total].dropna().reset_index(drop=True)
                n = min(len(gen_vals), len(orig_sorted))
                ax.plot(range(n), gen_vals.values[:n],
                        color=get_color(cfg, sk),
                        linestyle=get_ls(cfg, sk),
                        label=ds_label(cfg, lbl_key), linewidth=1.5, alpha=0.85)

        ax.set_xlabel(L(cfg, "axes", "sample_rank"))
        ax.set_ylabel(L(cfg, "axes", "score"))
        ax.set_title(L(cfg, "titles", "original_vs_janus"))
        ax.legend()
        if cfg["figures"]["grid"]: ax.grid(True, alpha=0.3)
        save(fig, os.path.join(out_dir, "exp2_lines_original_vs_janus.png"), cfg)

    # ── 2c. Diferença: original − 1B e original − 7B por atributo ────────────
    radar_attrs = [a for a in cfg["radar_attributes"] if
                   df_orig is not None and a in (df_orig.columns if df_orig is not None else [])]
    if df_orig is not None and radar_attrs:
        fig, ax = fig_base(cfg)
        x_pos  = np.arange(len(radar_attrs))
        width  = 0.35
        labels_r = [attr_label(cfg, a) for a in radar_attrs]

        for i, (df_gen, sk, lbl_key) in enumerate([(df_1b, "janus_1b", "janus_1b"),
                                                     (df_7b, "janus_7b", "janus_7b")]):
            if df_gen is not None:
                diffs = [df_orig[a].mean() - df_gen[a].mean()
                         for a in radar_attrs if a in df_gen.columns]
                offset = (i - 0.5) * width
                ax.bar(x_pos + offset, diffs, width,
                       label=ds_label(cfg, lbl_key),
                       color=get_color(cfg, sk),
                       hatch=get_hatch(cfg, sk),
                       edgecolor="black", alpha=0.8)

        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels_r, rotation=35, ha="right")
        ax.set_ylabel("Score Difference")
        ax.set_title(L(cfg, "titles", "score_diff"))
        ax.legend()
        if cfg["figures"]["grid"]: ax.grid(True, axis="y", alpha=0.3)
        save(fig, os.path.join(out_dir, "exp2_score_diff.png"), cfg)

    # ── 2d. Impacto da fonte de caption (2a vs 2b) ────────────────────────────
    for model_suffix, sk, lbl_key in [("Janus-Pro-1B", "janus_1b", "janus_1b"),
                                       ("Janus-Pro-7B", "janus_7b", "janus_7b")]:
        d2a = dfs.get(f"exp2a_{model_suffix}")
        d2b = dfs.get(f"exp2b_{model_suffix}")
        if d2a is not None and d2b is not None and total in d2a.columns:
            fig, ax = fig_base(cfg)
            for df_c, cap_key, offset in [(d2a, "exp2a", -0.2), (d2b, "exp2b", 0.2)]:
                data = df_c[total].dropna()
                bp = ax.boxplot(data, positions=[offset], widths=0.3, patch_artist=True)
                bp["boxes"][0].set_facecolor(get_color(cfg, sk))
                bp["boxes"][0].set_hatch(get_hatch(cfg, sk))
                bp["boxes"][0].set_alpha(0.7)
                bp["boxes"][0].set_edgecolor("black")

            ax.set_xticks([-0.2, 0.2])
            ax.set_xticklabels([ds_label(cfg, "exp2a"), ds_label(cfg, "exp2b")])
            ax.set_ylabel(L(cfg, "axes", "score"))
            ax.set_title(f"{L(cfg, 'titles', 'caption_source')} — {model_suffix}")
            if cfg["figures"]["grid"]: ax.grid(True, axis="y", alpha=0.3)
            save(fig, os.path.join(out_dir,
                 f"exp2_caption_source_{model_suffix.replace('-','_')}.png"), cfg)

            # Stat
            report.append(f"\n  Caption source ({model_suffix}):")
            run_stats(cfg,
                      {ds_label(cfg, "exp2a"): d2a[total],
                       ds_label(cfg, "exp2b"): d2b[total]},
                      total, report)


# ═══════════════════════════════════════════════════════════════════════════════
# EXP 3 — Arte vs. Não-arte
# ═══════════════════════════════════════════════════════════════════════════════

def exp3_analysis(cfg, out_dir: str, report: list):
    total   = "Total aesthetic score"
    exp1dir = os.path.join(cfg["paths"]["outputs"], "exp1_apdd")
    exp3dir = os.path.join(cfg["paths"]["outputs"], "exp3_mnist")

    df_art    = load_scores(exp1dir, "original")
    df_noart  = load_scores(exp3dir, "original")

    if df_art is None or df_noart is None:
        print("[exp3] Dados insuficientes — pulando.")
        return

    report.append("\n" + "═"*60)
    report.append("EXP 3 — Arte vs. Não-Arte (APDDv2 vs. MNIST)")
    report.append("═"*60)

    fig, ax = fig_base(cfg)
    for df_s, sk, lbl_key in [(df_art, "original", "original"),
                               (df_noart, "mnist",   "mnist")]:
        if total in df_s.columns:
            data = df_s[total].dropna()
            data.plot.kde(ax=ax,
                          color=get_color(cfg, sk),
                          linestyle=get_ls(cfg, sk),
                          label=ds_label(cfg, lbl_key),
                          linewidth=2)
            ax.fill_between(
                np.linspace(data.min(), data.max(), 200),
                0,
                [ax.lines[-1].get_ydata()[
                    np.argmin(np.abs(ax.lines[-1].get_xdata() - xi))]
                 for xi in np.linspace(data.min(), data.max(), 200)],
                alpha=0.15,
                hatch=get_hatch(cfg, sk),
                color=get_color(cfg, sk),
            )
            report.append(f"  {ds_label(cfg, lbl_key)}: n={len(data)}, "
                          f"mean={data.mean():.3f}, std={data.std():.3f}")

    ax.set_xlabel(L(cfg, "axes", "score"))
    ax.set_ylabel("Density")
    ax.set_title(L(cfg, "titles", "art_vs_noart"))
    ax.legend()
    if cfg["figures"]["grid"]: ax.grid(True, alpha=0.3)
    save(fig, os.path.join(out_dir, "exp3_art_vs_noart.png"), cfg)

    run_stats(cfg,
              {ds_label(cfg, "original"): df_art[total],
               ds_label(cfg, "mnist"):    df_noart[total]},
              total, report)


# ═══════════════════════════════════════════════════════════════════════════════
# EXP 4 — Robustez a ruído
# ═══════════════════════════════════════════════════════════════════════════════

def exp4_analysis(cfg, out_dir: str, report: list):
    total   = "Total aesthetic score"
    exp_dir = os.path.join(cfg["paths"]["outputs"], "exp4_noise")
    df      = load_scores(exp_dir, "original")
    meta    = load_pipeline_data(exp_dir)

    if df is None or not meta:
        print("[exp4] Dados insuficientes — pulando.")
        return

    meta_df = pd.DataFrame(meta)
    if "filename" not in meta_df.columns:
        print("[exp4] pipeline_data.json sem coluna 'filename' — pulando.")
        return

    # Junta scores com metadados (noise_type, noise_level)
    meta_df["filename"] = meta_df["filename"].apply(os.path.basename)
    df["filename"]      = df["filename"].apply(os.path.basename)
    merged = df.merge(meta_df[["filename", "noise_type", "noise_level"]], on="filename", how="left")

    report.append("\n" + "═"*60)
    report.append("EXP 4 — Impacto do Ruído")
    report.append("═"*60)

    if "noise_type" not in merged.columns or total not in merged.columns:
        print("[exp4] Colunas noise_type ou score ausentes.")
        return

    noise_types = merged["noise_type"].dropna().unique()
    lang_noise  = cfg["labels"][cfg["lang"]]["noise_types"]

    fig, ax = fig_base(cfg)
    style_keys_noise = ["janus_1b", "janus_7b", "mnist"]

    for i, nt in enumerate(noise_types):
        sk   = style_keys_noise[i % len(style_keys_noise)]
        sub  = merged[merged["noise_type"] == nt]
        mean = sub.groupby("noise_level")[total].mean()
        sem  = sub.groupby("noise_level")[total].sem().fillna(0).astype(float)
        lbl  = lang_noise.get(nt, nt)

        ax.plot(mean.index, mean.values.astype(float),
                color=get_color(cfg, sk),
                linestyle=get_ls(cfg, sk),
                marker=get_marker(cfg, sk),
                label=lbl, linewidth=2)
        ax.fill_between(mean.index.astype(float),
                        (mean.values - sem.values).astype(float),
                        (mean.values + sem.values).astype(float),
                        color=get_color(cfg, sk), alpha=0.15,
                        hatch=get_hatch(cfg, sk))

        report.append(f"\n  {lbl}:")
        for lvl, grp in sub.groupby("noise_level"):
            report.append(f"    level={lvl}: mean={grp[total].mean():.3f}, "
                          f"std={grp[total].std():.3f}, n={len(grp)}")

    ax.set_xlabel(L(cfg, "axes", "noise_level"))
    ax.set_ylabel(L(cfg, "axes", "score"))
    ax.set_title(L(cfg, "titles", "noise_impact"))
    ax.legend()
    if cfg["figures"]["grid"]: ax.grid(True, alpha=0.3)
    save(fig, os.path.join(out_dir, "exp4_noise_impact.png"), cfg)


# ═══════════════════════════════════════════════════════════════════════════════
# EXP 5a — Consistência temporal
# ═══════════════════════════════════════════════════════════════════════════════

def exp5a_analysis(cfg, out_dir: str, report: list):
    total   = "Total aesthetic score"
    exp_dir = os.path.join(cfg["paths"]["outputs"], "exp5a_temporal")
    df      = load_scores(exp_dir, "original")
    meta    = load_pipeline_data(exp_dir)

    if df is None or not meta:
        print("[exp5a] Dados insuficientes — pulando.")
        return

    meta_df = pd.DataFrame(meta)
    meta_df["filename"] = meta_df["filename"].apply(os.path.basename)
    df["filename"]      = df["filename"].apply(os.path.basename)
    merged = df.merge(meta_df[["filename", "video_id", "frame_idx"]], on="filename", how="left")

    report.append("\n" + "═"*60)
    report.append("EXP 5a — Consistência Temporal")
    report.append("═"*60)

    if "video_id" not in merged.columns:
        print("[exp5a] Coluna video_id ausente.")
        return

    # ── Linha: N vídeos selecionados aleatoriamente ───────────────────────────
    n_videos = cfg["temporal"]["n_videos_to_plot"]
    video_ids = sorted(merged["video_id"].dropna().unique())
    selected  = video_ids[:n_videos]

    fig, ax = fig_base(cfg)
    palette_list = list(cfg["palette"].values())
    ls_list      = list(cfg["linestyles"].values())

    for i, vid in enumerate(selected):
        sub = merged[merged["video_id"] == vid].sort_values("frame_idx")
        ax.plot(sub["frame_idx"], sub[total],
                color=palette_list[i % len(palette_list)],
                linestyle=ls_list[i % len(ls_list)],
                marker="o", markersize=4,
                label=f"Video {vid}", linewidth=1.5, alpha=0.85)

    # Média global por frame_idx
    mean_per_frame = merged.groupby("frame_idx")[total].mean()
    ax.plot(mean_per_frame.index, mean_per_frame.values,
            color="black", linestyle="solid", linewidth=2.5,
            marker="D", markersize=5, label="Mean (all videos)", zorder=10)

    ax.set_xlabel(L(cfg, "axes", "frame_idx"))
    ax.set_ylabel(L(cfg, "axes", "score"))
    ax.set_title(L(cfg, "titles", "temporal_consist"))
    ax.legend(fontsize=9)
    if cfg["figures"]["grid"]: ax.grid(True, alpha=0.3)
    save(fig, os.path.join(out_dir, "exp5a_temporal.png"), cfg)

    # Variância intra vs inter-vídeo
    intra = merged.groupby("video_id")[total].std().mean()
    inter = merged.groupby("video_id")[total].mean().std()
    report.append(f"  Intra-video std (mean): {intra:.4f}")
    report.append(f"  Inter-video std:        {inter:.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
# EXP 5b — Detecção de degradação progressiva
# ═══════════════════════════════════════════════════════════════════════════════

def exp5b_analysis(cfg, out_dir: str, report: list, baseline_dir: str = None):
    total   = "Total aesthetic score"
    exp_dir = os.path.join(cfg["paths"]["outputs"], "exp5b_temporal_error")
    df      = load_scores(exp_dir, "original")
    meta    = load_pipeline_data(exp_dir)

    if df is None or not meta:
        print("[exp5b] Dados insuficientes — pulando.")
        return

    meta_df = pd.DataFrame(meta)
    meta_df["filename"] = meta_df["filename"].apply(os.path.basename)
    df["filename"]      = df["filename"].apply(os.path.basename)

    merge_cols = ["filename"] + [c for c in ["noise_type", "degradation_pct", "frame_idx"]
                                  if c in meta_df.columns]
    merged = df.merge(meta_df[merge_cols], on="filename", how="left")

    report.append("\n" + "═"*60)
    report.append("EXP 5b — Detecção de Degradação Progressiva")
    report.append("═"*60)

    if "degradation_pct" not in merged.columns or "noise_type" not in merged.columns:
        print("[exp5b] Colunas degradation_pct ou noise_type ausentes.")
        return

    lang_noise   = cfg["labels"][cfg["lang"]]["noise_types"]
    noise_types  = merged["noise_type"].dropna().unique()
    style_keys_noise = ["janus_1b", "janus_7b", "mnist"]

    # Baseline exp5a — média global por frame_idx
    baseline_mean = None
    if baseline_dir and os.path.exists(baseline_dir):
        df_b   = load_scores(baseline_dir, "original")
        meta_b = load_pipeline_data(baseline_dir)
        if df_b is not None and meta_b:
            meta_b_df = pd.DataFrame(meta_b)
            meta_b_df["filename"] = meta_b_df["filename"].apply(os.path.basename)
            df_b["filename"]      = df_b["filename"].apply(os.path.basename)
            merged_b = df_b.merge(meta_b_df[["filename", "frame_idx"]], on="filename", how="left")
            if "frame_idx" in merged_b.columns:
                baseline_mean = merged_b.groupby("frame_idx")[total].mean()

    fig, ax = fig_base(cfg)

    # Baseline
    if baseline_mean is not None:
        # Mapeia frame_idx → degradation_pct (0 a 100)
        max_fi = baseline_mean.index.max()
        x_base = baseline_mean.index / max(max_fi, 1) * 100
        ax.plot(x_base, baseline_mean.values,
                color=get_color(cfg, "original"),
                linestyle=get_ls(cfg, "original"),
                marker=get_marker(cfg, "original"),
                markersize=3, linewidth=2,
                label=ds_label(cfg, "original") + " (baseline)")

    for i, nt in enumerate(noise_types):
        sk  = style_keys_noise[i % len(style_keys_noise)]
        sub = merged[merged["noise_type"] == nt]
        mean = sub.groupby("degradation_pct")[total].mean()
        sem  = sub.groupby("degradation_pct")[total].sem().fillna(0).astype(float)
        lbl  = lang_noise.get(nt, nt)

        ax.plot(mean.index.astype(float), mean.values.astype(float),
                color=get_color(cfg, sk),
                linestyle=get_ls(cfg, sk),
                marker=get_marker(cfg, sk),
                markersize=3, linewidth=2, label=lbl)
        ax.fill_between(mean.index.astype(float),
                        (mean.values - sem.values).astype(float),
                        (mean.values + sem.values).astype(float),
                        color=get_color(cfg, sk), alpha=0.12,
                        hatch=get_hatch(cfg, sk))

        report.append(f"\n  {lbl}: mean@0%={sub[sub['degradation_pct']<=1][total].mean():.3f}, "
                      f"mean@100%={sub[sub['degradation_pct']>=99][total].mean():.3f}")

    ax.set_xlabel(L(cfg, "axes", "degradation"))
    ax.set_ylabel(L(cfg, "axes", "score"))
    ax.set_title(L(cfg, "titles", "degradation_detect"))
    ax.legend()
    if cfg["figures"]["grid"]: ax.grid(True, alpha=0.3)
    save(fig, os.path.join(out_dir, "exp5b_degradation.png"), cfg)


# ═══════════════════════════════════════════════════════════════════════════════
# Amostras visuais — helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _show_image(ax, path, title=None):
    if path and os.path.exists(path):
        ax.imshow(Image.open(path).convert("RGB"))
    else:
        ax.set_facecolor("#CCCCCC")
        ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                transform=ax.transAxes, fontsize=10, color="#666666")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=9, pad=3)


def _show_text(ax, text, title=None, fontsize=8):
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=9, pad=3)
    wrapped = textwrap.fill(str(text) if text else "(sem texto)", width=38)
    ax.text(0.05, 0.95, wrapped, ha="left", va="top",
            transform=ax.transAxes, fontsize=fontsize, family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#F5F5F5",
                      edgecolor="#CCCCCC", alpha=0.8))


def _pick_samples(data, n=3, seed=42):
    rng  = random.Random(seed)
    pool = [s for s in data if s.get("path") and os.path.exists(s["path"])]
    return rng.sample(pool, min(n, len(pool)))


def _gen_path(exp_dir, model_name, filename):
    base = os.path.splitext(os.path.basename(filename))[0]
    return os.path.join(exp_dir, "generated", model_name, base + ".png")


def _frames_for_video(data, video_id, max_frames=None):
    frames = sorted(
        [s for s in data if s.get("video_id") == video_id],
        key=lambda s: s.get("frame_idx") or 0,
    )
    return frames[:max_frames] if max_frames else frames


# ═══════════════════════════════════════════════════════════════════════════════
# Amostras visuais — por experimento
# ═══════════════════════════════════════════════════════════════════════════════

def samples_exp1(cfg, samples_dir):
    exp_dir = os.path.join(cfg["paths"]["outputs"], "exp1_apdd")
    data    = load_pipeline_data(exp_dir)
    if not data:
        print("[samples/exp1] pipeline_data.json não encontrado — pulando.")
        return

    items   = _pick_samples(data, n=3)
    n_rows  = len(items)
    fig     = plt.figure(figsize=(16, 5 * n_rows))
    fig.suptitle("EXP 1 — APDDv2: Original → Caption → Generated", fontsize=13, y=1.01)
    gs = gridspec.GridSpec(n_rows, 4, figure=fig, hspace=0.4, wspace=0.15)

    for row, s in enumerate(items):
        _show_image(_ax(fig, gs, row, 0), s.get("path"),           title="Original (APDDv2)")
        _show_text (_ax(fig, gs, row, 1), s.get("caption", ""),    title="Caption (Janus-7B)")
        _show_image(_ax(fig, gs, row, 2), _gen_path(exp_dir, "Janus-Pro-1B", s["filename"]),
                    title="Generated — Janus-Pro-1B")
        _show_image(_ax(fig, gs, row, 3), _gen_path(exp_dir, "Janus-Pro-7B", s["filename"]),
                    title="Generated — Janus-Pro-7B")

    save(fig, os.path.join(samples_dir, "exp1_samples.png"), cfg)


def samples_exp2(cfg, samples_dir):
    for exp_name, cap_title in [("exp2a_portinari", "Caption (Janus-7B)"),
                                  ("exp2b_portinari_human", "Caption (humana/EN)")]:
        exp_dir = os.path.join(cfg["paths"]["outputs"], exp_name)
        data    = load_pipeline_data(exp_dir)
        if not data:
            print(f"[samples/{exp_name}] pipeline_data.json não encontrado — pulando.")
            continue

        items  = _pick_samples(data, n=3)
        n_rows = len(items)
        fig    = plt.figure(figsize=(20, 5 * n_rows))
        fig.suptitle(f"EXP 2 — Portinari ({exp_name})", fontsize=13, y=1.01)
        gs = gridspec.GridSpec(n_rows, 4, figure=fig, hspace=0.4, wspace=0.15)

        for row, s in enumerate(items):
            _show_text (_ax(fig, gs, row, 0), s.get("caption", ""),  title=cap_title)
            _show_image(_ax(fig, gs, row, 1), s.get("path"),          title="Original (Portinari)")
            _show_image(_ax(fig, gs, row, 2), _gen_path(exp_dir, "Janus-Pro-1B", s["filename"]),
                        title="Generated — Janus-Pro-1B")
            _show_image(_ax(fig, gs, row, 3), _gen_path(exp_dir, "Janus-Pro-7B", s["filename"]),
                        title="Generated — Janus-Pro-7B")

        save(fig, os.path.join(samples_dir, f"{exp_name}_samples.png"), cfg)


def samples_exp3(cfg, samples_dir):
    exp_dir = os.path.join(cfg["paths"]["outputs"], "exp3_mnist")
    data    = load_pipeline_data(exp_dir)
    if not data:
        print("[samples/exp3] pipeline_data.json não encontrado — pulando.")
        return

    pool    = [s for s in data if s.get("path") and os.path.exists(s["path"])]
    items   = random.Random(42).sample(pool, min(20, len(pool)))
    cols    = 5
    rows    = (len(items) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    fig.suptitle("EXP 3 — MNIST: Amostras", fontsize=13)
    axes = np.array(axes).flatten()

    for ax, s in zip(axes, items):
        _show_image(ax, s.get("path"), title=f"label={s.get('label','?')}")
    for ax in axes[len(items):]:
        ax.axis("off")

    plt.tight_layout()
    save(fig, os.path.join(samples_dir, "exp3_mnist_samples.png"), cfg)


def samples_exp4(cfg, samples_dir):
    from datasets.image import NOISE_FNS
    exp_dir = os.path.join(cfg["paths"]["outputs"], "exp4_noise")
    data    = load_pipeline_data(exp_dir)
    if not data:
        print("[samples/exp4] pipeline_data.json não encontrado — pulando.")
        return

    seen, base_items = set(), []
    for s in data:
        fn = s.get("filename")
        if fn not in seen and s.get("path") and os.path.exists(s["path"]):
            base_items.append(s)
            seen.add(fn)
        if len(base_items) == 3:
            break

    noise_types = ["gaussian", "blur", "shapes"]
    lang_noise  = cfg["labels"][cfg["lang"]]["noise_types"]
    n_cols      = 1 + len(noise_types)

    fig, axes = plt.subplots(len(base_items), n_cols,
                             figsize=(n_cols * 3.5, len(base_items) * 3.5))
    fig.suptitle("EXP 4 — Ruído: Original vs. Tipos de Ruído (nível 50)", fontsize=13)

    for row, s in enumerate(base_items):
        img = Image.open(s["path"]).convert("RGB")
        _show_image(axes[row, 0], s["path"], title="Original" if row == 0 else "")
        for col, nt in enumerate(noise_types, start=1):
            ax = axes[row, col]
            ax.imshow(NOISE_FNS[nt](img, 50))
            ax.axis("off")
            if row == 0:
                ax.set_title(lang_noise.get(nt, nt), fontsize=9)

    plt.tight_layout()
    save(fig, os.path.join(samples_dir, "exp4_noise_samples.png"), cfg)


def samples_exp5_grid(cfg, samples_dir):
    """Primeiros 5 frames de 3 vídeos diferentes."""
    exp_dir = os.path.join(cfg["paths"]["outputs"], "exp5a_temporal")
    data    = load_pipeline_data(exp_dir)
    if not data:
        print("[samples/exp5a] pipeline_data.json não encontrado — pulando.")
        return

    video_ids = sorted({s["video_id"] for s in data if "video_id" in s})[:3]
    n_frames  = 5
    fig, axes = plt.subplots(len(video_ids), n_frames,
                             figsize=(n_frames * 3, len(video_ids) * 3))
    fig.suptitle("EXP 5a — Primeiros 5 frames de 3 vídeos", fontsize=13)

    for row, vid in enumerate(video_ids):
        frames = _frames_for_video(data, vid, max_frames=n_frames)
        for col in range(n_frames):
            ax = axes[row, col]
            if col < len(frames):
                _show_image(ax, frames[col].get("frame_path"),
                            title=f"f{frames[col].get('frame_idx',col)}" if row == 0 else "")
            else:
                ax.axis("off")
        axes[row, 0].set_ylabel(f"Video {vid}", fontsize=9, rotation=90, labelpad=4)

    plt.tight_layout()
    save(fig, os.path.join(samples_dir, "exp5a_frame_grid.png"), cfg)


def samples_exp5_degradation(cfg, samples_dir):
    """Sequência de frames com degradação crescente (exp5b)."""
    from datasets.image import NOISE_FNS
    exp_dir = os.path.join(cfg["paths"]["outputs"], "exp5b_temporal_error")
    data    = load_pipeline_data(exp_dir)
    if not data:
        print("[samples/exp5b] pipeline_data.json não encontrado — pulando.")
        return

    video_ids = sorted({s["video_id"] for s in data if "video_id" in s})
    if not video_ids:
        return
    vid = video_ids[0]

    frames_all = sorted(
        [s for s in data if s.get("video_id") == vid and s.get("noise_type") == "gaussian"],
        key=lambda s: s.get("frame_idx") or 0,
    )
    step   = max(1, len(frames_all) // 8)
    frames = frames_all[::step][:8]

    fig, axes = plt.subplots(1, len(frames), figsize=(len(frames) * 3, 3.5))
    fig.suptitle(f"EXP 5b — Degradação progressiva (Video {vid}, Gaussian)", fontsize=12)

    for ax, s in zip(axes, frames):
        path = s.get("frame_path")
        if path and os.path.exists(path):
            img = Image.open(path).convert("RGB")
            lvl = int(s.get("noise_level", 0))
            if lvl > 0:
                img = NOISE_FNS["gaussian"](img, lvl)
            ax.imshow(img)
        else:
            ax.set_facecolor("#CCCCCC")
        ax.axis("off")
        ax.set_title(f"{float(s.get('degradation_pct') or 0):.0f}%", fontsize=9)

    plt.tight_layout()
    save(fig, os.path.join(samples_dir, "exp5b_degradation_sequence.png"), cfg)


def samples_exp5_noise_types(cfg, samples_dir):
    """Mesmo frame com os 3 tipos de ruído."""
    from datasets.image import NOISE_FNS
    exp_dir = os.path.join(cfg["paths"]["outputs"], "exp5a_temporal")
    data    = load_pipeline_data(exp_dir)
    if not data:
        return

    sample = next((s for s in data if s.get("frame_path") and
                   os.path.exists(s["frame_path"])), None)
    if not sample:
        print("[samples/exp5] Nenhum frame encontrado — pulando noise_types panel.")
        return

    noise_types = ["gaussian", "blur", "shapes"]
    lang_noise  = cfg["labels"][cfg["lang"]]["noise_types"]
    img_orig    = Image.open(sample["frame_path"]).convert("RGB")

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
    fig.suptitle("EXP 5 — Frame com diferentes tipos de ruído (nível 50)", fontsize=12)
    axes[0].imshow(img_orig); axes[0].axis("off"); axes[0].set_title("Original", fontsize=9)

    for ax, nt in zip(axes[1:], noise_types):
        ax.imshow(NOISE_FNS[nt](img_orig, 50))
        ax.axis("off")
        ax.set_title(lang_noise.get(nt, nt), fontsize=9)

    plt.tight_layout()
    save(fig, os.path.join(samples_dir, "exp5_noise_types.png"), cfg)


def samples_exp5_gif(cfg, samples_dir):
    """GIF animado com frames sequenciais de 3 vídeos."""
    exp_dir = os.path.join(cfg["paths"]["outputs"], "exp5a_temporal")
    data    = load_pipeline_data(exp_dir)
    if not data:
        return

    video_ids = sorted({s["video_id"] for s in data if "video_id" in s})[:3]
    for vid in video_ids:
        paths = [s.get("frame_path") for s in _frames_for_video(data, vid)
                 if s.get("frame_path") and os.path.exists(s["frame_path"])]
        if not paths:
            continue
        imgs     = [np.array(Image.open(p).convert("RGB").resize((256, 256))) for p in paths]
        gif_path = os.path.join(samples_dir, f"exp5a_video_{vid}.gif")
        imageio.mimsave(gif_path, imgs, fps=4, loop=0)
        print(f"  → {gif_path}")


# ── Tiny helper used only by samples functions ────────────────────────────────

def _ax(fig, gs, row, col):
    return fig.add_subplot(gs[row, col])


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/analysis.yaml")
    parser.add_argument("--skip-analysis", action="store_true",
                        help="Pula gráficos de análise; gera só amostras")
    parser.add_argument("--skip-samples", action="store_true",
                        help="Pula painéis de amostras; gera só análise")
    args = parser.parse_args()

    cfg      = load_cfg(args.config)
    fig_dir  = os.path.join(cfg["paths"]["reports"], "figures")
    samp_dir = os.path.join(cfg["paths"]["reports"], "samples")
    os.makedirs(fig_dir,  exist_ok=True)
    os.makedirs(samp_dir, exist_ok=True)

    # ── Análise ────────────────────────────────────────────────────────────────
    if not args.skip_analysis:
        report = ["RELATÓRIO DE ANÁLISE", "=" * 60]

        print("=== EXP 1 — APDDv2 Baseline ===")
        exp1_sampling_diagnostic(cfg, fig_dir)
        exp1_analysis(cfg, fig_dir, report)

        print("=== EXP 2 — Portinari ===")
        exp2_analysis(cfg, fig_dir, report)

        print("=== EXP 3 — Arte vs. Não-Arte ===")
        exp3_analysis(cfg, fig_dir, report)

        print("=== EXP 4 — Ruído ===")
        exp4_analysis(cfg, fig_dir, report)

        print("=== EXP 5a — Consistência Temporal ===")
        exp5a_analysis(cfg, fig_dir, report)

        print("=== EXP 5b — Degradação Progressiva ===")
        exp5b_analysis(
            cfg, fig_dir, report,
            baseline_dir=os.path.join(cfg["paths"]["outputs"], "exp5a_temporal"),
        )

        report_path = os.path.join(cfg["paths"]["reports"], "stats_report.txt")
        with open(report_path, "w") as f:
            f.write("\n".join(report))
        print(f"\nRelatório salvo em: {report_path}")
        print(f"Figuras salvas em:  {fig_dir}/")

    # ── Amostras visuais ───────────────────────────────────────────────────────
    if not args.skip_samples:
        print("\n=== EXP 1 — Amostras ===")
        samples_exp1(cfg, samp_dir)

        print("=== EXP 2 — Amostras ===")
        samples_exp2(cfg, samp_dir)

        print("=== EXP 3 — Amostras ===")
        samples_exp3(cfg, samp_dir)

        print("=== EXP 4 — Amostras ===")
        samples_exp4(cfg, samp_dir)

        print("=== EXP 5 — Amostras (frames, degradação, GIF) ===")
        samples_exp5_grid(cfg, samp_dir)
        samples_exp5_degradation(cfg, samp_dir)
        samples_exp5_noise_types(cfg, samp_dir)
        samples_exp5_gif(cfg, samp_dir)

        print(f"Amostras salvas em: {samp_dir}/")


if __name__ == "__main__":
    main()
