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
import textwrap
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import yaml
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
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/analysis.yaml")
    args = parser.parse_args()

    cfg     = load_cfg(args.config)
    out_dir = os.path.join(cfg["paths"]["reports"], "figures")
    os.makedirs(out_dir, exist_ok=True)

    report = ["RELATÓRIO DE ANÁLISE", "=" * 60]

    print("=== EXP 1 — APDDv2 Baseline ===")
    exp1_analysis(cfg, out_dir, report)

    print("=== EXP 2 — Portinari ===")
    exp2_analysis(cfg, out_dir, report)

    print("=== EXP 3 — Arte vs. Não-Arte ===")
    exp3_analysis(cfg, out_dir, report)

    print("=== EXP 4 — Ruído ===")
    exp4_analysis(cfg, out_dir, report)

    print("=== EXP 5a — Consistência Temporal ===")
    exp5a_analysis(cfg, out_dir, report)

    print("=== EXP 5b — Degradação Progressiva ===")
    exp5b_analysis(
        cfg, out_dir, report,
        baseline_dir=os.path.join(cfg["paths"]["outputs"], "exp5a_temporal"),
    )

    # Salva relatório estatístico
    report_path = os.path.join(cfg["paths"]["reports"], "stats_report.txt")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        f.write("\n".join(report))
    print(f"\nRelatório salvo em: {report_path}")
    print(f"Figuras salvas em:  {out_dir}/")


if __name__ == "__main__":
    main()
