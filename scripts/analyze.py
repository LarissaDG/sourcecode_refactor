"""
Análise e visualização estatística dos experimentos.

As amostras visuais (imagens de exemplo) rodam dentro do run.py, no cluster,
junto com o experimento — ver pipeline/samples.py.

Uso:
    python3 scripts/analyze.py --config configs/analysis.yaml
    python3 scripts/analyze.py --config configs/analysis.yaml --skip-analysis
"""

import argparse
import json
import os
import random
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from itertools import combinations
from scipy.stats import (friedmanchisquare, kendalltau, kruskal, ks_2samp, mannwhitneyu,
                         pearsonr, spearmanr, wasserstein_distance, wilcoxon)
from scipy.special import rel_entr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════════
# Config helpers
# ═══════════════════════════════════════════════════════════════════════════════

def load_cfg(path: str) -> dict:
    with open(path, encoding="utf-8-sig") as f:
        return yaml.safe_load(f)


def L(cfg, *keys):
    node = cfg["labels"][cfg["lang"]]
    for k in keys:
        node = node[k]
    return node


def attr_label(cfg, attr):
    return cfg["labels"][cfg["lang"]]["attributes"].get(attr, attr)


def ds_label(cfg, key):
    return cfg["labels"][cfg["lang"]]["datasets"].get(key, key)


# ═══════════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_scores(exp_dir: str, source: str) -> "pd.DataFrame | None":
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


def available_sources(exp_dir: str) -> list:
    scores_dir = os.path.join(exp_dir, "scores")
    if not os.path.isdir(scores_dir):
        return []
    sources = []
    for f in os.listdir(scores_dir):
        if f.startswith("scores_") and f.endswith(".csv"):
            sources.append(f.replace("scores_", "").replace(".csv", ""))
    return sources


def load_human_gt(cfg) -> "pd.DataFrame | None":
    """Carrega APDDv2-10023.csv como ground truth humano."""
    path = cfg["paths"].get("apddv2_csv", "")
    if not path or not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, encoding="ISO-8859-1")
    except Exception:
        try:
            df = pd.read_csv(path)
        except Exception:
            return None
    fn_col = next((c for c in df.columns if "filename" in c.lower()), None)
    if fn_col is None:
        fn_col = df.columns[0]
    df = df.rename(columns={fn_col: "filename"})
    df["stem"] = df["filename"].apply(_stem)
    return df


def _stem(filename) -> str:
    return os.path.splitext(os.path.basename(str(filename)))[0]


def _exp1_scores_dir(cfg, strategy: str = "uniform_bins") -> str:
    """
    exp1_apdd roda 2 estratégias de amostragem (sampling.strategies no YAML),
    cada uma na sua pasta: outputs/exp1_apdd_uniform_bins/,
    outputs/exp1_apdd_proportional_stratified/. Usa uniform_bins como o Exp1
    "principal" pras análises que citam só "exp1_apdd" (comparável ao antigo
    single-run); cai pra outputs/exp1_apdd/ sem sufixo se existir (setups
    antigos/locais que ainda não rodaram com sampling.strategies).
    """
    base = os.path.join(cfg["paths"]["outputs"], f"exp1_apdd_{strategy}")
    if os.path.isdir(base):
        return base
    return os.path.join(cfg["paths"]["outputs"], "exp1_apdd")


def _available_attrs(cfg, *dfs) -> list:
    """Retorna só os score_attributes presentes em todos os DataFrames fornecidos."""
    all_attrs = cfg["score_attributes"]
    dfs_valid = [d for d in dfs if d is not None]
    if not dfs_valid:
        return all_attrs
    return [a for a in all_attrs if all(a in d.columns for d in dfs_valid)]


# ═══════════════════════════════════════════════════════════════════════════════
# Visual helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _palette(cfg) -> dict:
    return cfg.get("palette", {
        "original": "#448FF2",
        "janus_1b": "#33A650",
        "janus_7b": "#F2A007",
        "mnist":    "#F23838",
        "highlight": "#1A00F2",
    })


def _hatches(cfg) -> dict:
    return cfg.get("hatches", {
        "original": "",
        "janus_1b": "///",
        "janus_7b": "xxx",
        "mnist":    "...",
        "highlight": "---",
    })


def _linestyles(cfg) -> dict:
    return cfg.get("linestyles", {
        "original": "solid",
        "janus_1b": "dashed",
        "janus_7b": "dotted",
        "mnist":    "dashdot",
    })


def _markers(cfg) -> dict:
    return cfg.get("markers", {
        "original": "o",
        "janus_1b": "s",
        "janus_7b": "^",
        "mnist":    "D",
    })


SOURCE_KEYS = {
    "original": "original",
    "Janus-Pro-1B": "janus_1b",
    "Janus-Pro-7B": "janus_7b",
    "mnist": "mnist",
    "Human_description": "original",
    "Gen_description": "janus_1b",
}


def _skey(source_name: str) -> str:
    return SOURCE_KEYS.get(source_name, "original")


def save(fig, path: str, cfg):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    dpi = cfg.get("figures", {}).get("dpi", 150)
    fmt = cfg.get("figures", {}).get("format", "png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight", format=fmt)
    plt.close(fig)


def _style_median(bp):
    for med in bp.get("medians", []):
        med.set_color("black")
        med.set_linewidth(2.5)


# ═══════════════════════════════════════════════════════════════════════════════
# Statistics: Friedman + Wilcoxon + CLD
# ═══════════════════════════════════════════════════════════════════════════════

def _compact_letters(group_names: list, means: dict, pval_dict: dict, alpha=0.05) -> dict:
    """Compact Letter Display: grupos sem diferença significativa compartilham uma letra."""
    if len(group_names) < 2:
        return {g: "a" for g in group_names}

    sorted_names = sorted(group_names, key=lambda g: means.get(g, 0), reverse=True)
    letter_groups = []  # list of sets

    for g in sorted_names:
        placed = []
        for idx, lg in enumerate(letter_groups):
            can_join = all(
                pval_dict.get(tuple(sorted([g, m])), 1.0) >= alpha
                for m in lg
            )
            if can_join:
                placed.append(idx)
        if placed:
            for idx in placed:
                letter_groups[idx].add(g)
        else:
            letter_groups.append({g})

    result = {g: "" for g in group_names}
    for idx, lg in enumerate(letter_groups):
        char = chr(ord("a") + idx)
        for g in lg:
            result[g] += char
    return result


def friedman_wilcoxon(groups_dict: dict, attrs: list, alpha=0.05) -> dict:
    """
    groups_dict: {group_name: DataFrame com coluna 'stem' e colunas de atributos}
    Retorna: {attr: {group_name: {"mean": float, "std": float, "letter": str, "n": int},
                     "_friedman_p": float}}
    """
    result = {}
    group_names = list(groups_dict.keys())

    for attr in attrs:
        dfs = []
        for name, df in groups_dict.items():
            if df is None or attr not in df.columns:
                continue
            if "stem" not in df.columns:
                df = df.copy()
                df["stem"] = df["filename"].apply(_stem)
            dfs.append(df[["stem", attr]].rename(columns={attr: name}))

        if len(dfs) < 2:
            continue

        merged = dfs[0]
        for d in dfs[1:]:
            merged = merged.merge(d, on="stem", how="inner")
        merged = merged.dropna()

        if len(merged) < 3:
            continue

        valid = [g for g in group_names if g in merged.columns]
        vals = [merged[g].values for g in valid]

        friedman_p = 1.0
        if len(vals) >= 3:
            try:
                _, friedman_p = friedmanchisquare(*vals)
            except Exception:
                pass

        pval_dict = {}
        for g1, g2 in combinations(valid, 2):
            pair = tuple(sorted([g1, g2]))
            try:
                _, p = wilcoxon(merged[g1].values, merged[g2].values)
                pval_dict[pair] = p
            except Exception:
                pval_dict[pair] = 1.0

        means = {g: float(merged[g].mean()) for g in valid}
        stds  = {g: float(merged[g].std())  for g in valid}
        letters = _compact_letters(valid, means, pval_dict, alpha)

        result[attr] = {
            g: {"mean": means[g], "std": stds[g], "letter": letters[g], "n": len(merged)}
            for g in valid
        }
        result[attr]["_friedman_p"] = friedman_p

    return result


def kruskal_mannwhitney(groups_dict: dict, attrs: list, alpha=0.05) -> dict:
    """
    Igual a friedman_wilcoxon, mas pra amostras INDEPENDENTES (não pareadas) —
    ex.: comparar atributos entre datasets diferentes (APDDv2 vs. Portinari vs.
    MNIST), onde não há como alinhar por 'stem' (imagens diferentes). Usa
    Kruskal-Wallis como teste ômnibus e Mann-Whitney U pareado a pareado (com o
    mesmo Compact Letter Display de friedman_wilcoxon) em vez de Friedman/Wilcoxon.

    groups_dict: {group_name: DataFrame com a coluna de cada atributo}.
    Retorna: {attr: {group_name: {"mean", "std", "letter", "n"}, "_kruskal_p": float}}
    """
    result = {}
    group_names = list(groups_dict.keys())

    for attr in attrs:
        vals_by_group = {}
        for name, df in groups_dict.items():
            if df is None or attr not in df.columns:
                continue
            v = df[attr].dropna().values
            if len(v) > 0:
                vals_by_group[name] = v

        valid = [g for g in group_names if g in vals_by_group]
        if len(valid) < 2:
            continue

        kruskal_p = 1.0
        if len(valid) >= 3:
            try:
                _, kruskal_p = kruskal(*[vals_by_group[g] for g in valid])
            except Exception:
                pass

        pval_dict = {}
        for g1, g2 in combinations(valid, 2):
            pair = tuple(sorted([g1, g2]))
            try:
                _, p = mannwhitneyu(vals_by_group[g1], vals_by_group[g2])
                pval_dict[pair] = p
            except Exception:
                pval_dict[pair] = 1.0

        means = {g: float(np.mean(vals_by_group[g])) for g in valid}
        stds  = {g: float(np.std(vals_by_group[g], ddof=1)) if len(vals_by_group[g]) > 1 else 0.0
                 for g in valid}
        letters = _compact_letters(valid, means, pval_dict, alpha)

        result[attr] = {
            g: {"mean": means[g], "std": stds[g], "letter": letters[g], "n": len(vals_by_group[g])}
            for g in valid
        }
        result[attr]["_kruskal_p"] = kruskal_p

    return result


def distribution_diff(s1: pd.Series, s2: pd.Series, name1="A", name2="B") -> "dict | None":
    a = s1.dropna().values
    b = s2.dropna().values
    if len(a) < 2 or len(b) < 2:
        return None
    ks_stat, ks_p = ks_2samp(a, b)
    w = wasserstein_distance(a, b)
    bins = 50
    r = (min(a.min(), b.min()), max(a.max(), b.max()))
    if r[0] == r[1]:
        kl = 0.0
    else:
        ph, _ = np.histogram(a, bins=bins, range=r, density=True)
        qh, _ = np.histogram(b, bins=bins, range=r, density=True)
        ph += 1e-10; qh += 1e-10
        kl = float(np.sum(rel_entr(ph, qh)))
    return {"pair": (name1, name2), "ks_stat": ks_stat, "ks_p": ks_p,
            "wasserstein": w, "kl": kl, "n1": len(a), "n2": len(b)}


def apply_nan_mask(df_ref: pd.DataFrame, df_target: pd.DataFrame, attr: str):
    """
    Alinha ref e target por stem e propaga o NaN do ref para o target,
    por atributo (view-only — não modifica os DataFrames originais).
    Retorna (ref_vals, target_vals) com NaN onde ref é NaN.
    """
    if "stem" not in df_ref.columns:
        df_ref = df_ref.copy(); df_ref["stem"] = df_ref["filename"].apply(_stem)
    if "stem" not in df_target.columns:
        df_target = df_target.copy(); df_target["stem"] = df_target["filename"].apply(_stem)
    merged = df_ref[["stem", attr]].merge(
        df_target[["stem", attr]].rename(columns={attr: attr + "_t"}),
        on="stem", how="inner"
    )
    mask = merged[attr].isna()
    ref_vals = merged[attr].copy()
    tgt_vals = merged[attr + "_t"].copy()
    tgt_vals[mask] = np.nan
    return ref_vals, tgt_vals


# ═══════════════════════════════════════════════════════════════════════════════
# Statistical table as PNG
# ═══════════════════════════════════════════════════════════════════════════════

def render_stat_table_png(fw_result: dict, attrs: list, group_names: list,
                           path: str, cfg, title="", best_is="highest"):
    """
    Renderiza a tabela Friedman/Wilcoxon como imagem PNG.
    Células: "mean ± std^letter"; melhor valor em negrito.
    """
    pal = _palette(cfg)
    col_labels = [ds_label(cfg, _skey(g)) if _skey(g) in cfg.get("labels", {}).get(cfg.get("lang", "en"), {}).get("datasets", {})
                  else g for g in group_names]

    # adiciona coluna p-value Friedman
    col_labels_full = col_labels + ["Friedman p"]
    row_labels = [attr_label(cfg, a) for a in attrs]

    cell_text = []
    cell_bold = []  # (row, col) to bold — filled in below, applied to the table after creation

    for i, attr in enumerate(attrs):
        row_text = []
        row_info = fw_result.get(attr, {})
        fp = row_info.get("_friedman_p", None)

        means = {g: row_info[g]["mean"] for g in group_names if g in row_info}
        if means:
            best_g = max(means, key=means.get) if best_is == "highest" else min(means, key=means.get)
        else:
            best_g = None

        for j, g in enumerate(group_names):
            if g not in row_info:
                row_text.append("—")
            else:
                m = row_info[g]["mean"]
                s = row_info[g]["std"]
                l = row_info[g]["letter"]
                cell = f"{m:.2f}±{s:.2f}{l}"
                if g == best_g:
                    cell_bold.append((i, j))
                row_text.append(cell)

        if fp is not None:
            row_text.append(f"{fp:.3f}" if fp >= 0.001 else "<0.001")
        else:
            row_text.append("—")
        cell_text.append(row_text)

    n_rows = len(cell_text)
    n_cols = len(col_labels_full)
    fig_w = max(8, 2 + n_cols * 2.8)
    fig_h = max(2, 0.5 + n_rows * 0.5)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    tbl = ax.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=col_labels_full,
        cellLoc="center",
        rowLoc="right",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.7)

    for j in range(n_cols):
        cell = tbl[(0, j)]
        cell.set_facecolor("#CCCCCC")
        cell.set_text_props(fontweight="bold")
    for i in range(n_rows):
        tbl[(i + 1, -1)].set_facecolor("#F5F5F5")
    for i, j in cell_bold:
        tbl[(i + 1, j)].set_text_props(fontweight="bold")

    if title:
        ax.set_title(title, pad=12, fontsize=11, fontweight="bold")

    plt.tight_layout()
    save(fig, path, cfg)


# ═══════════════════════════════════════════════════════════════════════════════
# Distribution difference table as PNG
# ═══════════════════════════════════════════════════════════════════════════════

def render_dist_diff_table(pairs_results: list, path: str, cfg, title=""):
    """
    pairs_results: list of dicts from distribution_diff()
    """
    if not pairs_results:
        return
    rows = []
    for r in pairs_results:
        if r is None:
            continue
        rows.append([
            f"{r['pair'][0]} vs {r['pair'][1]}",
            f"{r['ks_stat']:.3f}",
            f"{r['ks_p']:.3f}" if r['ks_p'] >= 0.001 else "<0.001",
            f"{r['wasserstein']:.3f}",
            f"{r['kl']:.3f}",
            str(r['n1']), str(r['n2']),
        ])
    if not rows:
        return
    col_labels = ["Pair", "KS stat", "KS p", "Wasserstein", "KL div", "n₁", "n₂"]
    n_rows = len(rows)
    n_cols = len(col_labels)
    fig_w = max(10, 2 + n_cols * 1.8)
    fig_h = max(2, 0.5 + n_rows * 0.45)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=col_labels, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.6)
    for j in range(n_cols):
        tbl[(0, j)].set_facecolor("#CCCCCC")
        tbl[(0, j)].set_text_props(fontweight="bold")
    if title:
        ax.set_title(title, pad=12, fontsize=11, fontweight="bold")
    plt.tight_layout()
    save(fig, path, cfg)


# ═══════════════════════════════════════════════════════════════════════════════
# Exp 1 — APDDv2
# ═══════════════════════════════════════════════════════════════════════════════

def analyse_exp1(cfg, out_dir: str):
    exp_dir = _exp1_scores_dir(cfg)
    attrs = cfg["score_attributes"]
    alpha = cfg["stats"]["alpha"]
    pal = _palette(cfg); ht = _hatches(cfg)
    ls = _linestyles(cfg); mk = _markers(cfg)

    df_orig  = load_scores(exp_dir, "original")
    df_1b    = load_scores(exp_dir, "Janus-Pro-1B")
    df_7b    = load_scores(exp_dir, "Janus-Pro-7B")
    df_human = load_human_gt(cfg)

    if df_orig is None:
        print("[exp1] scores não encontrados, pulando.")
        return

    total_attr = "Total aesthetic score"
    attrs = _available_attrs(cfg, df_orig, df_1b, df_7b)

    # ── 1. Distribuição de scores ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=cfg["figures"]["figsize"])
    for src, df, key in [
        ("original", df_orig, "original"),
        ("Janus-Pro-1B", df_1b, "janus_1b"),
        ("Janus-Pro-7B", df_7b, "janus_7b"),
    ]:
        if df is None or total_attr not in df.columns:
            continue
        vals = df[total_attr].dropna()
        ax.hist(vals, bins=30, alpha=0.6, color=pal[key],
                label=ds_label(cfg, key), hatch=ht[key], edgecolor="black")
    ax.set_xlabel(L(cfg, "axes", "score"))
    ax.set_ylabel("Count")
    ax.set_title(L(cfg, "titles", "dist_scores"))
    ax.legend()
    ax.grid(cfg["figures"]["grid"], alpha=0.3)
    save(fig, os.path.join(out_dir, "exp1_score_distributions.png"), cfg)

    # ── 2. Boxplot por fonte ────────────────────────────────────────────────
    sources = [("original", df_orig, "original"),
               ("Janus-Pro-1B", df_1b, "janus_1b"),
               ("Janus-Pro-7B", df_7b, "janus_7b")]
    available = [(n, d, k) for n, d, k in sources if d is not None and total_attr in d.columns]
    if available:
        fig, ax = plt.subplots(figsize=cfg["figures"]["figsize"])
        data_list = [d[total_attr].dropna().values for _, d, _ in available]
        labels = [ds_label(cfg, k) for _, _, k in available]
        bp = ax.boxplot(data_list, tick_labels=labels, patch_artist=True)
        _style_median(bp)
        for patch, (_, _, k) in zip(bp["boxes"], available):
            patch.set_facecolor(pal[k]); patch.set_hatch(ht[k])
        ax.set_ylabel(L(cfg, "axes", "score"))
        ax.set_title("APDDv2 — Score by Source")
        ax.grid(cfg["figures"]["grid"], alpha=0.3)
        save(fig, os.path.join(out_dir, "exp1_boxplot_sources.png"), cfg)

    # ── 3. Category table ───────────────────────────────────────────────────
    if df_orig is not None and "category" in df_orig.columns:
        _exp1_category_table(df_orig, cfg, out_dir, attrs, total_attr)

    # ── 4. Stat table (Friedman + Wilcoxon + CLD) ──────────────────────────
    groups = {}
    group_order = []
    for gname, df in [("original", df_orig), ("Janus-Pro-1B", df_1b), ("Janus-Pro-7B", df_7b)]:
        if df is not None:
            d = df.copy()
            if "stem" not in d.columns:
                d["stem"] = d["filename"].apply(_stem)
            groups[gname] = d
            group_order.append(gname)

    if len(groups) >= 2:
        fw = friedman_wilcoxon(groups, attrs, alpha)
        render_stat_table_png(
            fw, attrs, group_order,
            os.path.join(out_dir, "exp1_stat_table.png"), cfg,
            title="APDDv2 — Friedman + Wilcoxon (CLD)"
        )

    # ── 5. Score diff bar chart ─────────────────────────────────────────────
    if df_orig is not None and (df_1b is not None or df_7b is not None):
        _score_diff_bars(df_orig, df_1b, df_7b, cfg, out_dir,
                         prefix="exp1", title="APDDv2 — Score Difference (Original − Generated)")

    # ── 6. Cluster chart ────────────────────────────────────────────────────
    _cluster_chart(df_orig, cfg, out_dir, attrs, total_attr)

    # ── 7. Distribution differences ─────────────────────────────────────────
    if df_human is not None and df_orig is not None:
        _dist_diff_exp1(df_human, df_orig, df_1b, df_7b, cfg, out_dir, attrs)


def _exp1_category_table(df_orig, cfg, out_dir, attrs, total_attr):
    cats = df_orig["category"].dropna().unique()
    rows = []
    row_labels = []
    for cat in sorted(cats):
        sub = df_orig[df_orig["category"] == cat][total_attr].dropna()
        if len(sub) == 0:
            continue
        rows.append([f"{sub.mean():.2f}", f"{sub.std():.2f}", str(len(sub))])
        row_labels.append(str(cat))
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * len(rows) + 1)))
    ax.axis("off")
    tbl = ax.table(cellText=rows, rowLabels=row_labels,
                   colLabels=["Mean Score", "Std", "N"],
                   cellLoc="center", rowLoc="right", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.6)
    for j in range(3):
        tbl[(0, j)].set_facecolor("#CCCCCC")
        tbl[(0, j)].set_text_props(fontweight="bold")
    ax.set_title("APDDv2 — Score by Category", pad=12, fontsize=11, fontweight="bold")
    plt.tight_layout()
    save(fig, os.path.join(out_dir, "exp1_by_category_table.png"), cfg)


def _score_diff_bars(df_orig, df_1b, df_7b, cfg, out_dir, prefix="exp1", title="Score Difference"):
    attrs = cfg["score_attributes"]
    pal = _palette(cfg); ht = _hatches(cfg)

    # Só usa atributos presentes em pelo menos df_orig
    if df_orig is not None:
        attrs = [a for a in attrs if a in df_orig.columns]

    # Align by stem
    def align(df_a, df_b, attr):
        if df_a is None or df_b is None:
            return None, None
        a = df_a.copy(); b = df_b.copy()
        if "stem" not in a.columns: a["stem"] = a["filename"].apply(_stem)
        if "stem" not in b.columns: b["stem"] = b["filename"].apply(_stem)
        m = a[["stem", attr]].merge(b[["stem", attr]].rename(columns={attr: attr + "_b"}), on="stem")
        m = m.dropna()
        if len(m) == 0:
            return None, None
        return m[attr].values, m[attr + "_b"].values

    diffs_1b = []
    diffs_7b = []
    attr_labels = []
    for attr in attrs:
        a_vals, b1_vals = align(df_orig, df_1b, attr)
        _, b7_vals = align(df_orig, df_7b, attr)
        if a_vals is not None and b1_vals is not None:
            diffs_1b.append(float(np.mean(a_vals - b1_vals)))
        else:
            diffs_1b.append(None)
        if a_vals is not None and b7_vals is not None:
            diffs_7b.append(float(np.mean(a_vals - b7_vals)))
        else:
            diffs_7b.append(None)
        attr_labels.append(attr_label(cfg, attr))

    # Filter out None
    valid_idx = [i for i in range(len(attrs))
                 if diffs_1b[i] is not None or diffs_7b[i] is not None]
    if not valid_idx:
        return

    x = np.arange(len(valid_idx))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(10, len(valid_idx) * 0.9), 6))
    labels_v = [attr_labels[i] for i in valid_idx]
    d1 = [diffs_1b[i] if diffs_1b[i] is not None else 0 for i in valid_idx]
    d7 = [diffs_7b[i] if diffs_7b[i] is not None else 0 for i in valid_idx]

    bars1 = ax.bar(x - width/2, d1, width, label="Orig − Janus-1B",
                   color=pal["janus_1b"], hatch=ht["janus_1b"], edgecolor="black")
    bars7 = ax.bar(x + width/2, d7, width, label="Orig − Janus-7B",
                   color=pal["janus_7b"], hatch=ht["janus_7b"], edgecolor="black")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_v, rotation=40, ha="right")
    ax.set_ylabel("Average Score Difference")
    ax.set_title(title)
    ax.legend()
    ax.grid(cfg["figures"]["grid"], alpha=0.3, axis="y")
    plt.tight_layout()
    save(fig, os.path.join(out_dir, f"{prefix}_score_diff_bars.png"), cfg)


def _cluster_chart(df_orig, cfg, out_dir, attrs, total_attr):
    if df_orig is None:
        return
    n_clusters = cfg["clustering"]["n_clusters"]
    n_top = cfg["clustering"]["n_top"]

    sub = df_orig[attrs].dropna()
    if len(sub) < n_clusters * 2:
        return

    scaler = StandardScaler()
    X = scaler.fit_transform(sub.values)
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    pca = PCA(n_components=2, random_state=42)
    X2 = pca.fit_transform(X)

    cluster_colors = ["#448FF2", "#33A650", "#F2A007"][:n_clusters]

    total_scores = df_orig.loc[sub.index, total_attr].values
    top_idx  = np.argsort(total_scores)[-n_top:]
    bot_idx  = np.argsort(total_scores)[:n_top]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax_idx, (title_suffix, highlight_idx) in enumerate([
        (f"Top-{n_top}", top_idx), (f"Bottom-{n_top}", bot_idx)
    ]):
        ax = axes[ax_idx]
        for c in range(n_clusters):
            mask = labels == c
            ax.scatter(X2[mask, 0], X2[mask, 1], alpha=0.3, s=20,
                       color=cluster_colors[c], label=f"Cluster {c+1}")
        # Highlight: same cluster color, larger
        for i in highlight_idx:
            c = labels[i]
            ax.scatter(X2[i, 0], X2[i, 1], s=100, color=cluster_colors[c],
                       edgecolors="black", linewidths=1.5, zorder=5)
        ax.set_title(f"PCA Clusters — {title_suffix}")
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        ax.legend(fontsize=8)
        ax.grid(cfg["figures"]["grid"], alpha=0.3)

    plt.suptitle("APDDv2 — Cluster Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save(fig, os.path.join(out_dir, "exp1_clusters.png"), cfg)

    # Atributos médios por cluster (barras)
    fig, ax = plt.subplots(figsize=(max(10, len(attrs) * 0.9), 6))
    x = np.arange(len(attrs))
    width = 0.8 / n_clusters
    for c in range(n_clusters):
        mask = labels == c
        means = [float(sub.iloc[mask][a].mean()) for a in attrs]
        ax.bar(x + c * width, means, width, label=f"Cluster {c+1}",
               color=cluster_colors[c], edgecolor="black", alpha=0.85)
    ax.set_xticks(x + width * (n_clusters - 1) / 2)
    ax.set_xticklabels([attr_label(cfg, a) for a in attrs], rotation=40, ha="right")
    ax.set_ylabel(L(cfg, "axes", "score"))
    ax.set_title("Mean Attributes per Cluster")
    ax.legend()
    ax.grid(cfg["figures"]["grid"], alpha=0.3, axis="y")
    plt.tight_layout()
    save(fig, os.path.join(out_dir, "exp1_cluster_attrs.png"), cfg)


def _dist_diff_exp1(df_human, df_orig, df_1b, df_7b, cfg, out_dir, attrs):
    """Tabelas de diferença de distribuição para Exp1."""
    for attr in [cfg["score_attributes"][0]]:  # Total score apenas
        pairs = []
        if df_orig is not None:
            r, t = apply_nan_mask(df_human, df_orig, attr)
            pairs.append(distribution_diff(r.dropna(), t.dropna(), "Human GT", "Original"))
        if df_1b is not None:
            r, t = apply_nan_mask(df_human, df_1b, attr)
            pairs.append(distribution_diff(r.dropna(), t.dropna(), "Human GT", "Janus-1B"))
        if df_7b is not None:
            r, t = apply_nan_mask(df_human, df_7b, attr)
            pairs.append(distribution_diff(r.dropna(), t.dropna(), "Human GT", "Janus-7B"))
        pairs = [p for p in pairs if p is not None]
        if pairs:
            render_dist_diff_table(
                pairs, os.path.join(out_dir, "exp1_dist_diff.png"),
                cfg, title=f"APDDv2 — Distribution Differences ({attr_label(cfg, attr)})"
            )
    # Per-attribute full table
    all_pairs = []
    for attr in attrs:
        for gname, df in [("Janus-1B", df_1b), ("Janus-7B", df_7b)]:
            if df is None:
                continue
            r, t = apply_nan_mask(df_human, df, attr)
            res = distribution_diff(r.dropna(), t.dropna(),
                                    f"Human/{attr_label(cfg, attr)}",
                                    gname)
            if res:
                all_pairs.append(res)
    if all_pairs:
        render_dist_diff_table(
            all_pairs, os.path.join(out_dir, "exp1_dist_diff_full.png"),
            cfg, title="APDDv2 — Distribution Differences (all attributes)"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Exp 2 — Portinari
# ═══════════════════════════════════════════════════════════════════════════════

def analyse_exp2(cfg, out_dir: str):
    pal = _palette(cfg); ht = _hatches(cfg)
    attrs = cfg["score_attributes"]
    alpha = cfg["stats"]["alpha"]
    total_attr = "Total aesthetic score"

    def _load_exp2(name):
        d = os.path.join(cfg["paths"]["outputs"], name)
        return load_scores(d, "original"), load_scores(d, "Janus-Pro-1B"), load_scores(d, "Janus-Pro-7B")

    df_2a_o, df_2a_1b, df_2a_7b = _load_exp2("exp2a_portinari")
    df_2b_o, df_2b_1b, df_2b_7b = _load_exp2("exp2b_portinari_human")

    # ── 1. Boxplot Human vs Gen ─────────────────────────────────────────────
    # Reúne originais exp2a e exp2b como "Human_description" vs generated
    def _total(df):
        return df[total_attr].dropna().values if df is not None and total_attr in df.columns else None

    sources_box = [
        ("Human_description", df_2b_o, "original"),
        ("Gen_description",   df_2a_o, "janus_1b"),
        ("Janus-Pro-1B (2a)", df_2a_1b, "janus_1b"),
        ("Janus-Pro-7B (2a)", df_2a_7b, "janus_7b"),
        ("Janus-Pro-1B (2b)", df_2b_1b, "janus_1b"),
        ("Janus-Pro-7B (2b)", df_2b_7b, "janus_7b"),
    ]
    available_box = [(n, d, k) for n, d, k in sources_box if _total(d) is not None]
    if available_box:
        fig, ax = plt.subplots(figsize=(max(10, len(available_box) * 1.5), 6))
        data_list = [_total(d) for _, d, _ in available_box]
        labels = [n for n, _, _ in available_box]
        bp = ax.boxplot(data_list, tick_labels=labels, patch_artist=True)
        _style_median(bp)
        for patch, (_, _, k) in zip(bp["boxes"], available_box):
            patch.set_facecolor(pal[k]); patch.set_hatch(ht[k])
        ax.set_ylabel(L(cfg, "axes", "score"))
        ax.set_title("Portinari — Score by Source")
        ax.grid(cfg["figures"]["grid"], alpha=0.3)
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        save(fig, os.path.join(out_dir, "exp2_boxplot.png"), cfg)

    # ── 2. Stat tables ─────────────────────────────────────────────────────
    for exp_tag, df_o, df_1b, df_7b in [
        ("2a_gen_captions", df_2a_o, df_2a_1b, df_2a_7b),
        ("2b_human_captions", df_2b_o, df_2b_1b, df_2b_7b),
    ]:
        groups = {}; group_order = []
        for gname, df in [("original", df_o), ("Janus-Pro-1B", df_1b), ("Janus-Pro-7B", df_7b)]:
            if df is not None:
                d = df.copy()
                if "stem" not in d.columns: d["stem"] = d["filename"].apply(_stem)
                groups[gname] = d; group_order.append(gname)
        if len(groups) >= 2:
            fw = friedman_wilcoxon(groups, attrs, alpha)
            render_stat_table_png(fw, attrs, group_order,
                                  os.path.join(out_dir, f"exp2_{exp_tag}_stat_table.png"),
                                  cfg, title=f"Portinari ({exp_tag}) — Friedman + Wilcoxon")

    # ── 3. Score diff bars ──────────────────────────────────────────────────
    if df_2a_o is not None:
        _score_diff_bars(df_2a_o, df_2a_1b, df_2a_7b, cfg, out_dir,
                         prefix="exp2a", title="Portinari (AI Captions) — Score Difference")
    if df_2b_o is not None:
        _score_diff_bars(df_2b_o, df_2b_1b, df_2b_7b, cfg, out_dir,
                         prefix="exp2b", title="Portinari (Human Captions) — Score Difference")

    # ── 4. Distribution differences ─────────────────────────────────────────
    df_human = load_human_gt(cfg)
    all_pairs = []
    pair_defs = [
        ("Portinari orig (2a)", df_2a_o), ("Portinari orig (2b)", df_2b_o),
        ("Janus-1B (2a)", df_2a_1b), ("Janus-7B (2a)", df_2a_7b),
        ("Janus-1B (2b)", df_2b_1b), ("Janus-7B (2b)", df_2b_7b),
    ]
    if df_human is not None:
        for name, df in pair_defs:
            if df is None or total_attr not in df.columns:
                continue
            res = distribution_diff(
                df_human[total_attr].dropna() if total_attr in df_human.columns else pd.Series([]),
                df[total_attr].dropna(),
                "APDDv2-Human", name
            )
            if res:
                all_pairs.append(res)
    # Portinari intra-pairs
    for n1, d1 in pair_defs:
        for n2, d2 in pair_defs:
            if n1 >= n2: continue
            if d1 is None or d2 is None: continue
            if total_attr not in d1.columns or total_attr not in d2.columns: continue
            res = distribution_diff(d1[total_attr].dropna(), d2[total_attr].dropna(), n1, n2)
            if res:
                all_pairs.append(res)
    if all_pairs:
        render_dist_diff_table(
            all_pairs, os.path.join(out_dir, "exp2_dist_diff.png"),
            cfg, title="Portinari — Distribution Differences (Total Score)"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Exp 3 — MNIST
# ═══════════════════════════════════════════════════════════════════════════════

def analyse_exp3(cfg, out_dir: str):
    pal = _palette(cfg); ht = _hatches(cfg)
    total_attr = "Total aesthetic score"

    exp3_dir = os.path.join(cfg["paths"]["outputs"], "exp3_mnist")
    df_mnist = load_scores(exp3_dir, "original")

    df_apdd  = load_scores(_exp1_scores_dir(cfg), "original")

    exp2a_dir = os.path.join(cfg["paths"]["outputs"], "exp2a_portinari")
    df_port  = load_scores(exp2a_dir, "original")

    if df_mnist is None or total_attr not in (df_mnist.columns if df_mnist is not None else []):
        print("[exp3] scores não encontrados, pulando.")
        return

    # ── Art vs Non-Art boxplot ──────────────────────────────────────────────
    sources = []
    if df_apdd is not None and total_attr in df_apdd.columns:
        sources.append(("APDDv2", df_apdd[total_attr].dropna(), "original"))
    if df_port is not None and total_attr in df_port.columns:
        sources.append(("Portinari", df_port[total_attr].dropna(), "original"))
    if total_attr in df_mnist.columns:
        sources.append(("MNIST", df_mnist[total_attr].dropna(), "mnist"))

    if len(sources) >= 2:
        fig, ax = plt.subplots(figsize=(8, 6))
        bp = ax.boxplot([s[1].values for s in sources],
                        tick_labels=[s[0] for s in sources], patch_artist=True)
        _style_median(bp)
        for patch, (_, _, k) in zip(bp["boxes"], sources):
            patch.set_facecolor(pal[k]); patch.set_hatch(ht[k])
        ax.set_ylabel(L(cfg, "axes", "score"))
        ax.set_title(L(cfg, "titles", "art_vs_noart"))
        ax.grid(cfg["figures"]["grid"], alpha=0.3)
        save(fig, os.path.join(out_dir, "exp3_art_vs_noart.png"), cfg)

    # ── Distribution diffs ──────────────────────────────────────────────────
    pairs = []
    mnist_s = df_mnist[total_attr].dropna() if total_attr in df_mnist.columns else pd.Series([])
    if df_apdd is not None and total_attr in df_apdd.columns:
        pairs.append(distribution_diff(df_apdd[total_attr].dropna(), mnist_s, "APDDv2", "MNIST"))
    if df_port is not None and total_attr in df_port.columns:
        pairs.append(distribution_diff(df_port[total_attr].dropna(), mnist_s, "Portinari", "MNIST"))
    pairs = [p for p in pairs if p is not None]
    if pairs:
        render_dist_diff_table(
            pairs, os.path.join(out_dir, "exp3_dist_diff.png"),
            cfg, title="Art vs Non-Art — Distribution Differences"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Comparison figures (Fig 4.9 e Fig 4.10)
# ═══════════════════════════════════════════════════════════════════════════════

def analyse_comparisons(cfg, out_dir: str):
    pal = _palette(cfg); ht = _hatches(cfg); mk = _markers(cfg); ls = _linestyles(cfg)
    attrs = cfg["score_attributes"]
    total_attr = "Total aesthetic score"

    def _load(exp, source):
        d = os.path.join(cfg["paths"]["outputs"], exp)
        return load_scores(d, source)

    df_apdd = load_scores(_exp1_scores_dir(cfg), "original")
    df_port = _load("exp2a_portinari", "original")
    df_mnist = _load("exp3_mnist", "original")

    # ── Fig 4.9: APDDv2 vs Portinari por atributo ──────────────────────────
    if df_apdd is not None and df_port is not None:
        valid_attrs = [a for a in attrs if a in df_apdd.columns and a in df_port.columns]
        if valid_attrs:
            apdd_means = [df_apdd[a].dropna().mean() for a in valid_attrs]
            port_means = [df_port[a].dropna().mean() for a in valid_attrs]
            x = np.arange(len(valid_attrs))
            width = 0.35
            fig, ax = plt.subplots(figsize=(max(10, len(valid_attrs) * 0.9), 6))
            ax.bar(x - width/2, apdd_means, width, label="APDDv2", color=pal["original"],
                   hatch=ht["original"], edgecolor="black")
            ax.bar(x + width/2, port_means, width, label="Portinari", color=pal["janus_1b"],
                   hatch=ht["janus_1b"], edgecolor="black")
            ax.set_xticks(x)
            ax.set_xticklabels([attr_label(cfg, a) for a in valid_attrs], rotation=40, ha="right")
            ax.set_ylabel(L(cfg, "axes", "score"))
            ax.set_title("Fig 4.9 — APDDv2 vs Portinari: Mean Scores per Attribute")
            ax.legend(); ax.grid(cfg["figures"]["grid"], alpha=0.3, axis="y")
            plt.tight_layout()
            save(fig, os.path.join(out_dir, "fig49_apddv2_vs_portinari.png"), cfg)

    # ── Fig 4.10: Art vs MNIST por atributo ────────────────────────────────
    if df_mnist is not None:
        art_dfs = [(df_apdd, "APDDv2", "original"), (df_port, "Portinari", "original")]
        art_dfs = [(d, n, k) for d, n, k in art_dfs if d is not None]
        valid_attrs = [a for a in attrs if all(a in d.columns for d, _, _ in art_dfs)
                       and a in df_mnist.columns]
        if valid_attrs and art_dfs:
            x = np.arange(len(valid_attrs))
            width = 0.8 / (len(art_dfs) + 1)
            fig, ax = plt.subplots(figsize=(max(10, len(valid_attrs) * 0.9), 6))
            for i, (d, name, key) in enumerate(art_dfs):
                means = [d[a].dropna().mean() for a in valid_attrs]
                ax.bar(x + i * width, means, width, label=name,
                       color=pal[key], hatch=ht[key], edgecolor="black", alpha=0.85)
            mnist_means = [df_mnist[a].dropna().mean() for a in valid_attrs]
            ax.bar(x + len(art_dfs) * width, mnist_means, width, label="MNIST",
                   color=pal["mnist"], hatch=ht["mnist"], edgecolor="black", alpha=0.85)
            ax.set_xticks(x + width * len(art_dfs) / 2)
            ax.set_xticklabels([attr_label(cfg, a) for a in valid_attrs], rotation=40, ha="right")
            ax.set_ylabel(L(cfg, "axes", "score"))
            ax.set_title("Fig 4.10 — Art vs Non-Art: Mean Scores per Attribute")
            ax.legend(); ax.grid(cfg["figures"]["grid"], alpha=0.3, axis="y")
            plt.tight_layout()
            save(fig, os.path.join(out_dir, "fig410_art_vs_noart.png"), cfg)


# ═══════════════════════════════════════════════════════════════════════════════
# Exp 4 — Noise
# ═══════════════════════════════════════════════════════════════════════════════

def analyse_exp4(cfg, out_dir: str):
    exp_dir = os.path.join(cfg["paths"]["outputs"], "exp4_noise")
    df = load_scores(exp_dir, "original")
    if df is None:
        print("[exp4] scores não encontrados, pulando.")
        return
    if "noise_type" not in df.columns:
        print("[exp4] coluna noise_type ausente no CSV — reprocesse o scoring.")
        return

    pal = _palette(cfg)
    total_attr = "Total aesthetic score"
    alpha = cfg["stats"]["alpha"]

    noise_types = df["noise_type"].dropna().unique()
    noise_colors = {"gaussian": "#448FF2", "blur": "#33A650", "shapes": "#F2A007"}
    noise_hatches = {"gaussian": "///", "blur": "xxx", "shapes": "..."}
    noise_ls = {"gaussian": "solid", "blur": "dashed", "shapes": "dotted"}
    noise_mk = {"gaussian": "o", "blur": "s", "shapes": "^"}

    # ── Score vs noise level ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=cfg["figures"]["figsize"])
    for nt in sorted(noise_types):
        sub = df[df["noise_type"] == nt]
        if "noise_level" not in sub.columns or total_attr not in sub.columns:
            continue
        grp = sub.groupby("noise_level")[total_attr]
        levels = sorted(grp.groups.keys())
        means = [grp.get_group(l).mean() for l in levels]
        sems  = [grp.get_group(l).sem() for l in levels]
        color = noise_colors.get(nt, "#888888")
        hatch = noise_hatches.get(nt, "")
        linestyle = noise_ls.get(nt, "solid")
        marker = noise_mk.get(nt, "o")
        label = L(cfg, "noise_types", nt) if nt in cfg["labels"][cfg["lang"]].get("noise_types", {}) else nt
        sems_arr = np.array(sems, dtype=float)
        sems_arr = np.nan_to_num(sems_arr, 0)
        means_arr = np.array(means, dtype=float)
        ax.plot(levels, means_arr, color=color, linestyle=linestyle,
                marker=marker, label=label)
        ax.fill_between(levels, means_arr - sems_arr, means_arr + sems_arr,
                         alpha=0.15, color=color)
    ax.set_xlabel(L(cfg, "axes", "noise_level"))
    ax.set_ylabel(L(cfg, "axes", "score"))
    ax.set_title(L(cfg, "titles", "noise_impact"))
    ax.legend(); ax.grid(cfg["figures"]["grid"], alpha=0.3)
    save(fig, os.path.join(out_dir, "exp4_noise_impact.png"), cfg)

    # ── Boxplot per noise type ──────────────────────────────────────────────
    if total_attr in df.columns:
        fig, ax = plt.subplots(figsize=(8, 6))
        box_data = [df[df["noise_type"] == nt][total_attr].dropna().values
                    for nt in sorted(noise_types) if nt in df["noise_type"].values]
        box_labels = [L(cfg, "noise_types", nt) if nt in cfg["labels"][cfg["lang"]].get("noise_types", {}) else nt
                      for nt in sorted(noise_types) if nt in df["noise_type"].values]
        if box_data:
            bp = ax.boxplot(box_data, tick_labels=box_labels, patch_artist=True)
            _style_median(bp)
            for patch, nt in zip(bp["boxes"], sorted(noise_types)):
                patch.set_facecolor(noise_colors.get(nt, "#888888"))
                patch.set_hatch(noise_hatches.get(nt, ""))
            ax.set_ylabel(L(cfg, "axes", "score"))
            ax.set_title("Noise Types — Score Distribution")
            ax.grid(cfg["figures"]["grid"], alpha=0.3)
            save(fig, os.path.join(out_dir, "exp4_noise_boxplot.png"), cfg)

    # ── Distribution diffs vs baseline ─────────────────────────────────────
    df_base = load_scores(_exp1_scores_dir(cfg), "original")
    if df_base is not None and total_attr in df_base.columns:
        pairs = []
        for nt in sorted(noise_types):
            sub = df[df["noise_type"] == nt][total_attr].dropna()
            if len(sub) < 2:
                continue
            label_nt = L(cfg, "noise_types", nt) if nt in cfg["labels"][cfg["lang"]].get("noise_types", {}) else nt
            res = distribution_diff(df_base[total_attr].dropna(), sub,
                                    "APDDv2 Original", label_nt)
            if res:
                pairs.append(res)
        if pairs:
            render_dist_diff_table(
                pairs, os.path.join(out_dir, "exp4_dist_diff.png"),
                cfg, title="Noise — Distribution Differences vs APDDv2 Baseline"
            )


# ═══════════════════════════════════════════════════════════════════════════════
# Exp 5 — Temporal
# ═══════════════════════════════════════════════════════════════════════════════

def analyse_exp5(cfg, out_dir: str):
    pal = _palette(cfg)
    total_attr = "Total aesthetic score"
    alpha = cfg["stats"]["alpha"]

    exp5a_dir = os.path.join(cfg["paths"]["outputs"], "exp5a_temporal")
    exp5b_dir = os.path.join(cfg["paths"]["outputs"], "exp5b_temporal_error")
    df5a = load_scores(exp5a_dir, "original")
    df5b = load_scores(exp5b_dir, "original")

    # ── Exp5a: temporal consistency ────────────────────────────────────────
    if df5a is not None and "frame_idx" in df5a.columns and total_attr in df5a.columns:
        fig, ax = plt.subplots(figsize=cfg["figures"]["figsize"])
        grp = df5a.groupby("frame_idx")[total_attr]
        frames = sorted(grp.groups.keys())
        means = [grp.get_group(f).mean() for f in frames]
        sems  = np.nan_to_num(np.array([grp.get_group(f).sem() for f in frames], dtype=float), 0)
        ax.plot(frames, means, color=pal["original"], linewidth=2, marker="o",
                markersize=4, label="Mean score")
        ax.fill_between(frames, np.array(means) - sems, np.array(means) + sems,
                        alpha=0.2, color=pal["original"])
        ax.set_xlabel(L(cfg, "axes", "frame_idx"))
        ax.set_ylabel(L(cfg, "axes", "score"))
        ax.set_title(L(cfg, "titles", "temporal_consist"))
        ax.legend(); ax.grid(cfg["figures"]["grid"], alpha=0.3)
        save(fig, os.path.join(out_dir, "exp5a_temporal_consistency.png"), cfg)

    # ── Exp5b: degradation impact ───────────────────────────────────────────
    if df5b is not None and total_attr in df5b.columns:
        _exp5b_degradation(df5b, cfg, out_dir, total_attr, alpha, pal)


def _exp5b_degradation(df5b, cfg, out_dir, total_attr, alpha, pal):
    """Alarm visualization: score vs degradation %, threshold detection."""

    # Cenário 1: score por frame_idx (erro no frame 12)
    if "frame_idx" in df5b.columns:
        fig, ax = plt.subplots(figsize=cfg["figures"]["figsize"])
        grp = df5b.groupby("frame_idx")[total_attr]
        frames = sorted(grp.groups.keys())
        means = [float(grp.get_group(f).mean()) for f in frames]
        sems  = np.nan_to_num(np.array([grp.get_group(f).sem() for f in frames], dtype=float), 0)
        colors = ["#F23838" if f >= 12 else pal["original"] for f in frames]
        ax.plot(frames, means, color=pal["original"], linewidth=1.5, zorder=1)
        ax.scatter(frames, means, c=colors, s=40, zorder=2)
        ax.fill_between(frames, np.array(means) - sems, np.array(means) + sems,
                        alpha=0.15, color=pal["original"])
        ax.axvline(12, color="#F23838", linestyle="--", linewidth=1.5, label="Error starts (frame 12)")
        ax.set_xlabel(L(cfg, "axes", "frame_idx"))
        ax.set_ylabel(L(cfg, "axes", "score"))
        ax.set_title("Exp5b — Score Drop at Error Frame")
        ax.legend(); ax.grid(cfg["figures"]["grid"], alpha=0.3)
        save(fig, os.path.join(out_dir, "exp5b_frame_score.png"), cfg)

    # Cenário 3: threshold por degradação progressiva
    if "degradation_pct" in df5b.columns:
        grp_deg = df5b.groupby("degradation_pct")[total_attr]
        levels = sorted(grp_deg.groups.keys())
        if len(levels) < 2:
            return

        base_scores = grp_deg.get_group(levels[0]).dropna().values
        means = []; pvals = []; sig_bonf = []
        n_tests = len(levels) - 1
        for lvl in levels:
            sub = grp_deg.get_group(lvl).dropna().values
            means.append(float(np.mean(sub)))
            if lvl == levels[0] or len(sub) < 2 or len(base_scores) < 2:
                pvals.append(1.0)
            else:
                try:
                    # Wilcoxon only valid for paired; use Mann-Whitney as fallback
                    from scipy.stats import mannwhitneyu
                    _, p = mannwhitneyu(base_scores, sub, alternative="two-sided")
                    pvals.append(float(p))
                except Exception:
                    pvals.append(1.0)

        threshold_lvl = None
        for lvl, p in zip(levels[1:], pvals[1:]):
            if p < (alpha / n_tests):  # Bonferroni
                threshold_lvl = lvl
                break

        fig, ax = plt.subplots(figsize=cfg["figures"]["figsize"])
        ax.plot(levels, means, color=pal["original"], linewidth=2, marker="o")
        if threshold_lvl is not None:
            ax.axvline(threshold_lvl, color=pal["highlight"], linestyle="--", linewidth=2,
                       label=f"Significance threshold ({threshold_lvl}%)")
            ax.legend()
        ax.set_xlabel(L(cfg, "axes", "degradation"))
        ax.set_ylabel(L(cfg, "axes", "score"))
        ax.set_title(L(cfg, "titles", "degradation_detect"))
        ax.grid(cfg["figures"]["grid"], alpha=0.3)
        save(fig, os.path.join(out_dir, "exp5b_degradation.png"), cfg)


# ═══════════════════════════════════════════════════════════════════════════════
# Diagnósticos avançados (sem outro modelo pra comparar — só o ArtCLIP)
#   1. Monotonicidade:       score deve cair conforme o ruído aumenta (Exp4)
#   2. Validade discriminante: o ArtCLIP separa Humano / Sintético / MNIST?
#   3. Viés cultural:        score difere entre dentro e fora da base (Exp1 vs Exp2)
#   4. Grupos de dificuldade: Fácil/Médio/Difícil (adaptado de TRI sem múltiplos avaliadores)
# ═══════════════════════════════════════════════════════════════════════════════

def render_simple_table_png(rows: list, col_labels: list, path: str, cfg, title=""):
    """Tabela genérica (linhas já formatadas como string) renderizada como PNG."""
    if not rows:
        return
    n_rows = len(rows)
    n_cols = len(col_labels)
    fig_w = max(8, 2 + n_cols * 2.2)
    fig_h = max(2, 0.5 + n_rows * 0.45)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=col_labels, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.6)
    for j in range(n_cols):
        tbl[(0, j)].set_facecolor("#CCCCCC")
        tbl[(0, j)].set_text_props(fontweight="bold")
    if title:
        ax.set_title(title, pad=12, fontsize=11, fontweight="bold")
    plt.tight_layout()
    save(fig, path, cfg)


def _overlaid_density_chart(groups: list, out_path: str, cfg, title="", xlabel=""):
    """
    groups: lista de (label, valores, chave_de_cor) — chave resolvida via
    _palette()/_hatches()/_linestyles(). Histogramas de densidade sobrepostos
    (contorno + preenchimento com hachura), sem formas circulares/3D.
    """
    pal = _palette(cfg); ht = _hatches(cfg); ls = _linestyles(cfg)
    fig, ax = plt.subplots(figsize=cfg["figures"]["figsize"])
    for label, vals, key in groups:
        vals = np.asarray(vals, dtype=float)
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            continue
        color = pal.get(key, "#888888")
        hatch = ht.get(key, "")
        style = ls.get(key, "solid")
        ax.hist(vals, bins=30, density=True, histtype="step", linewidth=2.2,
                linestyle=style, color=color, label=f"{label} (n={len(vals)})")
        ax.hist(vals, bins=30, density=True, alpha=0.15, color=color, hatch=hatch)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Densidade")
    ax.set_title(title)
    ax.legend()
    ax.grid(cfg["figures"]["grid"], alpha=0.3)
    save(fig, out_path, cfg)


def analyse_monotonicity(cfg, out_dir: str):
    """
    Teste de Monotonicidade (Análise Sequencial de Ruído) — Exp4.
    Sem outro modelo pra comparar, valida o ArtCLIP contra a física do
    problema: o score deve cair conforme o ruído aumenta. Métrica: Spearman ρ
    e Kendall τ entre noise_level e score, por tipo de ruído.
    """
    exp4_dir = os.path.join(cfg["paths"]["outputs"], "exp4_noise")
    df = load_scores(exp4_dir, "original")
    total_attr = "Total aesthetic score"
    if df is None or "noise_type" not in df.columns or "noise_level" not in df.columns:
        print("[monotonicity] scores/metadados do exp4 não encontrados, pulando.")
        return

    alpha = cfg["stats"]["alpha"]
    rows = []
    for nt in sorted(df["noise_type"].dropna().unique()):
        sub = df[df["noise_type"] == nt][["noise_level", total_attr]].dropna()
        if len(sub) < 3:
            continue
        rho, p_s = spearmanr(sub["noise_level"], sub[total_attr])
        tau, p_k = kendalltau(sub["noise_level"], sub[total_attr])
        if p_s < alpha and rho < 0:
            verdict = "Monotônico (decrescente)"
        elif p_s < alpha and rho > 0:
            verdict = "Anômalo (crescente)"
        else:
            verdict = "Não monotônico"
        label_nt = L(cfg, "noise_types", nt) if nt in cfg["labels"][cfg["lang"]].get("noise_types", {}) else nt
        rows.append([
            label_nt, f"{rho:.3f}", f"{p_s:.3f}" if p_s >= 0.001 else "<0.001",
            f"{tau:.3f}", f"{p_k:.3f}" if p_k >= 0.001 else "<0.001",
            str(len(sub)), verdict,
        ])
    if not rows:
        print("[monotonicity] nenhum tipo de ruído com dados suficientes, pulando.")
        return
    render_simple_table_png(
        rows,
        ["Tipo de ruído", "Spearman ρ", "p (ρ)", "Kendall τ", "p (τ)", "n", "Diagnóstico"],
        os.path.join(out_dir, "monotonicity_table.png"), cfg,
        title="Teste de Monotonicidade — Score vs. Nível de Ruído (Exp4)\n"
              "(o gráfico exp4_noise_impact.png mostra a curva correspondente)"
    )


def analyse_discriminative_validity(cfg, out_dir: str):
    """
    Validade Discriminante — o ArtCLIP separa conteúdo estruturalmente
    diferente (pintura humana / sintética vs. dígito MNIST, que não é arte)?
    Sem outro modelo pra comparar, o teste é: as distribuições de score do
    próprio ArtCLIP pra esses grupos são estatisticamente diferentes (KS) e
    visualmente separáveis (densidade sobreposta)?
    """
    total_attr = "Total aesthetic score"
    exp1 = _exp1_scores_dir(cfg)
    df_human = load_scores(exp1, "original")
    df_1b    = load_scores(exp1, "Janus-Pro-1B")
    df_7b    = load_scores(exp1, "Janus-Pro-7B")
    df_mnist = load_scores(os.path.join(cfg["paths"]["outputs"], "exp3_mnist"), "original")

    groups = []
    for label, df, key in [
        ("Humano (APDDv2)", df_human, "original"),
        ("Sintético (Janus-1B)", df_1b, "janus_1b"),
        ("Sintético (Janus-7B)", df_7b, "janus_7b"),
        ("MNIST", df_mnist, "mnist"),
    ]:
        if df is not None and total_attr in df.columns:
            vals = df[total_attr].dropna().values
            if len(vals) > 0:
                groups.append((label, vals, key))

    if len(groups) < 2:
        print("[discriminative_validity] dados insuficientes, pulando.")
        return

    _overlaid_density_chart(
        groups, os.path.join(out_dir, "discriminative_validity_density.png"), cfg,
        title="Validade Discriminante — Distribuição de Score por Grupo",
        xlabel=L(cfg, "axes", "score"),
    )

    pairs = []
    for (n1, v1, _), (n2, v2, _) in combinations(groups, 2):
        res = distribution_diff(pd.Series(v1), pd.Series(v2), n1, n2)
        if res:
            pairs.append(res)
    if pairs:
        render_dist_diff_table(
            pairs, os.path.join(out_dir, "discriminative_validity_table.png"),
            cfg, title="Validade Discriminante — KS Test entre Grupos"
        )


def analyse_cultural_bias(cfg, out_dir: str):
    """
    Viés Geográfico/Cultural — compara o score do ArtCLIP em obras DENTRO da
    base de treino do ArtCLIP (APDDv2) vs. FORA da base (Portinari, arte
    brasileira). Um desvio sistemático indica viés de calibração cultural.
    """
    total_attr = "Total aesthetic score"
    df_in  = load_scores(_exp1_scores_dir(cfg), "original")
    df_out = load_scores(os.path.join(cfg["paths"]["outputs"], "exp2a_portinari"), "original")

    if (df_in is None or df_out is None
            or total_attr not in df_in.columns or total_attr not in df_out.columns):
        print("[cultural_bias] scores não encontrados, pulando.")
        return

    vals_in, vals_out = df_in[total_attr].dropna(), df_out[total_attr].dropna()
    if len(vals_in) == 0 or len(vals_out) == 0:
        print("[cultural_bias] sem valores válidos, pulando.")
        return

    pal = _palette(cfg); ht = _hatches(cfg)
    fig, ax = plt.subplots(figsize=(7, 6))
    bp = ax.boxplot([vals_in.values, vals_out.values],
                    tick_labels=["APDDv2\n(dentro da base)", "Portinari\n(fora da base)"],
                    patch_artist=True)
    _style_median(bp)
    for patch, key in zip(bp["boxes"], ["original", "janus_1b"]):
        patch.set_facecolor(pal[key]); patch.set_hatch(ht[key])
    ax.set_ylabel(L(cfg, "axes", "score"))
    ax.set_title("Viés Cultural — Score Dentro vs. Fora da Base de Treino")
    ax.grid(cfg["figures"]["grid"], alpha=0.3)
    save(fig, os.path.join(out_dir, "cultural_bias_boxplot.png"), cfg)

    res = distribution_diff(vals_in, vals_out, "APDDv2 (dentro)", "Portinari (fora)")
    if res:
        calibration_shift = float(vals_out.mean() - vals_in.mean())
        rows = [[
            f"{res['pair'][0]} vs {res['pair'][1]}",
            f"{vals_in.mean():.2f}", f"{vals_out.mean():.2f}",
            f"{calibration_shift:+.2f}",
            f"{res['ks_stat']:.3f}", f"{res['ks_p']:.3f}" if res['ks_p'] >= 0.001 else "<0.001",
            str(res['n1']), str(res['n2']),
        ]]
        render_simple_table_png(
            rows,
            ["Comparação", "Média dentro", "Média fora", "Desvio de calibração", "KS stat", "KS p", "n₁", "n₂"],
            os.path.join(out_dir, "cultural_bias_table.png"), cfg,
            title="Viés Cultural — Desvio de Calibração (fora − dentro da base)"
        )


def analyse_difficulty_groups(cfg, out_dir: str):
    """
    Cenário B adaptado (TRI sem múltiplos avaliadores/modelos): 3 grupos de
    dificuldade construídos por manipulação em vez de por consenso entre
    modelos — Fácil (humano original) / Médio (sintético limpo) / Difícil
    (sintético + corrupção estrutural, ruído "shapes" no nível máximo).
    Valida a regra de monotonicidade entre grupos (Fácil > Médio > Difícil) e
    sinaliza "erros de calibração": imagens do grupo Difícil que pontuam
    acima da mediana do grupo Fácil.
    """
    total_attr = "Total aesthetic score"
    exp1 = _exp1_scores_dir(cfg)
    df_easy = load_scores(exp1, "original")
    df_med  = load_scores(exp1, "Janus-Pro-7B")

    df4 = load_scores(os.path.join(cfg["paths"]["outputs"], "exp4_noise"), "original")
    df_hard = None
    if df4 is not None and "noise_type" in df4.columns and "noise_level" in df4.columns:
        shapes = df4[df4["noise_type"] == "shapes"]
        if not shapes.empty:
            max_level = shapes["noise_level"].max()
            df_hard = shapes[shapes["noise_level"] == max_level]

    groups_raw = [
        ("Fácil (Humano)", df_easy, "original"),
        ("Médio (Sintético limpo)", df_med, "janus_7b"),
        ("Difícil (Sintético + ruído estrutural)", df_hard, "highlight"),
    ]
    groups = []
    for name, d, key in groups_raw:
        if d is None or total_attr not in d.columns:
            continue
        vals = d[total_attr].dropna().values
        if len(vals) > 0:
            groups.append((name, vals, key))

    if len(groups) < 2:
        print("[difficulty_groups] dados insuficientes, pulando.")
        return

    # ── Matriz de confusão estética (densidade sobreposta) ──────────────────
    _overlaid_density_chart(
        groups, os.path.join(out_dir, "difficulty_groups_density.png"), cfg,
        title="Matriz de Confusão Estética — Grupos de Dificuldade",
        xlabel=L(cfg, "axes", "score"),
    )

    # ── Barras de média com regra de monotonicidade ─────────────────────────
    pal = _palette(cfg); ht = _hatches(cfg)
    fig, ax = plt.subplots(figsize=(8, 6))
    means = [float(np.mean(v)) for _, v, _ in groups]
    sems  = [float(np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0 for _, v, _ in groups]
    x = np.arange(len(groups))
    bars = ax.bar(x, means, yerr=sems, capsize=5,
                  color=[pal[k] for _, _, k in groups], edgecolor="black")
    for bar, (_, _, k) in zip(bars, groups):
        bar.set_hatch(ht[k])
    ax.set_xticks(x); ax.set_xticklabels([n for n, _, _ in groups], rotation=15, ha="right")
    ax.set_ylabel(L(cfg, "axes", "score"))
    monotonic = all(means[i] >= means[i + 1] for i in range(len(means) - 1))
    ax.set_title("Regra de Monotonicidade entre Grupos de Dificuldade\n"
                 + ("✓ Ordem preservada (Fácil > Médio > Difícil)" if monotonic
                    else "✗ Ordem violada — possível erro de calibração"))
    ax.grid(cfg["figures"]["grid"], alpha=0.3, axis="y")
    plt.tight_layout()
    save(fig, os.path.join(out_dir, "difficulty_groups_means.png"), cfg)

    # ── Tabela de estatísticas + erro de calibração ─────────────────────────
    easy_vals = groups[0][1]
    easy_median = float(np.median(easy_vals)) if len(easy_vals) else float("nan")
    rows = []
    for name, vals, _ in groups:
        n_above = int(np.sum(vals > easy_median)) if not np.isnan(easy_median) else 0
        pct_above = 100 * n_above / len(vals) if len(vals) else 0.0
        rows.append([
            name, str(len(vals)), f"{np.mean(vals):.2f}", f"{np.median(vals):.2f}",
            f"{np.std(vals):.2f}", f"{pct_above:.1f}%",
        ])
    render_simple_table_png(
        rows,
        ["Grupo", "n", "Média", "Mediana", "Std", "% acima da mediana do Fácil"],
        os.path.join(out_dir, "difficulty_groups_table.png"), cfg,
        title=f"Grupos de Dificuldade — Estatísticas e Erros de Calibração (mediana Fácil={easy_median:.2f})"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
# As amostras visuais (imagens de exemplo por experimento) não são mais geradas
# aqui: rodam dentro do próprio run.py, no cluster, junto com o experimento
# (ver pipeline/samples.py) — evita depender das bases de dados localmente.

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/analysis.yaml")
    parser.add_argument("--skip-analysis", action="store_true")
    args = parser.parse_args()

    cfg = load_cfg(args.config)

    fig_dir = os.path.join(cfg["paths"]["reports"], "figures")
    os.makedirs(fig_dir, exist_ok=True)

    rng_seed = 42
    random.seed(rng_seed)
    np.random.seed(rng_seed)

    if not args.skip_analysis:
        print("── Exp 1 (APDDv2) ─────────────────────────────────")
        analyse_exp1(cfg, fig_dir)
        print("── Exp 2 (Portinari) ───────────────────────────────")
        analyse_exp2(cfg, fig_dir)
        print("── Exp 3 (MNIST) ───────────────────────────────────")
        analyse_exp3(cfg, fig_dir)
        print("── Exp 4 (Noise) ───────────────────────────────────")
        analyse_exp4(cfg, fig_dir)
        print("── Exp 5 (Temporal) ────────────────────────────────")
        analyse_exp5(cfg, fig_dir)
        print("── Comparações (Fig 4.9, 4.10) ─────────────────────")
        analyse_comparisons(cfg, fig_dir)
        print("── Diagnósticos avançados ──────────────────────────")
        analyse_monotonicity(cfg, fig_dir)
        analyse_discriminative_validity(cfg, fig_dir)
        analyse_cultural_bias(cfg, fig_dir)
        analyse_difficulty_groups(cfg, fig_dir)

    print("\nConcluído.")


if __name__ == "__main__":
    main()
