"""
Análise fiel ao artigo ICCC 2025 — port de metricas.py.

Reproduz exatamente a metodologia do repositório original
(https://github.com/LarissaDG/ICCC), adaptada para os CSVs
gerados pelo pipeline refatorado.

Diferenças em relação ao original:
  - Caminhos de arquivo lidos do analysis.yaml em vez de hard-coded
  - Alinhamento por stem de filename (não por índice posicional)
  - Plotly substituído por matplotlib (outputs estáticos PNG)
  - Cada par (Human vs Janus-1B, Human vs Janus-7B) analisado separadamente

Uso:
    python scripts/analyze_iccc.py --config configs/analysis_local.yaml
    python scripts/analyze_iccc.py --config configs/analysis.yaml
"""

import argparse
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import yaml
from scipy.stats import pearsonr, spearmanr, ttest_ind, mannwhitneyu, f_oneway, norm as scipy_norm
from sklearn.preprocessing import MinMaxScaler


# ── Config ────────────────────────────────────────────────────────────────────

def load_cfg(path):
    with open(path, encoding="utf-8-sig") as f:
        return yaml.safe_load(f)


COLS = [
    "Total aesthetic score", "Theme and logic", "Creativity",
    "Layout and composition", "Space and perspective", "The sense of order",
    "Light and shadow", "Color", "Details and texture", "The overall", "Mood",
]

RADAR_COLS = [
    "Theme and logic", "Creativity", "Layout and composition",
    "Space and perspective", "Light and shadow", "Color",
    "Details and texture", "The overall", "Mood",
]


# ── Data loading ───────────────────────────────────────────────────────────────

def _stem(filename):
    return os.path.splitext(os.path.basename(str(filename)))[0]


def load_human_gt(cfg):
    path = cfg["paths"].get("apddv2_csv", "")
    if not path or not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, encoding="latin1")
    except Exception:
        df = pd.read_csv(path)
    fn_col = next((c for c in df.columns if "filename" in c.lower()), df.columns[0])
    df = df.rename(columns={fn_col: "filename"})
    df["stem"] = df["filename"].apply(_stem)
    df = df[~(df[COLS].fillna(0) == 0).all(axis=1)]  # remove linhas 100% zeradas
    return df


def load_artclip(exp_dir, source):
    path = os.path.join(exp_dir, "scores", f"scores_{source}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if "stem" not in df.columns:
        df["stem"] = df["filename"].apply(_stem)
    return df


def apply_nan_mask(df_human, df_artclip, col):
    """
    Alinha por stem e propaga o NaN do ground truth para df_artclip
    (mesma lógica de metricas.py, mas por stem em vez de índice posicional).
    Retorna (human_vals, artclip_vals) com NaN onde human é NaN.
    """
    merged = df_human[["stem", col]].merge(
        df_artclip[["stem", col]].rename(columns={col: col + "_gen"}),
        on="stem", how="inner"
    )
    mask = merged[col].isna()
    ref_vals  = merged[col].copy()
    gen_vals  = merged[col + "_gen"].copy()
    gen_vals[mask] = np.nan
    return ref_vals, gen_vals


def save(fig, path, dpi=150):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ── Statistical tests (fiel ao metricas.py) ────────────────────────────────────

def run_hypothesis_tests(df_human, df_artclip, label, out_dir, report_lines):
    """
    Reproduz os três blocos de hipóteses de metricas.py.
    """
    available_cols = [c for c in COLS if c in df_human.columns and c in df_artclip.columns]

    report_lines.append(f"\n{'='*60}")
    report_lines.append(f"Comparison: Human GT vs {label}")
    report_lines.append(f"{'='*60}")

    # 1. Correlação entre "The overall" e "Mood" (H0: sem correlação)
    report_lines.append("\n--- Hypothesis 1: Correlation between 'The overall' and 'Mood' ---")
    for df_label, df in [("Human GT", df_human), (label, df_artclip)]:
        sub = df.dropna(subset=["The overall", "Mood"]) \
            if "The overall" in df.columns and "Mood" in df.columns else None
        if sub is not None and len(sub) > 2:
            pc, pp = pearsonr(sub["The overall"], sub["Mood"])
            sc, sp = spearmanr(sub["The overall"], sub["Mood"])
            report_lines.append(
                f"  {df_label} — Pearson: r={pc:.3f} (p={pp:.3f})  "
                f"Spearman: ρ={sc:.3f} (p={sp:.3f})"
            )

    # 2. t-test + Mann-Whitney por atributo (H0: mesmas médias nas duas bases)
    report_lines.append("\n--- Hypothesis 2: t-test and Mann-Whitney per attribute ---")
    for col in available_cols:
        human_vals, gen_vals = apply_nan_mask(df_human, df_artclip, col)
        h = human_vals.dropna().values
        g = gen_vals.dropna().values
        if len(h) > 1 and len(g) > 1:
            t_stat, t_p = ttest_ind(h, g, equal_var=False, nan_policy="omit")
            mw_stat, mw_p = mannwhitneyu(h, g, alternative="two-sided")
            sig = "*" if t_p < 0.05 else " "
            report_lines.append(
                f"  {sig} {col:35s}  t-test p={t_p:.4f}  Mann-Whitney p={mw_p:.4f}"
            )

    # 3. ANOVA entre atributos no dataset gerado
    report_lines.append("\n--- Hypothesis 3: ANOVA across attributes in generated scores ---")
    anova_data = [df_artclip[c].dropna().values for c in available_cols
                  if c in df_artclip.columns and df_artclip[c].dropna().shape[0] > 1]
    if len(anova_data) >= 2:
        _, anova_p = f_oneway(*anova_data)
        report_lines.append(f"  ANOVA p={anova_p:.4f} ({'significant' if anova_p < 0.05 else 'not significant'})")

    return report_lines


# ── Visualizations (fiel ao metricas.py) ───────────────────────────────────────

def plot_histograms(df_human, df_artclip, label, out_dir):
    col = "Total aesthetic score"
    if col not in df_human.columns or col not in df_artclip.columns:
        return
    human_vals, gen_vals = apply_nan_mask(df_human, df_artclip, col)
    h = human_vals.dropna().values
    g = gen_vals.dropna().values

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(h, bins=30, color="#448FF2", alpha=0.7, edgecolor="black")
    axes[0].set_title(f"Total Aesthetic Score — Human GT\n(n={len(h)})")
    axes[0].set_xlabel("Score"); axes[0].set_ylabel("Count")
    axes[1].hist(g, bins=30, color="#33A650", alpha=0.7, edgecolor="black")
    axes[1].set_title(f"Total Aesthetic Score — {label}\n(n={len(g)})")
    axes[1].set_xlabel("Score"); axes[1].set_ylabel("Count")
    plt.tight_layout()
    save(fig, os.path.join(out_dir, f"iccc_hist_{label.replace(' ', '_')}.png"))


def plot_boxplot(df_human, df_artclip, label, out_dir):
    col = "Total aesthetic score"
    if col not in df_human.columns or col not in df_artclip.columns:
        return
    human_vals, gen_vals = apply_nan_mask(df_human, df_artclip, col)
    h = human_vals.dropna().values
    g = gen_vals.dropna().values

    fig, ax = plt.subplots(figsize=(7, 5))
    bp = ax.boxplot([h, g], tick_labels=["Human GT", label], patch_artist=True)
    bp["boxes"][0].set_facecolor("#448FF2")
    bp["boxes"][1].set_facecolor("#33A650")
    for med in bp["medians"]:
        med.set_color("black"); med.set_linewidth(2.5)
    ax.set_title(f"Total Aesthetic Score — Human GT vs {label}")
    ax.set_ylabel("Score")
    ax.grid(True, alpha=0.3)
    save(fig, os.path.join(out_dir, f"iccc_boxplot_{label.replace(' ', '_')}.png"))


def plot_scatter(df_human, df_artclip, label, out_dir):
    if "The overall" not in df_human.columns or "Mood" not in df_human.columns:
        return
    human_sub = df_human.dropna(subset=["The overall", "Mood"])
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(human_sub["The overall"], human_sub["Mood"],
               alpha=0.4, color="#448FF2", label="Human GT", s=20)
    if "The overall" in df_artclip.columns and "Mood" in df_artclip.columns:
        gen_sub = df_artclip.dropna(subset=["The overall", "Mood"])
        ax.scatter(gen_sub["The overall"], gen_sub["Mood"],
                   alpha=0.4, color="#33A650", label=label, s=20)
    ax.set_xlabel("The overall"); ax.set_ylabel("Mood")
    ax.set_title("Correlation: 'The overall' vs 'Mood'")
    ax.legend(); ax.grid(True, alpha=0.3)
    save(fig, os.path.join(out_dir, f"iccc_scatter_{label.replace(' ', '_')}.png"))


def plot_radar(df_human, df_artclip, label, out_dir):
    """Gráfico de radar — fiel ao metricas.py (nota: viola diretrizes de acessibilidade)."""
    available = [c for c in RADAR_COLS
                 if c in df_human.columns and c in df_artclip.columns]
    if len(available) < 3:
        return

    orig_means = [df_human[c].dropna().mean() for c in available]
    gen_means  = [df_artclip[c].dropna().mean() for c in available]

    angles = np.linspace(0, 2 * np.pi, len(available), endpoint=False).tolist()
    orig_means_c = orig_means + [orig_means[0]]
    gen_means_c  = gen_means  + [gen_means[0]]
    angles_c = angles + [angles[0]]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.fill(angles_c, orig_means_c, color="#448FF2", alpha=0.25, label="Human GT")
    ax.plot(angles_c, orig_means_c, color="#448FF2", linewidth=2)
    ax.fill(angles_c, gen_means_c, color="#33A650", alpha=0.25, label=label)
    ax.plot(angles_c, gen_means_c, color="#33A650", linewidth=2)
    ax.set_xticks(angles)
    ax.set_xticklabels(available, fontsize=9)
    ax.set_title(f"Mean Attribute Scores — Human GT vs {label}", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    save(fig, os.path.join(out_dir, f"iccc_radar_{label.replace(' ', '_')}.png"))


def plot_mean_bars(df_human, df_artclip_1b, df_artclip_7b, out_dir):
    """
    Gráfico de barras agrupadas: AVG(Human − Janus-1B) e AVG(Human − Janus-7B) por atributo.
    Reproduz o gráfico do notebook ICCC (Cell 106/107).
    """
    available = [c for c in RADAR_COLS if c in (df_human.columns if df_human is not None else [])]
    diffs_1b, diffs_7b, labels = [], [], []
    for col in available:
        if df_artclip_1b is not None and col in df_artclip_1b.columns:
            h, g = apply_nan_mask(df_human, df_artclip_1b, col)
            diff = (h - g).dropna()
            diffs_1b.append(diff.mean() if len(diff) > 0 else np.nan)
        else:
            diffs_1b.append(np.nan)
        if df_artclip_7b is not None and col in df_artclip_7b.columns:
            h, g = apply_nan_mask(df_human, df_artclip_7b, col)
            diff = (h - g).dropna()
            diffs_7b.append(diff.mean() if len(diff) > 0 else np.nan)
        else:
            diffs_7b.append(np.nan)
        labels.append(col)

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(12, len(labels) * 1.4), 6))
    ax.bar(x - width/2, diffs_1b, width, label="Base Human-(Janus-1B)",
           color="#33A650", hatch="///", edgecolor="white", alpha=0.85)
    ax.bar(x + width/2, diffs_7b, width, label="Base Human-(Janus-7B)",
           color="#F2A007", hatch="xxx", edgecolor="white", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=12)
    ax.set_ylabel("Average Score", fontsize=12)
    ax.set_title("Comparison of Aesthetic Scores Across Categories", fontsize=13)
    ax.legend(fontsize=12)
    ax.grid(False)
    plt.tight_layout()
    save(fig, os.path.join(out_dir, "iccc_score_diff_bars.png"))


# ── Summary table ──────────────────────────────────────────────────────────────

def summary_table(df_human, df_1b, df_7b, out_dir, report_lines):
    """
    Gera a Tabela 4.2 do artigo: média ± std por atributo para Human, 1B, 7B.
    """
    report_lines.append("\n" + "="*70)
    report_lines.append("Table 4.2 — Mean ± Std per attribute (ICCC methodology)")
    report_lines.append(f"{'Attribute':<35} {'Human':>14} {'Janus-1B':>14} {'Janus-7B':>14}")
    report_lines.append("-"*77)

    rows = []; row_labels = []
    for col in COLS:
        h_vals = df_human[col].dropna().values if (df_human is not None and col in df_human.columns) else np.array([])
        g1_vals = np.array([]); g7_vals = np.array([])
        if df_1b is not None and col in df_1b.columns and df_human is not None and col in df_human.columns:
            _, g1 = apply_nan_mask(df_human, df_1b, col)
            g1_vals = g1.dropna().values
        if df_7b is not None and col in df_7b.columns and df_human is not None and col in df_human.columns:
            _, g7 = apply_nan_mask(df_human, df_7b, col)
            g7_vals = g7.dropna().values

        h_str  = f"{h_vals.mean():.2f}±{h_vals.std():.2f}"  if len(h_vals)  > 0 else "—"
        g1_str = f"{g1_vals.mean():.2f}±{g1_vals.std():.2f}" if len(g1_vals) > 0 else "—"
        g7_str = f"{g7_vals.mean():.2f}±{g7_vals.std():.2f}" if len(g7_vals) > 0 else "—"
        report_lines.append(f"  {col:<33} {h_str:>14} {g1_str:>14} {g7_str:>14}")
        rows.append([h_str, g1_str, g7_str]); row_labels.append(col)

    # PNG da tabela
    if rows:
        fig, ax = plt.subplots(figsize=(10, max(3, 0.5 * len(rows) + 1)))
        ax.axis("off")
        tbl = ax.table(cellText=rows, rowLabels=row_labels,
                       colLabels=["Human GT", "Janus-Pro-1B", "Janus-Pro-7B"],
                       cellLoc="center", rowLoc="right", loc="center")
        tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 1.7)
        for j in range(3):
            tbl[(0, j)].set_facecolor("#CCCCCC")
            tbl[(0, j)].set_text_props(fontweight="bold")
        ax.set_title("Mean ± Std per Attribute (ICCC methodology)", pad=12,
                     fontsize=11, fontweight="bold")
        plt.tight_layout()
        save(fig, os.path.join(out_dir, "iccc_summary_table.png"))


# ── Extended visualizations (from ICCC.ipynb) ─────────────────────────────────

def plot_artistic_categories(df, out_dir, prefix=""):
    """
    Artistic subcategory proportions + missing values heatmap by category
    (EDA notebook cells 24-29).
    """
    col = "Artistic Categories"
    if col not in df.columns:
        return

    # ── subcategory one-hot ──────────────────────────────────────────────────
    dummies = df[col].dropna().str.replace("*", ", ", regex=False).str.get_dummies(sep=", ")
    if not dummies.empty:
        prop = dummies.sum().sort_values(ascending=False) / len(df) * 100
        fig, ax = plt.subplots(figsize=(max(8, len(prop) * 0.8), 5))
        prop.plot(kind="bar", ax=ax, color="#448FF2", alpha=0.85, edgecolor="black")
        ax.set_title("Artistic Subcategory Proportions (%)")
        ax.set_xlabel("Subcategory"); ax.set_ylabel("% of dataset")
        ax.tick_params(axis="x", rotation=45)
        plt.tight_layout()
        save(fig, os.path.join(out_dir, f"{prefix}subcategory_proportions.png"))

    # ── high-level category bar ───────────────────────────────────────────────
    counts = df[col].dropna().value_counts()
    if not counts.empty:
        fig, ax = plt.subplots(figsize=(max(8, len(counts) * 0.8), 5))
        counts.plot(kind="bar", ax=ax, color="#448FF2", alpha=0.85, edgecolor="black")
        ax.set_title("Artistic Categories Distribution")
        ax.set_xlabel("Category"); ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=45)
        plt.tight_layout()
        save(fig, os.path.join(out_dir, f"{prefix}categories.png"))

    # ── missing values heatmap by category ───────────────────────────────────
    score_cols = [c for c in COLS[1:] if c in df.columns]  # inclui The sense of order
    if score_cols and col in df.columns:
        try:
            missing = df.groupby(col)[score_cols].apply(lambda x: x.isna().sum())
            if not missing.empty and missing.values.max() > 0:
                fig, ax = plt.subplots(figsize=(max(10, len(score_cols) * 1.2),
                                                max(4, len(missing) * 0.5)))
                import seaborn as sns
                sns.heatmap(missing, cmap="Reds", annot=True, fmt="d",
                            linewidths=0.5, ax=ax)
                ax.set_title("Missing Values by Artistic Category")
                ax.set_xlabel("Score Column"); ax.set_ylabel("Category")
                plt.tight_layout()
                save(fig, os.path.join(out_dir, f"{prefix}missing_by_category.png"))
        except Exception:
            pass


def plot_boxplot_all_attrs(df_dict, out_dir, prefix=""):
    """
    Side-by-side boxplot of all attributes for multiple datasets
    (EDA notebook cells 26-27).
    df_dict: {label: dataframe}
    """
    cols = [c for c in RADAR_COLS if any(c in df.columns for df in df_dict.values())]
    if not cols:
        return
    records = []
    for label, df in df_dict.items():
        for col in cols:
            if col in df.columns:
                for v in df[col].dropna():
                    records.append({"Attribute": col, "Score": v, "Source": label})
    if not records:
        return
    import seaborn as sns
    long = pd.DataFrame(records)
    fig, ax = plt.subplots(figsize=(max(12, len(cols) * 1.4), 6))
    palette = {"Original": "#448FF2", "Human GT": "#448FF2",
               "Janus-1B": "#33A650", "Janus-7B": "#F2A007"}
    sns.boxplot(data=long, x="Attribute", y="Score", hue="Source",
                palette=palette, ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right", fontsize=8)
    ax.set_title("Score Distributions by Attribute")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    save(fig, os.path.join(out_dir, f"{prefix}boxplot_all_attrs.png"))


def plot_attribute_histograms_grid(df, title, out_dir, prefix=""):
    """3×3 grid of per-attribute histograms (notebook cells 16-19)."""
    cols = [c for c in RADAR_COLS if c in df.columns]
    if not cols:
        return
    ncols = 3
    nrows = (len(cols) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 3.5))
    axes = np.array(axes).flatten()
    for i, col in enumerate(cols):
        vals = df[col].dropna().values
        axes[i].hist(vals, bins=20, color="#448FF2", alpha=0.75, edgecolor="white")
        axes[i].set_title(col, fontsize=9)
        axes[i].set_xlabel("Score", fontsize=8)
        axes[i].set_ylabel("Count", fontsize=8)
        axes[i].tick_params(labelsize=7)
    for j in range(len(cols), len(axes)):
        axes[j].set_visible(False)
    fig.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    save(fig, os.path.join(out_dir, f"{prefix}attr_hist_grid.png"))


def plot_normalized_distributions(df_dict, out_dir, prefix=""):
    """
    All attributes overlaid after MinMaxScaler normalization (notebook cell 18).
    df_dict: {label: dataframe}
    """
    cols = RADAR_COLS
    scaler = MinMaxScaler()
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#448FF2", "#33A650", "#F2A007", "#E84393", "#a29bfe",
              "#fd79a8", "#00b894", "#e17055", "#74b9ff", "#fdcb6e"]
    for i, col in enumerate(cols):
        all_vals = []
        for df in df_dict.values():
            if col in df.columns:
                all_vals.extend(df[col].dropna().tolist())
        if not all_vals:
            continue
        vals_norm = scaler.fit_transform(np.array(all_vals).reshape(-1, 1)).flatten()
        ax.hist(vals_norm, bins=30, alpha=0.4, color=colors[i % len(colors)],
                label=col, density=True)
    ax.set_xlabel("Normalized Score (0-1)")
    ax.set_ylabel("Density")
    ax.set_title("Normalized Score Distributions — All Attributes")
    ax.legend(fontsize=7, ncol=3)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save(fig, os.path.join(out_dir, f"{prefix}normalized_distributions.png"))


def plot_correlation_heatmap(df, title, out_dir, prefix=""):
    """Upper-triangle correlation heatmap (notebook cell 20)."""
    cols = [c for c in RADAR_COLS if c in df.columns]
    if len(cols) < 2:
        return
    corr = df[cols].corr()
    mask = np.tril(np.ones_like(corr, dtype=bool), k=-1)
    corr_upper = corr.where(~mask)

    fig, ax = plt.subplots(figsize=(9, 7))
    cmap = plt.cm.coolwarm
    im = ax.imshow(corr_upper.values, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(cols))); ax.set_yticklabels(cols, fontsize=8)
    for i in range(len(cols)):
        for j in range(len(cols)):
            val = corr_upper.iloc[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=7, color="black" if abs(val) < 0.7 else "white")
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(title)
    plt.tight_layout()
    save(fig, os.path.join(out_dir, f"{prefix}correlation_heatmap.png"))


def plot_avg_score_gaussian(df, title, out_dir, prefix=""):
    """Avg Score histogram with gaussian overlay + skewness/kurtosis (notebook cells 22-26)."""
    cols = [c for c in RADAR_COLS if c in df.columns]
    if not cols:
        return
    avg_scores = df[cols].mean(axis=1).dropna()
    if len(avg_scores) < 5:
        return

    from scipy.stats import skew, kurtosis
    sk = skew(avg_scores)
    kurt = kurtosis(avg_scores)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(avg_scores, bins=20, density=True, color="#448FF2", alpha=0.7,
            edgecolor="white", label="Avg Score")
    x = np.linspace(avg_scores.min(), avg_scores.max(), 200)
    mu, std = avg_scores.mean(), avg_scores.std()
    ax.plot(x, scipy_norm.pdf(x, mu, std), color="#d63031", linewidth=2, label="Normal fit")
    ax.axvline(mu, color="black", linestyle="--", linewidth=1, label=f"μ={mu:.2f}")
    ax.set_xlabel("Average Score")
    ax.set_ylabel("Density")
    ax.set_title(f"{title}\nSkewness={sk:.3f}  Kurtosis={kurt:.3f}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save(fig, os.path.join(out_dir, f"{prefix}avg_score_gaussian.png"))


def plot_scatter_with_regression(df, label, out_dir, prefix=""):
    """Scatter of 'The overall' vs 'Mood' with OLS regression line (notebook cell 54)."""
    if "The overall" not in df.columns or "Mood" not in df.columns:
        return
    sub = df.dropna(subset=["The overall", "Mood"])
    if len(sub) < 3:
        return
    x = sub["The overall"].values
    y = sub["Mood"].values
    m, b = np.polyfit(x, y, 1)
    xline = np.linspace(x.min(), x.max(), 100)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(x, y, alpha=0.35, color="#448FF2", s=20, label=label)
    ax.plot(xline, m * xline + b, color="#d63031", linewidth=2,
            label=f"y={m:.2f}x+{b:.2f}")
    r, p = pearsonr(x, y)
    ax.set_xlabel("The overall")
    ax.set_ylabel("Mood")
    ax.set_title(f"Correlation: 'The overall' vs 'Mood'\nPearson r={r:.3f} (p={p:.4f})")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save(fig, os.path.join(out_dir, f"{prefix}scatter_regression.png"))


def plot_three_way_histograms(df_orig, df_1b, df_7b, out_dir, prefix=""):
    """3-way per-attribute histograms: original vs Janus-1B vs Janus-7B (notebook cell 78)."""
    cols = [c for c in RADAR_COLS
            if (df_orig is not None and c in df_orig.columns) or
               (df_1b is not None and c in df_1b.columns)]
    if not cols:
        return
    ncols = 3
    nrows = (len(cols) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 3.5))
    axes = np.array(axes).flatten()
    for i, col in enumerate(cols):
        ax = axes[i]
        if df_orig is not None and col in df_orig.columns:
            ax.hist(df_orig[col].dropna(), bins=20, alpha=0.5, color="#448FF2",
                    label="Original", density=True)
        if df_1b is not None and col in df_1b.columns:
            ax.hist(df_1b[col].dropna(), bins=20, alpha=0.5, color="#33A650",
                    label="Janus-1B", density=True)
        if df_7b is not None and col in df_7b.columns:
            ax.hist(df_7b[col].dropna(), bins=20, alpha=0.5, color="#F2A007",
                    label="Janus-7B", density=True)
        ax.set_title(col, fontsize=9)
        ax.set_xlabel("Score", fontsize=8); ax.set_ylabel("Density", fontsize=8)
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(fontsize=7)
    for j in range(len(cols), len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Score Distributions: Original vs Generated", fontsize=12, fontweight="bold")
    plt.tight_layout()
    save(fig, os.path.join(out_dir, f"{prefix}three_way_histograms.png"))


def plot_radar_three_way(df_orig, df_1b, df_7b, out_dir, prefix=""):
    """Radar chart comparing mean attribute scores for original, 1B, and 7B."""
    cols = [c for c in RADAR_COLS
            if (df_orig is not None and c in df_orig.columns)]
    if len(cols) < 3:
        return
    orig_means = [df_orig[c].dropna().mean() for c in cols]
    means_1b = [df_1b[c].dropna().mean() for c in cols] if df_1b is not None else None
    means_7b = [df_7b[c].dropna().mean() for c in cols] if df_7b is not None else None

    angles = np.linspace(0, 2 * np.pi, len(cols), endpoint=False).tolist()
    angles_c = angles + [angles[0]]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for means, color, label in [
        (orig_means, "#448FF2", "Original"),
        (means_1b,  "#33A650", "Janus-1B"),
        (means_7b,  "#F2A007", "Janus-7B"),
    ]:
        if means is None:
            continue
        vals = means + [means[0]]
        ax.plot(angles_c, vals, color=color, linewidth=2)
        ax.fill(angles_c, vals, color=color, alpha=0.15)
    ax.set_xticks(angles); ax.set_xticklabels(cols, fontsize=9)
    ax.set_title("Mean Attribute Scores — Original vs Generated", pad=20)
    ax.legend(["Original", "Janus-1B", "Janus-7B"],
              loc="upper right", bbox_to_anchor=(1.35, 1.1))
    save(fig, os.path.join(out_dir, f"{prefix}radar_three_way.png"))


def plot_mean_bars_three_way(df_orig, df_1b, df_7b, out_dir, prefix="",
                              title="Score Difference (Original − Generated)"):
    """Grouped bar chart: original − 1B and original − 7B per attribute."""
    cols = [c for c in RADAR_COLS if df_orig is not None and c in df_orig.columns]
    if not cols:
        return
    diffs_1b, diffs_7b = [], []
    for col in cols:
        for df_gen, out in [(df_1b, diffs_1b), (df_7b, diffs_7b)]:
            if df_gen is not None and col in df_gen.columns:
                orig_mean = df_orig[col].dropna().mean()
                gen_mean = df_gen[col].dropna().mean()
                out.append(orig_mean - gen_mean)
            else:
                out.append(np.nan)

    x = np.arange(len(cols))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(10, len(cols) * 1.1), 6))
    ax.bar(x - width/2, diffs_1b, width, label="Orig − Janus-1B",
           color="#33A650", hatch="///", edgecolor="black", alpha=0.85)
    ax.bar(x + width/2, diffs_7b, width, label="Orig − Janus-7B",
           color="#F2A007", hatch="xxx", edgecolor="black", alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x); ax.set_xticklabels(cols, rotation=40, ha="right")
    ax.set_ylabel("Average Score Difference")
    ax.set_title(title)
    ax.legend(); ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    save(fig, os.path.join(out_dir, f"{prefix}score_diff_bars.png"))


def plot_sampling_distribution(df_full, df_sampled, out_dir):
    """
    Dois subplots mostrando a distribuição do APDDv2 antes e após amostragem.
      - Esquerda: APDDv2 completo (10023 imagens), 30 bins
      - Direita:  subconjunto amostrado (paper ICCC), 10 bins com marcação das fronteiras

    Métrica por imagem = média dos 10 atributos individuais (excluindo "Total aesthetic score").
    """
    from scipy.stats import gaussian_kde

    ATTR_COLS = COLS[1:]  # os 10 atributos individuais

    full_avg    = df_full[ATTR_COLS].mean(axis=1, skipna=True).dropna()
    sampled_avg = df_sampled[ATTR_COLS].mean(axis=1, skipna=True).dropna()

    # escala fixa 0–10 (scores ArtCLIP)
    x_min, x_max = 0.0, 10.0

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # ── Subplot esquerdo: APDDv2 completo ────────────────────────────────────────
    ax0 = axes[0]
    n_bins_full = 30
    counts, edges, patches = ax0.hist(
        full_avg, bins=n_bins_full,
        range=(x_min, x_max),
        color="#448FF2", alpha=0.75, edgecolor="white", linewidth=0.5,
        label=f"APDDv2 (n={len(full_avg):,})"
    )
    ax0.axvline(full_avg.mean(), color="#d63031", ls="--", lw=1.8,
                label=f"Média = {full_avg.mean():.2f}")
    ax0.axvline(full_avg.median(), color="#6c5ce7", ls=":", lw=1.8,
                label=f"Mediana = {full_avg.median():.2f}")

    # KDE no eixo secundário
    xs = np.linspace(x_min, x_max, 400)
    kde0 = gaussian_kde(full_avg, bw_method="scott")
    ax0_twin = ax0.twinx()
    ax0_twin.plot(xs, kde0(xs), color="#0984e3", lw=2, alpha=0.8)
    ax0_twin.set_ylabel("Densidade KDE", fontsize=9, color="#0984e3")
    ax0_twin.tick_params(axis="y", labelcolor="#0984e3", labelsize=8)
    ax0_twin.set_ylim(0)

    ax0.set_xlim(x_min, x_max)
    ax0.set_xticks(np.arange(0, 11, 1))
    ax0.set_xlabel("Score Médio por Imagem", fontsize=11)
    ax0.set_ylabel("Frequência", fontsize=11)
    ax0.set_title(f"APDDv2 completo\n(n = {len(full_avg):,}  ·  30 bins)", fontsize=12)
    ax0.legend(fontsize=9, loc="upper left")
    ax0.grid(True, alpha=0.25, axis="y")

    # ── Subplot direito: subconjunto amostrado ───────────────────────────────────
    ax1 = axes[1]
    n_bins_sampled = 10
    _, bin_edges, _ = ax1.hist(
        sampled_avg, bins=n_bins_sampled,
        range=(x_min, x_max),
        color="#F2A007", alpha=0.75, edgecolor="white", linewidth=0.5,
        label=f"Amostra ICCC (n={len(sampled_avg)})"
    )

    # fronteiras dos bins — mostram a estratégia de amostragem uniforme
    for edge in bin_edges:
        ax1.axvline(edge, color="#636e72", ls=":", lw=0.9, alpha=0.65)

    ax1.axvline(sampled_avg.mean(), color="#d63031", ls="--", lw=1.8,
                label=f"Média = {sampled_avg.mean():.2f}")
    ax1.axvline(sampled_avg.median(), color="#6c5ce7", ls=":", lw=1.8,
                label=f"Mediana = {sampled_avg.median():.2f}")

    # KDE
    xs2 = np.linspace(x_min, x_max, 400)
    kde1 = gaussian_kde(sampled_avg, bw_method="scott")
    ax1_twin = ax1.twinx()
    ax1_twin.plot(xs2, kde1(xs2), color="#e17055", lw=2, alpha=0.8)
    ax1_twin.set_ylabel("Densidade KDE", fontsize=9, color="#e17055")
    ax1_twin.tick_params(axis="y", labelcolor="#e17055", labelsize=8)
    ax1_twin.set_ylim(0)

    ax1.set_xlim(x_min, x_max)
    ax1.set_xticks(np.arange(0, 11, 1))
    ax1.set_xlabel("Score Médio por Imagem", fontsize=11)
    ax1.set_ylabel("Frequência", fontsize=11)
    ax1.set_title(
        f"Subconjunto amostrado — Paper ICCC\n(n = {len(sampled_avg)}  ·  10 bins, proporcional)",
        fontsize=12
    )
    ax1.legend(fontsize=9, loc="upper left")
    ax1.grid(True, alpha=0.25, axis="y")

    fig.suptitle(
        "Distribuição do Score Médio — APDDv2 Antes e Após Amostragem Proporcional por Bins",
        fontsize=13, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    save(fig, os.path.join(out_dir, "iccc_sampling_distribution.png"))


def plot_attr_before_after(df_full, df_sampled_filtered, out_dir):
    """
    Grade 2 colunas × 10 linhas.
    df_full             — APDDv2 completo (10023 imagens)
    df_sampled_filtered — APDDv2 filtrado pelas imagens que entraram na amostra
                          (obtido via filename-match com sampled_BIG/SMALL)
    Eixo X padronizado: global min/max sobre TODOS os atributos e ambos conjuntos.
    Salva: iccc_attr_before_after.png
    """
    df_sampled = df_sampled_filtered
    ATTR_COLS = COLS[1:]  # 10 atributos individuais
    available  = [c for c in ATTR_COLS
                  if c in df_full.columns and c in df_sampled.columns]
    if not available:
        return

    # ── Eixo X fixo 0–10 (escala ArtCLIP) ──────────────────────────────────────
    global_min  = 0.0
    global_max  = 10.0
    global_bins = np.linspace(global_min, global_max, 26)

    n = len(available)
    fig, axes = plt.subplots(n, 2, figsize=(13, 3.4 * n),
                             sharex=False, sharey=False)
    if n == 1:
        axes = [axes]

    color_full    = "#448FF2"
    color_sampled = "#F2A007"

    for i, col in enumerate(available):
        vals_full    = df_full[col].dropna()
        vals_sampled = df_sampled[col].dropna()

        if vals_full.empty:
            for ax in (axes[i][0], axes[i][1]):
                ax.text(0.5, 0.5, "Sem dados", ha="center", va="center",
                        transform=ax.transAxes, fontsize=9, color="#636e72")
                ax.set_axis_off()
            continue

        x_min = global_min
        x_max = global_max
        bins  = global_bins

        mean_full    = vals_full.mean()
        mean_sampled = vals_sampled.mean() if not vals_sampled.empty else float("nan")
        delta        = mean_sampled - mean_full if not np.isnan(mean_sampled) else float("nan")

        # ── esquerda: APDDv2 completo ─────────────────────────────────────────
        ax_l = axes[i][0]
        ax_l.hist(vals_full, bins=bins, color=color_full, alpha=0.75,
                  edgecolor="white", linewidth=0.4)
        ax_l.axvline(mean_full, color="#d63031", ls="--", lw=1.6,
                     label=f"μ = {mean_full:.2f}")
        ax_l.set_ylabel("Frequência", fontsize=8)
        ax_l.set_xlabel(col, fontsize=8)
        ax_l.set_xlim(x_min, x_max)
        ax_l.set_xticks(np.arange(0, 11, 1))
        ax_l.legend(fontsize=8, loc="upper left")
        ax_l.grid(True, alpha=0.2, axis="y")
        ax_l.tick_params(labelsize=7)
        if i == 0:
            ax_l.set_title("APDDv2 completo\n(n = {:,})".format(len(vals_full)),
                           fontsize=10, fontweight="bold", color=color_full)

        # ── direita: subconjunto amostrado ────────────────────────────────────
        ax_r = axes[i][1]
        if vals_sampled.empty:
            ax_r.text(0.5, 0.5, f"Sem anotação na amostra\n(NaN em {col})",
                      ha="center", va="center", transform=ax_r.transAxes,
                      fontsize=8, color="#636e72", style="italic")
            ax_r.set_axis_off()
        else:
            ax_r.hist(vals_sampled, bins=bins, color=color_sampled, alpha=0.75,
                      edgecolor="white", linewidth=0.4)
            ax_r.axvline(mean_sampled, color="#d63031", ls="--", lw=1.6,
                         label=f"μ = {mean_sampled:.2f}")

            if not np.isnan(delta):
                sign        = "▲" if delta > 0 else "▼"
                color_delta = "#00b894" if delta > 0 else "#e17055"
                ax_r.text(0.97, 0.88, f"{sign} {abs(delta):.2f}",
                          transform=ax_r.transAxes, ha="right", fontsize=9,
                          color=color_delta, fontweight="bold")

            ax_r.set_xlabel(col, fontsize=8)
            ax_r.set_xlim(x_min, x_max)
            ax_r.set_xticks(np.arange(0, 11, 1))
            ax_r.legend(fontsize=8, loc="upper left")
            ax_r.grid(True, alpha=0.2, axis="y")
            ax_r.tick_params(labelsize=7)

        if i == 0:
            n_label = len(vals_sampled) if not vals_sampled.empty else 0
            ax_r.set_title("Subconjunto amostrado — Paper ICCC\n(n = {})".format(n_label),
                           fontsize=10, fontweight="bold", color=color_sampled)

    fig.suptitle("Distribuição por Atributo Estético — Antes e Após Amostragem",
                 fontsize=13, fontweight="bold", y=1.005)
    fig.text(
        0.5, -0.008,
        f"Eixo X padronizado: [{global_min:.1f}, {global_max:.1f}] — "
        f"mínimo e máximo globais sobre todos os atributos e ambos os conjuntos de dados.",
        ha="center", fontsize=8, color="#636e72", style="italic"
    )
    plt.tight_layout()
    save(fig, os.path.join(out_dir, "iccc_attr_before_after.png"))


def run_full_analysis(df_human, df_orig, df_1b, df_7b, out_dir, exp_label, has_human=True):
    """Runs all extended visualizations for one experiment."""
    prefix = exp_label.replace(" ", "_") + "_" if exp_label else ""

    if has_human and df_human is not None:
        plot_artistic_categories(df_human, out_dir)
        plot_attribute_histograms_grid(df_human, "Human GT — Per-Attribute Histograms",
                                       out_dir, f"{prefix}human_")
        plot_normalized_distributions({"Human GT": df_human}, out_dir, f"{prefix}human_")
        plot_correlation_heatmap(df_human, "Human GT — Correlation Matrix",
                                 out_dir, f"{prefix}human_")
        plot_avg_score_gaussian(df_human, "Human GT — Avg Score Distribution",
                                out_dir, f"{prefix}human_")
        plot_scatter_with_regression(df_human, "Human GT", out_dir, f"{prefix}human_")

    if df_orig is not None:
        plot_attribute_histograms_grid(df_orig, f"{exp_label} — Original (ArtCLIP) Histograms",
                                       out_dir, f"{prefix}orig_")
        plot_correlation_heatmap(df_orig, f"{exp_label} — Original Correlation Matrix",
                                 out_dir, f"{prefix}orig_")
        plot_avg_score_gaussian(df_orig, f"{exp_label} — Original Avg Score",
                                out_dir, f"{prefix}orig_")
        plot_scatter_with_regression(df_orig, f"{exp_label} — Original", out_dir, f"{prefix}orig_")

    plot_three_way_histograms(df_orig, df_1b, df_7b, out_dir, prefix)
    plot_radar_three_way(df_orig, df_1b, df_7b, out_dir, prefix)
    plot_mean_bars_three_way(df_orig, df_1b, df_7b, out_dir, prefix)

    # side-by-side boxplot
    box_dict = {}
    if df_orig is not None: box_dict["Original"] = df_orig
    if df_1b   is not None: box_dict["Janus-1B"] = df_1b
    if df_7b   is not None: box_dict["Janus-7B"] = df_7b
    if box_dict:
        plot_boxplot_all_attrs(box_dict, out_dir, prefix)


# ── Main ──────────────────────────────────────────────────────────────────────

# ── Legacy ICCC data (sampled_SMALL / sampled_BIG CSVs) ───────────────────────

def load_legacy_csv(path):
    """Carrega sampled_SMALL_with_gen_scored.csv ou sampled_BIG_with_gen_scored.csv."""
    if not path or not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, encoding="latin1")
    except Exception:
        df = pd.read_csv(path)
    if "stem" not in df.columns:
        df["stem"] = df["filename"].apply(_stem)
    return df


def plot_comparison_bars(df_human, df_1b_gen, df_7b_gen,
                         df_orig_artclip,
                         out_path, title_suffix=""):
    """
    Gera DOIS painéis lado a lado:
      Painel A: Human annotation (APDDv2-10023) − ArtCLIP(gerado)  [ICCC metricas.py]
      Painel B: ArtCLIP(original) − ArtCLIP(gerado)                [Tabela 4.2 do artigo]
    """
    available = [c for c in RADAR_COLS if c in (df_human.columns if df_human is not None else [])]

    def _diffs(df_ref, df_gen):
        d1, d7 = [], []
        for col in available:
            for d_gen, out in [(df_1b_gen, d1), (df_7b_gen, d7)]:
                if df_ref is not None and d_gen is not None and col in df_ref.columns and col in d_gen.columns:
                    h, g = apply_nan_mask(df_ref, d_gen, col)
                    diff = (h - g).dropna()
                    out.append(float(diff.mean()) if len(diff) > 0 else np.nan)
                else:
                    out.append(np.nan)
        return d1, d7

    x = np.arange(len(available))
    width = 0.35

    has_human = df_human is not None
    has_orig  = df_orig_artclip is not None
    n_panels  = int(has_human) + int(has_orig)
    if n_panels == 0:
        return

    fig, axes = plt.subplots(1, n_panels, figsize=(max(10, len(available)) * n_panels, 6),
                             squeeze=False)
    panel = 0

    if has_human:
        d1, d7 = _diffs(df_human, df_1b_gen)
        ax = axes[0][panel]
        ax.bar(x - width/2, d1, width, label="Human − Janus-1B",
               color="#33A650", hatch="///", edgecolor="black", alpha=0.85)
        ax.bar(x + width/2, d7, width, label="Human − Janus-7B",
               color="#F2A007", hatch="xxx", edgecolor="black", alpha=0.85)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x); ax.set_xticklabels(available, rotation=40, ha="right")
        ax.set_ylabel("Avg Score Difference")
        ax.set_title(f"Panel A — Human annotation − ArtCLIP(generated)\n{title_suffix}")
        ax.legend(); ax.grid(True, alpha=0.3, axis="y")
        panel += 1

    if has_orig:
        d1, d7 = _diffs(df_orig_artclip, df_1b_gen)
        ax = axes[0][panel]
        ax.bar(x - width/2, d1, width, label="ArtCLIP orig − Janus-1B",
               color="#448FF2", hatch="", edgecolor="black", alpha=0.85)
        ax.bar(x + width/2, d7, width, label="ArtCLIP orig − Janus-7B",
               color="#448FF2", hatch="xxx", edgecolor="black", alpha=0.85)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x); ax.set_xticklabels(available, rotation=40, ha="right")
        ax.set_ylabel("Avg Score Difference")
        ax.set_title(f"Panel B — ArtCLIP(original) − ArtCLIP(generated)\n{title_suffix}")
        ax.legend(); ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle("Comparison of Aesthetic Scores Across Categories", fontsize=13, fontweight="bold")
    plt.tight_layout()
    save(fig, out_path)


def analyse_legacy(cfg, args, out_dir):
    """Análise usando os CSVs legados do ICCC (sampled_SMALL / sampled_BIG)."""
    df_small = load_legacy_csv(args.iccc_small)  # Janus-1B
    df_big   = load_legacy_csv(args.iccc_big)    # Janus-7B
    df_human = load_human_gt(cfg)

    if df_small is None and df_big is None:
        return

    # ArtCLIP scores on original paintings (from pipeline)
    exp1_dir = os.path.join(cfg["paths"]["outputs"], "exp1_apdd")
    df_orig_artclip = load_artclip(exp1_dir, "original")

    print("── Legacy ICCC data ────────────────────────────────────")

    # Gráfico comparativo (dois painéis)
    plot_comparison_bars(
        df_human, df_small, df_big, df_orig_artclip,
        out_path=os.path.join(out_dir, "iccc_legacy_comparison_bars.png"),
        title_suffix="(Legacy ICCC data: sampled_SMALL=1B, sampled_BIG=7B)"
    )
    print("  ✓ iccc_legacy_comparison_bars.png")

    # Tabela sumária com os dados legados
    if df_human is not None:
        report_lines = ["Legacy ICCC Data — Summary Table", "=" * 60]
        summary_table(df_human, df_small, df_big, out_dir, report_lines)
        # Renomeia para não sobrescrever a tabela do pipeline
        src = os.path.join(out_dir, "iccc_summary_table.png")
        dst = os.path.join(out_dir, "iccc_legacy_summary_table.png")
        if os.path.exists(src):
            os.replace(src, dst)
        print("  ✓ iccc_legacy_summary_table.png")

        # Stat tests
        for label, df_gen in [("Janus-1B (legacy)", df_small), ("Janus-7B (legacy)", df_big)]:
            if df_gen is not None:
                run_hypothesis_tests(df_human, df_gen, label, out_dir, report_lines)
        rpath = os.path.join(cfg["paths"]["reports"], "iccc_legacy_stats_report.txt")
        with open(rpath, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines))
        print(f"  ✓ {rpath}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/analysis.yaml")
    parser.add_argument("--iccc-small", default=None,
                        help="Caminho para sampled_SMALL_with_gen_scored.csv (Janus-1B)")
    parser.add_argument("--iccc-big", default=None,
                        help="Caminho para sampled_BIG_with_gen_scored.csv (Janus-7B)")
    parser.add_argument("--exp-dir", default=None,
                        help="Pasta de experimento do pipeline (ex: outputs/exp1_apdd). "
                             "Gera o gráfico de comparação usando scores_original/1B/7B.csv")
    parser.add_argument("--all-viz", action="store_true",
                        help="Gera todas as visualizações do notebook ICCC.ipynb")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    out_dir = os.path.join(cfg["paths"]["reports"], "figures_iccc")
    os.makedirs(out_dir, exist_ok=True)

    exp1_dir  = os.path.join(cfg["paths"]["outputs"], "exp1_apdd")
    exp2a_dir = os.path.join(cfg["paths"]["outputs"], "exp2a_portinari")
    exp2b_dir = os.path.join(cfg["paths"]["outputs"], "exp2b_portinari_human")

    df_human = load_human_gt(cfg)
    df_1b    = load_artclip(exp1_dir, "Janus-Pro-1B")
    df_7b    = load_artclip(exp1_dir, "Janus-Pro-7B")
    df_orig  = load_artclip(exp1_dir, "original")

    if df_human is None:
        print("[iccc] APDDv2-10023.csv não encontrado — análise com human GT pulada.")
        print("       Defina paths.apddv2_csv no YAML para ativar.")

    report_lines = [
        "ICCC Faithful Analysis Report",
        "=" * 60,
        "Methodology: t-test, Mann-Whitney U, ANOVA (metricas.py)",
        "NaN masking: per-attribute, aligned by filename stem",
        "",
    ]

    # ── Gráficos de resultados do Paper_iccc.html — fontes: legacy CSVs ──────────
    # sampled_SMALL = Janus-1B, sampled_BIG = Janus-7B
    df_legacy_1b = load_legacy_csv(cfg["paths"].get("sampled_small", ""))
    df_legacy_7b = load_legacy_csv(cfg["paths"].get("sampled_big",   ""))

    for label, df_artclip in [("Janus-Pro-1B", df_legacy_1b), ("Janus-Pro-7B", df_legacy_7b)]:
        if df_artclip is None:
            print(f"[iccc] {label} (legacy) não encontrado, pulando.")
            continue
        print(f"── {label} ──────────────────────────────────")
        if df_human is not None:
            run_hypothesis_tests(df_human, df_artclip, label, out_dir, report_lines)
            plot_histograms(df_human, df_artclip, label, out_dir)
            plot_boxplot(df_human, df_artclip, label, out_dir)
            plot_scatter(df_human, df_artclip, label, out_dir)
            plot_radar(df_human, df_artclip, label, out_dir)

    if df_human is not None:
        summary_table(df_human, df_legacy_1b, df_legacy_7b, out_dir, report_lines)
        plot_mean_bars(df_human, df_legacy_1b, df_legacy_7b, out_dir)

    # Gráficos de amostragem: APDDv2 completo vs amostra gerada por sample_apddv2.py
    if df_human is not None:
        sample_csv = cfg.get("sampling", {}).get("output_csv", "")
        sampled_big_path = cfg["paths"].get("sampled_big", "")

        df_sample_ref = None
        if sample_csv and os.path.exists(sample_csv):
            df_sample_ref = pd.read_csv(sample_csv, encoding="utf-8")
            print(f"\n── Gráficos de amostragem (sample_apddv2.py) ───────")
            print(f"  Usando: {sample_csv} (n={len(df_sample_ref)})")
        elif sampled_big_path and os.path.exists(sampled_big_path):
            df_sample_ref = load_legacy_csv(sampled_big_path)
            print(f"\n── Gráficos de amostragem (legado fallback) ────────")
            print(f"  Usando: {sampled_big_path}")

        if df_sample_ref is not None:
            fn_col = next((c for c in df_sample_ref.columns if "filename" in c.lower()), None)
            if fn_col:
                stems = set(df_sample_ref[fn_col].apply(
                    lambda x: os.path.splitext(os.path.basename(str(x)))[0]))
                df_human_sampled = df_human[df_human["stem"].isin(stems)].copy()
            else:
                df_human_sampled = df_sample_ref
            print(f"  APDDv2 filtrado: {len(df_human_sampled)} imagens")
            plot_sampling_distribution(df_human, df_human_sampled, out_dir)
            print("  ✓ iccc_sampling_distribution.png")
            plot_attr_before_after(df_human, df_human_sampled, out_dir)
            print("  ✓ iccc_attr_before_after.png")

    report_path = os.path.join(cfg["paths"]["reports"], "iccc_stats_report.txt")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"\n✓ Relatório salvo: {report_path}")
    print(f"✓ Figuras salvas: {out_dir}")

    # Extended visualizations (all notebook viz)
    if args.all_viz:
        print("\n── Extended visualizations (ICCC.ipynb) ────────────")
        run_full_analysis(df_human, df_orig, df_1b, df_7b, out_dir,
                          "exp1_apdd", has_human=(df_human is not None))
        print("  ✓ exp1_apdd extended visualizations")

        for exp_label, exp_dir_path in [("exp2a", exp2a_dir), ("exp2b", exp2b_dir)]:
            if not os.path.isdir(exp_dir_path):
                continue
            p_orig = load_artclip(exp_dir_path, "original")
            p_1b   = load_artclip(exp_dir_path, "Janus-Pro-1B")
            p_7b   = load_artclip(exp_dir_path, "Janus-Pro-7B")
            if p_orig is None and p_1b is None:
                continue
            run_full_analysis(None, p_orig, p_1b, p_7b, out_dir, exp_label, has_human=False)
            print(f"  ✓ {exp_label} extended visualizations")

    # Dados legados do ICCC (sampled_SMALL / sampled_BIG)
    if args.iccc_small or args.iccc_big:
        analyse_legacy(cfg, args, out_dir)

    # Gráfico de comparação direto de uma pasta de experimento do pipeline
    if args.exp_dir:
        exp_name = os.path.basename(args.exp_dir.rstrip("/\\"))
        df_orig_e  = load_artclip(args.exp_dir, "original")
        df_1b_e  = load_artclip(args.exp_dir, "Janus-Pro-1B")
        df_7b_e  = load_artclip(args.exp_dir, "Janus-Pro-7B")
        out_path = os.path.join(out_dir, f"pipeline_comparison_{exp_name}.png")
        plot_comparison_bars(
            df_human, df_1b_e, df_7b_e, df_orig_e,
            out_path=out_path,
            title_suffix=f"(Pipeline: {exp_name})"
        )
        print(f"  ✓ pipeline_comparison_{exp_name}.png")


if __name__ == "__main__":
    main()
