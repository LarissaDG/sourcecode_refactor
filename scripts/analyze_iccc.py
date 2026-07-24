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
import numpy as np
import pandas as pd
import yaml
from scipy.stats import pearsonr, spearmanr, ttest_ind, mannwhitneyu, f_oneway


# ── Config ────────────────────────────────────────────────────────────────────

def load_cfg(path):
    with open(path, encoding="utf-8") as f:
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
    Reproduz o gráfico de barras do artigo:
    AVG(Human − Janus-1B) e AVG(Human − Janus-7B) por atributo.
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
    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 1.1), 6))
    ax.bar(x - width/2, diffs_1b, width, label="Human − Janus-1B",
           color="#33A650", hatch="///", edgecolor="black", alpha=0.85)
    ax.bar(x + width/2, diffs_7b, width, label="Human − Janus-7B",
           color="#F2A007", hatch="xxx", edgecolor="black", alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right")
    ax.set_ylabel("Average Score Difference")
    ax.set_title("Comparison of Aesthetic Scores Across Categories\n(Positive = Human scores higher)")
    ax.legend(); ax.grid(True, alpha=0.3, axis="y")
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


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/analysis.yaml")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    out_dir = os.path.join(cfg["paths"]["reports"], "figures_iccc")
    os.makedirs(out_dir, exist_ok=True)

    exp1_dir = os.path.join(cfg["paths"]["outputs"], "exp1_apdd")

    df_human = load_human_gt(cfg)
    df_1b    = load_artclip(exp1_dir, "Janus-Pro-1B")
    df_7b    = load_artclip(exp1_dir, "Janus-Pro-7B")

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

    for label, df_artclip in [("Janus-Pro-1B", df_1b), ("Janus-Pro-7B", df_7b)]:
        if df_artclip is None:
            print(f"[iccc] {label} scores não encontrados, pulando.")
            continue
        print(f"── {label} ──────────────────────────────────")
        if df_human is not None:
            run_hypothesis_tests(df_human, df_artclip, label, out_dir, report_lines)
            plot_histograms(df_human, df_artclip, label, out_dir)
            plot_boxplot(df_human, df_artclip, label, out_dir)
            plot_scatter(df_human, df_artclip, label, out_dir)
            plot_radar(df_human, df_artclip, label, out_dir)

    if df_human is not None:
        summary_table(df_human, df_1b, df_7b, out_dir, report_lines)
        plot_mean_bars(df_human, df_1b, df_7b, out_dir)

    report_path = os.path.join(cfg["paths"]["reports"], "iccc_stats_report.txt")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"\n✓ Relatório salvo: {report_path}")
    print(f"✓ Figuras salvas: {out_dir}")


if __name__ == "__main__":
    main()
