"""
Visualizações e tabelas da narrativa de Paper_iccc.html e exp1_apdd.html —
EDA do APDDv2, análise da amostragem (antes/depois, por estratégia) e as
perguntas sobre impacto do tamanho do modelo (Human vs. Janus-1B vs. Janus-7B).

Reaproveita helpers já existentes em scripts/analyze.py (Friedman+Wilcoxon+CLD,
diferença de distribuição, carregamento de scores/ground-truth) — não duplica
lógica estatística.

Toda tabela é salva em 2 formatos: <nome>.png (imagem) e <nome>.tex.txt (LaTeX).

Uso:
    python scripts/analyze_paper.py --config configs/analysis_local.yaml
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm as scipy_norm, skew

from analyze import (
    load_cfg, load_scores, load_human_gt, _stem, attr_label,
    friedman_wilcoxon, render_stat_table_png, distribution_diff,
    render_dist_diff_table, save as save_fig,
)

# ═══════════════════════════════════════════════════════════════════════════
# Constantes
# ═══════════════════════════════════════════════════════════════════════════

TOTAL_ATTR = "Total aesthetic score"

# Os 10 atributos individuais do APDDv2 (sem o Total)
ALL_ATTRS = [
    "Theme and logic", "Creativity", "Layout and composition",
    "Space and perspective", "The sense of order", "Light and shadow",
    "Color", "Details and texture", "The overall", "Mood",
]

# 9 atributos usados na métrica de amostragem — exclui "The sense of order"
# (modelo 6 do ArtCLIP tem bug conhecido, ver README). Mesmo critério de
# datasets/apddv2.py::BIN_ATTRIBUTES.
BIN_ATTRS = [a for a in ALL_ATTRS if a != "The sense of order"]

COLOR_BEFORE   = "#33A650"  # verde — antes da amostragem
COLOR_AFTER    = "#F1C40F"  # amarelo — depois da amostragem
COLOR_HUMAN_1B = "#448FF2"  # azul — Human − Janus-1B
COLOR_HUMAN_7B = "#F23838"  # vermelho — Human − Janus-7B
COLOR_UNIFORM  = "#448FF2"  # azul — estratégia uniform_bins
COLOR_STRAT    = "#F2A007"  # laranja — estratégia proportional_stratified

STRATEGIES = ["uniform_bins", "proportional_stratified"]
STRATEGY_LABELS = {
    "uniform_bins": "Uniforme",
    "proportional_stratified": "Estratificado Proporcional",
}


# ═══════════════════════════════════════════════════════════════════════════
# Helpers de dados
# ═══════════════════════════════════════════════════════════════════════════

def _exp_scores_dir(cfg, base_name, strategy):
    """
    <outputs>/<base_name>_<strategy>/ com fallback pra <outputs>/<base_name>/
    (experimentos que ainda não rodaram com sampling.strategies).
    """
    suffixed = os.path.join(cfg["paths"]["outputs"], f"{base_name}_{strategy}")
    if os.path.isdir(suffixed):
        return suffixed
    return os.path.join(cfg["paths"]["outputs"], base_name)


def _avg_score(df):
    """Média dos 9 atributos de BIN_ATTRS por linha (métrica usada na amostragem)."""
    cols = [c for c in BIN_ATTRS if c in df.columns]
    if not cols:
        return pd.Series([], dtype=float)
    return df[cols].mean(axis=1, skipna=True)


def _title_case(s):
    return str(s).strip().title() if pd.notna(s) else "—"


def _parse_categories(df):
    """Separa 'Artistic Categories' ('medium*style*subject') em 3 colunas."""
    parts = df["Artistic Categories"].fillna("").str.split("*", n=2, expand=True)
    for i, c in enumerate(["Medium", "Style", "Subject"]):
        df = df.copy()
        df[c] = parts[i] if i in parts.columns else None
    return df


# ═══════════════════════════════════════════════════════════════════════════
# Tabelas — export duplo PNG + LaTeX
# ═══════════════════════════════════════════════════════════════════════════

def _latex_escape(s):
    return (str(s).replace("\\", r"\textbackslash{}").replace("&", r"\&")
            .replace("%", r"\%").replace("_", r"\_").replace("#", r"\#"))


def save_table(rows, col_labels, out_dir, name, cfg, title="", row_labels=None):
    """
    Salva uma tabela simples em 2 formatos:
      <out_dir>/<name>.png     — imagem (matplotlib table)
      <out_dir>/<name>.tex.txt — LaTeX (booktabs)

    rows: lista de listas de strings já formatadas (sem markup).
    """
    os.makedirs(out_dir, exist_ok=True)
    n_rows, n_cols = len(rows), len(col_labels)
    if n_rows == 0:
        return

    # ── PNG ──────────────────────────────────────────────────────────────
    fig_w = max(8, 1.5 + (n_cols + (1 if row_labels else 0)) * 2.0)
    fig_h = max(2, 0.6 + n_rows * 0.42)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    kwargs = dict(cellText=rows, colLabels=col_labels, cellLoc="center", loc="center")
    if row_labels:
        kwargs["rowLabels"] = row_labels
        kwargs["rowLoc"] = "right"
    tbl = ax.table(**kwargs)
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.5)
    for j in range(n_cols):
        tbl[(0, j)].set_facecolor("#CCCCCC")
        tbl[(0, j)].set_text_props(fontweight="bold")
    if title:
        ax.set_title(title, pad=12, fontsize=11, fontweight="bold")
    plt.tight_layout()
    save_fig(fig, os.path.join(out_dir, f"{name}.png"), cfg)

    # ── LaTeX ────────────────────────────────────────────────────────────
    n_cols_tex = n_cols + (1 if row_labels else 0)
    lines = [r"\begin{table}[ht]", r"\centering"]
    if title:
        lines.append(f"\\caption{{{_latex_escape(title)}}}")
    lines.append(r"\begin{tabular}{" + "l" * n_cols_tex + "}")
    lines.append(r"\toprule")
    header = ([""] if row_labels else []) + [_latex_escape(c) for c in col_labels]
    lines.append(" & ".join(header) + r" \\")
    lines.append(r"\midrule")
    for i, row in enumerate(rows):
        cells = [_latex_escape(c) for c in row]
        rlabel = [_latex_escape(row_labels[i])] if row_labels else []
        lines.append(" & ".join(rlabel + cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(os.path.join(out_dir, f"{name}.tex.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ═══════════════════════════════════════════════════════════════════════════
# 1. Análise exploratória do dataset (EDA)
# ═══════════════════════════════════════════════════════════════════════════

def build_eda(cfg, out_dir):
    df = load_human_gt(cfg)
    if df is None or "Artistic Categories" not in df.columns:
        print("[eda] APDDv2-10023.csv (ou coluna 'Artistic Categories') não encontrado, pulando.")
        return
    df = _parse_categories(df)
    total = len(df)

    # ── Tabela cruzada Medium x Style x Subject ─────────────────────────
    grp = (df.groupby(["Medium", "Style", "Subject"], dropna=False)
             .size().reset_index(name="Count")
             .sort_values(["Medium", "Style", "Subject"]))
    rows = [
        [_title_case(r["Medium"]), _title_case(r["Style"]), _title_case(r["Subject"]),
         f"{int(r['Count']):,}", f"{100 * r['Count'] / total:.1f}%"]
        for _, r in grp.iterrows()
    ]
    save_table(rows, ["Medium", "Style", "Subject", "Count", "% do dataset"],
               out_dir, "eda_category_crosstab", cfg,
               title="Tabela cruzada Medium × Style × Subject")

    # ── Tabelas marginais ────────────────────────────────────────────────
    for dim in ["Medium", "Style", "Subject"]:
        counts = df[dim].apply(_title_case).value_counts()
        rows = [[k, f"{v:,}", f"{100 * v / total:.1f}%"] for k, v in counts.items()]
        save_table(rows, ["Categoria", "n", "%"], out_dir, f"eda_{dim.lower()}_summary", cfg,
                   title=f"Distribuição por {dim}")
    print("  ✓ EDA: tabela cruzada + distribuições Medium/Style/Subject")


def build_missing_values(cfg, out_dir):
    df = load_human_gt(cfg)
    if df is None or "Artistic Categories" not in df.columns:
        print("[missing_values] dados insuficientes, pulando.")
        return
    df = _parse_categories(df)
    df["Category"] = (df["Medium"].apply(_title_case) + " / " +
                       df["Style"].apply(_title_case) + " / " +
                       df["Subject"].apply(_title_case))
    score_cols = [c for c in ALL_ATTRS if c in df.columns]
    if not score_cols:
        return
    missing = df.groupby("Category")[score_cols].apply(lambda x: x.isna().sum())
    if missing.empty:
        return

    fig, ax = plt.subplots(figsize=(max(10, len(score_cols) * 1.2), max(6, len(missing) * 0.42)))
    vmax = max(1, int(missing.values.max()))
    im = ax.imshow(missing.values, cmap="Reds", aspect="auto", vmin=0, vmax=vmax)
    ax.set_xticks(range(len(score_cols)))
    ax.set_xticklabels([attr_label(cfg, c) for c in score_cols], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(missing)))
    ax.set_yticklabels(missing.index, fontsize=8)
    for i in range(missing.shape[0]):
        for j in range(missing.shape[1]):
            val = int(missing.values[i, j])
            ax.text(j, i, str(val), ha="center", va="center", fontsize=7,
                    color="white" if val > vmax * 0.6 else "black")
    plt.colorbar(im, ax=ax, shrink=0.7, label="Valores ausentes")
    ax.set_title("Valores Ausentes por Categoria Artística")
    plt.tight_layout()
    save_fig(fig, os.path.join(out_dir, "eda_missing_by_category.png"), cfg)

    row_labels = list(missing.index)
    rows = [[str(int(v)) for v in missing.values[i]] for i in range(len(missing))]
    save_table(rows, [attr_label(cfg, c) for c in score_cols], out_dir,
               "eda_missing_by_category_table", cfg,
               title="Valores Ausentes por Categoria Artística", row_labels=row_labels)
    print("  ✓ Missing values por categoria")


# ═══════════════════════════════════════════════════════════════════════════
# 3. Amostragem — antes/depois, por estratégia
# ═══════════════════════════════════════════════════════════════════════════

def _sampling_distribution_chart(df_full, df_sampled, strategy, out_dir, cfg, show_full_stats):
    full_avg = _avg_score(df_full).dropna()
    sampled_avg = _avg_score(df_sampled).dropna()
    if full_avg.empty or sampled_avg.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    panels = [
        (axes[0], full_avg, COLOR_BEFORE, "Antes (APDDv2 completo)", True),
        (axes[1], sampled_avg, COLOR_AFTER, f"Depois ({STRATEGY_LABELS[strategy]})", show_full_stats),
    ]
    for ax, vals, color, label, show_stats in panels:
        ax.hist(vals, bins=30, range=(0, 10), density=True, color=color, alpha=0.75, edgecolor="white")
        mu, med = vals.mean(), vals.median()
        ax.axvline(mu, color="#2d3436", ls="--", lw=1.6, label=f"Média = {mu:.2f}")
        ax.axvline(med, color="#6c5ce7", ls=":", lw=1.6, label=f"Mediana = {med:.2f}")
        title = f"{label}\n(n = {len(vals):,})"
        if show_stats and len(vals) > 3:
            sk, kt, std = skew(vals), kurtosis(vals), vals.std()
            xs = np.linspace(0, 10, 300)
            ax.plot(xs, scipy_norm.pdf(xs, mu, std), color="#d63031", lw=2, label="Ajuste normal")
            title += f"\nAssimetria = {sk:.3f}   Curtose = {kt:.3f}"
        ax.set_xlim(0, 10)
        ax.set_xticks(np.arange(0, 11, 1))
        ax.set_xlabel("Score Médio por Imagem")
        ax.set_ylabel("Frequência (densidade)")
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.25, axis="y")
    fig.suptitle("Distribuição do Score Médio — Antes vs. Depois da Amostragem", fontsize=13, fontweight="bold")
    plt.tight_layout()
    save_fig(fig, os.path.join(out_dir, f"sampling_dist_{strategy}.png"), cfg)


def _bin_distribution_table(df_full, df_sampled, out_dir, cfg, name, n_bins=30):
    full_avg = _avg_score(df_full).dropna()
    sampled_avg = _avg_score(df_sampled).dropna()
    if full_avg.empty:
        return
    edges = np.linspace(0, 10, n_bins + 1)
    full_binned = pd.cut(full_avg, bins=edges, include_lowest=True)
    sampled_binned = pd.cut(sampled_avg, bins=edges, include_lowest=True) if not sampled_avg.empty else None
    full_counts = full_binned.value_counts().sort_index()
    sampled_counts = sampled_binned.value_counts().sort_index() if sampled_binned is not None else None

    rows = []
    for i, interval in enumerate(full_counts.index):
        n_sampled = int(sampled_counts.iloc[i]) if sampled_counts is not None else 0
        rows.append([str(i + 1), f"[{interval.left:.2f}, {interval.right:.2f}]",
                     str(int(full_counts.iloc[i])), str(n_sampled)])
    save_table(
        rows, ["Bin", "Intervalo de score", "Nb (APDDv2)", "nb (amostrado)"],
        out_dir, name, cfg,
        title=(
            "Distribuição de Imagens por Bin — "
            "Bin: índice da faixa; Intervalo de score: limites do score médio nessa faixa; "
            "Nb (APDDv2): total de imagens do dataset completo nessa faixa; "
            "nb (amostrado): quantas dessa faixa entraram na amostra."
        ),
    )


def _attr_before_after_grid(df_full, df_sampled, out_dir, cfg, filename):
    cols = BIN_ATTRS
    bins = np.linspace(0, 10, 26)
    n = len(cols)
    fig, axes = plt.subplots(n, 2, figsize=(13, 3.0 * n))
    for i, col in enumerate(cols):
        vf = df_full[col].dropna() if col in df_full.columns else pd.Series([], dtype=float)
        vs = df_sampled[col].dropna() if col in df_sampled.columns else pd.Series([], dtype=float)
        ax_l, ax_r = axes[i][0], axes[i][1]

        if not vf.empty:
            ax_l.hist(vf, bins=bins, color=COLOR_BEFORE, alpha=0.8, edgecolor="white", linewidth=0.4)
            ax_l.axvline(vf.mean(), color="#d63031", ls="--", lw=1.4, label=f"μ={vf.mean():.2f}")
            ax_l.legend(fontsize=7, loc="upper left")
        ax_l.set_xlim(0, 10); ax_l.set_xticks(np.arange(0, 11, 1))
        ax_l.set_ylabel(attr_label(cfg, col), fontsize=8)
        ax_l.tick_params(labelsize=7)
        ax_l.grid(True, alpha=0.2, axis="y")

        if not vs.empty:
            ax_r.hist(vs, bins=bins, color=COLOR_AFTER, alpha=0.8, edgecolor="white", linewidth=0.4)
            ax_r.axvline(vs.mean(), color="#d63031", ls="--", lw=1.4, label=f"μ={vs.mean():.2f}")
            ax_r.legend(fontsize=7, loc="upper left")
        else:
            ax_r.text(0.5, 0.5, "Sem dados", ha="center", va="center", transform=ax_r.transAxes,
                      fontsize=8, color="#636e72")
        ax_r.set_xlim(0, 10); ax_r.set_xticks(np.arange(0, 11, 1))
        ax_r.tick_params(labelsize=7)
        ax_r.grid(True, alpha=0.2, axis="y")

        if not vf.empty and not vs.empty:
            delta = vs.mean() - vf.mean()
            arrow, arrow_color = ("▲", "#2e7d32") if delta >= 0 else ("▼", "#c0392b")
            ax_r.text(0.98, 0.95, f"{arrow} {delta:+.2f}", ha="right", va="top",
                      transform=ax_r.transAxes, fontsize=9, fontweight="bold", color=arrow_color)

        if i == 0:
            ax_l.set_title(f"Antes (n = {len(vf):,})", fontsize=10, fontweight="bold", color=COLOR_BEFORE)
            ax_r.set_title(f"Depois (n = {len(vs):,})", fontsize=10, fontweight="bold", color="#a68b00")

    fig.suptitle("Distribuição por Atributo Estético — Antes vs. Depois da Amostragem",
                 fontsize=13, fontweight="bold", y=1.002)
    plt.tight_layout()
    save_fig(fig, os.path.join(out_dir, filename), cfg)


def build_sampling_section(cfg, base_name, strategies, out_dir):
    df_full = load_human_gt(cfg)
    if df_full is None:
        print("[sampling] APDDv2-10023.csv não encontrado, pulando.")
        return
    n_bins = cfg.get("sampling", {}).get("num_bins", 30)

    for strategy in strategies:
        exp_dir = _exp_scores_dir(cfg, base_name, strategy)
        df_scores = load_scores(exp_dir, "original")
        if df_scores is None:
            print(f"[sampling] {base_name} ({strategy}): scores não encontrados, pulando.")
            continue
        stems = set(df_scores["filename"].apply(_stem))
        df_sampled = df_full[df_full["stem"].isin(stems)]

        show_stats = (strategy == "proportional_stratified")
        _sampling_distribution_chart(df_full, df_sampled, strategy, out_dir, cfg, show_stats)
        _bin_distribution_table(df_full, df_sampled, out_dir, cfg,
                                f"sampling_bin_table_{strategy}", n_bins=n_bins)
        _attr_before_after_grid(df_full, df_sampled, out_dir, cfg,
                                f"sampling_attr_grid_{strategy}.png")
        print(f"  ✓ Amostragem ({strategy}): n_amostrado={len(df_sampled)}")


def build_strategy_comparison(cfg, out_dir):
    """8ª pergunta (só Exp1): compara as distribuições uniform_bins vs. proportional_stratified."""
    dir_u = _exp_scores_dir(cfg, "exp1_apdd", "uniform_bins")
    dir_s = _exp_scores_dir(cfg, "exp1_apdd", "proportional_stratified")
    df_u_scores = load_scores(dir_u, "original")
    df_s_scores = load_scores(dir_s, "original")
    df_full = load_human_gt(cfg)
    if df_u_scores is None or df_s_scores is None or df_full is None:
        print("[strategy_comparison] dados insuficientes, pulando.")
        return

    stems_u = set(df_u_scores["filename"].apply(_stem))
    stems_s = set(df_s_scores["filename"].apply(_stem))
    avg_u = _avg_score(df_full[df_full["stem"].isin(stems_u)]).dropna()
    avg_s = _avg_score(df_full[df_full["stem"].isin(stems_s)]).dropna()
    if avg_u.empty or avg_s.empty:
        print("[strategy_comparison] amostras vazias, pulando.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(avg_u, bins=30, range=(0, 10), density=True, histtype="step", linewidth=2.2,
            color=COLOR_UNIFORM, label=f"Uniforme (n={len(avg_u)})")
    ax.hist(avg_u, bins=30, range=(0, 10), density=True, alpha=0.15, color=COLOR_UNIFORM, hatch="///")
    ax.hist(avg_s, bins=30, range=(0, 10), density=True, histtype="step", linewidth=2.2,
            linestyle="dashed", color=COLOR_STRAT, label=f"Estratificado Proporcional (n={len(avg_s)})")
    ax.hist(avg_s, bins=30, range=(0, 10), density=True, alpha=0.15, color=COLOR_STRAT, hatch="xxx")
    ax.set_xlim(0, 10); ax.set_xticks(np.arange(0, 11, 1))
    ax.set_xlabel("Score Médio por Imagem"); ax.set_ylabel("Densidade")
    ax.set_title("Uniforme vs. Estratificado Proporcional — Comparação das Distribuições")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_fig(fig, os.path.join(out_dir, "sampling_strategy_comparison.png"), cfg)

    res = distribution_diff(avg_u, avg_s, "Uniforme", "Estratificado Proporcional")
    if res:
        render_dist_diff_table(
            [res], os.path.join(out_dir, "sampling_strategy_comparison_table.png"), cfg,
            title="Uniforme vs. Estratificado Proporcional — Diferença de Distribuição"
        )
    print("  ✓ Comparação de estratégias de amostragem (8ª pergunta)")


# ═══════════════════════════════════════════════════════════════════════════
# Perguntas 1-3 — impacto do tamanho do modelo (Human vs. Janus-1B vs. Janus-7B)
# ═══════════════════════════════════════════════════════════════════════════

def _save_q1_latex(fw, attrs, group_names, out_dir, cfg, name):
    rows, row_labels = [], []
    for attr in attrs:
        row_info = fw.get(attr, {})
        means = {g: row_info[g]["mean"] for g in group_names if g in row_info}
        best_g = max(means, key=means.get) if means else None
        row = []
        for g in group_names:
            if g not in row_info:
                row.append("—")
                continue
            m, s, l = row_info[g]["mean"], row_info[g]["std"], row_info[g]["letter"]
            cell = f"{m:.2f}$\\pm${s:.2f}$^{{{l}}}$"
            if g == best_g:
                cell = f"\\textbf{{{cell}}}"
            row.append(cell)
        rows.append(row)
        row_labels.append(attr_label(cfg, attr))

    lines = [
        r"\begin{table}[ht]", r"\centering",
        r"\caption{Média ± desvio padrão de cada atributo estético para Human, Janus-Pro-1B e "
        r"Janus-Pro-7B. Quanto maior o score, melhor. Letras sobrescritas diferentes indicam "
        r"diferença estatisticamente significativa (p < 0,05). Valores em negrito indicam o "
        r"melhor score para aquele atributo.}",
        r"\begin{tabular}{l" + "c" * len(group_names) + "}",
        r"\toprule",
        "Atributo & " + " & ".join(group_names) + r" \\",
        r"\midrule",
    ]
    for rl, row in zip(row_labels, rows):
        lines.append(f"{rl} & " + " & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(os.path.join(out_dir, f"{name}.tex.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def build_questions(cfg, base_name, strategy, out_dir, suffix=""):
    """Perguntas 1, 2 e 3 — Friedman/Wilcoxon, barras de diferença, tabela de diferença."""
    exp_dir = _exp_scores_dir(cfg, base_name, strategy)
    df_1b = load_scores(exp_dir, "Janus-Pro-1B")
    df_7b = load_scores(exp_dir, "Janus-Pro-7B")
    df_human = load_human_gt(cfg)
    if df_human is None or (df_1b is None and df_7b is None):
        print(f"[questions] {base_name} ({strategy}): dados insuficientes, pulando.")
        return
    attrs = [a for a in ALL_ATTRS if a in df_human.columns]

    groups = {"Human": df_human}
    if df_1b is not None:
        d = df_1b.copy(); d["stem"] = d["filename"].apply(_stem); groups["Janus-1B"] = d
    if df_7b is not None:
        d = df_7b.copy(); d["stem"] = d["filename"].apply(_stem); groups["Janus-7B"] = d

    # ── Pergunta 1 ───────────────────────────────────────────────────────
    if len(groups) >= 2:
        fw = friedman_wilcoxon(groups, attrs, cfg["stats"]["alpha"])
        name = f"q1_friedman_wilcoxon_table{suffix}"
        render_stat_table_png(
            fw, attrs, list(groups.keys()), os.path.join(out_dir, f"{name}.png"), cfg,
            title="Tabela 1 — Friedman + Wilcoxon (Human vs. Janus-1B vs. Janus-7B)"
        )
        _save_q1_latex(fw, attrs, list(groups.keys()), out_dir, cfg, name)

    # ── Perguntas 2 e 3 ──────────────────────────────────────────────────
    _score_diff_bars_hr(df_human, df_1b, df_7b, attrs, out_dir, cfg, suffix)
    print(f"  ✓ Perguntas 1-3 ({base_name}, {strategy})")


def _score_diff_bars_hr(df_human, df_1b, df_7b, attrs, out_dir, cfg, suffix=""):
    def align(df_ref, df_gen, attr):
        if df_ref is None or df_gen is None:
            return None, None
        if attr not in df_ref.columns or attr not in df_gen.columns:
            return None, None  # ex: "The sense of order" — agente falhou ao carregar no scoring
        a, b = df_ref.copy(), df_gen.copy()
        if "stem" not in a.columns: a["stem"] = a["filename"].apply(_stem)
        if "stem" not in b.columns: b["stem"] = b["filename"].apply(_stem)
        m = a[["stem", attr]].merge(b[["stem", attr]].rename(columns={attr: attr + "_g"}), on="stem").dropna()
        if len(m) == 0:
            return None, None
        return m[attr].values, m[attr + "_g"].values

    diffs_1b, diffs_7b, labels = [], [], []
    for attr in attrs:
        h1, g1 = align(df_human, df_1b, attr)
        h7, g7 = align(df_human, df_7b, attr)
        diffs_1b.append(float(np.mean(h1 - g1)) if h1 is not None else None)
        diffs_7b.append(float(np.mean(h7 - g7)) if h7 is not None else None)
        labels.append(attr_label(cfg, attr))

    valid = [i for i in range(len(attrs)) if diffs_1b[i] is not None or diffs_7b[i] is not None]
    if not valid:
        return

    x = np.arange(len(valid)); width = 0.35
    d1 = [diffs_1b[i] if diffs_1b[i] is not None else 0 for i in valid]
    d7 = [diffs_7b[i] if diffs_7b[i] is not None else 0 for i in valid]
    fig, ax = plt.subplots(figsize=(max(10, len(valid) * 0.9), 6))
    ax.bar(x - width/2, d1, width, label="Human − Janus-1B",
           color=COLOR_HUMAN_1B, hatch="///", edgecolor="white")
    ax.bar(x + width/2, d7, width, label="Human − Janus-7B",
           color=COLOR_HUMAN_7B, hatch="xxx", edgecolor="white")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x); ax.set_xticklabels([labels[i] for i in valid], rotation=40, ha="right")
    ax.set_ylabel("Average Score")
    ax.set_title("Diferença Média de Score (Human − Gerado) por Atributo")
    ax.legend(); ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    save_fig(fig, os.path.join(out_dir, f"q2_score_diff_bars{suffix}.png"), cfg)

    rows = [
        [labels[i],
         f"{d1[j]:+.3f}" if diffs_1b[i] is not None else "—",
         f"{d7[j]:+.3f}" if diffs_7b[i] is not None else "—"]
        for j, i in enumerate(valid)
    ]
    save_table(rows, ["Atributo", "Human − Janus-1B", "Human − Janus-7B"], out_dir,
               f"q3_score_diff_table{suffix}", cfg, title="Tabela de Diferenças de Score")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/analysis_local.yaml")
    args = parser.parse_args()
    cfg = load_cfg(args.config)

    base_dir = os.path.join(cfg["paths"]["reports"], "figures_paper")
    shared_dir = os.path.join(base_dir, "shared")
    iccc_dir = os.path.join(base_dir, "iccc")
    exp1_dir_out = os.path.join(base_dir, "exp1")
    for d in (shared_dir, iccc_dir, exp1_dir_out):
        os.makedirs(d, exist_ok=True)

    print("── EDA (compartilhado entre Paper e Exp1) ────────────")
    build_eda(cfg, shared_dir)
    build_missing_values(cfg, shared_dir)

    print("── Paper ICCC (exp0_iccc) ─────────────────────────────")
    build_sampling_section(cfg, "exp0_iccc", ["proportional_stratified"], iccc_dir)
    build_questions(cfg, "exp0_iccc", "proportional_stratified", iccc_dir, suffix="_proportional_stratified")

    print("── Exp 1 (exp1_apdd) — ambas as estratégias ───────────")
    build_sampling_section(cfg, "exp1_apdd", STRATEGIES, exp1_dir_out)
    for strategy in STRATEGIES:
        build_questions(cfg, "exp1_apdd", strategy, exp1_dir_out, suffix=f"_{strategy}")
    build_strategy_comparison(cfg, exp1_dir_out)

    print(f"\n✓ Figuras/tabelas salvas em: {base_dir}")


if __name__ == "__main__":
    main()
