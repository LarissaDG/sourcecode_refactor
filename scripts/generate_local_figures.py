# -*- coding: utf-8 -*-
"""
Regenera localmente, em arquivos PNG/.tex.txt permanentes (fora do scratchpad
temporário), as mesmas figuras e tabelas hoje embutidas como base64 em
docs/Paper_iccc.html, exp1_apdd.html, exp2_portinari.html, exp3_mnist.html,
exp4_noise.html, exp5_temporal.html e ai_measurement_science.html — pra usar
direto na dissertação sem depender do GitHub Pages carregar.

Não escreve nada em docs/*.html. Saída em <cfg.paths.reports>/figures_paper/<exp>/.

Reaproveita os mesmos helpers já usados pra gerar o site (scripts/analyze.py,
scripts/analyze_paper.py) — sem duplicar lógica estatística/visual.

Uso:
    python scripts/generate_local_figures.py --config configs/analysis_local.yaml
    python scripts/generate_local_figures.py --only exp2,exp4
"""
import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, mannwhitneyu, skew

import analyze as az
import analyze_paper as ap

ATTRS = [
    "Theme and logic", "Creativity", "Layout and composition", "Space and perspective",
    "Light and shadow", "Color", "Details and texture", "The overall", "Mood",
]
# Rótulo padronizado por critério (pedido explícito da usuária em 2026-08-16):
# Title Case em todo mundo, EXCETO "The overall", que fica exatamente assim —
# não "The Overall", não "Overall". "The sense of order" também entra aqui
# (vira "Sense of Order", sem o "The") pra cobrir os poucos lugares que usam
# ALL_ATTRS (10 atributos) em vez do ATTRS de 9 acima.
ATTR_LABEL = {
    "Theme and logic":        "Theme and Logic",
    "Creativity":              "Creativity",
    "Layout and composition":  "Layout and Composition",
    "Space and perspective":   "Space and Perspective",
    "The sense of order":      "Sense of Order",
    "Light and shadow":        "Light and Shadow",
    "Color":                   "Color",
    "Details and texture":     "Details and Texture",
    "The overall":             "The overall",
    "Mood":                    "Mood",
}
OLD_BIN_ATTRS = [  # métrica usada na amostragem real do Exp4 (antes da correção "The sense of order")
    "Theme and logic", "Creativity", "Layout and composition",
    "Space and perspective", "The sense of order", "Light and shadow",
    "Color", "Details and texture", "Mood",
]


def _copy(src, dst_dir, dst_name=None):
    if not os.path.exists(src):
        print(f"  [aviso] não encontrado, pulando: {src}")
        return
    os.makedirs(dst_dir, exist_ok=True)
    shutil.copy2(src, os.path.join(dst_dir, dst_name or os.path.basename(src)))


def _align(df_ref, df_gen, attr, key="stem"):
    if df_ref is None or df_gen is None or attr not in df_ref.columns or attr not in df_gen.columns:
        return None, None
    m = df_ref[[key, attr]].merge(
        df_gen[[key, attr]].rename(columns={attr: attr + "_g"}), on=key
    ).dropna()
    if len(m) == 0:
        return None, None
    return m[attr].values, m[attr + "_g"].values


def _diff_bars(diffs_by_label, attrs, colors, hatches, out_path, title, xlabel_map=ATTR_LABEL,
               ylim=None, yticks=None):
    labels = [xlabel_map[a] for a in attrs]
    x = np.arange(len(attrs))
    n = len(diffs_by_label)
    width = 0.8 / n
    fig, ax = plt.subplots(figsize=(max(10, len(attrs) * 0.9 + n * 0.2), 6))
    for i, (label, diffs) in enumerate(diffs_by_label.items()):
        vals = [diffs[a] if diffs.get(a) is not None else 0 for a in attrs]
        offset = (i - (n - 1) / 2) * width
        ax.bar(x + offset, vals, width, label=label, color=colors[i], hatch=hatches[i], edgecolor="white")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=40, ha="right")
    if ylim is not None:
        ax.set_ylim(*ylim)
    if yticks is not None:
        ax.set_yticks(yticks)
    ax.set_ylabel("Average Score")
    ax.set_title(title)
    ax.legend(); ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out_path}")


def _friedman_table(groups, attrs, out_dir, cfg, name, title):
    fw = az.friedman_wilcoxon(groups, attrs, cfg["stats"]["alpha"])
    az.render_stat_table_png(fw, attrs, list(groups.keys()), os.path.join(out_dir, f"{name}.png"), cfg, title=title)
    ap._save_q1_latex(fw, attrs, list(groups.keys()), out_dir, cfg, name)
    print(f"  -> {name}.png / {name}.tex.txt")
    return fw


def _diffs_from_groups(df_ref, groups, attrs, key="stem"):
    """{label: {attr: mean(ref - group)}} pareado por `key`."""
    out = {}
    for label, df in groups.items():
        out[label] = {}
        for attr in attrs:
            r, g = _align(df_ref, df, attr, key=key)
            out[label][attr] = float(np.mean(r - g)) if r is not None else None
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Paper ICCC (exp0_iccc) + Exp1 — mesma lógica de scripts/analyze_paper.py::main(),
# só que escrita como função pra entrar no mesmo comando único.
# ═══════════════════════════════════════════════════════════════════════════

def gen_paper_and_exp1(cfg, base_dir):
    print("\n=== Paper ICCC + Exp1 — APDDv2 ===")
    shared_dir = os.path.join(base_dir, "shared")
    iccc_dir = os.path.join(base_dir, "iccc")
    exp1_dir_out = os.path.join(base_dir, "exp1")
    for d in (shared_dir, iccc_dir, exp1_dir_out):
        os.makedirs(d, exist_ok=True)

    ap.build_eda(cfg, shared_dir)
    ap.build_missing_values(cfg, shared_dir)
    print("  -> EDA + missing values (shared/)")

    # outputs/exp0_iccc_*/exp1_apdd_* no cluster ainda foram amostrados com a
    # métrica ANTIGA de binning (excluía "The overall", não "The sense of
    # order" — mesmo bug já contornado no Exp4). As seções de amostragem
    # (histograma antes/depois, grid por atributo, comparação uniforme x
    # estratificado) recalculam o score médio pra binning — usar a métrica
    # nova aqui re-binaria os MESMOS 469/502/500 pontos numa grade que a
    # amostragem real nunca respeitou, estourando a contagem por bin (~19 em
    # vez de ~16). Enquanto ela não reroda no cluster com o bug corrigido,
    # usar a métrica antiga aqui é o que reflete fielmente a amostragem real.
    old_bin_attrs = ap.BIN_ATTRS
    ap.BIN_ATTRS = OLD_BIN_ATTRS
    try:
        ap.build_sampling_section(cfg, "exp0_iccc", ap.STRATEGIES, iccc_dir)
        ap.build_sampling_section(cfg, "exp1_apdd", ap.STRATEGIES, exp1_dir_out)
        ap.build_strategy_comparison(cfg, exp1_dir_out)
    finally:
        ap.BIN_ATTRS = old_bin_attrs

    for strategy in ap.STRATEGIES:
        ap.build_questions(cfg, "exp0_iccc", strategy, iccc_dir, suffix=f"_{strategy}",
                            q2_ylim=ap.PAPER_Q2_YLIM, q2_yticks=ap.PAPER_Q2_YTICKS)
    print("  -> Paper ICCC (iccc/), ambas as estratégias (Q2 eixo Y fixo, igual Portinari)")

    for strategy in ap.STRATEGIES:
        ap.build_questions(cfg, "exp1_apdd", strategy, exp1_dir_out, suffix=f"_{strategy}")
    print("  -> Exp1 (exp1/), ambas as estratégias + comparação")


# ═══════════════════════════════════════════════════════════════════════════
# Exp2 — Portinari
# ═══════════════════════════════════════════════════════════════════════════

def gen_exp2(cfg, base_dir):
    print("\n=== Exp2 — Portinari ===")
    out_dir = os.path.join(base_dir, "exp2")
    os.makedirs(out_dir, exist_ok=True)
    OUT_ROOT = cfg["paths"]["outputs"]

    def load_exp(exp_dir_name):
        exp_dir = os.path.join(OUT_ROOT, exp_dir_name)
        human = az.load_scores(exp_dir, "original")
        d1b = az.load_scores(exp_dir, "Janus-Pro-1B")
        d7b = az.load_scores(exp_dir, "Janus-Pro-7B")
        for d in (human, d1b, d7b):
            if d is not None and "stem" not in d.columns:
                d["stem"] = d["filename"].apply(az._stem)
        return human, d1b, d7b

    human_2a, d1b_2a, d7b_2a = load_exp("exp2a_portinari")
    human_2b, d1b_2b, d7b_2b = load_exp("exp2b_portinari_human")
    exp1_dir = ap._exp_scores_dir(cfg, "exp1_apdd", "uniform_bins")
    human_1u = az.load_human_gt(cfg)
    d1b_1u = az.load_scores(exp1_dir, "Janus-Pro-1B")
    d7b_1u = az.load_scores(exp1_dir, "Janus-Pro-7B")
    for d in (d1b_1u, d7b_1u):
        if d is not None and "stem" not in d.columns:
            d["stem"] = d["filename"].apply(az._stem)

    diffs_7b_by_exp = {}
    for key, human, d1b, d7b, label in [
        ("exp2a", human_2a, d1b_2a, d7b_2a, "Descrições Sintéticas"),
        ("exp2b", human_2b, d1b_2b, d7b_2b, "Descrições Humanas"),
    ]:
        attrs_present = [a for a in ATTRS if a in human.columns]
        groups = {"Human": human}
        if d1b is not None: groups["Janus-1B"] = d1b
        if d7b is not None: groups["Janus-7B"] = d7b
        _friedman_table(groups, attrs_present, out_dir, cfg, f"q1_friedman_{key}",
                         f"Tabela 1 — Friedman + Wilcoxon ({label})")

        diffs = _diffs_from_groups(human, {"Janus-1B": d1b, "Janus-7B": d7b}, attrs_present)
        diffs_7b_by_exp[key] = diffs["Janus-7B"]
        ap.save_table(
            [[ATTR_LABEL[a], f"{diffs['Janus-1B'][a]:+.3f}" if diffs["Janus-1B"][a] is not None else "—",
              f"{diffs['Janus-7B'][a]:+.3f}" if diffs["Janus-7B"][a] is not None else "—"] for a in attrs_present],
            ["Atributo", "Human − Janus-1B", "Human − Janus-7B"], out_dir, f"q3_table_{key}", cfg,
            title=f"Tabela de Diferenças — {label}",
        )
        print(f"  -> q3_table_{key}.png / .tex.txt")

    # Q2 — barras de diferença, eixo Y compartilhado entre exp2a/exp2b
    all_vals = [v for d in diffs_7b_by_exp.values() for v in d.values() if v is not None]
    diffs_1b_by_exp = {}
    for key, human, d1b in [("exp2a", human_2a, d1b_2a), ("exp2b", human_2b, d1b_2b)]:
        d1 = {}
        for a in ATTRS:
            r, g = _align(human, d1b, a)
            d1[a] = float(np.mean(r - g)) if r is not None else None
        diffs_1b_by_exp[key] = d1
        all_vals += [v for v in d1.values() if v is not None]

    y_min = np.floor(min(0, min(all_vals)) / 0.1) * 0.1 - 0.02
    y_max = np.ceil(max(all_vals) / 0.1) * 0.1 + 0.05
    y_ticks = np.arange(np.floor(y_min / 0.1) * 0.1, y_max + 0.001, 0.1)
    for key, label in [("exp2a", "Descrições Sintéticas"), ("exp2b", "Descrições Humanas")]:
        _diff_bars(
            {"Human − Janus-1B": diffs_1b_by_exp[key], "Human − Janus-7B": diffs_7b_by_exp[key]},
            ATTRS, [ap.COLOR_HUMAN_1B, ap.COLOR_HUMAN_7B], ["///", "xxx"],
            os.path.join(out_dir, f"q2_bars_{key}.png"),
            f"Diferença Média de Score (Human − Gerado) por Atributo — {label}",
            ylim=(y_min, y_max), yticks=y_ticks,
        )

    # Q4 — dist diff Exp2a x Exp2b (1B e 7B)
    attrs_present = [a for a in ATTRS if a in d1b_2a.columns and a in d1b_2b.columns]
    dfs_q4 = {"Exp2a-1B": d1b_2a, "Exp2b-1B": d1b_2b, "Exp2a-7B": d7b_2a, "Exp2b-7B": d7b_2b}
    ap.distribution_diff_table_per_attr(
        dfs_q4,
        [("Janus-Pro-1B: Descrições Sintéticas × Humanas", "Exp2a-1B", "Exp2b-1B"),
         ("Janus-Pro-7B: Descrições Sintéticas × Humanas", "Exp2a-7B", "Exp2b-7B")],
        attrs_present, out_dir, cfg, "q4_dist_diff_exp2a_exp2b",
        title="Diferença de Distribuição — Descrições Sintéticas vs. Descrições Humanas",
    )
    print("  -> q4_dist_diff_exp2a_exp2b.png / .tex.txt")

    # Q5 — deviation line: Exp2a x Exp2b
    ap.deviation_line_graph(
        {"Descrições Sintéticas": diffs_7b_by_exp["exp2a"], "Descrições Humanas": diffs_7b_by_exp["exp2b"]},
        attrs_present, out_dir, cfg, "q5_deviation_line.png",
        title="Gráfico de Desvio por Atributo — Descrições Sintéticas vs. Humanas (Human − Janus-7B)",
    )
    print("  -> q5_deviation_line.png")

    # Q6/Q7 — Exp1(uniforme) x Exp2a x Exp2b
    attrs_67 = [a for a in ATTRS if a in d7b_1u.columns and a in d7b_2a.columns and a in d7b_2b.columns]
    dfs_q6 = {"Exp1-Uniforme-7B": d7b_1u, "Exp2a-7B": d7b_2a, "Exp2b-7B": d7b_2b}
    ap.distribution_diff_table_per_attr(
        dfs_q6,
        [("APDDv2(Exp1-Unif.) × Descrições Sintéticas", "Exp1-Uniforme-7B", "Exp2a-7B"),
         ("APDDv2(Exp1-Unif.) × Descrições Humanas", "Exp1-Uniforme-7B", "Exp2b-7B"),
         ("Descrições Sintéticas × Humanas", "Exp2a-7B", "Exp2b-7B")],
        attrs_67, out_dir, cfg, "q6_dist_diff_exp1_exp2",
        title="Diferença de Distribuição — APDDv2 (Exp1 Uniforme) vs. Portinari (Descrições Sintéticas/Humanas), Janus-7B",
    )
    print("  -> q6_dist_diff_exp1_exp2.png / .tex.txt")

    diffs_7b_1u = {}
    for a in attrs_67:
        r, g = _align(human_1u, d7b_1u, a)
        diffs_7b_1u[a] = float(np.mean(r - g)) if r is not None else None
    ap.deviation_line_graph(
        {"Exp1 Uniforme (APDDv2)": diffs_7b_1u, "Portinari (Descrições Sintéticas)": diffs_7b_by_exp["exp2a"],
         "Portinari (Descrições Humanas)": diffs_7b_by_exp["exp2b"]},
        attrs_67, out_dir, cfg, "q7_deviation_line.png",
        title="Gráfico de Desvio por Atributo — Exp1(Uniforme) vs. Portinari (Sintéticas/Humanas) (Human − Janus-7B)",
    )
    print("  -> q7_deviation_line.png")

    # Amostra: distribuição só-da-amostra + tamanho de legendas (amostra + acervo completo)
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    acervo = pd.read_csv(os.path.join(ROOT, "data", "portinari", "acervoPortinari.csv"))
    n_total = len(acervo)
    avg = human_2a[ATTRS].mean(axis=1).dropna()
    n = len(avg)
    mu, med, sk, kt = avg.mean(), avg.median(), skew(avg), kurtosis(avg)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.hist(avg, bins=25, range=(0, 10), density=False, color="#00b894", alpha=0.8, edgecolor="white")
    ax.axvline(mu, color="#2d3436", ls="--", lw=1.6, label=f"Média = {mu:.2f}")
    ax.axvline(med, color="#6c5ce7", ls=":", lw=1.6, label=f"Mediana = {med:.2f}")
    ax.set_xlim(0, 10); ax.set_xticks(np.arange(0, 11, 1))
    ax.set_xlabel("Score Médio por Imagem"); ax.set_ylabel("Contagem")
    ax.set_title(f"Distribuição do Score Médio — Amostra Portinari (n={n}, {n/n_total*100:.1f}% do acervo)\n"
                 f"Assimetria = {sk:.3f}   Curtose = {kt:.3f}", fontsize=11)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.25, axis="y")
    plt.tight_layout()
    p = os.path.join(out_dir, "sample_dist.png")
    fig.savefig(p, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {p}")

    def caption_stats(pipeline_json_path):
        with open(pipeline_json_path, encoding="utf-8") as f:
            data = json.load(f)
        caps = [d["caption"] for d in data if d.get("caption")]
        n_chars = [len(c) for c in caps]
        n_words = [len(c.split()) for c in caps]
        return dict(n=len(caps), chars=n_chars, words=n_words)

    cap_2a = caption_stats(os.path.join(OUT_ROOT, "exp2a_portinari", "pipeline_data.json"))
    cap_2b = caption_stats(os.path.join(OUT_ROOT, "exp2b_portinari_human", "pipeline_data.json"))
    desc = acervo["Descrição"].dropna(); desc = desc[desc.str.strip() != ""]
    full_words = desc.str.split().apply(len); full_chars = desc.str.len()

    def caption_table(name, title, words, chars, n):
        rows = [
            ["Mínimo", str(min(words)), str(min(chars))],
            ["Máximo", str(max(words)), str(max(chars))],
            ["Média", f"{np.mean(words):.1f}", f"{np.mean(chars):.1f}"],
            ["Desvio padrão", f"{np.std(words):.1f}", f"{np.std(chars):.1f}"],
        ]
        ap.save_table(rows, ["Métrica", "Palavras", "Caracteres"], out_dir, name, cfg, title=f"{title} (n={n})")
        print(f"  -> {name}.png / .tex.txt")

    caption_table("samp5_captions_exp2a", "Tamanho das Legendas — Descrições Sintéticas (Janus-7B)", cap_2a["words"], cap_2a["chars"], cap_2a["n"])
    caption_table("samp5_captions_exp2b", "Tamanho das Legendas — Descrições Humanas (EN)", cap_2b["words"], cap_2b["chars"], cap_2b["n"])
    caption_table("samp5b_captions_full_archive", "Tamanho das Legendas — Acervo Completo (PT, antes da amostragem)",
                  full_words.tolist(), full_chars.tolist(), len(desc))

    # cópias das amostras visuais já geradas pelo pipeline
    for key, folder in [("exp2a", "exp2a_portinari"), ("exp2b", "exp2b_portinari_human")]:
        _copy(os.path.join(OUT_ROOT, folder, "samples", "sample_panel.png"), os.path.join(out_dir, "samples"),
              f"sample_panel_{key}.png")


# ═══════════════════════════════════════════════════════════════════════════
# Exp3 — MNIST
# ═══════════════════════════════════════════════════════════════════════════

def gen_exp3(cfg, base_dir):
    print("\n=== Exp3 — MNIST ===")
    out_dir = os.path.join(base_dir, "exp3")
    os.makedirs(out_dir, exist_ok=True)
    OUT_ROOT = cfg["paths"]["outputs"]
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    human_apdd = az.load_human_gt(cfg)
    human_portinari = az.load_scores(os.path.join(OUT_ROOT, "exp2a_portinari"), "original")
    mnist = az.load_scores(os.path.join(OUT_ROOT, "exp3_mnist"), "original")

    # contagem por dígito
    with open(os.path.join(OUT_ROOT, "exp3_mnist", "pipeline_data.json"), encoding="utf-8") as f:
        pipeline_data = json.load(f)
    from datasets.mnist import MNISTDataset
    raw = MNISTDataset(root=os.path.join(ROOT, "data", "mnist", "_raw"), train=True)
    idx_to_digit = dict(zip(raw.df["index"], raw.df["digit"]))
    digits = []
    for d in pipeline_data:
        fn = d["filename"]
        idx = int(fn.replace("mnist_", "").replace(".png", ""))
        digits.append(idx_to_digit.get(idx))
    digit_counts = pd.Series(digits).value_counts().sort_index()
    ap.save_table(
        [[str(d), str(c)] for d, c in digit_counts.items()], ["Dígito", "Contagem"], out_dir,
        "samp2b_digit_table", cfg, title=f"Distribuição por Dígito (n={int(digit_counts.sum())})",
    )
    print("  -> samp2b_digit_table.png / .tex.txt")

    # distribuição só-da-amostra (MNIST)
    avg = mnist[ATTRS].mean(axis=1).dropna()
    n = len(avg)
    mu, med, sk, kt = avg.mean(), avg.median(), skew(avg), kurtosis(avg)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.hist(avg, bins=25, range=(0, 10), density=False, color="#a29bfe", alpha=0.85, edgecolor="white")
    ax.axvline(mu, color="#2d3436", ls="--", lw=1.6, label=f"Média = {mu:.2f}")
    ax.axvline(med, color="#6c5ce7", ls=":", lw=1.6, label=f"Mediana = {med:.2f}")
    ax.set_xlim(0, 10); ax.set_xticks(np.arange(0, 11, 1))
    ax.set_xlabel("Score Médio por Imagem"); ax.set_ylabel("Contagem")
    ax.set_title(f"Distribuição do Score Médio — Amostra MNIST (n={n})\n"
                 f"Assimetria = {sk:.3f}   Curtose = {kt:.3f}", fontsize=11)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.25, axis="y")
    plt.tight_layout()
    p = os.path.join(out_dir, "sample_dist.png")
    fig.savefig(p, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {p}")

    # Q1a — boxplot grid
    dfs_3way = {"APDDv2": human_apdd, "Portinari": human_portinari, "MNIST": mnist}
    ap.boxplot_grid_by_attribute(
        dfs_3way, ATTRS, out_dir, cfg, "q1a_boxplot_grid.png",
        title="Distribuição de Notas por Atributo — APDDv2 × Portinari × MNIST",
    )
    print("  -> q1a_boxplot_grid.png")

    # Q1b — dist diff MNIST x APDDv2 / MNIST x Portinari
    ap.distribution_diff_table_per_attr(
        dfs_3way, [("MNIST × APDDv2", "MNIST", "APDDv2"), ("MNIST × Portinari", "MNIST", "Portinari")],
        ATTRS, out_dir, cfg, "q1b_dist_diff",
        title="Diferença de Distribuição — MNIST vs. APDDv2 e MNIST vs. Portinari",
    )
    print("  -> q1b_dist_diff.png / .tex.txt")

    # Q1c — Mann-Whitney Human x MNIST
    rows = []
    for attr in ATTRS:
        h = human_apdd[attr].dropna(); m = mnist[attr].dropna()
        u, p = mannwhitneyu(h, m)
        rows.append([ATTR_LABEL[attr], f"{h.mean():.2f} ± {h.std():.2f}", f"{m.mean():.2f} ± {m.std():.2f}",
                     f"{p:.2e}" if p >= 0.0001 else "<0.0001"])
    ap.save_table(rows, ["Atributo", "Human (APDDv2)", "MNIST", "p (Mann-Whitney)"], out_dir,
                  "q1c_stats_table", cfg, title="Human vs. MNIST — Mann-Whitney U (não pareado)")
    print("  -> q1c_stats_table.png / .tex.txt")

    # Q2 — linhas (médias) APDDv2 x Portinari x MNIST
    means = {"APDDv2 (Human)": human_apdd[ATTRS].mean(), "Portinari (Human)": human_portinari[ATTRS].mean(),
             "MNIST": mnist[ATTRS].mean()}
    labels = [ATTR_LABEL[a] for a in ATTRS]
    x = np.arange(len(ATTRS))
    colors = ["#33A650", "#e17055", "#a29bfe"]; markers = ["o", "s", "^"]; linestyles = ["-", "--", "-."]
    fig, ax = plt.subplots(figsize=(max(9, len(ATTRS) * 0.9), 5.5))
    for i, (label, vals) in enumerate(means.items()):
        ax.plot(x, [vals[a] for a in ATTRS], marker=markers[i], linestyle=linestyles[i], color=colors[i],
                linewidth=2, markersize=7, label=label)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=40, ha="right")
    ax.set_ylim(0, 10); ax.set_yticks(np.arange(0, 11, 1))
    ax.set_ylabel("Score Médio (Average Score)")
    ax.set_title("Nota Média por Atributo — APDDv2 vs. Portinari vs. MNIST", fontsize=12, fontweight="bold")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p = os.path.join(out_dir, "q2_lines.png")
    fig.savefig(p, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {p}")

    # Q3 — tabela de diferença de médias (não pareada) APDDv2 - MNIST
    means_apdd = human_apdd[ATTRS].mean(); means_mnist = mnist[ATTRS].mean()
    diffs = {a: float(means_apdd[a] - means_mnist[a]) for a in ATTRS}
    ap.save_table([[ATTR_LABEL[a], f"{diffs[a]:+.3f}"] for a in ATTRS], ["Atributo", "APDDv2 − MNIST"],
                  out_dir, "q3_table", cfg, title="Diferença de Médias (não pareada) — APDDv2 − MNIST")
    print("  -> q3_table.png / .tex.txt")

    _copy(os.path.join(OUT_ROOT, "exp3_mnist", "samples", "sample_panel.png"), os.path.join(out_dir, "samples"))


# ═══════════════════════════════════════════════════════════════════════════
# Exp4 — Ruído
# ═══════════════════════════════════════════════════════════════════════════

def gen_exp4(cfg, base_dir):
    print("\n=== Exp4 — Ruído ===")
    out_dir = os.path.join(base_dir, "exp4")
    os.makedirs(out_dir, exist_ok=True)
    OUT_ROOT = cfg["paths"]["outputs"]

    old_bin_attrs = ap.BIN_ATTRS
    ap.BIN_ATTRS = OLD_BIN_ATTRS
    try:
        noise_df = az.load_scores(os.path.join(OUT_ROOT, "exp4_noise"), "original")
        noise_df["stem"] = noise_df["filename"].apply(az._stem)
        human_apdd = az.load_human_gt(cfg)
        stems = set(noise_df["stem"].unique())
        df_sampled = human_apdd[human_apdd["stem"].isin(stems)]

        ap._sampling_distribution_chart(human_apdd, df_sampled, "uniform_bins", out_dir, cfg, False)
        ap._attr_before_after_grid(human_apdd, df_sampled, out_dir, cfg, "samp3_attr_grid.png")
        print("  -> sampling_dist_uniform_bins.png / samp3_attr_grid.png")
    finally:
        ap.BIN_ATTRS = old_bin_attrs

    NOISE_LEVEL_REF = 100
    groups = {"Human": human_apdd}
    noise_dfs_100 = {}
    for nt in ["blur", "gaussian", "shapes"]:
        d = noise_df[(noise_df["noise_type"] == nt) & (noise_df["noise_level"] == NOISE_LEVEL_REF)].copy()
        label = nt.capitalize()
        groups[label] = d; noise_dfs_100[label] = d

    attrs_present = [a for a in ATTRS if a in human_apdd.columns]
    _friedman_table(groups, attrs_present, out_dir, cfg, "q1_friedman",
                     f"Tabela 1 — Friedman + Wilcoxon (Human vs. Blur/Gaussian/Shapes, nível {NOISE_LEVEL_REF})")

    dfs_q1b = {"Human": human_apdd, **noise_dfs_100}
    ap.distribution_diff_table_per_attr(
        dfs_q1b, [("Human × Blur", "Human", "Blur"), ("Human × Gaussian", "Human", "Gaussian"),
                  ("Human × Shapes", "Human", "Shapes")],
        attrs_present, out_dir, cfg, "q1b_dist_diff",
        title=f"Diferença de Distribuição — Human vs. Blur/Gaussian/Shapes (nível {NOISE_LEVEL_REF})",
    )
    print("  -> q1b_dist_diff.png / .tex.txt")

    diffs = _diffs_from_groups(human_apdd, noise_dfs_100, attrs_present)
    _diff_bars(
        {f"Human − {k}": v for k, v in diffs.items()}, attrs_present,
        [ap.COLOR_HUMAN_1B, ap.COLOR_HUMAN_7B, "#f39c12"], ["///", "xxx", "..."],
        os.path.join(out_dir, "q2_bars.png"),
        f"Diferença Média de Score (Human − Ruído) por Atributo — nível {NOISE_LEVEL_REF}",
    )
    rows = [[ATTR_LABEL[a]] + [f"{diffs[k][a]:+.3f}" if diffs[k][a] is not None else "—"
                                   for k in ["Blur", "Gaussian", "Shapes"]] for a in attrs_present]
    ap.save_table(rows, ["Atributo", "Human − Blur", "Human − Gaussian", "Human − Shapes"], out_dir,
                  "q3_table", cfg, title=f"Tabela de Diferenças — nível {NOISE_LEVEL_REF}")
    print("  -> q3_table.png / .tex.txt")

    # bônus — curva de degradação
    levels = sorted(noise_df["noise_level"].unique())
    colors_curve = {"blur": ap.COLOR_HUMAN_1B, "gaussian": ap.COLOR_HUMAN_7B, "shapes": "#f39c12"}
    markers_curve = {"blur": "o", "gaussian": "s", "shapes": "^"}
    noise_label_pt = {"gaussian": "Ruído Gaussiano", "blur": "Desfoque (Blur)", "shapes": "Formas (Shapes)"}

    def _degradation_curve(ax, attr, legend=False):
        human_mean = human_apdd[attr].mean()
        ax.axhline(human_mean, color="#2d3436", linestyle="--", linewidth=1.5,
                   label=f"Human (baseline) = {human_mean:.2f}" if legend else None)
        for nt in ["blur", "gaussian", "shapes"]:
            by_level = [noise_df[(noise_df["noise_type"] == nt) & (noise_df["noise_level"] == lv)][attr]
                        for lv in levels]
            means = [s.mean() for s in by_level]
            stds = [s.std() for s in by_level]
            ax.errorbar(levels, means, yerr=stds, marker=markers_curve[nt], color=colors_curve[nt],
                       linewidth=2, markersize=6, capsize=3, elinewidth=1, alpha=0.9,
                       label=noise_label_pt[nt] if legend else None)
        ax.set_ylim(0, 10); ax.set_xticks(levels)

    avg_attr = "The overall"
    fig, ax = plt.subplots(figsize=(9, 5.5))
    _degradation_curve(ax, avg_attr, legend=True)
    ax.set_yticks(np.arange(0, 11, 1))
    ax.set_xlabel("Nível de Ruído"); ax.set_ylabel(f"Score Médio ({ATTR_LABEL[avg_attr]})")
    ax.set_title("Curva de Degradação — Score vs. Nível de Ruído", fontsize=12, fontweight="bold")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p = os.path.join(out_dir, "degradation_curve.png")
    fig.savefig(p, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {p}")

    fig, axes = plt.subplots(3, 3, figsize=(4.2 * 3, 3.6 * 3))
    axes = np.array(axes).reshape(-1)
    for i, attr in enumerate(attrs_present):
        ax = axes[i]
        _degradation_curve(ax, attr, legend=(i == 0))
        ax.set_title(ATTR_LABEL[attr], fontsize=10, fontweight="bold")
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.25)
        if i == 0:
            ax.legend(fontsize=7)
    fig.supxlabel("Nível de Ruído", fontsize=10)
    fig.supylabel("Score Médio", fontsize=10)
    fig.suptitle("Curva de Degradação — Score vs. Nível de Ruído, por Atributo",
                 fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    p = os.path.join(out_dir, "degradation_curve_grid.png")
    fig.savefig(p, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {p}")

    for i in [1, 2, 3]:
        _copy(os.path.join(OUT_ROOT, "exp4_noise", "samples", f"noise_grid_0{i}.png"), os.path.join(out_dir, "samples"))


# ═══════════════════════════════════════════════════════════════════════════
# Exp5 — Temporal
# ═══════════════════════════════════════════════════════════════════════════

def gen_exp5(cfg, base_dir):
    print("\n=== Exp5 — Temporal ===")
    out_dir = os.path.join(base_dir, "exp5")
    os.makedirs(out_dir, exist_ok=True)
    OUT_ROOT = cfg["paths"]["outputs"]

    exp5a = az.load_scores(os.path.join(OUT_ROOT, "exp5a_temporal"), "original")
    exp5b = az.load_scores(os.path.join(OUT_ROOT, "exp5b_temporal_error"), "original")

    # Q1a — spaghetti plot
    np.random.seed(7)
    sample_videos = np.random.choice(exp5a["video_id"].unique(), size=15, replace=False)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for vid in sample_videos:
        sub = exp5a[exp5a["video_id"] == vid].sort_values("frame_idx")
        ax.plot(sub["frame_idx"], sub["The overall"], alpha=0.6, linewidth=1.3)
    ax.set_xlabel("Frame"); ax.set_ylabel("Score (The overall)")
    ax.set_ylim(0, 10); ax.set_yticks(np.arange(0, 11, 1))
    ax.set_title("Consistência Temporal — Trajetória de Score por Frame (15 vídeos, atributo 'The overall')",
                 fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p = os.path.join(out_dir, "q1a_trajectories.png")
    fig.savefig(p, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {p}")

    # Q1b — grade de desvio-padrão por atributo
    std_by_video = exp5a.groupby("video_id")[ATTRS].std()
    fig, axes = plt.subplots(3, 3, figsize=(4.2 * 3, 3.2 * 3))
    axes = np.array(axes).reshape(-1)
    for i, attr in enumerate(ATTRS):
        ax = axes[i]
        vals = std_by_video[attr].dropna()
        ax.hist(vals, bins=25, color="#0984e3", alpha=0.8, edgecolor="white")
        ax.axvline(vals.mean(), color="#d63031", linestyle="--", linewidth=1.4, label=f"média={vals.mean():.2f}")
        ax.set_title(ATTR_LABEL[attr], fontsize=10, fontweight="bold")
        ax.set_xlabel("Desvio-padrão intra-vídeo"); ax.tick_params(labelsize=7)
        ax.legend(fontsize=7); ax.grid(True, alpha=0.25, axis="y")
    fig.suptitle("Consistência Temporal por Atributo — Desvio-padrão de Score dentro de cada Vídeo",
                 fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    p = os.path.join(out_dir, "q1b_std_grid.png")
    fig.savefig(p, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {p}")

    # Q2 — Friedman/Wilcoxon Human(0%) x Blur/Gaussian/Shapes(100%)
    baseline = exp5b[exp5b["degradation_pct"] == 0].drop_duplicates("video_id").copy()
    baseline["stem"] = baseline["video_id"]
    groups = {"Human": baseline}
    final_by_type = {}
    for nt in ["blur", "gaussian", "shapes"]:
        d = exp5b[(exp5b["noise_type"] == nt) & (exp5b["degradation_pct"] == 100)].copy()
        d["stem"] = d["video_id"]
        label = nt.capitalize()
        groups[label] = d; final_by_type[label] = d

    _friedman_table(groups, ATTRS, out_dir, cfg, "q2_friedman",
                     "Tabela 2 — Friedman + Wilcoxon (Frame 0% vs. Blur/Gaussian/Shapes a 100%)")

    dfs_q2b = {"Human": baseline, **final_by_type}
    ap.distribution_diff_table_per_attr(
        dfs_q2b, [("Human × Blur", "Human", "Blur"), ("Human × Gaussian", "Human", "Gaussian"),
                  ("Human × Shapes", "Human", "Shapes")],
        ATTRS, out_dir, cfg, "q2b_dist_diff", title="Diferença de Distribuição — Frame 0% vs. Frame 100% de degradação",
    )
    print("  -> q2b_dist_diff.png / .tex.txt")

    # Q3/Q4 — diferença pareada por vídeo
    diffs = _diffs_from_groups(baseline, final_by_type, ATTRS)
    _diff_bars(
        {f"Frame 0% − {k} 100%": v for k, v in diffs.items()}, ATTRS,
        [ap.COLOR_HUMAN_1B, ap.COLOR_HUMAN_7B, "#f39c12"], ["///", "xxx", "..."],
        os.path.join(out_dir, "q3_bars.png"),
        "Diferença Média de Score (Frame 0% − Frame 100% de degradação) por Atributo",
    )
    rows = [[ATTR_LABEL[a]] + [f"{diffs[k][a]:+.3f}" if diffs[k][a] is not None else "—"
                                   for k in ["Blur", "Gaussian", "Shapes"]] for a in ATTRS]
    ap.save_table(rows, ["Atributo", "Frame0 − Blur", "Frame0 − Gaussian", "Frame0 − Shapes"], out_dir,
                  "q4_table", cfg, title="Tabela de Diferenças — Frame 0% vs. 100% de degradação")
    print("  -> q4_table.png / .tex.txt")

    # Q5 — ponto de detecção estatística (usa "blur" como representativo)
    rep = exp5b[exp5b["noise_type"] == "blur"].copy()
    pct_values = sorted(rep["degradation_pct"].unique())
    baseline_by_video = rep[rep["degradation_pct"] == 0].set_index("video_id")
    detection = {a: None for a in ATTRS}
    pvals_by_attr = {a: [] for a in ATTRS}
    for attr in ATTRS:
        base_vals = baseline_by_video[attr]
        for pct in pct_values:
            if pct == 0:
                pvals_by_attr[attr].append((pct, 1.0)); continue
            cur = rep[rep["degradation_pct"] == pct].set_index("video_id")[attr]
            common = base_vals.index.intersection(cur.index)
            b_vals = base_vals.loc[common].dropna(); c_vals = cur.loc[common].dropna()
            common2 = b_vals.index.intersection(c_vals.index)
            if len(common2) < 8:
                pvals_by_attr[attr].append((pct, np.nan)); continue
            try:
                _, p = mannwhitneyu(b_vals.loc[common2], c_vals.loc[common2])
            except ValueError:
                p = np.nan
            pvals_by_attr[attr].append((pct, p))
            if detection[attr] is None and p < cfg["stats"]["alpha"]:
                detection[attr] = pct

    fig, ax = plt.subplots(figsize=(9, 5.5))
    pcts, ps = zip(*pvals_by_attr["The overall"])
    ax.plot(pcts, ps, marker="o", color="#d63031", linewidth=1.8, markersize=5)
    ax.axhline(cfg["stats"]["alpha"], color="black", linestyle="--", linewidth=1.2, label=f"α = {cfg['stats']['alpha']}")
    if detection["The overall"] is not None:
        ax.axvline(detection["The overall"], color="#0984e3", linestyle=":", linewidth=1.5,
                   label=f"1º ponto significativo: {detection['The overall']:.1f}%")
    ax.set_yscale("log")
    ax.set_xlabel("Nível de degradação (%)"); ax.set_ylabel("p-valor (Mann-Whitney vs. frame 0%, escala log)")
    ax.set_title("Ponto de Detecção Estatística — Atributo 'The overall'", fontsize=12, fontweight="bold")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p = os.path.join(out_dir, "q5_detection_overall.png")
    fig.savefig(p, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {p}")

    fig, axes = plt.subplots(3, 3, figsize=(4.2 * 3, 3.2 * 3))
    axes = np.array(axes).reshape(-1)
    for i, attr in enumerate(ATTRS):
        ax = axes[i]
        pcts, ps = zip(*pvals_by_attr[attr])
        ax.plot(pcts, ps, marker="o", color="#d63031", linewidth=1.4, markersize=3.5)
        ax.axhline(cfg["stats"]["alpha"], color="black", linestyle="--", linewidth=1)
        if detection[attr] is not None:
            ax.axvline(detection[attr], color="#0984e3", linestyle=":", linewidth=1.3)
        ax.set_yscale("log")
        ax.set_title(f"{ATTR_LABEL[attr]} ({detection[attr]:.0f}%)" if detection[attr] is not None
                     else f"{ATTR_LABEL[attr]} (n.s.)", fontsize=9, fontweight="bold")
        ax.tick_params(labelsize=7); ax.grid(True, alpha=0.25)
    fig.suptitle("Ponto de Detecção Estatística por Atributo (p-valor vs. % de degradação)",
                 fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    p = os.path.join(out_dir, "q5_detection_grid.png")
    fig.savefig(p, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {p}")

    rows = [[ATTR_LABEL[a], f"{detection[a]:.1f}%" if detection[a] is not None else "não detectado"] for a in ATTRS]
    ap.save_table(rows, ["Atributo", "1º ponto significativo (p<0.05)"], out_dir, "q5_table", cfg,
                  title="Ponto de Detecção Estatística por Atributo")
    print("  -> q5_table.png / .tex.txt")

    # Q6 — RMSE por vídeo: consistência natural (original vs. média do próprio
    # vídeo — "consistência perfeita" seria uma linha horizontal) vs.
    # sensibilidade ao ruído (original vs. degradação progressiva), separado
    # por tipo de ruído e por atributo. Ideia da usuária, 2026-08-16.
    colors_nt = {"blur": ap.COLOR_HUMAN_1B, "gaussian": ap.COLOR_HUMAN_7B, "shapes": "#f39c12"}
    markers_nt = {"blur": "o", "gaussian": "s", "shapes": "^"}
    noise_label_pt = {"gaussian": "Ruído Gaussiano", "blur": "Desfoque (Blur)", "shapes": "Formas (Shapes)"}

    rmse_rows = []  # {video_id, noise_type, attr, rmse_self, rmse_noise}
    for vid, g in exp5a.groupby("video_id"):
        g = g.sort_values("frame_idx")
        self_rmse = {
            attr: float(np.sqrt(((g[attr] - g[attr].mean()) ** 2).mean()))
            for attr in ATTRS if g[attr].notna().sum() >= 2
        }
        for nt in ["blur", "gaussian", "shapes"]:
            noisy = exp5b[(exp5b["video_id"] == vid) & (exp5b["noise_type"] == nt)]
            merged = g[["frame_idx", *ATTRS]].merge(
                noisy[["frame_idx", *ATTRS]], on="frame_idx", suffixes=("_clean", "_noisy")
            )
            for attr in ATTRS:
                if attr not in self_rmse:
                    continue
                pair = merged[[attr + "_clean", attr + "_noisy"]].dropna()
                if len(pair) < 2:
                    continue
                rmse_noise = float(np.sqrt(((pair[attr + "_clean"] - pair[attr + "_noisy"]) ** 2).mean()))
                rmse_rows.append({
                    "video_id": vid, "noise_type": nt, "attr": attr,
                    "rmse_self": self_rmse[attr], "rmse_noise": rmse_noise,
                })
    rmse_df = pd.DataFrame(rmse_rows)

    def _rmse_scatter(ax, attr, legend=False):
        sub = rmse_df[rmse_df["attr"] == attr]
        for nt in ["blur", "gaussian", "shapes"]:
            s = sub[sub["noise_type"] == nt]
            ax.scatter(s["rmse_self"], s["rmse_noise"], color=colors_nt[nt], marker=markers_nt[nt],
                       s=42, alpha=0.85, edgecolor="white", linewidth=0.6,
                       label=noise_label_pt[nt] if legend else None)
        lim = max(0.01, sub["rmse_self"].max() if len(sub) else 0, sub["rmse_noise"].max() if len(sub) else 0) * 1.08
        ax.plot([0, lim], [0, lim], color="#636e72", linestyle="--", linewidth=1, alpha=0.6)
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)

    fig, ax = plt.subplots(figsize=(7, 6.5))
    _rmse_scatter(ax, "The overall", legend=True)
    ax.set_xlabel("RMSE — consistência natural (original vs. média do vídeo)")
    ax.set_ylabel("RMSE — sensibilidade ao ruído (original vs. degradação progressiva)")
    ax.set_title("Consistência Temporal vs. Sensibilidade ao Ruído por Vídeo — atributo 'The overall'",
                 fontsize=11, fontweight="bold")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p = os.path.join(out_dir, "q6_rmse_overall.png")
    fig.savefig(p, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {p}")

    fig, axes = plt.subplots(3, 3, figsize=(4.2 * 3, 4.0 * 3))
    axes = np.array(axes).reshape(-1)
    for i, attr in enumerate(ATTRS):
        ax = axes[i]
        _rmse_scatter(ax, attr, legend=(i == 0))
        ax.set_title(ATTR_LABEL[attr], fontsize=10, fontweight="bold")
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.25)
        if i == 0:
            ax.legend(fontsize=7)
    fig.supxlabel("RMSE — consistência natural (original vs. média do vídeo)", fontsize=10)
    fig.supylabel("RMSE — sensibilidade ao ruído (original vs. degradação progressiva)", fontsize=10)
    fig.suptitle("Consistência Temporal vs. Sensibilidade ao Ruído por Vídeo, por Atributo",
                 fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    p = os.path.join(out_dir, "q6_rmse_grid.png")
    fig.savefig(p, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {p}")

    rows_tbl = []
    for attr in ATTRS:
        for nt in ["blur", "gaussian", "shapes"]:
            s = rmse_df[(rmse_df["attr"] == attr) & (rmse_df["noise_type"] == nt)]
            if len(s) == 0:
                continue
            rmse_self_mean = s["rmse_self"].mean()
            rmse_noise_mean = s["rmse_noise"].mean()
            ratio = rmse_noise_mean / rmse_self_mean if rmse_self_mean > 0 else float("nan")
            rows_tbl.append([
                ATTR_LABEL[attr], noise_label_pt[nt],
                f"{rmse_self_mean:.3f}", f"{rmse_noise_mean:.3f}",
                f"{ratio:.2f}x" if not np.isnan(ratio) else "—",
            ])
    ap.save_table(
        rows_tbl, ["Atributo", "Tipo de Ruído", "RMSE Consist. Natural", "RMSE Sensib. ao Ruído", "Razão"],
        out_dir, "q6_table", cfg,
        title="RMSE — Consistência Natural vs. Sensibilidade ao Ruído (média por vídeo)",
        col_widths=[1.3, 1.3, 1.2, 1.2, 0.8],
    )
    print("  -> q6_table.png / .tex.txt")

    # Copia os GIFs pelos video_id reais (TimeCraft) — não mais nomes numéricos
    # (0001/0003), que eram do dataset antigo @ArtsyLolaCo e não existem mais.
    # Exp5b agora tem 1 GIF por (vídeo, tipo de ruído), não 1 representante só.
    sample_video_ids = sorted(exp5a["video_id"].unique())[:3]
    for vid in sample_video_ids:
        _copy(os.path.join(OUT_ROOT, "exp5a_temporal", "samples", f"sequence_{vid}.gif"),
              os.path.join(out_dir, "samples"))
        for nt in ["blur", "gaussian", "shapes"]:
            _copy(os.path.join(OUT_ROOT, "exp5b_temporal_error", "samples", f"degradation_{vid}_{nt}.gif"),
                  os.path.join(out_dir, "samples"))
    _copy(os.path.join(OUT_ROOT, "exp5a_temporal", "samples", "frame_grid_last6.png"), os.path.join(out_dir, "samples"))
    _copy(os.path.join(OUT_ROOT, "exp5b_temporal_error", "samples", "frame_grid_uniform6.png"), os.path.join(out_dir, "samples"))


# ═══════════════════════════════════════════════════════════════════════════
# Exp5c — Temporal (janela macro, 1 fps, sem ruído)
# ═══════════════════════════════════════════════════════════════════════════

def gen_exp5c(cfg, base_dir):
    print("\n=== Exp5c — Temporal (janela macro) ===")
    out_dir = os.path.join(base_dir, "exp5c")
    os.makedirs(out_dir, exist_ok=True)
    OUT_ROOT = cfg["paths"]["outputs"]

    df = az.load_scores(os.path.join(OUT_ROOT, "exp5c_temporal_macro"), "original")
    if df is None:
        print("  [aviso] outputs/exp5c_temporal_macro/scores/scores_original.csv não encontrado, pulando "
              "(rode slurm_extract_exp5c_frames.sh + slurm_exp5c_temporal_macro.sh no cluster primeiro).")
        return

    video_ids = sorted(df["video_id"].unique())
    n_videos = len(video_ids)

    # Q1 — pedido principal: grade 3x3, uma linha clarinha por vídeo + média em destaque
    fig, axes = plt.subplots(3, 3, figsize=(4.6 * 3, 3.4 * 3))
    axes = np.array(axes).reshape(-1)
    for i, attr in enumerate(ATTRS):
        ax = axes[i]
        for vid in video_ids:
            sub = df[df["video_id"] == vid].sort_values("frame_idx")
            ax.plot(sub["frame_idx"], sub[attr], color="#74b9ff", alpha=0.25, linewidth=0.9)
        mean_by_frame = df.groupby("frame_idx")[attr].mean()
        ax.plot(mean_by_frame.index, mean_by_frame.values, color="#2d3436", linewidth=2.2,
               label="Média entre vídeos" if i == 0 else None)
        ax.set_ylim(0, 10); ax.set_yticks(np.arange(0, 11, 2))
        ax.set_title(ATTR_LABEL[attr], fontsize=10, fontweight="bold")
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.25)
        if i == 0:
            ax.legend(fontsize=8)
    fig.supxlabel("Frame (segundo do vídeo)", fontsize=10)
    fig.supylabel("Score", fontsize=10)
    fig.suptitle(f"Consistência Temporal em Nível Macro — {n_videos} Vídeos, por Atributo",
                 fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    p = os.path.join(out_dir, "q1_macro_trajectories_grid.png")
    fig.savefig(p, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {p}")

    # Q2 — sugestão: a dispersão ENTRE vídeos cresce ou diminui conforme o
    # vídeo avança? (faixa média ± desvio-padrão por segundo, agregando os
    # N vídeos em cada frame_idx -- diferente do Q1, que mostra vídeo por
    # vídeo; aqui é "o quão de acordo os vídeos estão entre si, a cada
    # segundo", pra ver se a resposta do ArtCLIP fica menos previsível à
    # medida que a pintura avança).
    avg_attr = "The overall"
    by_frame = df.groupby("frame_idx")[avg_attr]
    mean_line, std_line = by_frame.mean(), by_frame.std()
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(mean_line.index, mean_line.values, color="#2d3436", linewidth=2, label="Média entre vídeos")
    ax.fill_between(mean_line.index, mean_line - std_line, mean_line + std_line,
                    color="#74b9ff", alpha=0.35, label="± 1 desvio-padrão")
    ax.set_ylim(0, 10); ax.set_yticks(np.arange(0, 11, 1))
    ax.set_xlabel("Frame (segundo do vídeo)"); ax.set_ylabel(f"Score ({ATTR_LABEL[avg_attr]})")
    ax.set_title("Dispersão entre Vídeos ao Longo do Tempo — atributo 'The overall'",
                 fontsize=12, fontweight="bold")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p = os.path.join(out_dir, "q2_dispersion_over_time.png")
    fig.savefig(p, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {p}")

    # Q3 — sugestão: desvio-padrão INTRA-vídeo por atributo, na janela macro
    # (mesma lógica do Q1b do Exp5a, só que numa sequência bem mais longa —
    # se a consistência cai numa janela maior, aparece aqui como std maior).
    std_by_video = df.groupby("video_id")[ATTRS].std()
    fig, axes = plt.subplots(3, 3, figsize=(4.2 * 3, 3.2 * 3))
    axes = np.array(axes).reshape(-1)
    for i, attr in enumerate(ATTRS):
        ax = axes[i]
        vals = std_by_video[attr].dropna()
        ax.hist(vals, bins=15, color="#0984e3", alpha=0.8, edgecolor="white")
        ax.axvline(vals.mean(), color="#d63031", linestyle="--", linewidth=1.4, label=f"média={vals.mean():.2f}")
        ax.set_title(ATTR_LABEL[attr], fontsize=10, fontweight="bold")
        ax.set_xlabel("Desvio-padrão intra-vídeo", fontsize=8); ax.tick_params(labelsize=7)
        ax.legend(fontsize=7); ax.grid(True, alpha=0.25, axis="y")
    fig.suptitle("Consistência Temporal Macro por Atributo — Desvio-padrão dentro de cada Vídeo",
                 fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    p = os.path.join(out_dir, "q3_std_grid.png")
    fig.savefig(p, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {p}")

    rows = [[ATTR_LABEL[a], f"{std_by_video[a].mean():.3f}"] for a in ATTRS]
    ap.save_table(rows, ["Atributo", "Desvio-padrão intra-vídeo médio (macro)"], out_dir, "q3_table", cfg,
                  title="Consistência Temporal Macro — Desvio-padrão Médio por Atributo")
    print("  -> q3_table.png / .tex.txt")

    for vid in video_ids[:3]:
        _copy(os.path.join(OUT_ROOT, "exp5c_temporal_macro", "samples", f"sequence_{vid}.gif"),
              os.path.join(out_dir, "samples"))
    _copy(os.path.join(OUT_ROOT, "exp5c_temporal_macro", "samples", "frame_grid_uniform5.png"),
          os.path.join(out_dir, "samples"))


# ═══════════════════════════════════════════════════════════════════════════
# Exp5d — Temporal (janelas micro, 10x3s espalhadas pelo vídeo, sem ruído)
# ═══════════════════════════════════════════════════════════════════════════

def _contiguous_segments(frame_idx):
    """Quebra uma sequência ORDENADA de frame_idx em blocos contíguos (gap=1),
    devolvendo uma lista de arrays de índices POSICIONAIS (não os frame_idx em
    si) -- um array por bloco de 3s. É isso que faz a linha do gráfico "parar"
    entre 0-3s e 30-33s em vez de ligar os dois com um segmento reto."""
    idx = np.asarray(frame_idx)
    if len(idx) == 0:
        return []
    breaks = np.where(np.diff(idx) > 1)[0] + 1
    return np.split(np.arange(len(idx)), breaks)


def gen_exp5d(cfg, base_dir):
    print("\n=== Exp5d — Temporal (janelas micro) ===")
    out_dir = os.path.join(base_dir, "exp5d")
    videos_dir = os.path.join(out_dir, "videos")
    os.makedirs(videos_dir, exist_ok=True)
    OUT_ROOT = cfg["paths"]["outputs"]

    df = az.load_scores(os.path.join(OUT_ROOT, "exp5d_temporal_micro"), "original")
    if df is None:
        print("  [aviso] outputs/exp5d_temporal_micro/scores/scores_original.csv não encontrado, pulando "
              "(rode slurm_extract_exp5d_frames.sh + slurm_exp5d_temporal_micro.sh no cluster primeiro).")
        return

    video_ids = sorted(df["video_id"].unique())

    # Um gráfico POR VÍDEO: 9 subplots (um por atributo), dentro de cada um as
    # ~10 sub-amostras de 3s aparecem como segmentos SEPARADOS (sem ligar os
    # buracos entre janelas não-adjacentes), mais uma linha de referência
    # tracejada na média do vídeo inteiro pro atributo (destaque pedido —
    # como as janelas não compartilham posição temporal contígua entre si,
    # uma média ponto-a-ponto não faz sentido; a média global do vídeo é o
    # "quão consistente" comparável, análoga às linhas de referência "Human"
    # já usadas no resto do site).
    for vid in video_ids:
        sub = df[df["video_id"] == vid].sort_values("frame_idx")
        frame_idx = sub["frame_idx"].to_numpy()
        segments = _contiguous_segments(frame_idx)
        n_blocks = len(segments)

        fig, axes = plt.subplots(3, 3, figsize=(4.6 * 3, 3.4 * 3))
        axes = np.array(axes).reshape(-1)
        for i, attr in enumerate(ATTRS):
            ax = axes[i]
            vals = sub[attr].to_numpy()
            for seg in segments:
                ax.plot(frame_idx[seg], vals[seg], color="#74b9ff", alpha=0.55, linewidth=1.3,
                       marker="o", markersize=2.5)
            mean_val = np.nanmean(vals)
            ax.axhline(mean_val, color="#2d3436", linestyle="--", linewidth=1.8,
                       label=f"Média do vídeo ({mean_val:.2f})" if i == 0 else None)
            ax.set_ylim(0, 10); ax.set_yticks(np.arange(0, 11, 2))
            ax.set_title(ATTR_LABEL[attr], fontsize=10, fontweight="bold")
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.25)
            if i == 0:
                ax.legend(fontsize=8)
        fig.supxlabel("Frame (segundo real no vídeo original)", fontsize=10)
        fig.supylabel("Score", fontsize=10)
        fig.suptitle(f"Consistência Temporal em Nível Micro — {vid} ({n_blocks} janelas de 3s)",
                     fontsize=12, fontweight="bold", y=1.02)
        plt.tight_layout()
        p = os.path.join(videos_dir, f"q1_micro_trajectories_{vid}.png")
        fig.savefig(p, dpi=110, bbox_inches="tight")
        plt.close(fig)
    print(f"  -> {len(video_ids)} gráficos individuais em {videos_dir}/q1_micro_trajectories_<video_id>.png")

    for vid in video_ids[:3]:
        _copy(os.path.join(OUT_ROOT, "exp5d_temporal_micro", "samples", f"sequence_{vid}.gif"),
              os.path.join(out_dir, "samples"))
    _copy(os.path.join(OUT_ROOT, "exp5d_temporal_micro", "samples", "frame_grid_micro10.png"),
          os.path.join(out_dir, "samples"))


# ═══════════════════════════════════════════════════════════════════════════
# AI Measurement Science
# ═══════════════════════════════════════════════════════════════════════════

def gen_aims(cfg, base_dir):
    print("\n=== AI Measurement Science ===")
    out_dir = os.path.join(base_dir, "aims")
    os.makedirs(out_dir, exist_ok=True)
    OUT_ROOT = cfg["paths"]["outputs"]

    exp1_dir = ap._exp_scores_dir(cfg, "exp1_apdd", "uniform_bins")
    df_artclip = az.load_scores(exp1_dir, "original")
    df_artclip["stem"] = df_artclip["filename"].apply(az._stem)
    df_human = az.load_human_gt(cfg)

    rows = []
    for attr in ATTRS:
        m = df_human[["stem", attr]].merge(
            df_artclip[["stem", attr]].rename(columns={attr: attr + "_ac"}), on="stem"
        ).dropna()
        err = m[attr + "_ac"] - m[attr]
        mae = float(err.abs().mean()); rmse = float(np.sqrt((err ** 2).mean()))
        bias = float(err.mean()); corr = float(m[attr].corr(m[attr + "_ac"]))
        rows.append([ATTR_LABEL[attr], f"{mae:.3f}", f"{rmse:.3f}", f"{bias:+.3f}", f"{corr:.3f}", str(len(m))])
    ap.save_table(rows, ["Atributo", "MAE", "RMSE", "Bias", "r (Pearson)", "n"], out_dir, "aims6_error_table", cfg,
                  title="Erro do Modelo — ArtCLIP (original) vs. Human GT (APDDv2)")
    print("  -> aims6_error_table.png / .tex.txt")

    human_portinari = az.load_scores(os.path.join(OUT_ROOT, "exp2a_portinari"), "original")
    mnist = az.load_scores(os.path.join(OUT_ROOT, "exp3_mnist"), "original")

    fig, ax = plt.subplots(figsize=(7, 5.5))
    data = [df_human["Creativity"].dropna(), human_portinari["Creativity"].dropna(), mnist["Creativity"].dropna()]
    labels = ["APDDv2", "Portinari", "MNIST"]; colors = ["#33A650", "#e17055", "#a29bfe"]
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.6)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.75)
    ax.set_ylim(0, 10); ax.set_yticks(np.arange(0, 11, 1))
    ax.set_ylabel("Creativity")
    ax.set_title("Distribuição de Creativity por Dataset", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    p = os.path.join(out_dir, "aims7_creativity_boxplot.png")
    fig.savefig(p, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {p}")

    corr_matrix = df_human[ATTRS].corr()
    creativity_corr = corr_matrix["Creativity"].drop("Creativity").sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    labels_c = [ATTR_LABEL[a] for a in creativity_corr.index]
    colors_c = ["#0984e3" if v >= 0 else "#d63031" for v in creativity_corr.values]
    ax.barh(labels_c, creativity_corr.values, color=colors_c, alpha=0.85)
    ax.set_xlabel("Correlação de Pearson com Creativity"); ax.set_xlim(-1, 1)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Correlação de Creativity com os Outros Atributos (APDDv2)", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    p = os.path.join(out_dir, "aims7_creativity_corr.png")
    fig.savefig(p, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {p}")

    # AIMS-1/2/3/4/5 já existem prontos (analyse_* em scripts/analyze.py) — só copia
    figs_dir = os.path.join(cfg["paths"]["reports"], "figures")
    for name in ["discriminative_validity_density.png", "discriminative_validity_table.png",
                 "cultural_bias_boxplot.png", "cultural_bias_table.png",
                 "difficulty_groups_table.png", "difficulty_groups_density.png", "difficulty_groups_means.png",
                 "monotonicity_table.png", "exp1_clusters.png"]:
        _copy(os.path.join(figs_dir, name), out_dir)


# ═══════════════════════════════════════════════════════════════════════════
# APDDv2 x Portinari x MNIST — tabelas com teste estatístico único (Kruskal-
# Wallis + Mann-Whitney par-a-par) entre TODOS os grupos de cada linha, ao
# invés de reaproveitar letras de testes separados por bloco. APDDv2 = amostra
# de exp0_iccc_proportional_stratified (502 imgs, reprodução estratificada do
# Paper ICCC) — pedido explícito da usuária em 2026-08-06.
# ═══════════════════════════════════════════════════════════════════════════

def _render_kw_table_png(rows_raw, col_labels, out_path, title):
    """rows_raw: lista de linhas, cada célula = (texto_plano, is_bold)."""
    n_rows, n_cols = len(rows_raw), len(col_labels)
    fig_w = max(9, 1.2 + n_cols * 1.7)
    fig_h = max(2, 0.7 + n_rows * 0.42)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    text_rows = [[c[0] for c in row] for row in rows_raw]
    tbl = ax.table(cellText=text_rows, colLabels=col_labels, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.5)
    for j in range(n_cols):
        tbl[(0, j)].set_facecolor("#CCCCCC")
        tbl[(0, j)].set_text_props(fontweight="bold")
    for i, row in enumerate(rows_raw):
        for j, (_, is_bold) in enumerate(row):
            if is_bold:
                tbl[(i + 1, j)].set_text_props(fontweight="bold")
    ax.set_title(title, pad=12, fontsize=11, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out_path}")


def _kw_row_cells(row_info, col_order):
    means = {g: row_info[g]["mean"] for g in col_order if g in row_info}
    best_g = max(means, key=means.get) if means else None
    tex_cells, png_cells = [], []
    for g in col_order:
        if g not in row_info:
            tex_cells.append("\\textemdash"); png_cells.append(("—", False)); continue
        m, s, l = row_info[g]["mean"], row_info[g]["std"], row_info[g]["letter"]
        is_best = g == best_g
        tex = f"{m:.2f}$\\pm${s:.2f}$^{{{l}}}$"
        if is_best:
            tex = f"\\textbf{{{m:.2f}}}$\\pm${s:.2f}$^{{{l}}}$"
        tex_cells.append(tex)
        png_cells.append((f"{m:.2f}±{s:.2f}^{l}", is_best))
    return tex_cells, png_cells


def gen_apddv2_portinari_mnist_tables(cfg, base_dir):
    print("\n=== APDDv2 x Portinari x MNIST — tabelas Kruskal-Wallis ===")
    out_dir = os.path.join(base_dir, "shared")
    os.makedirs(out_dir, exist_ok=True)
    OUT_ROOT = cfg["paths"]["outputs"]

    apdd_full = az.load_human_gt(cfg)
    iccc_dir = ap._exp_scores_dir(cfg, "exp0_iccc", "proportional_stratified")
    iccc_1b = az.load_scores(iccc_dir, "Janus-Pro-1B")
    stems_iccc = set(iccc_1b["filename"].apply(az._stem)) if iccc_1b is not None else set()
    apdd_strat = apdd_full[apdd_full["stem"].isin(stems_iccc)].copy()
    print(f"  APDDv2 (Paper ICCC estratificado): {len(apdd_strat)} imagens (esperado 502)")

    def load_exp(exp_dir_name):
        exp_dir = os.path.join(OUT_ROOT, exp_dir_name)
        human = az.load_scores(exp_dir, "original")
        d1b = az.load_scores(exp_dir, "Janus-Pro-1B")
        d7b = az.load_scores(exp_dir, "Janus-Pro-7B")
        for d in (human, d1b, d7b):
            if d is not None and "stem" not in d.columns:
                d["stem"] = d["filename"].apply(az._stem)
        return human, d1b, d7b

    human_2a, d1b_2a, d7b_2a = load_exp("exp2a_portinari")
    human_2b, d1b_2b, d7b_2b = load_exp("exp2b_portinari_human")
    mnist = az.load_scores(os.path.join(OUT_ROOT, "exp3_mnist"), "original")

    # ── Tabela 1: APDDv2 x Portinari-Sintética(H,1B,7B) x Portinari-Humana(H,1B,7B) ──
    groups_t1 = {
        "APDDv2": apdd_strat,
        "Exp2a-Human": human_2a, "Exp2a-1B": d1b_2a, "Exp2a-7B": d7b_2a,
        "Exp2b-Human": human_2b, "Exp2b-1B": d1b_2b, "Exp2b-7B": d7b_2b,
    }
    km1 = az.kruskal_mannwhitney(groups_t1, ATTRS, cfg["stats"]["alpha"])
    col_order1 = ["APDDv2", "Exp2a-Human", "Exp2a-1B", "Exp2a-7B", "Exp2b-Human", "Exp2b-1B", "Exp2b-7B"]
    col_labels1 = ["Atributo", "APDDv2", "Human", "Janus-1B", "Janus-7B", "Human", "Janus-1B", "Janus-7B"]

    tex_rows1, png_rows1 = [], []
    for attr in ATTRS:
        tex_cells, png_cells = _kw_row_cells(km1.get(attr, {}), col_order1)
        tex_rows1.append([ATTR_LABEL[attr]] + tex_cells)
        png_rows1.append([(ATTR_LABEL[attr], False)] + png_cells)

    tex1 = r"""\begin{table*}[t]
\centering
\caption{Comparação dos escores do ArtCLIP por atributo para APDDv2 (amostra
estratificada proporcional, reprodução do Paper ICCC, $n=502$), Portinari com
descrição sintética e Portinari com descrição humana. Letras
sobrescritas diferentes indicam diferença estatisticamente significativa entre
os \emph{sete} grupos da linha (Kruskal-Wallis + Mann-Whitney par-a-par,
$p<0{,}05$, tratado como amostras independentes mesmo onde há pareamento
parcial, para permitir uma única comparação válida cobrindo também APDDv2
vs. Portinari). Em negrito, o melhor valor de cada linha.}
\label{tab:apddv2_portinari_kw}
\resizebox{\textwidth}{!}{%
\begin{tabular}{l c ccc ccc}
\toprule
 & APDDv2 & \multicolumn{3}{c}{Portinari --- Descrição Sintética} & \multicolumn{3}{c}{Portinari --- Descrição Humana} \\
\cmidrule(lr){2-2} \cmidrule(lr){3-5} \cmidrule(lr){6-8}
""" + " & ".join(col_labels1) + r""" \\
\midrule
"""
    for cells in tex_rows1:
        tex1 += " & ".join(cells) + r" \\" + "\n"
    tex1 += "\\bottomrule\n\\end{tabular}%\n}\n\\end{table*}\n"
    with open(os.path.join(out_dir, "tab_apddv2_portinari_kw.tex.txt"), "w", encoding="utf-8") as f:
        f.write(tex1)
    _render_kw_table_png(
        png_rows1, ["Atributo", "APDDv2", "Sint.-Human", "Sint.-1B", "Sint.-7B", "Hum.-Human", "Hum.-1B", "Hum.-7B"],
        os.path.join(out_dir, "tab_apddv2_portinari_kw.png"),
        "APDDv2 x Portinari (Sintética/Humana) — Kruskal-Wallis + Mann-Whitney (7 grupos/linha)",
    )
    print("  -> tab_apddv2_portinari_kw.png / .tex.txt")

    # ── Tabela 2: APDDv2(Human) x Portinari(Human) x MNIST ──────────────────
    groups_t2 = {"APDDv2 (Human)": apdd_strat, "Portinari (Human)": human_2a, "MNIST": mnist}
    km2 = az.kruskal_mannwhitney(groups_t2, ATTRS, cfg["stats"]["alpha"])
    col_order2 = ["APDDv2 (Human)", "Portinari (Human)", "MNIST"]

    tex_rows2, png_rows2 = [], []
    for attr in ATTRS:
        tex_cells, png_cells = _kw_row_cells(km2.get(attr, {}), col_order2)
        tex_rows2.append([ATTR_LABEL[attr]] + tex_cells)
        png_rows2.append([(ATTR_LABEL[attr], False)] + png_cells)

    tex2 = r"""\begin{table}[t]
\centering
\caption{Comparação dos escores do ArtCLIP (\textit{Human/original}) por
atributo entre APDDv2 (amostra estratificada proporcional, reprodução do
Paper ICCC, $n=502$), Portinari ($n=500$) e MNIST ($n=500$) --- três amostras
independentes. Letras sobrescritas diferentes indicam diferença
estatisticamente significativa (Kruskal-Wallis + Mann-Whitney par-a-par,
$p<0{,}05$). Em negrito, o melhor valor de cada linha.}
\label{tab:apddv2_portinari_mnist_kw}
\begin{tabular}{lccc}
\toprule
Atributo & APDDv2 (Human) & Portinari (Human) & MNIST \\
\midrule
"""
    for cells in tex_rows2:
        tex2 += " & ".join(cells) + r" \\" + "\n"
    tex2 += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    with open(os.path.join(out_dir, "tab_apddv2_portinari_mnist_kw.tex.txt"), "w", encoding="utf-8") as f:
        f.write(tex2)
    _render_kw_table_png(
        png_rows2, ["Atributo", "APDDv2 (Human)", "Portinari (Human)", "MNIST"],
        os.path.join(out_dir, "tab_apddv2_portinari_mnist_kw.png"),
        "APDDv2 x Portinari x MNIST (Human) — Kruskal-Wallis + Mann-Whitney",
    )
    print("  -> tab_apddv2_portinari_mnist_kw.png / .tex.txt")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

GENERATORS = {
    "paper_exp1": gen_paper_and_exp1,
    "exp2": gen_exp2, "exp3": gen_exp3, "exp4": gen_exp4, "exp5": gen_exp5, "exp5c": gen_exp5c,
    "exp5d": gen_exp5d,
    "aims": gen_aims,
    "kw_tables": gen_apddv2_portinari_mnist_tables,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/analysis_local.yaml")
    parser.add_argument("--only", default=None, help="lista separada por vírgula, ex: exp2,exp4")
    args = parser.parse_args()
    cfg = az.load_cfg(args.config)

    base_dir = os.path.join(cfg["paths"]["reports"], "figures_paper")
    os.makedirs(base_dir, exist_ok=True)

    keys = args.only.split(",") if args.only else list(GENERATORS.keys())
    for key in keys:
        GENERATORS[key](cfg, base_dir)

    print(f"\nTudo salvo em: {base_dir}")


if __name__ == "__main__":
    main()
