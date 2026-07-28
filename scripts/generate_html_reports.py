"""
Gera relatórios HTML navegáveis para GitHub Pages — um arquivo por experimento.

  python scripts/generate_html_reports.py --config configs/analysis_local.yaml

Estrutura gerada:
  reports/index.html
  reports/exp1_apdd.html
  reports/exp2a_portinari.html
  reports/exp2b_portinari_human.html
  reports/exp3_mnist.html
  reports/exp4_noise.html
  reports/exp5_temporal.html
  reports/legacy_iccc.html
"""

import argparse
import base64
import os
import yaml


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_cfg(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def img_b64(path):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    ext = os.path.splitext(path)[1].lstrip(".").lower()
    mime = "image/gif" if ext == "gif" else "image/png"
    return f"data:{mime};base64,{data}"


def img_tag(path, caption="", width="100%", css_class=""):
    uri = img_b64(path)
    if uri is None:
        return f'<p class="missing">⚠ Figura não encontrada: <code>{os.path.basename(path)}</code></p>'
    alt = caption or os.path.basename(path)
    cls = f' class="{css_class}"' if css_class else ""
    return (
        f'<figure{cls}>'
        f'<img src="{uri}" alt="{alt}" style="width:{width};max-width:1400px">'
        f'{"<figcaption>" + caption + "</figcaption>" if caption else ""}'
        f'</figure>'
    )


def section(anchor, title, badge, body):
    b = f' <span class="badge">{badge}</span>' if badge else ""
    return f'<section id="{anchor}"><h2>{title}{b}</h2>{body}</section>\n'


def row(*items):
    return '<div class="grid">' + "".join(items) + "</div>\n"


def row3(*items):
    return '<div class="grid-3">' + "".join(items) + "</div>\n"


def full(content):
    return f'<div class="full">{content}</div>'


# ── CSS + page template ────────────────────────────────────────────────────────

CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'Segoe UI', Arial, sans-serif; background: #f5f6fa;
       color: #2d3436; line-height: 1.6; }
header { background: #2d3436; color: white; padding: 1.5rem 2rem; }
header h1 { font-size: 1.4rem; }
header p  { font-size: 0.85rem; opacity: 0.75; margin-top: 0.3rem; }
nav { background: #636e72; padding: 0.4rem 2rem; display: flex; gap: 0.6rem;
      flex-wrap: wrap; }
nav a { color: #dfe6e9; text-decoration: none; padding: 0.25rem 0.7rem;
        border-radius: 4px; font-size: 0.82rem; }
nav a:hover, nav a.active { background: #2d3436; }
.container { max-width: 1400px; margin: 0 auto; padding: 2rem; }
section { margin-bottom: 3rem; }
h2 { font-size: 1.2rem; color: #2d3436; border-left: 4px solid var(--accent,#0984e3);
     padding-left: 0.8rem; margin-bottom: 1rem; margin-top: 2rem; }
h3 { font-size: 1rem; color: #636e72; margin: 1.2rem 0 0.5rem; }
.grid   { display: grid; grid-template-columns: repeat(auto-fit, minmax(480px, 1fr));
          gap: 1.2rem; }
.grid-3 { display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr));
          gap: 1rem; }
figure { background: white; border-radius: 8px; padding: 1rem;
         box-shadow: 0 2px 8px rgba(0,0,0,0.08); }
figure img { width: 100%; height: auto; border-radius: 4px; }
figcaption { font-size: 0.8rem; color: #636e72; margin-top: 0.5rem; text-align: center; }
.full { grid-column: 1 / -1; }
.info { background: #dfe6e9; border-radius: 6px; padding: 0.8rem 1rem;
        font-size: 0.88rem; margin-bottom: 1rem; }
.info strong { color: var(--accent,#0984e3); }
.missing { color: #d63031; font-style: italic; padding: 0.4rem; font-size: 0.85rem; }
.badge { display: inline-block; background: var(--accent,#0984e3); color: white;
         font-size: 0.72rem; padding: 0.1rem 0.45rem; border-radius: 10px;
         margin-left: 0.4rem; vertical-align: middle; }
footer { text-align: center; padding: 1.5rem; color: #636e72; font-size: 0.82rem;
         border-top: 1px solid #dfe6e9; margin-top: 2rem; }
"""


def page(title, subtitle, nav_links, body, accent="#0984e3", active_href=""):
    nav_html = ""
    for label, href in nav_links:
        cls = ' class="active"' if href == active_href else ""
        nav_html += f'<a href="{href}"{cls}>{label}</a>'
    return f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
:root {{ --accent: {accent}; }}
{CSS}
</style>
</head>
<body>
<header style="border-bottom: 4px solid {accent};">
  <h1>{title}</h1>
  <p>{subtitle}</p>
</header>
<nav>{nav_html}</nav>
<div class="container">
{body}
</div>
<footer>
  Gerado automaticamente · Larissa Gomide, Lucas Nascimento Ferreira, Wagner Meira Jr. · ICCC 2025
</footer>
</body>
</html>"""


# ── Navigation shared across all pages ────────────────────────────────────────

NAV = [
    ("Início",        "index.html"),
    ("Legacy ICCC",   "legacy_iccc.html"),
    ("Exp 1 APDDv2",  "exp1_apdd.html"),
    ("Exp 2a Portinari (AI)",  "exp2a_portinari.html"),
    ("Exp 2b Portinari (Human)", "exp2b_portinari_human.html"),
    ("Exp 3 MNIST",   "exp3_mnist.html"),
    ("Exp 4 Ruído",   "exp4_noise.html"),
    ("Exp 5 Temporal","exp5_temporal.html"),
]


# ── Per-experiment page builders ──────────────────────────────────────────────

def build_index(reports_dir):
    cards = [
        ("#6c5ce7", "legacy_iccc.html",           "Legacy ICCC",
         "Metodologia original ICCC 2025.<br>APDDv2-10023 vs sampled_SMALL/BIG.<br>t-test · Mann-Whitney · ANOVA · Radar"),
        ("#0984e3", "exp1_apdd.html",              "Exp 1 — APDDv2",
         "Baseline APDDv2 — 448 imagens amostradas.<br>Friedman · Wilcoxon · CLD · Radar · Clusters"),
        ("#00b894", "exp2a_portinari.html",        "Exp 2a — Portinari (AI Captions)",
         "500 imagens de Portinari.<br>Captions geradas pelo Janus-Pro-7B."),
        ("#e17055", "exp2b_portinari_human.html",  "Exp 2b — Portinari (Human Captions)",
         "498 imagens de Portinari.<br>Captions escritas por humanos."),
        ("#a29bfe", "exp3_mnist.html",             "Exp 3 — MNIST",
         "Arte vs. Não-Arte.<br>Discriminação ArtCLIP em dígitos manuscritos."),
        ("#fd79a8", "exp4_noise.html",             "Exp 4 — Ruído",
         "Impacto de ruído visual no score estético.<br>Gaussiano · Blur · Formas geométricas."),
        ("#fdcb6e", "exp5_temporal.html",          "Exp 5 — Temporal",
         "Consistência temporal em GIFs.<br>Detecção de degradação progressiva."),
    ]
    grid = ""
    for color, href, title, desc in cards:
        grid += f"""
    <a href="{href}" style="text-decoration:none;">
      <div style="background:white;border-radius:12px;padding:1.2rem;
                  box-shadow:0 4px 16px rgba(0,0,0,0.1);
                  border-top:4px solid {color};height:100%;">
        <h3 style="color:{color};margin-bottom:.4rem;">{title}</h3>
        <p style="color:#636e72;font-size:.87rem;">{desc}</p>
      </div>
    </a>"""
    body = f"""
<section style="max-width:1200px;margin:2rem auto;">
  <h2 style="border:none;font-size:1.6rem;margin-bottom:0.5rem;">Resultados da Dissertação</h2>
  <p style="color:#636e72;margin-bottom:1.5rem;font-size:.9rem;">
    Automatic Aesthetic Evaluation and Prompt Controllability in Generative Image Models<br>
    Larissa Gomide · Lucas Nascimento Ferreira · Wagner Meira Jr. · ICCC 2025
  </p>
  <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:1rem;">
    {grid}
  </div>
  <div style="margin-top:1.5rem;background:#dfe6e9;border-radius:8px;
              padding:0.8rem 1rem;font-size:.84rem;color:#636e72;">
    <strong>Reprodutibilidade:</strong>
    Consulte <code>REPRODUCIBILITY.txt</code> no repositório para detalhes metodológicos.
  </div>
</section>"""
    return page("Análise Estética — ICCC 2025", "Dissertação de Mestrado",
                NAV, body, accent="#2d3436", active_href="index.html")


def build_legacy(fi):
    def f(name, caption=""):
        return img_tag(os.path.join(fi, name), caption)

    body = (
        section("overview", "Metodologia", "",
            '<div class="info"><strong>ICCC 2025:</strong> '
            'Compara anotações humanas (APDDv2-10023.csv) com scores ArtCLIP de imagens geradas '
            'pelo Janus-Pro-1B (sampled_SMALL) e Janus-Pro-7B (sampled_BIG). '
            'Testes: Welch\'s t-test, Mann-Whitney U, ANOVA. Alinhamento por stem de filename.</div>') +

        section("comparison", "Comparação de Scores", "principal",
            row(full(f("iccc_legacy_comparison_bars.png",
                       "Human − Gerado (painel A) · ArtCLIP(original) − ArtCLIP(gerado) (painel B)")),
                full(f("iccc_score_diff_bars.png", "AVG(Human − Janus) por atributo")))) +

        section("distrib", "Distribuições de Score", "",
            row(f("iccc_hist_Janus-Pro-1B.png", "Histograma: Human GT vs Janus-Pro-1B"),
                f("iccc_hist_Janus-Pro-7B.png", "Histograma: Human GT vs Janus-Pro-7B")) +
            '<h3>Histogramas por atributo</h3>' +
            row(f("exp1_apdd_human_attr_hist_grid.png", "Grade — Human GT"),
                f("exp1_apdd_orig_attr_hist_grid.png", "Grade — ArtCLIP Original")) +
            '<h3>Avg Score com ajuste normal</h3>' +
            row(f("exp1_apdd_human_avg_score_gaussian.png", "Human GT"),
                f("exp1_apdd_orig_avg_score_gaussian.png", "ArtCLIP Original")) +
            '<h3>Distribuições normalizadas (MinMaxScaler)</h3>' +
            row(full(f("exp1_apdd_human_normalized_distributions.png", "Human GT — todos os atributos")))) +

        section("eda", "Análise Exploratória APDDv2", "",
            row(f("subcategory_proportions.png", "Proporção de subcategorias artísticas"),
                f("categories.png", "Distribuição de categorias")) +
            row(full(f("missing_by_category.png", "Valores ausentes por categoria artística")))) +

        section("corr", "Correlações", "",
            row(f("exp1_apdd_human_correlation_heatmap.png", "Correlação — Human GT"),
                f("exp1_apdd_orig_correlation_heatmap.png", "Correlação — ArtCLIP Original")) +
            row(f("iccc_scatter_Janus-Pro-1B.png", "Scatter: Human GT vs Janus-1B"),
                f("iccc_scatter_Janus-Pro-7B.png", "Scatter: Human GT vs Janus-7B"),
                f("exp1_apdd_human_scatter_regression.png", "Scatter com regressão — Human GT"),
                f("exp1_apdd_orig_scatter_regression.png", "Scatter com regressão — Original"))) +

        section("stats", "Tabela Estatística", "",
            row(full(f("iccc_summary_table.png", "Tabela 4.2 — Média ± Std (novo pipeline)")),
                full(f("iccc_legacy_summary_table.png", "Tabela — dados legacy (sampled_SMALL/BIG)")))) +

        section("boxplots", "Boxplots", "",
            row(f("iccc_boxplot_Janus-Pro-1B.png", "Boxplot: Human GT vs Janus-1B"),
                f("iccc_boxplot_Janus-Pro-7B.png", "Boxplot: Human GT vs Janus-7B"))) +

        section("radar", "Radar — Médias por Atributo", "",
            row(f("iccc_radar_Janus-Pro-1B.png", "Radar: Human GT vs Janus-1B"),
                f("iccc_radar_Janus-Pro-7B.png", "Radar: Human GT vs Janus-7B"),
                full(f("exp1_apdd_radar_three_way.png", "Radar 3-vias: Original vs Janus-1B vs Janus-7B"))))
    )
    return page("Legacy ICCC", "APDDv2-10023 vs sampled_SMALL/BIG · Metodologia ICCC 2025",
                NAV, body, accent="#6c5ce7", active_href="legacy_iccc.html")


def _exp_page(fi_iccc, fi_fig, fi_smp,
              exp_prefix_iccc, panel_a, panel_b,
              stat_table, boxplot_fig, dist_diff_fig, score_diff_fig,
              sample_a, sample_b,
              title, subtitle, accent, href, info_text):
    """Generic builder for Exp 1 / 2a / 2b style pages."""
    p = exp_prefix_iccc

    def fi(name, caption=""):
        return img_tag(os.path.join(fi_iccc, name), caption)
    def fn(name, caption=""):
        return img_tag(os.path.join(fi_fig, name), caption)
    def fs(name, caption=""):
        return img_tag(os.path.join(fi_smp, name), caption)

    body = (
        section("overview", "Visão Geral", "",
            f'<div class="info"><strong>Experimento:</strong> {info_text}</div>') +

        section("samples", "Amostras de Imagens", "",
            row(fs(sample_a, "Imagens originais (amostra)"),
                fs(sample_b, "Imagens geradas (amostra)"))) +

        section("comparison", "Diferença de Score", "principal",
            row(full(fi(f"{p}score_diff_bars.png",
                        "Original − Gerado por atributo (visualização ICCC)")),
                full(fn(score_diff_fig, "Score diff — metodologia nova")))) +

        section("distrib", "Distribuições por Atributo", "",
            row(full(fi(f"{p}three_way_histograms.png",
                        "Histogramas: Original vs Janus-1B vs Janus-7B"))) +
            row(fi(f"{p}orig_attr_hist_grid.png", "Grade — Original"),
                fi(f"{p}orig_avg_score_gaussian.png", "Avg Score — Original")) +
            row(fn(dist_diff_fig, "Diferença de distribuição (KS + Wasserstein + KL)"))) +

        section("stats", "Análise Estatística", "",
            row(full(fn(stat_table, "Tabela Estatística (Friedman + Wilcoxon + CLD)"))) +
            row(fn(boxplot_fig, "Boxplot por fonte"))) +

        section("corr", "Correlações", "",
            row(fi(f"{p}orig_correlation_heatmap.png", "Correlação — Original"),
                fi(f"{p}orig_scatter_regression.png", "'The overall' vs 'Mood'"))) +

        section("radar", "Radar — Médias por Atributo", "",
            row(full(fi(f"{p}radar_three_way.png",
                        "Radar: Original vs Janus-1B vs Janus-7B"))))
    )
    return page(title, subtitle, NAV, body, accent=accent, active_href=href)


def build_exp1(fi_iccc, fi_fig, fi_smp):
    def fi(name, caption=""):
        return img_tag(os.path.join(fi_iccc, name), caption)
    def fn(name, caption=""):
        return img_tag(os.path.join(fi_fig, name), caption)
    def fs(name, caption=""):
        return img_tag(os.path.join(fi_smp, name), caption)

    body = (
        section("overview", "Visão Geral", "",
            '<div class="info"><strong>Exp 1 — APDDv2:</strong> '
            '448 imagens amostradas (uniform bins, 30 bins). '
            'Compara Human GT (APDDv2-10023.csv) vs ArtCLIP(gerado). '
            'Análise ICCC + Friedman/Wilcoxon/CLD + distribuições.</div>') +

        section("samples", "Amostras de Imagens", "",
            row(fs("exp1_panel_a.png", "Originais (amostra)"),
                fs("exp1_panel_b.png", "Geradas (amostra)"))) +

        section("comparison", "Comparação de Scores", "principal",
            row(full(fi("pipeline_comparison_exp1_apdd.png",
                        "Painel A: Human − ArtCLIP(gerado) · Painel B: ArtCLIP(orig) − ArtCLIP(gerado)")),
                full(fi("exp1_apdd_score_diff_bars.png", "Diferença média por atributo (viz ICCC)")),
                full(fn("exp1_score_diff_bars.png", "Diferença de score — metodologia nova")))) +

        section("distrib", "Distribuições por Atributo", "",
            row(full(fi("exp1_apdd_three_way_histograms.png",
                        "Histogramas: Original vs Janus-1B vs Janus-7B"))) +
            row(fi("exp1_apdd_human_attr_hist_grid.png", "Grade — Human GT"),
                fi("exp1_apdd_orig_attr_hist_grid.png", "Grade — Original")) +
            row(fi("exp1_apdd_human_avg_score_gaussian.png", "Avg Score — Human GT"),
                fi("exp1_apdd_orig_avg_score_gaussian.png", "Avg Score — Original")) +
            '<h3>Distribuições normalizadas</h3>' +
            row(full(fi("exp1_apdd_human_normalized_distributions.png", "MinMaxScaler — Human GT"))) +
            row(fn("exp1_score_distributions.png", "Distribuições de score — metodologia nova"),
                fn("exp1_dist_diff.png", "Diferença de distribuição (KS + Wasserstein + KL)"))) +

        section("stats", "Análise Estatística", "",
            row(full(fn("exp1_stat_table.png", "Tabela (Friedman + Wilcoxon + CLD)")),
                full(fi("iccc_summary_table.png", "Tabela 4.2 — Média ± Std (metodologia ICCC)"))) +
            row(fn("exp1_boxplot_sources.png", "Boxplot por fonte"),
                fn("exp1_clusters.png", "Clusters por score"),
                fn("exp1_cluster_attrs.png", "Clusters por atributo"))) +

        section("corr", "Correlações e Scatter", "",
            row(fi("exp1_apdd_human_correlation_heatmap.png", "Correlação — Human GT"),
                fi("exp1_apdd_orig_correlation_heatmap.png", "Correlação — Original")) +
            row(fi("iccc_scatter_Janus-Pro-1B.png", "Scatter: Human GT vs Janus-1B"),
                fi("iccc_scatter_Janus-Pro-7B.png", "Scatter: Human GT vs Janus-7B"),
                fi("exp1_apdd_human_scatter_regression.png", "Scatter com regressão — Human GT"),
                fi("exp1_apdd_orig_scatter_regression.png", "Scatter com regressão — Original"))) +

        section("boxplot_iccc", "Boxplot — Metodologia ICCC", "",
            row(fi("iccc_boxplot_Janus-Pro-1B.png", "Boxplot: Human GT vs Janus-1B"),
                fi("iccc_boxplot_Janus-Pro-7B.png", "Boxplot: Human GT vs Janus-7B"),
                fi("exp1_apdd_boxplot_all_attrs.png", "Boxplot side-by-side todos os atributos"))) +

        section("radar", "Radar — Médias por Atributo", "",
            row(fi("iccc_radar_Janus-Pro-1B.png", "Radar: Human GT vs Janus-1B"),
                fi("iccc_radar_Janus-Pro-7B.png", "Radar: Human GT vs Janus-7B"),
                full(fi("exp1_apdd_radar_three_way.png",
                        "Radar 3-vias: Original vs Janus-1B vs Janus-7B"))))
    )
    return page("Exp 1 — APDDv2", "Baseline APDDv2 · Human GT vs Janus-Pro-1B/7B",
                NAV, body, accent="#0984e3", active_href="exp1_apdd.html")


def build_exp2(fi_iccc, fi_fig, fi_smp, variant):
    """variant: 'a' or 'b'"""
    prefix   = f"exp2{'a' if variant == 'a' else 'b'}_"
    fig_pfx  = f"exp2{'a' if variant == 'a' else 'b'}"
    href     = f"exp2{'a' if variant == 'a' else 'b'}_portinari.html"
    accent   = "#00b894" if variant == "a" else "#e17055"
    label    = "AI Captions" if variant == "a" else "Human Captions"
    subtitle = f"500 imagens de Portinari · {label}"
    title    = f"Exp 2{'a' if variant == 'a' else 'b'} — Portinari ({label})"
    tag      = f"exp2{'a' if variant == 'a' else 'b'}"

    def fi(name, caption=""):
        return img_tag(os.path.join(fi_iccc, name), caption)
    def fn(name, caption=""):
        return img_tag(os.path.join(fi_fig, name), caption)
    def fs(name, caption=""):
        return img_tag(os.path.join(fi_smp, name), caption)

    body = (
        section("overview", "Visão Geral", "",
            f'<div class="info"><strong>{title}:</strong> '
            f'Imagens do acervo Portinari geradas com {label.lower()}. '
            'Compara ArtCLIP(original) vs ArtCLIP(gerado).</div>') +

        section("samples", "Amostras de Imagens", "",
            row(fs(f"exp2_panel_a.png", "Originais (amostra)"),
                fs(f"exp2_panel_b.png", "Geradas (amostra)"))) +

        section("comparison", "Diferença de Score", "principal",
            row(full(fi(f"{prefix}score_diff_bars.png",
                        "Diferença média por atributo (viz ICCC)")),
                full(fn(f"{fig_pfx}_score_diff_bars.png",
                        "Score diff — metodologia nova")))) +

        section("distrib", "Distribuições por Atributo", "",
            row(full(fi(f"{prefix}three_way_histograms.png",
                        "Histogramas: Original vs Janus-1B vs Janus-7B"))) +
            row(fi(f"{prefix}orig_attr_hist_grid.png", "Grade — Original"),
                fi(f"{prefix}orig_avg_score_gaussian.png", "Avg Score — Original")) +
            row(fn("exp2_dist_diff.png", "Diferença de distribuição (KS + Wasserstein + KL)"))) +

        section("stats", "Análise Estatística", "",
            row(full(fn(f"exp2_2{'a' if variant == 'a' else 'b'}_{tag}_captions_stat_table.png" if variant == 'a'
                        else f"exp2_2b_human_captions_stat_table.png",
                        "Tabela Estatística (Friedman + Wilcoxon + CLD)")),
                fn("exp2_boxplot.png", "Boxplot por fonte"))) +

        section("corr", "Correlações", "",
            row(fi(f"{prefix}orig_correlation_heatmap.png", "Correlação — Original"),
                fi(f"{prefix}orig_scatter_regression.png", "'The overall' vs 'Mood'"))) +

        section("radar", "Radar — Médias por Atributo", "",
            row(full(fi(f"{prefix}radar_three_way.png",
                        "Radar: Original vs Janus-1B vs Janus-7B"))))
    )
    return page(title, subtitle, NAV, body, accent=accent, active_href=href)


def build_exp3(fi_fig, fi_smp):
    def fn(name, caption=""):
        return img_tag(os.path.join(fi_fig, name), caption)
    def fs(name, caption=""):
        return img_tag(os.path.join(fi_smp, name), caption)

    body = (
        section("overview", "Visão Geral", "",
            '<div class="info"><strong>Exp 3 — MNIST:</strong> '
            'Avalia a capacidade do ArtCLIP de discriminar imagens artísticas (Portinari/APDDv2) '
            'de imagens não-artísticas (dígitos MNIST). '
            'Expectativa: scores significativamente mais altos para obras de arte.</div>') +

        section("samples", "Amostras MNIST", "",
            row(full(fs("exp3_mnist_samples.png", "Amostras de dígitos MNIST")))) +

        section("results", "Resultados", "principal",
            row(full(fn("exp3_art_vs_noart.png", "Arte vs. Não-Arte — distribuições de score")),
                full(fn("exp3_dist_diff.png", "Diferença de distribuição (KS + Wasserstein + KL)"))))
    )
    return page("Exp 3 — MNIST", "Arte vs. Não-Arte · Discriminação ArtCLIP",
                NAV, body, accent="#a29bfe", active_href="exp3_mnist.html")


def build_exp4(fi_fig, fi_smp):
    def fn(name, caption=""):
        return img_tag(os.path.join(fi_fig, name), caption)
    def fs(name, caption=""):
        return img_tag(os.path.join(fi_smp, name), caption)

    body = (
        section("overview", "Visão Geral", "",
            '<div class="info"><strong>Exp 4 — Ruído:</strong> '
            'Aplica três tipos de ruído (Gaussiano, Blur+Salt&Pepper, Formas Geométricas) '
            'progressivamente às imagens originais e mede o impacto no score ArtCLIP.</div>') +

        section("samples", "Amostras de Ruído", "",
            row(full(fs("exp4_noise_samples.png", "Exemplos de imagens com diferentes tipos de ruído")))) +

        section("results", "Impacto do Ruído", "principal",
            row(full(fn("exp4_noise_impact.png", "Impacto do ruído no score estético por nível")),
                full(fn("exp4_noise_boxplot.png", "Boxplot por tipo de ruído")),
                full(fn("exp4_dist_diff.png", "Diferença de distribuição por tipo de ruído"))))
    )
    return page("Exp 4 — Ruído", "Impacto de Ruído Visual no Score Estético",
                NAV, body, accent="#fd79a8", active_href="exp4_noise.html")


def build_exp5(fi_fig, fi_smp):
    def fn(name, caption=""):
        return img_tag(os.path.join(fi_fig, name), caption)
    def fs(name, caption=""):
        return img_tag(os.path.join(fi_smp, name), caption)

    body = (
        section("overview", "Visão Geral", "",
            '<div class="info"><strong>Exp 5 — Temporal:</strong> '
            'Avalia consistência temporal de scores ArtCLIP em GIFs de storyboard (5a) '
            'e detecta degradação progressiva em vídeos (5b).</div>') +

        section("exp5a", "Exp 5a — Consistência Temporal", "",
            row(full(fs("exp5a_frame_grid.png", "Grade de frames — GIFs de storyboard")),
                full(fn("exp5a_temporal_consistency.png", "Consistência temporal: score por frame")))) +

        section("exp5b", "Exp 5b — Detecção de Degradação", "",
            row(full(fs("exp5b_degradation_sequence.png", "Sequência de degradação progressiva")),
                full(fn("exp5b_frame_score.png", "Score por frame — detecção de anomalia")),
                full(fn("exp5b_degradation.png", "Degradação progressiva detectada"))))
    )
    return page("Exp 5 — Temporal", "Consistência Temporal · Detecção de Degradação",
                NAV, body, accent="#fdcb6e", active_href="exp5_temporal.html")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/analysis_local.yaml")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    reports = cfg["paths"]["reports"]
    fi_iccc = os.path.join(reports, "figures_iccc")
    fi_fig  = os.path.join(reports, "figures")
    fi_smp  = os.path.join(reports, "samples")

    os.makedirs(reports, exist_ok=True)

    pages = {
        "index.html":               build_index(reports),
        "legacy_iccc.html":         build_legacy(fi_iccc),
        "exp1_apdd.html":           build_exp1(fi_iccc, fi_fig, fi_smp),
        "exp2a_portinari.html":     build_exp2(fi_iccc, fi_fig, fi_smp, "a"),
        "exp2b_portinari_human.html": build_exp2(fi_iccc, fi_fig, fi_smp, "b"),
        "exp3_mnist.html":          build_exp3(fi_fig, fi_smp),
        "exp4_noise.html":          build_exp4(fi_fig, fi_smp),
        "exp5_temporal.html":       build_exp5(fi_fig, fi_smp),
    }

    for fname, html in pages.items():
        out = os.path.join(reports, fname)
        with open(out, "w", encoding="utf-8") as f:
            f.write(html)
        size_kb = os.path.getsize(out) // 1024
        print(f"[html] OK {fname}  ({size_kb} KB)")

    print(f"\nRelatórios em: {reports}")
    print("Para GitHub Pages, copie os HTMLs para docs/ ou gh-pages branch.")


if __name__ == "__main__":
    main()
