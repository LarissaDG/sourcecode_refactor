"""
Gera relatórios HTML navegáveis para GitHub Pages.

  python scripts/generate_html_reports.py --config configs/analysis_local.yaml

Gera dois arquivos:
  reports/legacy_report.html    — análise ICCC (APDDv2-10023 vs sampled_SMALL/BIG)
  reports/pipeline_report.html  — análise novo pipeline (APDDv2-10023 vs exp1 scores)
  reports/index.html            — página de entrada com links para os dois
"""

import argparse
import base64
import os
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ── helpers ───────────────────────────────────────────────────────────────────

def load_cfg(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def img_b64(path):
    """Converte PNG para data URI base64."""
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    return f"data:image/png;base64,{data}"


def img_tag(path, caption="", width="100%"):
    uri = img_b64(path)
    if uri is None:
        return f'<p class="missing">⚠ Figura não encontrada: {os.path.basename(path)}</p>'
    alt = caption or os.path.basename(path)
    return (
        f'<figure>'
        f'<img src="{uri}" alt="{alt}" style="width:{width};max-width:1200px">'
        f'{"<figcaption>" + caption + "</figcaption>" if caption else ""}'
        f'</figure>'
    )


CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'Segoe UI', Arial, sans-serif; background: #f5f6fa;
       color: #2d3436; line-height: 1.6; }
header { background: #2d3436; color: white; padding: 1.5rem 2rem; }
header h1 { font-size: 1.5rem; }
header p  { font-size: 0.9rem; opacity: 0.8; margin-top: 0.3rem; }
nav { background: #636e72; padding: 0.5rem 2rem; display: flex; gap: 1rem;
      flex-wrap: wrap; }
nav a { color: #dfe6e9; text-decoration: none; padding: 0.3rem 0.8rem;
        border-radius: 4px; font-size: 0.9rem; }
nav a:hover { background: #2d3436; }
.container { max-width: 1400px; margin: 0 auto; padding: 2rem; }
section { margin-bottom: 3rem; }
h2 { font-size: 1.3rem; color: #2d3436; border-left: 4px solid #0984e3;
     padding-left: 0.8rem; margin-bottom: 1rem; margin-top: 2rem; }
h3 { font-size: 1.05rem; color: #636e72; margin: 1.2rem 0 0.5rem; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(480px, 1fr));
        gap: 1.5rem; }
.grid-3 { display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
           gap: 1rem; }
figure { background: white; border-radius: 8px; padding: 1rem;
         box-shadow: 0 2px 8px rgba(0,0,0,0.08); }
figure img { width: 100%; height: auto; border-radius: 4px; }
figcaption { font-size: 0.82rem; color: #636e72; margin-top: 0.5rem;
             text-align: center; }
.full { grid-column: 1 / -1; }
.info { background: #dfe6e9; border-radius: 6px; padding: 1rem 1.2rem;
        font-size: 0.9rem; margin-bottom: 1rem; }
.info strong { color: #0984e3; }
.missing { color: #d63031; font-style: italic; padding: 0.5rem; }
.badge { display: inline-block; background: #0984e3; color: white;
         font-size: 0.75rem; padding: 0.1rem 0.5rem; border-radius: 10px;
         margin-left: 0.5rem; vertical-align: middle; }
table { width: 100%; border-collapse: collapse; background: white;
        border-radius: 8px; overflow: hidden;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08); font-size: 0.88rem; }
th { background: #2d3436; color: white; padding: 0.6rem 1rem; text-align: left; }
td { padding: 0.5rem 1rem; border-bottom: 1px solid #f0f0f0; }
tr:last-child td { border-bottom: none; }
tr:hover td { background: #f5f6fa; }
footer { text-align: center; padding: 2rem; color: #636e72; font-size: 0.85rem;
         border-top: 1px solid #dfe6e9; margin-top: 2rem; }
"""

NAV_LEGACY = """
<nav>
  <a href="#overview">Visão Geral</a>
  <a href="#comparison">Comparação de Scores</a>
  <a href="#distributions">Distribuições</a>
  <a href="#stats">Tabela Estatística</a>
  <a href="#boxplots">Boxplots</a>
  <a href="#scatter">Correlação</a>
  <a href="#radar">Radar</a>
  <a href="pipeline_report.html">→ Novo Pipeline</a>
  <a href="index.html">↩ Início</a>
</nav>
"""

NAV_PIPELINE = """
<nav>
  <a href="#overview">Visão Geral</a>
  <a href="#comparison">Comparação APDDv2</a>
  <a href="#exp1">Exp 1 — APDDv2</a>
  <a href="#exp2">Exp 2 — Portinari</a>
  <a href="#exp3">Exp 3 — MNIST</a>
  <a href="#crossexp">Comparações</a>
  <a href="legacy_report.html">→ Legacy ICCC</a>
  <a href="index.html">↩ Início</a>
</nav>
"""


def page(title, subtitle, nav, body, color="#0984e3"):
    return f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>{CSS}</style>
</head>
<body>
<header style="border-bottom: 4px solid {color};">
  <h1>{title}</h1>
  <p>{subtitle}</p>
</header>
{nav}
<div class="container">
{body}
</div>
<footer>
  Gerado automaticamente por generate_html_reports.py —
  Larissa Gomide, Lucas Nascimento Ferreira, Wagner Meira Jr. · ICCC 2025
</footer>
</body>
</html>"""


# ── Legacy report ─────────────────────────────────────────────────────────────

def build_legacy_report(fi):
    """fi = figures_iccc directory"""

    def f(name, caption=""):
        return img_tag(os.path.join(fi, name), caption)

    body = f"""
<section id="overview">
  <div class="info">
    <strong>Metodologia ICCC:</strong>
    Compara anotações humanas do APDDv2-10023.csv com scores ArtCLIP de imagens geradas
    pelo Janus-Pro-1B (sampled_SMALL) e Janus-Pro-7B (sampled_BIG).
    Testes estatísticos: Welch's t-test, Mann-Whitney U, ANOVA.
    Alinhamento por stem de filename.
  </div>
</section>

<section id="comparison">
  <h2>Comparação de Scores — APDDv2-10023 vs Geradas <span class="badge">Principal</span></h2>
  <p style="font-size:.9rem;color:#636e72;margin-bottom:1rem;">
    Painel A: Human annotation (APDDv2-10023.csv) − ArtCLIP(gerado) &nbsp;|&nbsp;
    Painel B: ArtCLIP(original) − ArtCLIP(gerado)
  </p>
  <div class="grid">
    <figure class="full">{img_tag(os.path.join(fi, "iccc_legacy_comparison_bars.png"),
      "Comparação: Human − Gerado (painel A) e ArtCLIP(original) − ArtCLIP(gerado) (painel B)")}</figure>
  </div>
  <h3>Diferença de score (Human − Gerado) por atributo</h3>
  <div class="grid">
    <figure class="full">{img_tag(os.path.join(fi, "iccc_score_diff_bars.png"),
      "AVG(Human − Janus) por atributo — metodologia ICCC")}</figure>
  </div>
</section>

<section id="stats">
  <h2>Tabela Estatística</h2>
  <div class="grid">
    <figure class="full">{img_tag(os.path.join(fi, "iccc_summary_table.png"),
      "Tabela 4.2 — médias Human GT vs ArtCLIP(gerado)")}</figure>
    <figure class="full">{img_tag(os.path.join(fi, "iccc_legacy_summary_table.png"),
      "Tabela resumo — dados legacy (sampled_SMALL/BIG)")}</figure>
  </div>
</section>

<section id="distributions">
  <h2>Distribuições de Score</h2>
  <div class="grid">
    {f("iccc_hist_Janus-Pro-1B.png", "Histogramas: Human GT vs Janus-Pro-1B")}
    {f("iccc_hist_Janus-Pro-7B.png", "Histogramas: Human GT vs Janus-Pro-7B")}
  </div>
</section>

<section id="boxplots">
  <h2>Boxplots por Atributo</h2>
  <div class="grid">
    {f("iccc_boxplot_Janus-Pro-1B.png", "Boxplot: Human GT vs Janus-Pro-1B")}
    {f("iccc_boxplot_Janus-Pro-7B.png", "Boxplot: Human GT vs Janus-Pro-7B")}
  </div>
</section>

<section id="scatter">
  <h2>Correlação (Scatter)</h2>
  <div class="grid">
    {f("iccc_scatter_Janus-Pro-1B.png", "Scatter: Human GT vs Janus-Pro-1B")}
    {f("iccc_scatter_Janus-Pro-7B.png", "Scatter: Human GT vs Janus-Pro-7B")}
  </div>
</section>

<section id="radar">
  <h2>Radar — Médias por Atributo</h2>
  <div class="grid">
    {f("iccc_radar_Janus-Pro-1B.png", "Radar: Human GT vs Janus-Pro-1B")}
    {f("iccc_radar_Janus-Pro-7B.png", "Radar: Human GT vs Janus-Pro-7B")}
  </div>
</section>
"""
    return page(
        "Análise Legacy ICCC",
        "APDDv2-10023.csv vs sampled_SMALL (Janus-1B) / sampled_BIG (Janus-7B) · Metodologia ICCC 2025",
        NAV_LEGACY, body, color="#6c5ce7"
    )


# ── Pipeline report ───────────────────────────────────────────────────────────

def build_pipeline_report(fi_iccc, fi_new):
    """fi_iccc = figures_iccc dir, fi_new = figures dir"""

    def fi(name, caption=""):
        return img_tag(os.path.join(fi_iccc, name), caption)

    def fn(name, caption=""):
        return img_tag(os.path.join(fi_new, name), caption)

    body = f"""
<section id="overview">
  <div class="info">
    <strong>Novo Pipeline:</strong>
    Execução completa Sampling → Captioning → Generation → Scoring.
    Análise com Friedman + Wilcoxon + CLD. Alinhamento por stem de filename.
    Experimentos: Exp1 (APDDv2), Exp2a/2b (Portinari), Exp3 (MNIST).
  </div>
</section>

<section id="comparison">
  <h2>Comparação APDDv2-10023 vs Exp1 <span class="badge">Principal</span></h2>
  <p style="font-size:.9rem;color:#636e72;margin-bottom:1rem;">
    APDDv2-10023.csv (human GT) vs scores_Janus-Pro-1B/7B.csv do exp1_apdd
  </p>
  <div class="grid">
    <figure class="full">{img_tag(os.path.join(fi_iccc, "pipeline_comparison_exp1_apdd.png"),
      "Painel A: Human − ArtCLIP(gerado) | Painel B: ArtCLIP(orig) − ArtCLIP(gerado)")}</figure>
  </div>
</section>

<section id="exp1">
  <h2>Exp 1 — APDDv2 Baseline</h2>
  <div class="grid">
    {fn("exp1_stat_table.png", "Tabela estatística (Friedman + Wilcoxon + CLD)")}
    {fn("exp1_score_diff_bars.png", "Diferença de score: Original − Gerado")}
  </div>
  <div class="grid">
    {fn("exp1_dist_diff.png", "Diferença de distribuição (KS + Wasserstein + KL)")}
    {fn("exp1_score_distributions.png", "Distribuições de score")}
  </div>
  <div class="grid">
    {fn("exp1_boxplot_sources.png", "Boxplot por fonte")}
    {fn("exp1_clusters.png", "Clusters por score")}
  </div>
</section>

<section id="exp2">
  <h2>Exp 2 — Portinari (Caption AI vs Caption Humana)</h2>
  <div class="grid">
    {fn("exp2_2a_gen_captions_stat_table.png", "Exp2a — Captions geradas por IA")}
    {fn("exp2_2b_human_captions_stat_table.png", "Exp2b — Captions humanas")}
  </div>
  <div class="grid">
    {fn("exp2a_score_diff_bars.png", "Exp2a — Diferença Original − Gerado (AI captions)")}
    {fn("exp2b_score_diff_bars.png", "Exp2b — Diferença Original − Gerado (Human captions)")}
  </div>
  <div class="grid">
    {fn("exp2_boxplot.png", "Boxplot Exp2")}
    {fn("exp2_dist_diff.png", "Diferença de distribuição Exp2")}
  </div>
</section>

<section id="exp3">
  <h2>Exp 3 — MNIST (Arte vs Não-Arte)</h2>
  <div class="grid">
    {fn("exp3_art_vs_noart.png", "Discriminação Arte vs Não-Arte")}
    {fn("exp3_dist_diff.png", "Diferença de distribuição Exp3")}
  </div>
</section>

<section id="crossexp">
  <h2>Comparações Cross-Experimento</h2>
  <div class="grid">
    {fn("fig49_apddv2_vs_portinari.png", "Fig 4.9 — APDDv2 vs Portinari")}
    {fn("fig410_art_vs_noart.png", "Fig 4.10 — Arte vs MNIST")}
  </div>
</section>
"""
    return page(
        "Análise Novo Pipeline",
        "exp1_apdd · exp2a/2b_portinari · exp3_mnist — Friedman + Wilcoxon + CLD",
        NAV_PIPELINE, body, color="#0984e3"
    )


# ── Index ─────────────────────────────────────────────────────────────────────

def build_index():
    body = """
<section style="max-width:800px;margin:3rem auto;text-align:center;">
  <h2 style="border:none;text-align:center;font-size:1.8rem;margin-bottom:1rem;">
    Resultados da Dissertação
  </h2>
  <p style="color:#636e72;margin-bottom:2rem;">
    Automatic Aesthetic Evaluation and Prompt Controllability in Generative Image Models<br>
    Larissa Gomide · Lucas Nascimento Ferreira · Wagner Meira Jr. · ICCC 2025
  </p>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:1.5rem;text-align:left;">
    <a href="legacy_report.html" style="text-decoration:none;">
      <div style="background:white;border-radius:12px;padding:1.5rem;
                  box-shadow:0 4px 16px rgba(0,0,0,0.1);
                  border-top:4px solid #6c5ce7;">
        <h3 style="color:#6c5ce7;margin-bottom:.5rem;">📄 Legacy ICCC</h3>
        <p style="color:#636e72;font-size:.9rem;">
          Metodologia original do artigo publicado.<br>
          APDDv2-10023.csv vs sampled_SMALL/BIG.<br>
          t-test · Mann-Whitney · ANOVA · Radar
        </p>
      </div>
    </a>
    <a href="pipeline_report.html" style="text-decoration:none;">
      <div style="background:white;border-radius:12px;padding:1.5rem;
                  box-shadow:0 4px 16px rgba(0,0,0,0.1);
                  border-top:4px solid #0984e3;">
        <h3 style="color:#0984e3;margin-bottom:.5rem;">🔬 Novo Pipeline</h3>
        <p style="color:#636e72;font-size:.9rem;">
          Pipeline refatorado — dissertação de mestrado.<br>
          Exp1 (APDDv2) · Exp2 (Portinari) · Exp3 (MNIST).<br>
          Friedman · Wilcoxon · CLD · KS · Wasserstein
        </p>
      </div>
    </a>
  </div>
  <div style="margin-top:2rem;background:#dfe6e9;border-radius:8px;
              padding:1rem;font-size:.85rem;color:#636e72;">
    <strong>Reprodutibilidade:</strong>
    Consulte <code>REPRODUCIBILITY.txt</code> no repositório para detalhes sobre
    diferenças metodológicas e como reproduzir os resultados.
  </div>
</section>
"""
    return page(
        "Análise Estética — ICCC 2025",
        "Resultados da dissertação de mestrado",
        '<nav><a href="legacy_report.html">Legacy ICCC</a>'
        '<a href="pipeline_report.html">Novo Pipeline</a></nav>',
        body
    )


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/analysis_local.yaml")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    reports_dir = cfg["paths"]["reports"]
    fi_iccc = os.path.join(reports_dir, "figures_iccc")
    fi_new  = os.path.join(reports_dir, "figures")

    os.makedirs(reports_dir, exist_ok=True)

    print(f"[html] reports_dir: {reports_dir}")
    print(f"[html] figures_iccc: {fi_iccc}")
    print(f"[html] figures_new:  {fi_new}")

    # Legacy report
    out = os.path.join(reports_dir, "legacy_report.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(build_legacy_report(fi_iccc))
    print(f"[html] OK {out}")

    # Pipeline report
    out = os.path.join(reports_dir, "pipeline_report.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(build_pipeline_report(fi_iccc, fi_new))
    print(f"[html] OK {out}")

    # Index
    out = os.path.join(reports_dir, "index.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(build_index())
    print(f"[html] OK {out}")

    print("\nRelatorios gerados. Para GitHub Pages, copie os 3 HTMLs para docs/ ou gh-pages branch.")


if __name__ == "__main__":
    main()
