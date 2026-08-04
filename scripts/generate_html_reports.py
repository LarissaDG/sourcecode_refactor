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
  reports/Paper_iccc.html
"""

import argparse
import base64
import os
import yaml


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_cfg(path):
    with open(path, encoding="utf-8-sig") as f:
        return yaml.safe_load(f)


STRATEGY_LABELS = {
    "uniform_bins": "Uniforme",
    "proportional_stratified": "Estratificado Proporcional",
}


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


def toggle_buttons(group_id, options):
    """options: lista de (key, label). O primeiro fica ativo por padrão."""
    btns = "".join(
        f'<button class="toggle-btn{" active" if i == 0 else ""}" '
        f'data-group="{group_id}" data-show="{key}">{label}</button>'
        for i, (key, label) in enumerate(options)
    )
    return f'<div class="toggle-bar">{btns}</div>'


def toggle_panel(group_id, key, content, default=False):
    style = "" if default else ' style="display:none"'
    return f'<div class="toggle-panel" data-group="{group_id}" data-strategy="{key}"{style}>{content}</div>'


def toggle_section(anchor, title, badge, content_fn, strategies, group_id="strategy"):
    """
    content_fn(strategy) -> HTML. Se `strategies` tiver mais de 1 item, mostra
    botões de alternância (+ opção "ambos lado a lado"); senão, mostra só o
    conteúdo da única estratégia, sem botão.
    """
    if len(strategies) > 1:
        options = [(s, STRATEGY_LABELS.get(s, s)) for s in strategies] + [("both", "Ambos lado a lado")]
        body = toggle_buttons(group_id, options)
        for i, s in enumerate(strategies):
            body += toggle_panel(group_id, s, content_fn(s), default=(i == 0))
    else:
        body = content_fn(strategies[0])
    return section(anchor, title, badge, body)


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
.toggle-bar { margin-bottom: 1rem; }
.toggle-btn { background: white; border: 1px solid #b2bec3; color: #2d3436;
              padding: 0.4rem 0.9rem; border-radius: 6px; margin-right: 0.5rem;
              font-size: 0.85rem; cursor: pointer; }
.toggle-btn:hover { border-color: var(--accent,#0984e3); }
.toggle-btn.active { background: var(--accent,#0984e3); color: white; border-color: var(--accent,#0984e3); }
pre { white-space: pre-wrap; font-size: 0.78rem; line-height: 1.4; }
"""

TOGGLE_JS = """
document.querySelectorAll('.toggle-btn').forEach(function (btn) {
  btn.addEventListener('click', function () {
    var group = btn.getAttribute('data-group');
    var show = btn.getAttribute('data-show');
    document.querySelectorAll('.toggle-btn[data-group="' + group + '"]').forEach(function (b) {
      b.classList.toggle('active', b === btn);
    });
    document.querySelectorAll('.toggle-panel[data-group="' + group + '"]').forEach(function (panel) {
      var strat = panel.getAttribute('data-strategy');
      panel.style.display = (show === 'both' || strat === show) ? '' : 'none';
    });
  });
});
"""


def page(title, subtitle, nav_links, body, accent="#0984e3", active_href="", header_title=None):
    nav_html = ""
    for label, href in nav_links:
        cls = ' class="active"' if href == active_href else ""
        nav_html += f'<a href="{href}"{cls}>{label}</a>'
    h1 = header_title or title
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
  <h1>{h1}</h1>
  <p>{subtitle}</p>
</header>
<nav>{nav_html}</nav>
<div class="container">
{body}
</div>
<footer>
  Gerado automaticamente · Larissa Gomide, Lucas Nascimento Ferreira, Wagner Meira Jr. · ICCC 2025
</footer>
<script>{TOGGLE_JS}</script>
</body>
</html>"""


# ── Navigation shared across all pages ────────────────────────────────────────

NAV = [
    ("Início",                    "index.html"),
    ("Paper ICCC",                "Paper_iccc.html"),
    ("Exp 1 APDDv2",              "exp1_apdd.html"),
    ("Exp 2a Portinari (AI)",     "exp2a_portinari.html"),
    ("Exp 2b Portinari (Human)",  "exp2b_portinari_human.html"),
    ("Exp 3 MNIST",               "exp3_mnist.html"),
    ("Exp 4 Ruído",               "exp4_noise.html"),
    ("Exp 5 Temporal",            "exp5_temporal.html"),
]


# ── Per-experiment page builders ──────────────────────────────────────────────

def build_index(reports_dir):
    cards = [
        ("#6c5ce7", "Paper_iccc.html",            "Paper ICCC",
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
    <b>Aluna:</b> Larissa Dolabella Gomide  <br>
    <b>Professores:</b> Lucas Nascimento Ferreira · Wagner Meira Jr.
    <br>
    <b>Resumo do trabalho:</b> Este trabalho investiga a avaliação estética objetiva de pinturas geradas por inteligência artificial no âmbito da criatividade computacional, propondo o modelo ArtCLIP, treinado com anotações de especialistas em arte, como uma heurística formal para medir o valor estético. Por meio de cinco núcleos experimentais que testaram desde a comparação entre obras humanas e
    sintéticas até análises de riqueza semântica, discriminação artística e estabilidade a ruídos e modificações locais, os autores demonstraram que o ArtCLIP alinha-se de forma consistente ao julgamento humano especializado. Com isso, o estudo conclui que heurísticas computacionais baseadas no conhecimento de especialistas oferecem uma alternativa escalável e robusta para superar a
    subjetividade na avaliação de arte gerada por IA. Esse trabalho gerou como resultado um paper que foi apresentado no 16th International Conference on Computational Creativity, ICCC'25, em Campinas - Brasil. Além de ter participado, também do Early Career Symposium da 17th International Conference on Computational Creativity, ICCC'26, em Coimbra - Portugal.
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
    header_title = "Avaliação Estética Automática de Imagens Artísticas: Uma abordagem heurística para apoiar processos criativos em artes visuais"
    return page("Análise Estética — ICCC 2025", "Dissertação de Mestrado",
                NAV, body, accent="#2d3436", active_href="index.html",
                header_title=header_title)


# ── Blocos de conteúdo compartilhados entre Paper_iccc.html e exp1_apdd.html ──
# (ver scripts/analyze_paper.py — é quem gera as figuras/tabelas referenciadas aqui)

def _eda_html(fi_shared):
    def f(name, caption=""):
        return img_tag(os.path.join(fi_shared, f"{name}.png"), caption)
    return (
        section("eda", "1. Análise Exploratória do Dataset", "",
            '<div class="info"><strong>Fonte:</strong> <code>APDDv2-10023.csv</code> '
            '(10.023 imagens, inclui o atributo <strong>The sense of order</strong>).</div>' +
            row(full(f("eda_category_crosstab", "Tabela cruzada Medium × Style × Subject"))) +
            row(f("eda_medium_summary", "Distribuição por Medium"),
                f("eda_style_summary", "Distribuição por Style"),
                f("eda_subject_summary", "Distribuição por Subject"))) +
        section("missing", "2. Valores Ausentes por Categoria Artística", "",
            '<div class="info">Eixo X = atributo estético, eixo Y = uma das 24 combinações '
            'Medium/Style/Subject. Célula = quantidade de valores ausentes.</div>' +
            row(full(f("eda_missing_by_category", "Valores ausentes por categoria artística"))))
    )


def _sampling_html(fi_dir, strategies):
    def content(strategy):
        def f(name, caption=""):
            return img_tag(os.path.join(fi_dir, f"{name}_{strategy}.png"), caption)
        return (
            row(full(f("sampling_dist", "Distribuição do score médio — antes vs. depois da amostragem"))) +
            row(full(f("sampling_bin_table", "Distribuição de imagens por bin"))) +
            row(full(f("sampling_attr_grid", "Distribuição por atributo — antes vs. depois")))
        )
    return toggle_section("sampling", "3. Amostragem", "", content, strategies)


def _samples_html(outputs_dir, base_name, strategies):
    def content(strategy):
        smp_dir = os.path.join(outputs_dir, f"{base_name}_{strategy}", "samples")
        if not os.path.isdir(smp_dir):
            smp_dir = os.path.join(outputs_dir, base_name, "samples")  # fallback single-strategy
        return row(full(img_tag(os.path.join(smp_dir, "sample_panel.png"),
                                f"Amostras de exemplo — {STRATEGY_LABELS.get(strategy, strategy)}")))
    return toggle_section("samples", "4. Exemplo do Conjunto de Dados", "", content, strategies)


def _questions_html(fi_dir, strategies):
    def q1(strategy):
        return row(full(img_tag(os.path.join(fi_dir, f"q1_friedman_wilcoxon_table_{strategy}.png"),
                                "Tabela 1 — Friedman + Wilcoxon (CLD)")))
    def q2(strategy):
        return row(full(img_tag(os.path.join(fi_dir, f"q2_score_diff_bars_{strategy}.png"),
                                "Diferença média de score (Human − Gerado) por atributo")))
    def q3(strategy):
        return row(full(img_tag(os.path.join(fi_dir, f"q3_score_diff_table_{strategy}.png"),
                                "Tabela de diferenças (Human − Janus-1B / Human − Janus-7B)")))
    return (
        toggle_section("q1", "5. Pergunta 1 — Impacto do Tamanho do Modelo", "principal", q1, strategies) +
        toggle_section("q2", "6. Pergunta 2 — Diferença Média de Score", "", q2, strategies) +
        toggle_section("q3", "7. Pergunta 3 — A Diferença é Consistente?", "", q3, strategies)
    )


def _strategy_comparison_html(fi_exp1_dir):
    return section("strategy_comparison", "8. Impacto da Estratégia de Amostragem", "",
        '<div class="info">Comparação direta entre as duas estratégias de amostragem do Exp1 — '
        'mostra se a escolha uniform_bins vs. proportional_stratified afeta a distribuição de scores '
        'da amostra resultante.</div>' +
        row(full(img_tag(os.path.join(fi_exp1_dir, "sampling_strategy_comparison.png"),
                         "Uniforme vs. Estratificado Proporcional — distribuições sobrepostas")),
            full(img_tag(os.path.join(fi_exp1_dir, "sampling_strategy_comparison_table.png"),
                         "Diferença de distribuição (KS + Wasserstein + KL)"))))


def _legacy_validation_html(outputs_dir):
    val_dir = os.path.join(outputs_dir, "exp_legacy_validation")
    fig_dir = os.path.join(val_dir, "figures_validation")
    report_path = os.path.join(val_dir, "validation_report.txt")
    report_html = ""
    if os.path.exists(report_path):
        with open(report_path, encoding="utf-8") as rf:
            report_html = f'<figure><pre>{rf.read()}</pre></figure>'
    return section("legacy_validation", "Validação de Reprodutibilidade", "",
        '<div class="info">Re-executamos o pipeline novo com as mesmas 502 imagens e captions do '
        'experimento original do ICCC 2025 e comparamos os scores atributo a atributo — validando '
        'que o pipeline refatorado reproduz os resultados publicados.</div>' +
        row(img_tag(os.path.join(fig_dir, "validation_means.png"), "Scores lado a lado — legado vs. novo pipeline"),
            img_tag(os.path.join(fig_dir, "validation_diff.png"), "Diferença (novo − legado)")) +
        report_html)


def build_legacy(outputs_dir, reports_dir):
    fig_paper = os.path.join(reports_dir, "figures_paper")
    fi_shared = os.path.join(fig_paper, "shared")
    fi_iccc = os.path.join(fig_paper, "iccc")

    body = (
        _eda_html(fi_shared) +
        _sampling_html(fi_iccc, ["proportional_stratified"]) +
        _samples_html(outputs_dir, "exp0_iccc", ["proportional_stratified"]) +
        _questions_html(fi_iccc, ["proportional_stratified"]) +
        _legacy_validation_html(outputs_dir)
    )
    return page("Paper ICCC", "APDDv2-10023 · Amostra original do ICCC 2025 (exp0_iccc)",
                NAV, body, accent="#6c5ce7", active_href="Paper_iccc.html")


def build_exp1(outputs_dir, reports_dir):
    fig_paper = os.path.join(reports_dir, "figures_paper")
    fi_shared = os.path.join(fig_paper, "shared")
    fi_exp1 = os.path.join(fig_paper, "exp1")
    strategies = ["proportional_stratified", "uniform_bins"]

    body = (
        _eda_html(fi_shared) +
        _sampling_html(fi_exp1, strategies) +
        _samples_html(outputs_dir, "exp1_apdd", strategies) +
        _questions_html(fi_exp1, strategies) +
        _strategy_comparison_html(fi_exp1)
    )
    return page("Exp 1 — APDDv2", "Baseline APDDv2 · Human GT vs Janus-Pro-1B/7B · uniform_bins vs. proportional_stratified",
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
    parser.add_argument("--out", default=None,
                        help="Pasta de saída dos HTMLs (padrão: docs/ no repositório)")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    reports = cfg["paths"]["reports"]
    outputs = cfg["paths"]["outputs"]
    fi_iccc = os.path.join(reports, "figures_iccc")
    fi_fig  = os.path.join(reports, "figures")
    fi_smp  = os.path.join(reports, "samples")

    # HTMLs sempre vão para docs/ no repositório (GitHub Pages)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    html_out  = args.out or os.path.join(repo_root, "docs")
    os.makedirs(html_out, exist_ok=True)

    pages = {
        "index.html":                 build_index(reports),
        "Paper_iccc.html":            build_legacy(outputs, reports),
        "exp1_apdd.html":             build_exp1(outputs, reports),
        "exp2a_portinari.html":       build_exp2(fi_iccc, fi_fig, fi_smp, "a"),
        "exp2b_portinari_human.html": build_exp2(fi_iccc, fi_fig, fi_smp, "b"),
        "exp3_mnist.html":            build_exp3(fi_fig, fi_smp),
        "exp4_noise.html":            build_exp4(fi_fig, fi_smp),
        "exp5_temporal.html":         build_exp5(fi_fig, fi_smp),
    }

    for fname, html in pages.items():
        out = os.path.join(html_out, fname)
        with open(out, "w", encoding="utf-8") as f:
            f.write(html)
        size_kb = os.path.getsize(out) // 1024
        print(f"[html] OK {fname}  ({size_kb} KB)")

    print(f"\nHTMLs gerados em: {html_out}")


if __name__ == "__main__":
    main()
