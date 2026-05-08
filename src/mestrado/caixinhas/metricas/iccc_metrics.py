import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import scipy.stats as stats
from scipy.stats import pearsonr, spearmanr, ttest_ind, mannwhitneyu, f_oneway
import matplotlib.pyplot as plt


# -------------------------------
# 1) CARREGAR BASES
# -------------------------------
def load_datasets(file_original, file_new):
    df_original = pd.read_csv(file_original, encoding="latin1")
    df_new = pd.read_csv(file_new, encoding="utf-8")
    return df_original, df_new


# -------------------------------
# 2) DEFINIR COLUNAS DE INTERESSE
# -------------------------------
def get_columns_to_compare():
    return [
        "Total aesthetic score", "Theme and logic", "Creativity",
        "Layout and composition", "Space and perspective",
        "The sense of order", "Light and shadow", "Color",
        "Details and texture", "The overall", "Mood"
    ]


# -------------------------------
# 3) LIMPEZA E ALINHAMENTO DAS BASES
# -------------------------------
def preprocess_datasets(df_original, df_new, cols_to_compare):
    # Remover linhas zeradas na nova base
    df_new = df_new[~(df_new[cols_to_compare] == 0).all(axis=1)]

    # Resetar índices
    df_original = df_original.reset_index(drop=True)
    df_new = df_new.reset_index(drop=True)

    # Garantir NaN onde a base original não tem valor
    for col in cols_to_compare:
        if col in df_original.columns and col in df_new.columns:
            mask = df_original[col].isna()
            df_new.loc[mask, col] = np.nan

    return df_original, df_new


# -------------------------------
# 4) TESTE T ENTRE BASES
# -------------------------------
def run_ttests(df_original, df_new, cols_to_compare):
    results = {}
    for col in cols_to_compare:
        if df_original[col].notna().sum() > 0:
            results[col] = stats.ttest_ind(
                df_original[col].dropna(),
                df_new[col].dropna(),
                equal_var=False,
                nan_policy="omit"
            )
    return results


# -------------------------------
# 5) ANÁLISE DE CORRELAÇÃO
# -------------------------------
def correlation_analysis(df, dataset_name):
    df = df.dropna(subset=["The overall", "Mood"])

    pearson_corr, pearson_p = pearsonr(df["The overall"], df["Mood"])
    spearman_corr, spearman_p = spearmanr(df["The overall"], df["Mood"])

    print(f"{dataset_name} - Correlação de Pearson: {pearson_corr:.3f} (p={pearson_p:.3f})")
    print(f"{dataset_name} - Correlação de Spearman: {spearman_corr:.3f} (p={spearman_p:.3f})")


# -------------------------------
# 6) TESTES DE HIPÓTESE ENTRE BASES
# -------------------------------
def hypothesis_tests(df_original, df_new, cols_to_compare):
    print("----- Testes de Hipótese -----")

    # Correlação Overall vs Mood
    correlation_analysis(df_original, "Base Original")
    correlation_analysis(df_new, "Nova Base")

    # Testes t e Mann-Whitney
    for col in cols_to_compare:
        orig_col = df_original[col].dropna()
        new_col = df_new[col].dropna()

        if len(orig_col) > 1 and len(new_col) > 1:
            t_stat, t_p = ttest_ind(orig_col, new_col, equal_var=False, nan_policy="omit")
            mw_stat, mw_p = mannwhitneyu(orig_col, new_col)
            print(f"{col} - Teste t: p={t_p:.3f}, Mann-Whitney U: p={mw_p:.3f}")

    # ANOVA
    anova_p = f_oneway(
        *[df_new[col].dropna() for col in cols_to_compare if col in df_new.columns]
    )[1]
    print(f"ANOVA entre categorias na nova base: p={anova_p:.3f}")


# -------------------------------
# 7) VISUALIZAÇÕES GERAIS
# -------------------------------
def plot_distributions(df_original, df_new, cols_to_compare):
    fig1 = px.histogram(
        df_original[cols_to_compare].melt(),
        x="value",
        color="variable",
        title="Distribuição das Notas - Base Original"
    )

    fig2 = px.histogram(
        df_new[cols_to_compare].melt(),
        x="value",
        color="variable",
        title="Distribuição das Notas - Nova Base"
    )

    fig1.show()
    fig2.show()


# -------------------------------
# 8) VISUALIZAÇÕES TOTAL AESTHETIC SCORE
# -------------------------------
def plot_total_score_comparison(df_original, df_new):
    col = "Total aesthetic score"

    original_scores = df_original[col].dropna()
    new_scores = df_new[col].dropna()

    df_scores = pd.DataFrame({
        "Total Aesthetic Score": np.concatenate([original_scores, new_scores]),
        "Base": ["Original"] * len(original_scores) + ["Nova"] * len(new_scores)
    })

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.boxplot(x="Base", y="Total Aesthetic Score", data=df_scores, ax=axes[0])
    axes[0].set_title("Boxplot - Comparação do Total Aesthetic Score")

    sns.histplot(original_scores, color="blue", label="Original", kde=True, alpha=0.5, ax=axes[1])
    sns.histplot(new_scores, color="red", label="Nova", kde=True, alpha=0.5, ax=axes[1])
    axes[1].legend()
    axes[1].set_title("Histograma - Distribuição do Total Aesthetic Score")

    plt.tight_layout()
    plt.show()

    print(f"Média Original: {original_scores.mean():.3f}, Média Nova: {new_scores.mean():.3f}")
    print(f"Mediana Original: {original_scores.median():.3f}, Mediana Nova: {new_scores.median():.3f}")


# -------------------------------
# 9) SCATTER OVERALL VS MOOD
# -------------------------------
def plot_overall_vs_mood(df_original, df_new):
    sns.scatterplot(data=df_original, x="The overall", y="Mood", alpha=0.5, label="Base Original")
    sns.scatterplot(data=df_new, x="The overall", y="Mood", alpha=0.5, label="Nova Base")
    plt.title("Relação entre 'The overall' e 'Mood'")
    plt.legend()
    plt.show()


# -------------------------------
# 10) GRÁFICO DE RADAR
# -------------------------------
def plot_radar(df_original, df_new):
    cols = [
        "Theme and logic", "Creativity", "Layout and composition",
        "Space and perspective", "Light and shadow", "Color",
        "Details and texture", "The overall", "Mood"
    ]

    orig_means = [df_original[col].mean() for col in cols]
    new_means = [df_new[col].mean() for col in cols]

    angles = np.linspace(0, 2 * np.pi, len(cols), endpoint=False).tolist()
    orig_means += orig_means[:1]
    new_means += new_means[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.fill(angles, orig_means, alpha=0.25, label="Base Original")
    ax.fill(angles, new_means, alpha=0.25, label="Nova Base")
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(cols, fontsize=10)
    plt.title("Comparação das Médias das Categorias")
    plt.legend()
    plt.show()


# -------------------------------
# 11) EXECUÇÃO PRINCIPAL
# -------------------------------
def main():
    file_original = "/home_cerberus/disk3/larissa.gomide/APDDv2/APDDv2-10023.csv"
    file_new = "/home_cerberus/disk3/larissa.gomide/updated_sampled_dataset.csv"

    df_original, df_new = load_datasets(file_original, file_new)
    cols_to_compare = get_columns_to_compare()

    df_original, df_new = preprocess_datasets(df_original, df_new, cols_to_compare)

    run_ttests(df_original, df_new, cols_to_compare)
    hypothesis_tests(df_original, df_new, cols_to_compare)

    plot_distributions(df_original, df_new, cols_to_compare)
    plot_total_score_comparison(df_original, df_new)
    plot_overall_vs_mood(df_original, df_new)
    plot_radar(df_original, df_new)


if __name__ == "__main__":
    main()
