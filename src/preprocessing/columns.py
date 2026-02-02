import pandas as pd
import numpy as np

def add_score_columns(df, cols_to_compare):
    df = df.copy()

    df["Total Score"] = df[cols_to_compare].sum(axis=1)
    df["Avg Score"] = df["Total Score"] / len(cols_to_compare)

    return df

# -------------------------------
# 3) INICIALIZAR NOVAS COLUNAS
# -------------------------------
def initialize_columns(df: pd.DataFrame):
    new_columns = [
        'Total aesthetic score', 'Theme and logic', 'Creativity', 'Layout and composition',
        'Space and perspective', 'The sense of order', 'Light and shadow', 'Color',
        'Details and texture', 'The overall', 'Mood', 'generated_filename', 'Description'
    ]
    for col in new_columns:
        df[col] = ''  # Inicializando com valores vazios
    return df
