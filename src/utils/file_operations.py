import pandas as pd 
import zipfile
from pathlib import Path

# -------------------------------
# DESCOMPACTAR ZIP
# -------------------------------
def unzip_file(zip_path: str, unzip_dir: str):
    print("Descompactando o arquivo ZIP...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(unzip_dir)
    print("Descompactação concluída.")

# -------------------------------
# CARREGAR CSV
# -------------------------------
def load_csv(csv_path: str) -> pd.DataFrame:
    print(f"Carregando o arquivo CSV: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
        print("Arquivo CSV carregado com sucesso.")
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding="latin1")
        #df = pd.read_csv(csv_path, encoding='ISO-8859-1')
    return df

# -------------------------------
# SALVAR CSV
# -------------------------------
def save_csv(df: pd.DataFrame, output_csv_path: str):
    df.to_csv(output_csv_path, index=False)
    print(f"Arquivo CSV atualizado salvo em {output_csv_path}")
