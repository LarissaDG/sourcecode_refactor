"""
Traduz a coluna 'Descrição' do CSV do Portinari para inglês (→ Description_en).
Usa deep-translator (Google Tradutor) — sem necessidade de token ou modelo local.

Uso:
    python3 scripts/portinari_translate.py \
        --csv  data/portinari/acervoPortinari.csv \
        --out  data/portinari/MiniBasePortinari_Translated.csv \
        --n    500 \
        --seed 42

Flags:
    --csv    CSV de entrada (acervoPortinari.csv)
    --out    CSV de saída com coluna Description_en
    --n      Número de amostras (padrão: 500, 0 = todas)
    --seed   Semente para amostragem (padrão: 42)
    --batch  Tamanho do batch de tradução (padrão: 16)

Dependências:
    pip install deep-translator pandas
"""

import argparse
import time
from pathlib import Path

import pandas as pd
from deep_translator import GoogleTranslator


def translate_batch(texts: list[str], batch_size: int) -> list[str]:
    translator = GoogleTranslator(source="pt", target="en")
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        for text in batch:
            try:
                translated = translator.translate(text[:5000]) if text.strip() else ""
            except Exception as e:
                print(f"  [!] Erro ao traduzir: {e}. Mantendo original.")
                translated = text
            results.append(translated)
        print(f"  {min(i + batch_size, len(texts))}/{len(texts)} descrições traduzidas...")
        time.sleep(0.5)  # evita rate limit do Google
    return results


def main():
    parser = argparse.ArgumentParser(description="Traduz descrições do Portinari PT→EN")
    parser.add_argument("--csv",   required=True,         help="CSV de entrada")
    parser.add_argument("--out",   required=True,         help="CSV de saída")
    parser.add_argument("--n",     type=int, default=500, help="Número de amostras (0 = todas)")
    parser.add_argument("--seed",  type=int, default=42,  help="Semente (padrão: 42)")
    parser.add_argument("--batch", type=int, default=16,  help="Batch de tradução (padrão: 16)")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    print(f"CSV carregado: {len(df)} registros.")

    if "Descrição" not in df.columns:
        raise ValueError(f"Coluna 'Descrição' não encontrada. Colunas: {list(df.columns)}")

    if args.n > 0 and args.n < len(df):
        df = df.sample(n=args.n, random_state=args.seed).reset_index(drop=True)
        print(f"Amostrados {len(df)} registros (seed={args.seed}).")

    print(f"Traduzindo {len(df)} descrições via Google Tradutor...")
    textos = df["Descrição"].fillna("").astype(str).tolist()
    df["Description_en"] = translate_batch(textos, args.batch)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nFINALIZADO → {out}")
    print(f"  Colunas: {list(df.columns)}")


if __name__ == "__main__":
    main()
