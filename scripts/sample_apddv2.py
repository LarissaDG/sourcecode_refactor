"""
sample_apddv2.py — Gera a amostra do APDDv2 para uso nos experimentos.

Uso:
    python scripts/sample_apddv2.py --config configs/analysis_local.yaml

Métodos disponíveis (definidos em sampling.method no YAML):
    gaussian  — proporcional por bins (pd.cut + frac = n_target/N)
    uniform   — fixo por bin         (pd.cut + n = floor(n_target/num_bins))
"""

import argparse
import os

import pandas as pd
import yaml


ATTR_COLS = [
    "Theme and logic", "Creativity", "Layout and composition",
    "Space and perspective", "The sense of order", "Light and shadow",
    "Color", "Details and texture", "The overall", "Mood",
]


def load_cfg(path):
    with open(path, encoding="utf-8-sig") as f:
        return yaml.safe_load(f)


def compute_avg(df):
    cols = [c for c in ATTR_COLS if c in df.columns]
    df = df.copy()
    df["_avg"] = df[cols].mean(axis=1, skipna=True)
    return df.dropna(subset=["_avg"]).copy()


def sample_gaussian(df, num_bins, n_target, random_state):
    """Proporcional: cada bin contribui com frac = n_target/N das suas observações."""
    frac = n_target / len(df)
    df["_bin"] = pd.cut(df["_avg"], bins=num_bins)
    sampled = (
        df.groupby("_bin", group_keys=False)
          .apply(lambda x: x.sample(frac=frac, replace=False, random_state=random_state))
          .reset_index(drop=True)
    )
    return sampled.drop(columns=["_avg", "_bin"])


def sample_uniform(df, num_bins, n_target, random_state):
    """Uniforme: cada bin contribui com k = floor(n_target/num_bins) observações."""
    k = n_target // num_bins
    df["_bin"] = pd.cut(df["_avg"], bins=num_bins)
    sampled = (
        df.groupby("_bin", group_keys=False)
          .apply(lambda x: x.sample(n=min(k, len(x)), replace=False, random_state=random_state))
          .reset_index(drop=True)
    )
    return sampled.drop(columns=["_avg", "_bin"])


def main():
    parser = argparse.ArgumentParser(description="Gera amostra do APDDv2")
    parser.add_argument("--config", default="configs/analysis_local.yaml")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    s   = cfg.get("sampling", {})

    apddv2_csv   = cfg["paths"].get("apddv2_csv", "")
    method       = s.get("method",       "gaussian")
    num_bins     = s.get("num_bins",     10)
    n_target     = s.get("n_target",     500)
    random_state = s.get("random_state", 42)
    output_csv   = s.get("output_csv",   "")

    if not apddv2_csv or not os.path.exists(apddv2_csv):
        print(f"[erro] APDDv2 não encontrado: {apddv2_csv}")
        print("       Verifique paths.apddv2_csv no YAML.")
        return

    if not output_csv:
        print("[erro] sampling.output_csv não definido no YAML.")
        return

    print(f"Carregando APDDv2: {apddv2_csv}")
    df = pd.read_csv(apddv2_csv, encoding="latin1")
    print(f"  Total: {len(df):,} imagens")

    df = compute_avg(df)
    print(f"  Com _avg válido: {len(df):,} imagens")
    print(f"  Intervalo avg score: [{df['_avg'].min():.3f}, {df['_avg'].max():.3f}]")

    print(f"\nMétodo: {method} | num_bins={num_bins} | n_target={n_target} | seed={random_state}")

    if method == "gaussian":
        sampled = sample_gaussian(df, num_bins, n_target, random_state)
    elif method == "uniform":
        sampled = sample_uniform(df, num_bins, n_target, random_state)
    else:
        print(f"[erro] Método desconhecido: '{method}'. Use 'gaussian' ou 'uniform'.")
        return

    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    sampled.to_csv(output_csv, index=False, encoding="utf-8")

    print(f"\n✓ Amostra salva: {output_csv}")
    print(f"  n = {len(sampled)} imagens")
    print(f"  avg score — média: {sampled[[c for c in ATTR_COLS if c in sampled.columns]].mean(axis=1).mean():.3f}")


if __name__ == "__main__":
    main()
