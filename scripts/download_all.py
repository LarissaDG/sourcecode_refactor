"""
Orquestrador de download de todos os datasets do pipeline.

Uso:
    python3 scripts/download_all.py --out data/
    python3 scripts/download_all.py --out data/ --only portinari
    python3 scripts/download_all.py --out data/ --skip-temporal

Datasets:
    portinari   → data/portinari/   (download_portinari.py)
    mnist       → data/mnist/       (download_mnist.py)
    temporal    → data/temporal/    (download_temporal.py — requer yt-dlp)
    apddv2      → manual (link de download não está mais disponível)

Flags:
    --out              Diretório raiz de saída
    --only <nome>      Baixa só este dataset (portinari | mnist | temporal)
    --skip-portinari   Pula o Portinari
    --skip-mnist       Pula o MNIST
    --skip-temporal    Pula o Temporal
"""

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parent


def run(cmd: list[str], label: str):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\nERRO ao executar: {label}")
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(description="Download de todos os datasets")
    parser.add_argument("--out",            required=True,       help="Diretório raiz de saída")
    parser.add_argument("--only",           default=None,        help="Baixa só este dataset")
    parser.add_argument("--skip-portinari", action="store_true", help="Pula o Portinari")
    parser.add_argument("--skip-mnist",     action="store_true", help="Pula o MNIST")
    parser.add_argument("--skip-temporal",  action="store_true", help="Pula o Temporal")
    args = parser.parse_args()

    out = Path(args.out)

    do_portinari = not args.skip_portinari and (args.only in (None, "portinari"))
    do_mnist     = not args.skip_mnist     and (args.only in (None, "mnist"))
    do_temporal  = not args.skip_temporal  and (args.only in (None, "temporal"))

    if do_portinari:
        run(
            [sys.executable, str(SCRIPTS_DIR / "download_portinari.py"),
             "--out", str(out / "portinari")],
            "Portinari — imagens + CSV"
        )

    if do_mnist:
        run(
            [sys.executable, str(SCRIPTS_DIR / "download_mnist.py"),
             "--out", str(out / "mnist")],
            "MNIST — 500 amostras"
        )

    if do_temporal:
        run(
            [sys.executable, str(SCRIPTS_DIR / "download_temporal.py"),
             "--out", str(out / "temporal")],
            "Temporal — vídeos @ArtsyLolaCo + frames"
        )

    print("\n" + "="*60)
    print("DOWNLOADS FINALIZADOS")
    print("="*60)
    if do_portinari: print(f"  {out}/portinari/")
    if do_mnist:     print(f"  {out}/mnist/")
    if do_temporal:  print(f"  {out}/temporal/")

    print("""
APDDv2 — download manual necessário:
  O link público das imagens não está mais disponível.
  Repositório: https://github.com/BestiVictory/APDDv2
  Use uma cópia local e aponte dataset.path no YAML do exp1/exp4.
""")


if __name__ == "__main__":
    main()
