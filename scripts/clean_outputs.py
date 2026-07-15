"""
Limpa os outputs da última execução do pipeline.

Uso:
    python3 scripts/clean_outputs.py                    # limpa outputs/ inteiro
    python3 scripts/clean_outputs.py --exp exp1_apdd   # limpa só um experimento
    python3 scripts/clean_outputs.py --dry-run          # mostra o que seria deletado

Flags:
    --exp       Nome do experimento (pasta dentro de outputs/)
    --dry-run   Mostra o que seria deletado sem deletar nada
"""

import argparse
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Limpa outputs do pipeline")
    parser.add_argument("--exp",     default=None,        help="Experimento específico a limpar")
    parser.add_argument("--dry-run", action="store_true", help="Só mostra, não deleta")
    args = parser.parse_args()

    outputs_dir = Path("outputs")

    if not outputs_dir.exists():
        print("Pasta outputs/ não encontrada. Nada a limpar.")
        return

    if args.exp:
        targets = [outputs_dir / args.exp]
    else:
        targets = [p for p in outputs_dir.iterdir() if p.is_dir()]

    if not targets:
        print("Nenhum output encontrado.")
        return

    for target in targets:
        if not target.exists():
            print(f"  Não encontrado: {target}")
            continue
        size = sum(f.stat().st_size for f in target.rglob("*") if f.is_file())
        size_mb = size / (1024 * 1024)
        if args.dry_run:
            print(f"  [dry-run] Deletaria: {target}/ ({size_mb:.1f} MB)")
        else:
            shutil.rmtree(target)
            print(f"  Deletado: {target}/ ({size_mb:.1f} MB)")

    if not args.dry_run:
        print("\nOutputs limpos. Próxima execução começa do zero.")


if __name__ == "__main__":
    main()
