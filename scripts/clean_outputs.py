"""
Limpa os outputs da última execução do pipeline e diretórios de cache obsoletos.

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

CASA_DIR = Path("/sonic_home/larissa.gomide/casa")


def _delete(path: Path, dry_run: bool):
    size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    size_mb = size / (1024 * 1024)
    if dry_run:
        print(f"  [dry-run] Deletaria: {path}/ ({size_mb:.1f} MB)")
    else:
        shutil.rmtree(path)
        print(f"  Deletado: {path}/ ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Limpa outputs do pipeline")
    parser.add_argument("--exp",           default=None,        help="Experimento específico a limpar")
    parser.add_argument("--dry-run",       action="store_true", help="Só mostra, não deleta")
    parser.add_argument("--clean-legacy",  action="store_true", help="Deleta diretório legado casa/ se existir")
    args = parser.parse_args()

    # Limpa diretório legado 'casa' se solicitado
    if args.clean_legacy:
        if CASA_DIR.exists():
            print(f"Diretório legado encontrado: {CASA_DIR}")
            _delete(CASA_DIR, args.dry_run)
        else:
            print(f"Diretório legado não encontrado: {CASA_DIR}")

    outputs_dir = Path("/snfs1/speed/larissa.gomide/outputs")

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
        _delete(target, args.dry_run)

    if not args.dry_run:
        print("\nOutputs limpos. Próxima execução começa do zero.")


if __name__ == "__main__":
    main()
