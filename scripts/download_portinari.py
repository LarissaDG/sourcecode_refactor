"""
Baixa as imagens do Portinari (ZIPs via Google Drive) e o CSV do acervo (Google Sheets).
Após extrair, verifica se cada linha do CSV tem uma imagem correspondente.

Uso:
    python3 scripts/download_portinari.py --out /sonic_home/larissa.gomide/data/portinari

Flags disponíveis:
    --skip-images   só baixa o CSV
    --skip-csv      só baixa as imagens
    --keep-zips     mantém os ZIPs após extração
    --verify-only   só verifica consistência (sem baixar nada)

Dependências:
    pip install gdown requests pandas
"""

import argparse
import os
import re
import shutil
import sys
import zipfile
from pathlib import Path

import pandas as pd
import requests

SHEET_ID   = "1t3LdHHJHTXoTbx7TWIiH07gtnjtySgQUMUWngqE3qvw"
SHEET_GID  = "1640515601"
CSV_EXPORT = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={SHEET_GID}"
LINKS_FILE = Path(__file__).parent / "portinari_links.txt"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}


def extrair_id(url: str):
    m = re.search(r"/d/([A-Za-z0-9_-]+)/", url)
    return m.group(1) if m else None


def download_csv(out_dir: Path) -> Path:
    csv_path = out_dir / "acervoPortinari.csv"
    print(f"Baixando CSV → {csv_path}")
    r = requests.get(CSV_EXPORT)
    r.raise_for_status()
    csv_path.write_bytes(r.content)
    df = pd.read_csv(csv_path)
    print(f"  {len(df)} registros, {len(df.columns)} colunas.")
    return csv_path


def download_zips(zip_dir: Path, done_file: Path) -> list[Path]:
    if not LINKS_FILE.exists():
        raise FileNotFoundError(f"Arquivo de links não encontrado: {LINKS_FILE}")

    with open(LINKS_FILE) as f:
        links = [l.strip() for l in f if l.strip()]

    ids = []
    for link in links:
        gid = extrair_id(link)
        if gid:
            ids.append(gid)
        else:
            print(f"  AVISO: link ignorado (sem ID): {link}")

    print(f"Baixando {len(ids)} ZIPs → {zip_dir}")
    zip_dir.mkdir(parents=True, exist_ok=True)

    # done_file fica fora de zip_dir para sobreviver ao rmtree
    done = set(done_file.read_text().splitlines()) if done_file.exists() else set()

    falhas = []
    for i, gid in enumerate(ids, 1):
        if gid in done:
            print(f"  [{i}/{len(ids)}] já baixado: {gid}")
            continue

        print(f"  [{i}/{len(ids)}] {gid}")
        ret = os.system(f'gdown "{gid}" --output "{zip_dir}/"')
        if ret != 0:
            print(f"  AVISO: falha ao baixar ID {gid}")
            falhas.append(gid)
        else:
            done.add(gid)
            done_file.write_text("\n".join(sorted(done)))

    if falhas:
        import time
        print(f"\n  {len(falhas)} falha(s). Aguardando 60s e tentando novamente...")
        time.sleep(60)
        still_failing = []
        for gid in falhas:
            ret = os.system(f'gdown "{gid}" --output "{zip_dir}/"')
            if ret != 0:
                print(f"  ERRO persistente: {gid}")
                still_failing.append(gid)
            else:
                done.add(gid)
                done_file.write_text("\n".join(sorted(done)))
        if still_failing:
            print(f"\n  {len(still_failing)} ZIP(s) não baixado(s): {still_failing}")

    zips = list(zip_dir.glob("*.zip"))
    print(f"  {len(zips)} arquivo(s) ZIP baixado(s).")
    return zips


def extract_zips(zips: list[Path], images_dir: Path) -> int:
    images_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = images_dir.parent / "_temp_extract"
    total = 0

    for zp in zips:
        print(f"Extraindo {zp.name}...")
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir()

        try:
            with zipfile.ZipFile(zp, "r") as z:
                z.extractall(temp_dir)
        except Exception as e:
            print(f"  AVISO: falha ao extrair {zp.name}: {e}")
            continue

        copiadas = 0
        for src in temp_dir.rglob("*"):
            if src.is_file() and src.suffix.lower() in IMAGE_EXTS:
                dst = images_dir / src.name
                if not dst.exists():
                    shutil.copy2(src, dst)
                    copiadas += 1
        print(f"  {copiadas} imagens copiadas.")
        total += copiadas

    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    return total


def verify(csv_path: Path, images_dir: Path):
    print("\nVerificando consistência CSV ↔ imagens...")
    df = pd.read_csv(csv_path)

    if "Numero da Obra" not in df.columns:
        raise ValueError(f"Coluna 'Numero da Obra' não encontrada. Colunas: {list(df.columns)}")

    esperados   = df["Numero da Obra"].astype(str).tolist()
    encontradas = {p.name for p in images_dir.iterdir() if p.is_file()}

    faltando = [n for n in esperados if n not in encontradas]
    extras   = [n for n in encontradas if n not in set(esperados)]

    print(f"  Registros no CSV : {len(esperados)}")
    print(f"  Imagens na pasta : {len(encontradas)}")
    print(f"  Faltando         : {len(faltando)}")
    print(f"  Extras (sem CSV) : {len(extras)}")

    if faltando:
        print("\nERRO: as seguintes imagens estão no CSV mas não foram encontradas:")
        for nome in faltando[:20]:
            print(f"  {nome}")
        if len(faltando) > 20:
            print(f"  ... e mais {len(faltando) - 20}.")
        sys.exit(1)

    print("  OK — todas as imagens do CSV foram encontradas.")


def main():
    parser = argparse.ArgumentParser(description="Download Portinari dataset")
    parser.add_argument("--out",         required=True,       help="Diretório de saída")
    parser.add_argument("--skip-images", action="store_true", help="Só baixa o CSV")
    parser.add_argument("--skip-csv",    action="store_true", help="Só baixa as imagens")
    parser.add_argument("--keep-zips",   action="store_true", help="Mantém os ZIPs após extração")
    parser.add_argument("--verify-only", action="store_true", help="Só verifica consistência")
    args = parser.parse_args()

    out        = Path(args.out)
    zip_dir    = out / "_zips"
    images_dir = out / "Imagens"
    csv_path   = out / "acervoPortinari.csv"
    out.mkdir(parents=True, exist_ok=True)

    if args.verify_only:
        verify(csv_path, images_dir)
        return

    if not args.skip_csv:
        csv_path = download_csv(out)

    if not args.skip_images:
        done_file = out / ".downloaded_ids"   # fora de zip_dir — sobrevive ao rmtree
        zips  = download_zips(zip_dir, done_file)
        total = extract_zips(zips, images_dir)
        print(f"\nTotal de imagens extraídas: {total}")
        if not args.keep_zips:
            shutil.rmtree(zip_dir)
            print("ZIPs temporários removidos (progresso salvo em .downloaded_ids).")

    if not args.skip_csv and not args.skip_images:
        verify(csv_path, images_dir)

    print("\nFINALIZADO")
    print(f"  {csv_path}")
    print(f"  {images_dir}/")


if __name__ == "__main__":
    main()
