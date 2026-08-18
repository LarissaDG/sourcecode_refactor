"""
Exp 5d — consistência temporal em nível MICRO: 10 sub-amostras aleatórias de
3 segundos por vídeo (30 frames/vídeo no total, a 1 frame/segundo), em vez da
janela contínua truncada do Exp5c.

Diferente do Exp5c, as 10 janelas de 3s são escolhidas aleatoriamente ao
longo de TODO o vídeo (não só o começo), mas SEM embaralhar a ordem: cada
frame extraído guarda seu índice/segundo REAL no vídeo original (com
lacunas entre as janelas), pra permitir avaliar a consistência ao longo do
tempo mesmo com amostragem esparsa. Não baixa vídeo nenhum — reusa os .mp4
já baixados por download_timecraft.py, igual ao Exp5c.

Mede a duração/fps reais por conta própria (não depende do resultado do job
do Exp5c) — pode rodar em paralelo com ele, ou mesmo antes. Se um vídeo não
tiver 30s (10 janelas de 3s) de duração real, usa quantas janelas não
sobrepostas couberem nele (menos de 10) em vez de pular o vídeo inteiro;
vídeos com menos de 3s (nem 1 janela cabe) são pulados e ficam registrados
no metadata.csv, mas não interrompem os demais.

A escolha das janelas é determinística por vídeo (seed = --seed + nome do
vídeo), então rodar de novo escolhe exatamente as mesmas janelas — resumível
sem precisar persistir a escolha em separado.

Uso:
    python3 scripts/extract_exp5d_frames.py --videos-dir data/temporal/videos --out data/temporal_micro

Flags:
    --videos-dir     Pasta com os .mp4 já baixados (padrão: data/temporal/videos)
    --out            Diretório de saída (frames/metadata.csv/duration_report.csv)
    --repo-dir       Onde está clonado o repositório timecraft (padrão: data/timecraft_repo)
    --block-sec      Duração de cada sub-amostra, em segundos (padrão: 3)
    --n-blocks       Quantas sub-amostras por vídeo, quando couberem (padrão: 10)
    --seed           Semente pra escolha determinística das janelas (padrão: 42)
    --scale          Fator de redimensionamento após o crop (padrão: 0.5)
    --skip-measure   Reusa duration_report.csv já existente
    --skip-extract   Só mede e grava duration_report.csv, não extrai frames
"""

import argparse
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import download_timecraft as dt
import extract_exp5c_frames as e5c  # reusa measure_durations() / print_duration_summary()


def pick_blocks(duration_sec: float, block_sec: int, n_blocks_wanted: int, seed_key: str) -> list[int]:
    """Escolhe até n_blocks_wanted índices de bloco de block_sec segundos,
    sem sobreposição, dentro de [0, duration_sec). Determinístico por
    seed_key (ex: "42_video_nome"). Devolve lista ORDENADA (ordem temporal),
    não embaralhada."""
    n_available = int(duration_sec // block_sec)
    if n_available <= 0:
        return []
    rng = random.Random(seed_key)
    k = min(n_blocks_wanted, n_available)
    chosen = rng.sample(range(n_available), k=k)
    return sorted(chosen)


def blocks_already_extracted(vid_name: str, blocks: list[int], block_sec: int, frames_dir: Path) -> bool:
    vid_dir = frames_dir / vid_name
    if not vid_dir.is_dir():
        return False
    existing = {p.name for p in vid_dir.glob("*.png")}
    expected = {
        f"{vid_name}_frame_{block * block_sec + k:04d}.png"
        for block in blocks for k in range(block_sec)
    }
    return expected.issubset(existing)


def extract_blocks(video_path: Path, vid_name: str, fps_real: float, blocks: list[int], block_sec: int,
                    crop_info: dict, scale: float, frames_dir: Path) -> int:
    """Extrai block_sec frames (1/s) de cada bloco escolhido, nomeando pelo
    segundo REAL no vídeo original (com lacunas entre blocos não-adjacentes).
    """
    import imageio
    from PIL import Image

    vid_dir = frames_dir / vid_name
    vid_dir.mkdir(parents=True, exist_ok=True)
    crop = crop_info.get(vid_name)

    saved = 0
    try:
        reader = imageio.get_reader(str(video_path), "ffmpeg")
        for block in blocks:
            for k in range(block_sec):
                true_second = block * block_sec + k
                raw_idx = round(true_second * fps_real)
                try:
                    frame = reader.get_data(raw_idx)
                except Exception:
                    continue
                img = Image.fromarray(frame).convert("RGB")
                if crop is not None:
                    x0, y0 = crop["crop_start_xy"]
                    x1, y1 = crop["crop_end_xy"]
                    x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
                    if x1 > x0 and y1 > y0:
                        img = img.crop((x0, y0, x1, y1))
                if scale != 1.0:
                    img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))))
                img.save(vid_dir / f"{vid_name}_frame_{true_second:04d}.png")
                saved += 1
        reader.close()
    except Exception as e:
        print(f"  AVISO: erro ao extrair frames de {video_path.name}: {e}")
    return saved


def main():
    parser = argparse.ArgumentParser(description="Exp5d: 10 sub-amostras aleatórias de 3s por vídeo, sem shuffle")
    parser.add_argument("--videos-dir",   default="data/temporal/videos",   help="Pasta com os .mp4 já baixados")
    parser.add_argument("--out",          default="data/temporal_micro",   help="Diretório de saída")
    parser.add_argument("--repo-dir",     default="data/timecraft_repo",   help="Onde está clonado o timecraft (crop info)")
    parser.add_argument("--block-sec",    type=int, default=3,             help="Duração de cada sub-amostra, em segundos")
    parser.add_argument("--n-blocks",     type=int, default=10,            help="Sub-amostras por vídeo, quando couberem")
    parser.add_argument("--seed",         type=int, default=42,           help="Semente pra escolha determinística")
    parser.add_argument("--scale",        type=float, default=0.5,        help="Fator de redimensionamento")
    parser.add_argument("--skip-measure", action="store_true",            help="Reusa duration_report.csv já existente")
    parser.add_argument("--skip-extract", action="store_true",            help="Só mede, não extrai frames")
    args = parser.parse_args()

    videos_dir  = Path(args.videos_dir)
    out         = Path(args.out)
    frames_dir  = out / "frames"
    duration_csv = out / "duration_report.csv"
    meta_csv    = out / "metadata.csv"
    out.mkdir(parents=True, exist_ok=True)

    # ── 1. Medição real de duração/fps (independente do Exp5c) ─────────────
    if args.skip_measure and duration_csv.exists():
        print(f"Reusando {duration_csv} (--skip-measure).")
        dur_df = pd.read_csv(duration_csv)
    else:
        print(f"Medindo duração real dos vídeos em {videos_dir}...")
        dur_df = e5c.measure_durations(videos_dir)
        dur_df.to_csv(duration_csv, index=False)
        print(f"Medidos {len(dur_df)} vídeos -> {duration_csv}")

    e5c.print_duration_summary(dur_df)

    if args.skip_extract:
        print("\n--skip-extract: parando aqui.")
        return

    valid = dur_df[dur_df["duration_sec"] > 0]
    if valid.empty:
        print("Nenhum vídeo com duração válida -- não há o que extrair.")
        return

    min_needed = args.block_sec * args.n_blocks
    print(f"\nMeta: até {args.n_blocks} janelas de {args.block_sec}s ({min_needed}s) por vídeo, "
          f"sem sobreposição, escolhidas aleatoriamente (sem shuffle na ordem final).")

    print("\nCarregando crop info do TimeCraft (.pkl)...")
    infos = dt.load_video_infos(Path(args.repo_dir))
    crop_info = {i["vid_name"]: i for i in infos}

    rows = []
    for _, row in valid.iterrows():
        vid_name = row["vid_name"]
        fps_real = row["fps_real"] or 24.0
        duration = row["duration_sec"]
        video_path = videos_dir / f"{vid_name}.mp4"
        if not video_path.exists():
            continue

        blocks = pick_blocks(duration, args.block_sec, args.n_blocks, seed_key=f"{args.seed}_{vid_name}")
        if not blocks:
            rows.append({
                "vid_name": vid_name, "fps_real": fps_real, "duration_sec": duration,
                "n_blocks_disponiveis": 0, "n_blocks_usados": 0, "blocos": "",
                "n_frames_extraidos": 0, "status": "video_curto_demais",
            })
            pd.DataFrame(rows).to_csv(meta_csv, index=False)
            print(f"  [video_curto_demais] {vid_name}: {duration:.1f}s, não cabe nem 1 janela de {args.block_sec}s")
            continue

        n_expected = len(blocks) * args.block_sec
        if blocks_already_extracted(vid_name, blocks, args.block_sec, frames_dir):
            n_saved = n_expected
            status = "ok_ja_existia"
        else:
            n_saved = extract_blocks(video_path, vid_name, fps_real, blocks, args.block_sec,
                                      crop_info, args.scale, frames_dir)
            status = "ok" if n_saved > 0 else "sem_frames"

        n_available = int(duration // args.block_sec)
        rows.append({
            "vid_name": vid_name, "fps_real": fps_real, "duration_sec": duration,
            "n_blocks_disponiveis": n_available, "n_blocks_usados": len(blocks),
            "blocos": ",".join(str(b) for b in blocks),
            "n_frames_extraidos": n_saved, "status": status,
        })
        pd.DataFrame(rows).to_csv(meta_csv, index=False)
        flag = "" if len(blocks) == args.n_blocks else f" (só {len(blocks)}/{args.n_blocks} coube)"
        print(f"  [{status}] {vid_name}: {n_saved}/{n_expected} frames em {len(blocks)} janelas{flag}")

    df = pd.DataFrame(rows)
    n_ok = int(df["status"].isin(["ok", "ok_ja_existia"]).sum()) if len(df) else 0
    n_full = int((df["n_blocks_usados"] == args.n_blocks).sum()) if len(df) else 0
    print(f"\nFINALIZADO: {n_ok}/{len(valid)} vídeos com frames extraídos "
          f"({n_full} com as {args.n_blocks} janelas completas).")
    print(f"  {frames_dir}/    <- frames PNG (subpasta por vid_name, índice = segundo real no vídeo)")
    print(f"  {meta_csv}       <- metadados da extração (blocos usados por vídeo)")
    print(f"  {duration_csv}   <- duração/fps reais medidos (passo 1)")


if __name__ == "__main__":
    main()
