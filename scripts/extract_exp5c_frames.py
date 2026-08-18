"""
Exp 5c — mede a duração real dos vídeos já baixados por download_timecraft.py
e extrai 1 frame/segundo, truncado ao mesmo tamanho em todos os vídeos, pra
uma análise de consistência temporal em nível macro (o vídeo inteiro, não só
os ~24 primeiros frames usados no Exp5a/5b).

Não baixa vídeo nenhum — reusa os .mp4 já baixados em <videos-dir> (a mesma
pasta que download_timecraft.py já preenche). Faz 2 passos:

  1. Mede a duração e o fps REAIS de cada vídeo (via metadados do próprio
     arquivo, com imageio/ffmpeg — não é mais estimativa a partir da lista
     framenums do .pkl). Grava esses números em duration_report.csv e
     imprime um resumo (mín/máx/média/mediana) ANTES de extrair qualquer
     frame, pra você conferir os números reais antes de confiar neles.
  2. Determina uma duração comum (por padrão, o mínimo real entre os vídeos
     medidos, com uma margem de segurança de 1s) e extrai 1 frame/segundo
     dessa duração em todos os vídeos — mesmo recorte/escala do
     download_timecraft.py (usa o crop_start_xy/crop_end_xy do .pkl do
     TimeCraft), pra ficar visualmente comparável ao Exp5a/5b.

Uso:
    python3 scripts/extract_exp5c_frames.py --videos-dir data/temporal/videos --out data/temporal_1fps

Flags:
    --videos-dir       Pasta com os .mp4 já baixados (padrão: data/temporal/videos)
    --out              Diretório de saída (frames/metadata.csv/duration_report.csv)
    --repo-dir         Onde está clonado o repositório timecraft (padrão: data/timecraft_repo)
    --duration-sec     Força uma duração comum específica, em vez de auto-detectar o mínimo real
    --margin-sec       Margem de segurança subtraída do mínimo real detectado (padrão: 1s)
    --scale            Fator de redimensionamento após o crop (padrão: 0.5, igual ao download_timecraft.py)
    --skip-measure     Pula a medição (reusa duration_report.csv já existente)
    --skip-extract     Só mede e grava duration_report.csv, não extrai frames

Dependências: as mesmas do download_timecraft.py (imageio, imageio-ffmpeg, pandas, pillow, numpy)
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import download_timecraft as dt  # reusa load_video_infos() (crop/pkl) sem duplicar


def measure_durations(videos_dir: Path) -> pd.DataFrame:
    """Lê fps/duração reais de cada .mp4 em videos_dir via metadados do
    próprio arquivo (imageio/ffmpeg) -- não é estimativa, é medição direta."""
    import imageio

    rows = []
    mp4s = sorted(videos_dir.glob("*.mp4"))
    for path in mp4s:
        vid_name = path.stem
        try:
            reader = imageio.get_reader(str(path), "ffmpeg")
            meta = reader.get_meta_data()
            reader.close()
            fps = float(meta.get("fps") or 0)
            duration = float(meta.get("duration") or 0)
        except Exception as e:
            print(f"  AVISO: não consegui ler metadados de {path.name}: {e}")
            fps, duration = 0.0, 0.0
        rows.append({"vid_name": vid_name, "fps_real": fps, "duration_sec": duration})
    return pd.DataFrame(rows)


def print_duration_summary(df: pd.DataFrame):
    valid = df[df["duration_sec"] > 0]
    if valid.empty:
        print("  Nenhum vídeo com duração válida medida.")
        return
    d = valid["duration_sec"]
    print(f"\n  {len(valid)}/{len(df)} vídeos com duração medida com sucesso.")
    print(f"  mínimo = {d.min():.1f}s ({d.min()/60:.2f} min)  -- {valid.loc[d.idxmin(), 'vid_name']}")
    print(f"  máximo = {d.max():.1f}s ({d.max()/60:.2f} min)  -- {valid.loc[d.idxmax(), 'vid_name']}")
    print(f"  média  = {d.mean():.1f}s ({d.mean()/60:.2f} min)")
    print(f"  mediana = {d.median():.1f}s ({d.median()/60:.2f} min)")
    print(f"  fps real: mín={valid['fps_real'].min():.2f}  máx={valid['fps_real'].max():.2f}  média={valid['fps_real'].mean():.2f}")


def frames_already_extracted(vid_name: str, n_frames: int, frames_dir: Path) -> bool:
    vid_dir = frames_dir / vid_name
    if not vid_dir.is_dir():
        return False
    existing = {p.name for p in vid_dir.glob("*.png")}
    expected = {f"{vid_name}_frame_{i:04d}.png" for i in range(n_frames)}
    return expected.issubset(existing)


def extract_1fps(video_path: Path, vid_name: str, fps_real: float, n_frames: int,
                  crop_info: dict, scale: float, frames_dir: Path) -> int:
    """Extrai 1 frame/segundo (0s, 1s, ..., n_frames-1 s), recorta pelo crop
    do .pkl (se disponível pra esse vid_name) e salva como
    <vid_name>_frame_XXXX.png (XXXX = segundo, 0..n_frames-1)."""
    import imageio
    from PIL import Image

    vid_dir = frames_dir / vid_name
    vid_dir.mkdir(parents=True, exist_ok=True)

    crop = crop_info.get(vid_name)
    saved = 0
    try:
        reader = imageio.get_reader(str(video_path), "ffmpeg")
        for t in range(n_frames):
            raw_idx = round(t * fps_real)
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
            img.save(vid_dir / f"{vid_name}_frame_{t:04d}.png")
            saved += 1
        reader.close()
    except Exception as e:
        print(f"  AVISO: erro ao extrair frames de {video_path.name}: {e}")
    return saved


def main():
    parser = argparse.ArgumentParser(description="Exp5c: mede duração real e extrai 1 fps truncado")
    parser.add_argument("--videos-dir",   default="data/temporal/videos", help="Pasta com os .mp4 já baixados")
    parser.add_argument("--out",          default="data/temporal_1fps",  help="Diretório de saída")
    parser.add_argument("--repo-dir",     default="data/timecraft_repo", help="Onde está clonado o timecraft (crop info)")
    parser.add_argument("--duration-sec", type=float, default=None,     help="Força duração comum (senão auto-detecta o mínimo real)")
    parser.add_argument("--margin-sec",   type=float, default=1.0,      help="Margem de segurança subtraída do mínimo real (padrão: 1s)")
    parser.add_argument("--scale",        type=float, default=0.5,      help="Fator de redimensionamento (padrão: 0.5)")
    parser.add_argument("--skip-measure", action="store_true",          help="Reusa duration_report.csv já existente")
    parser.add_argument("--skip-extract", action="store_true",          help="Só mede, não extrai frames")
    args = parser.parse_args()

    videos_dir = Path(args.videos_dir)
    out         = Path(args.out)
    frames_dir  = out / "frames"
    duration_csv = out / "duration_report.csv"
    meta_csv    = out / "metadata.csv"
    out.mkdir(parents=True, exist_ok=True)

    # ── 1. Medição real de duração/fps ───────────────────────────────────────
    if args.skip_measure and duration_csv.exists():
        print(f"Reusando {duration_csv} (--skip-measure).")
        dur_df = pd.read_csv(duration_csv)
    else:
        print(f"Medindo duração real dos vídeos em {videos_dir}...")
        dur_df = measure_durations(videos_dir)
        dur_df.to_csv(duration_csv, index=False)
        print(f"Medidos {len(dur_df)} vídeos -> {duration_csv}")

    print_duration_summary(dur_df)

    if args.skip_extract:
        print("\n--skip-extract: parando aqui.")
        return

    # ── 2. Duração comum e extração 1 fps ────────────────────────────────────
    valid = dur_df[dur_df["duration_sec"] > 0]
    if valid.empty:
        print("Nenhum vídeo com duração válida -- não há o que extrair.")
        return

    if args.duration_sec is not None:
        common_sec = args.duration_sec
        print(f"\nDuração comum forçada por --duration-sec: {common_sec:.1f}s")
    else:
        common_sec = max(1.0, valid["duration_sec"].min() - args.margin_sec)
        print(f"\nDuração comum auto-detectada (mínimo real - {args.margin_sec:.0f}s de margem): {common_sec:.1f}s")

    n_frames = int(common_sec)
    if n_frames < 10:
        print(f"  AVISO: duração comum de só {n_frames} frames (~{n_frames}s) -- "
              f"confira se algum vídeo muito curto está distorcendo o mínimo antes de prosseguir.")
    print(f"  -> {n_frames} frames (1/s) por vídeo.")

    print("\nCarregando crop info do TimeCraft (.pkl)...")
    infos = dt.load_video_infos(Path(args.repo_dir))
    crop_info = {i["vid_name"]: i for i in infos}

    rows = []
    for _, row in valid.iterrows():
        vid_name = row["vid_name"]
        fps_real = row["fps_real"] or 24.0
        video_path = videos_dir / f"{vid_name}.mp4"
        if not video_path.exists():
            continue

        if frames_already_extracted(vid_name, n_frames, frames_dir):
            n_saved = n_frames
            status = "ok_ja_existia"
        else:
            n_saved = extract_1fps(video_path, vid_name, fps_real, n_frames, crop_info, args.scale, frames_dir)
            status = "ok" if n_saved > 0 else "sem_frames"

        rows.append({
            "vid_name": vid_name, "fps_real": fps_real, "duration_sec": row["duration_sec"],
            "common_duration_sec": common_sec, "n_frames_extraidos": n_saved, "status": status,
        })
        pd.DataFrame(rows).to_csv(meta_csv, index=False)
        print(f"  [{status}] {vid_name}: {n_saved}/{n_frames} frames")

    n_ok = sum(1 for r in rows if r["status"] in ("ok", "ok_ja_existia"))
    print(f"\nFINALIZADO: {n_ok}/{len(valid)} vídeos com frames extraídos com sucesso.")
    print(f"  {frames_dir}/    <- frames PNG (subpasta por vid_name, {n_frames} frames cada, 1/s)")
    print(f"  {meta_csv}       <- metadados da extração")
    print(f"  {duration_csv}   <- duração/fps reais medidos (passo 1)")


if __name__ == "__main__":
    main()
