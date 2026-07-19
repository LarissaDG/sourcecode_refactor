"""
Baixa vídeos do canal @ArtsyLolaCo (YouTube Shorts), filtra por título,
amostra 500, extrai frames (1/seg) e salva metadados em CSV.

Uso:
    python3 scripts/download_temporal.py --out data/temporal

Flags:
    --out        Diretório de saída
    --n          Número de vídeos a amostrar (padrão: 500)
    --seed       Semente aleatória (padrão: 42)
    --skip-download  Só extrai frames de vídeos já baixados
    --skip-frames    Só baixa vídeos, sem extrair frames

Dependências:
    pip install yt-dlp imageio imageio-ffmpeg pandas tqdm
"""

import argparse
import csv
import json
import os
import random
import re
import subprocess
import unicodedata
from pathlib import Path

import pandas as pd
from tqdm import tqdm

CHANNEL_URL = "https://www.youtube.com/@ArtsyLolaCo/shorts"

EXCLUDE_WORDS = [
    "tips", "full tutorial", "available now", "shop", "sale",
    "discount", "merch", "collab", "sponsor", "ad", "review",
    "unboxing", "haul", "vlog", "q&a", "announcement",
]


def slugify(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-zA-Z0-9_-]+", "_", text)
    return text.strip("_") or "video"


def should_exclude(title: str) -> bool:
    title_lower = title.lower()
    return any(word in title_lower for word in EXCLUDE_WORDS)


def list_channel_videos() -> list[dict]:
    print("Listando vídeos do canal...")
    result = subprocess.run(
        ["yt-dlp", "-j", "--flat-playlist", CHANNEL_URL],
        capture_output=True, text=True
    )
    videos = []
    for line in result.stdout.splitlines():
        try:
            data = json.loads(line)
            vid_id = data.get("id")
            title  = data.get("title", "")
            if vid_id and not should_exclude(title):
                videos.append({"video_id": vid_id, "title": title})
        except json.JSONDecodeError:
            continue
    print(f"  {len(videos)} vídeos após filtro de título.")
    return videos


def fetch_metadata(video_id: str) -> dict:
    result = subprocess.run(
        ["yt-dlp", "--print", "%(duration)s\t%(upload_date)s\t%(view_count)s\t%(like_count)s\t%(title)s",
         f"https://www.youtube.com/watch?v={video_id}"],
        capture_output=True, text=True
    )
    line = result.stdout.strip()
    if not line:
        return {}
    parts = line.split("\t")
    if len(parts) < 5:
        return {}
    return {
        "duration_sec": parts[0],
        "upload_date":  parts[1],
        "view_count":   parts[2],
        "like_count":   parts[3],
        "title":        parts[4],
    }


def download_video(video_id: str, out_dir: Path, index: int) -> Path | None:
    out_path = out_dir / f"{index:04d}.mp4"
    if out_path.exists():
        return out_path
    result = subprocess.run([
        "yt-dlp",
        "-o", str(out_path),
        "--merge-output-format", "mp4",
        f"https://www.youtube.com/watch?v={video_id}"
    ])
    return out_path if result.returncode == 0 else None


def extract_frames(video_path: Path, frames_dir: Path):
    import imageio
    frames_dir.mkdir(parents=True, exist_ok=True)

    try:
        reader = imageio.get_reader(str(video_path), "ffmpeg")
        meta = reader.get_meta_data()
        fps = meta.get("fps", 25)
        duration = meta.get("duration", 0.0)
        total_secs = int(duration)

        for sec in range(total_secs):
            idx = int(round(sec * fps))
            try:
                from PIL import Image
                frame = reader.get_data(idx)
                img = Image.fromarray(frame).convert("RGB")
                name = f"{video_path.stem}_frame_{sec:04d}.png"
                img.save(frames_dir / name)
            except Exception:
                break
        reader.close()
    except Exception as e:
        print(f"  AVISO: erro ao extrair frames de {video_path.name}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Download temporal dataset (@ArtsyLolaCo)")
    parser.add_argument("--out",           required=True,        help="Diretório de saída")
    parser.add_argument("--n",             type=int, default=500, help="Número de vídeos (padrão: 500)")
    parser.add_argument("--seed",          type=int, default=42,  help="Semente (padrão: 42)")
    parser.add_argument("--skip-download", action="store_true",   help="Pula download, só extrai frames")
    parser.add_argument("--skip-frames",   action="store_true",   help="Pula extração de frames")
    args = parser.parse_args()

    out       = Path(args.out)
    videos_dir = out / "videos"
    frames_dir = out / "frames"
    csv_path   = out / "metadata.csv"
    out.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(exist_ok=True)

    # ── Lista e filtra vídeos ─────────────────────────────────────────────────
    videos = list_channel_videos()
    random.seed(args.seed)
    random.shuffle(videos)
    sampled = videos[:args.n]
    print(f"Amostrando {len(sampled)} vídeos (seed={args.seed}).")

    # ── Baixa metadados e vídeos ──────────────────────────────────────────────
    rows = []
    for i, vid in enumerate(tqdm(sampled, desc="Baixando vídeos"), start=1):
        meta = fetch_metadata(vid["video_id"])
        row = {"index": i, "video_id": vid["video_id"], **meta}

        if not args.skip_download:
            path = download_video(vid["video_id"], videos_dir, i)
            row["local_path"] = str(path) if path else ""

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print(f"\nMetadados salvos → {csv_path}")

    # ── Extrai frames ─────────────────────────────────────────────────────────
    if not args.skip_frames:
        print("\nExtraindo frames (1/seg por vídeo)...")
        mp4s = sorted(videos_dir.glob("*.mp4"))
        for mp4 in tqdm(mp4s, desc="Extraindo frames"):
            vid_frames_dir = frames_dir / mp4.stem
            extract_frames(mp4, vid_frames_dir)
        total = sum(1 for _ in frames_dir.rglob("*.png"))
        print(f"  {total} frames extraídos em {frames_dir}/")

    print("\nFINALIZADO")
    print(f"  {videos_dir}/   ← vídeos MP4 (nomeados por índice)")
    print(f"  {frames_dir}/   ← frames PNG (subpasta por vídeo)")
    print(f"  {csv_path}      ← metadados")


if __name__ == "__main__":
    main()
