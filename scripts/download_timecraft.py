"""
Baixa o dataset "Digital Paintings" do TimeCraft (github.com/xamyzhao/timecraft,
CVPR 2020 — "Painting Many Pasts: Synthesizing Time Lapse Videos of Paintings"),
substituindo o dataset @ArtsyLolaCo (download_temporal.py) como fonte do Exp 5.

O repositório do TimeCraft tem dois datasets:
  - Watercolors: pinturas físicas reais, só liberado por e-mail aos autores
    (não automatizável — não é o que este script baixa).
  - Digital Paintings: vídeos reais de pessoas pintando digitalmente, com
    metadados (.pkl) já publicados no repo — é este que baixamos aqui.

O repo publica 83 arquivos .pkl (um por vídeo/"peça"), cada um com o ID do
vídeo no YouTube e a lista de índices de frame do vídeo original ("framenums",
essencialmente contígua — não são frames-chave pré-selecionados). Como cada
vídeo tem milhares de frames, extraímos só uma janela de `--n-frames` frames
a partir do índice `--offset` dentro dessa lista (padrão: pula os primeiros
24 frames e pega os 24 seguintes) — isso evita o "quadro já finalizado,
paradinho" que alguns vídeos mostram como abertura antes de reiniciar do
zero. Nem todo vídeo de 2019 ainda está disponível no YouTube hoje; os que
falharem são pulados (mesmo comportamento do notebook original do TimeCraft).

Retomável: pode ser interrompido a qualquer momento (ex: walltime do SLURM) e
rodado de novo com o mesmo comando — vídeos já baixados não são rebaixados,
vídeos já extraídos não são reextraídos, e metadata.csv é reescrito a cada
vídeo processado (não só no final), então nada fica sem registro.

Uso:
    python3 scripts/download_timecraft.py --out data/temporal

Flags:
    --out             Diretório de saída (frames/videos/metadata.csv)
    --repo-dir        Onde clonar/usar o repositório timecraft (padrão: data/timecraft_repo)
    --n-frames        Frames por vídeo (padrão: 24)
    --offset          Quantos frames iniciais da lista pular por vídeo (padrão: 24)
    --scale           Fator de redimensionamento após o crop (padrão: 0.5, igual ao notebook original)
    --skip-clone      Não clona/atualiza o repositório (usa o que já existir em --repo-dir)
    --skip-download   Só extrai frames de vídeos já baixados
    --skip-frames     Só baixa vídeos, sem extrair frames

Dependências:
    pip install yt-dlp imageio imageio-ffmpeg pandas tqdm pillow numpy
    git (para clonar o repositório)
"""

import argparse
import pickle
import subprocess
import sys
import zipfile
from pathlib import Path

import pandas as pd
from tqdm import tqdm

REPO_URL = "https://github.com/xamyzhao/timecraft.git"
PKL_ZIP_REL_PATH = "dataset/digital_vid_caches_minimal.zip"
YT_DLP_CMD = [sys.executable, "-m", "yt_dlp"]


def clone_or_update_repo(repo_dir: Path):
    if repo_dir.exists() and (repo_dir / ".git").exists():
        print(f"Repositório já existe em {repo_dir}, atualizando...")
        subprocess.run(["git", "-C", str(repo_dir), "pull"], check=False)
    else:
        print(f"Clonando {REPO_URL} em {repo_dir}...")
        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", REPO_URL, str(repo_dir)], check=True)


def load_video_infos(repo_dir: Path) -> list[dict]:
    """Lê o zip de .pkl do repo e devolve uma lista de dicts (um por vídeo/peça)."""
    zip_path = repo_dir / PKL_ZIP_REL_PATH
    if not zip_path.exists():
        raise FileNotFoundError(
            f"{zip_path} não encontrado — o repositório foi clonado corretamente?"
        )
    infos = []
    with zipfile.ZipFile(zip_path) as z:
        for name in sorted(z.namelist()):
            if not name.endswith(".pkl"):
                continue
            obj = pickle.loads(z.read(name))
            infos.append({
                "vid_name":       obj["vid_name"],
                "vid_id":         obj["vid_id"],
                "framenums":      sorted(obj["framenums"]),
                "crop_start_xy":  obj["crop_start_xy"],
                "crop_end_xy":    obj["crop_end_xy"],
            })
    print(f"  {len(infos)} vídeos/peças listados em {zip_path.name}.")
    return infos


def _try_download(url: str, out_path: Path) -> bool:
    result = subprocess.run([
        *YT_DLP_CMD,
        "-o", str(out_path),
        "--merge-output-format", "mp4",
        "-f", "mp4",
        url,
    ])
    return result.returncode == 0 and out_path.exists()


def download_video(vid_id: str, vid_name: str, out_dir: Path) -> Path | None:
    """Baixa vid_id do YouTube; se falhar e o id for puramente numérico
    (padrão de ID do Vimeo — o README do TimeCraft avisa que alguns vídeos
    digitais vêm do Vimeo), tenta vimeo.com/<id> como fallback."""
    out_path = out_dir / f"{vid_name}.mp4"
    if out_path.exists():
        return out_path

    if _try_download(f"https://www.youtube.com/watch?v={vid_id}", out_path):
        return out_path

    if vid_id.isdigit() and _try_download(f"https://vimeo.com/{vid_id}", out_path):
        return out_path

    return None


def frames_already_extracted(info: dict, n_frames: int, frames_dir: Path) -> bool:
    """True se os n_frames PNGs já existem em disco para esse vídeo — usado
    para pular reextração ao retomar um download interrompido."""
    vid_frames_dir = frames_dir / info["vid_name"]
    if not vid_frames_dir.is_dir():
        return False
    existing = {p.name for p in vid_frames_dir.glob("*.png")}
    expected = {f"{info['vid_name']}_frame_{i:04d}.png" for i in range(n_frames)}
    return expected.issubset(existing)


def extract_frame_window(video_path: Path, info: dict, offset: int, n_frames: int,
                          scale: float, frames_dir: Path) -> int:
    """Extrai info['framenums'][offset : offset+n_frames], recorta pelo crop
    salvo no .pkl e salva como <vid_name>_frame_XXXX.png (XXXX = índice
    relativo 0..n_frames-1, não o índice bruto do vídeo). Devolve quantos
    frames foram salvos."""
    import imageio
    from PIL import Image

    framenums = info["framenums"]
    window = framenums[offset:offset + n_frames]
    if not window:
        return 0

    x0, y0 = info["crop_start_xy"]
    x1, y1 = info["crop_end_xy"]
    x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

    vid_frames_dir = frames_dir / info["vid_name"]
    vid_frames_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    try:
        reader = imageio.get_reader(str(video_path), "ffmpeg")
        for rel_idx, raw_idx in enumerate(window):
            try:
                frame = reader.get_data(raw_idx)
            except Exception:
                continue
            img = Image.fromarray(frame).convert("RGB")
            if x1 > x0 and y1 > y0:
                img = img.crop((x0, y0, x1, y1))
            if scale != 1.0:
                img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))))
            name = f"{info['vid_name']}_frame_{rel_idx:04d}.png"
            img.save(vid_frames_dir / name)
            saved += 1
        reader.close()
    except Exception as e:
        print(f"  AVISO: erro ao extrair frames de {video_path.name}: {e}")
    return saved


def main():
    parser = argparse.ArgumentParser(description="Download do dataset Digital Paintings (TimeCraft)")
    parser.add_argument("--out",           required=True,          help="Diretório de saída")
    parser.add_argument("--repo-dir",      default="data/timecraft_repo", help="Onde clonar o repositório")
    parser.add_argument("--n-frames",      type=int, default=24,   help="Frames por vídeo (padrão: 24)")
    parser.add_argument("--offset",        type=int, default=24,   help="Frames iniciais a pular por vídeo (padrão: 24)")
    parser.add_argument("--scale",         type=float, default=0.5, help="Fator de redimensionamento (padrão: 0.5)")
    parser.add_argument("--skip-clone",    action="store_true",    help="Não clona/atualiza o repositório")
    parser.add_argument("--skip-download", action="store_true",    help="Pula download, só extrai frames")
    parser.add_argument("--skip-frames",   action="store_true",    help="Pula extração de frames")
    args = parser.parse_args()

    out         = Path(args.out)
    repo_dir    = Path(args.repo_dir)
    videos_dir  = out / "videos"
    frames_dir  = out / "frames"
    csv_path    = out / "metadata.csv"
    out.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(exist_ok=True)

    # ── Clona o repositório ──────────────────────────────────────────────────
    if not args.skip_clone:
        clone_or_update_repo(repo_dir)

    # ── Lê a lista de vídeos/peças a partir dos .pkl do repo ────────────────
    infos = load_video_infos(repo_dir)

    # ── Baixa vídeos e extrai a janela de frames ─────────────────────────────
    # Retomável: se o job for interrompido (ex: walltime do SLURM), rodar de
    # novo pula vídeos já baixados (download_video checa o .mp4 em disco) e
    # pula reextração de vídeos cujos n_frames PNGs já existem — só refaz o
    # que realmente falta. metadata.csv é reescrito a cada vídeo processado,
    # não só no final, então uma interrupção não deixa o resumo em branco.
    rows = []
    did_not_download = []
    for info in tqdm(infos, desc="Processando vídeos"):
        vid_name = info["vid_name"]
        n_available = len(info["framenums"])

        already_extracted = (not args.skip_frames) and frames_already_extracted(
            info, args.n_frames, frames_dir
        )

        video_path = videos_dir / f"{vid_name}.mp4"
        if already_extracted:
            video_path = video_path if video_path.exists() else None
        elif not args.skip_download:
            video_path = download_video(info["vid_id"], vid_name, videos_dir)
            if video_path is None:
                did_not_download.append(vid_name)
                rows.append({
                    "vid_name": vid_name, "vid_id": info["vid_id"],
                    "n_framenums_disponiveis": n_available,
                    "frames_extraidos": 0, "status": "download_falhou",
                })
                pd.DataFrame(rows).to_csv(csv_path, index=False)
                continue
        elif not video_path.exists():
            did_not_download.append(vid_name)
            rows.append({
                "vid_name": vid_name, "vid_id": info["vid_id"],
                "n_framenums_disponiveis": n_available,
                "frames_extraidos": 0, "status": "video_nao_encontrado",
            })
            pd.DataFrame(rows).to_csv(csv_path, index=False)
            continue

        if already_extracted:
            n_saved = args.n_frames
            status = "ok_ja_existia"
        else:
            n_saved = 0
            if not args.skip_frames:
                n_saved = extract_frame_window(
                    video_path, info, args.offset, args.n_frames, args.scale, frames_dir
                )
            status = "ok" if n_saved > 0 else ("nao_extraido" if args.skip_frames else "sem_frames_na_janela")

        rows.append({
            "vid_name": vid_name, "vid_id": info["vid_id"],
            "n_framenums_disponiveis": n_available,
            "frames_extraidos": n_saved,
            "status": status,
        })
        pd.DataFrame(rows).to_csv(csv_path, index=False)

    df = pd.DataFrame(rows)
    n_ok = int(df["status"].isin(["ok", "ok_ja_existia"]).sum())
    print(f"\nMetadados salvos -> {csv_path}")
    print(f"Vídeos que falharam no download ({len(did_not_download)}): {did_not_download}")

    print("\nFINALIZADO")
    print(f"  {videos_dir}/   <- vídeos MP4 (nomeados por vid_name)")
    print(f"  {frames_dir}/   <- frames PNG (subpasta por vid_name, {args.n_frames} frames cada a partir do offset {args.offset})")
    print(f"  {csv_path}      <- metadados")
    print(f"  {n_ok}/{len(infos)} vídeos com frames extraídos com sucesso")


if __name__ == "__main__":
    main()
