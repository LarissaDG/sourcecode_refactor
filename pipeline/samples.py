"""
Caixinha extra — Amostras visuais.

Gera exemplos reais de entrada/saída de cada experimento (3 instâncias por
padrão), rodando junto com o próprio experimento (dentro de run.py, no
cluster) — assim não depende de ter as bases de dados localmente, nem
duplica imagens além dos painéis/GIFs de amostra já compostos.

A escolha das instâncias é determinística (as N primeiras após ordenar por
filename/video_id, sem sorteio) para que reruns do mesmo experimento sempre
mostrem os mesmos exemplos, e para que o Exp2b (que reusa exatamente as
imagens do Exp2a via reuse_from) coincida automaticamente com o Exp2a.

Saída: outputs/<experimento>/samples/
"""
import os
import textwrap

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image, ImageDraw

from datasets.noise import NOISE_FNS

SAMPLE_N_INSTANCES = 3
NOISE_ROW_TYPES = [("blur", "Blur"), ("gaussian", "Gaussian"), ("shapes", "Shapes")]
NOISE_LEVELS = list(range(10, 101, 10))  # 10 colunas (0% já aparece no painel do Exp1/ICCC)


# ── Helpers genéricos ─────────────────────────────────────────────────────────

def _open_img(path, size=(224, 224)):
    try:
        img = Image.open(path).convert("RGB")
        img.thumbnail(size, Image.LANCZOS)
        canvas = Image.new("RGB", size, (230, 230, 230))
        ox = (size[0] - img.width) // 2
        oy = (size[1] - img.height) // 2
        canvas.paste(img, (ox, oy))
        return canvas
    except Exception:
        img = Image.new("RGB", size, (180, 180, 180))
        draw = ImageDraw.Draw(img)
        draw.text((10, size[1] // 2 - 8), "N/A", fill=(100, 100, 100))
        return img


def _wrap(text, width=42, max_lines=6):
    if not text:
        return "(sem descrição)"
    lines = textwrap.wrap(str(text), width=width)[:max_lines]
    return "\n".join(lines)


def _pick_instances(data, key="filename", n=SAMPLE_N_INSTANCES):
    """Escolhe as `n` primeiras instâncias únicas de `data`, ordenadas por `key` — determinístico."""
    seen = {}
    for item in data:
        k = item.get(key)
        if k is not None and k not in seen:
            seen[k] = item
    return [seen[k] for k in sorted(seen.keys())[:n]]


def _pick_video_ids(data, n=SAMPLE_N_INSTANCES):
    ids = sorted({item.get("video_id") for item in data if item.get("video_id")})
    return ids[:n]


def _frames_for_video(data, video_id, noise_type=None):
    frames = [d for d in data if d.get("video_id") == video_id]
    if noise_type is not None:
        frames = [d for d in frames if d.get("noise_type") == noise_type]
    return sorted(frames, key=lambda d: d.get("frame_idx", 0))


# ── Exp 0 (ICCC) / Exp 1 / Exp 2a / Exp 2b ─────────────────────────────────────
# Original | Descrição | Gerada Janus-1B | Gerada Janus-7B

def _panel_original_caption_generated(data, out_dir, caption_label, n=SAMPLE_N_INSTANCES):
    instances = _pick_instances(data, "filename", n)
    if not instances:
        print("[samples] sem dados para o painel de amostras, pulando.")
        return

    col_titles = ["Original", caption_label, "Janus-Pro-1B", "Janus-Pro-7B"]
    thumb = (224, 224)
    n_rows = len(instances)
    fig = plt.figure(figsize=(4 * 3.0, n_rows * 3.0))
    gs = gridspec.GridSpec(n_rows, 4, figure=fig, hspace=0.5, wspace=0.15)

    for row, item in enumerate(instances):
        gen1b = (item.get("generated_Janus-Pro-1B") or [None])[0]
        gen7b = (item.get("generated_Janus-Pro-7B") or [None])[0]
        cell_values = [item.get("path"), None, gen1b, gen7b]
        for col, val in enumerate(cell_values):
            ax = fig.add_subplot(gs[row, col])
            ax.axis("off")
            if col == 1:
                ax.text(0.5, 0.5, _wrap(item.get("caption")), ha="center", va="center",
                        fontsize=8, transform=ax.transAxes)
            else:
                ax.imshow(np.array(_open_img(val, thumb)))
            if row == 0:
                ax.set_title(col_titles[col], fontsize=10, fontweight="bold")

    fig.suptitle("Amostras — Original / Descrição / Gerado", fontsize=12, fontweight="bold")
    out_path = os.path.join(out_dir, "sample_panel.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [samples] painel salvo: {out_path}")


# ── Exp 3 (MNIST) ───────────────────────────────────────────────────────────────

def _mnist_panel(data, out_dir, n=SAMPLE_N_INSTANCES):
    instances = _pick_instances(data, "filename", n)
    if not instances:
        print("[samples] sem dados MNIST, pulando.")
        return

    thumb = (140, 140)
    fig, axes = plt.subplots(1, len(instances), figsize=(len(instances) * 2.4, 2.8))
    if len(instances) == 1:
        axes = [axes]
    for ax, item in zip(axes, instances):
        ax.imshow(np.array(_open_img(item.get("path"), thumb)))
        ax.axis("off")
        ax.set_title(f"Label: {item.get('digit', '?')}", fontsize=10)
    fig.suptitle("Exp 3 — MNIST: amostras", fontsize=12, fontweight="bold")
    out_path = os.path.join(out_dir, "sample_panel.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [samples] painel salvo: {out_path}")


# ── Exp 4 (Ruído) ────────────────────────────────────────────────────────────────
# 1 grid por instância: linhas = tipo de ruído, colunas = nível (10%-100%)

def _noise_grids(data, out_dir, n=SAMPLE_N_INSTANCES):
    instances = _pick_instances(data, "filename", n)
    if not instances:
        print("[samples] sem dados para grids de ruído, pulando.")
        return

    for idx, item in enumerate(instances, start=1):
        path = item.get("path")
        try:
            base_img = Image.open(path).convert("RGB")
        except Exception:
            print(f"  [samples] imagem base não encontrada, pulando instância: {path}")
            continue

        n_rows, n_cols = len(NOISE_ROW_TYPES), len(NOISE_LEVELS)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.5, n_rows * 1.8))
        for r, (noise_key, label) in enumerate(NOISE_ROW_TYPES):
            fn = NOISE_FNS[noise_key]
            for c, level in enumerate(NOISE_LEVELS):
                ax = axes[r][c]
                noisy = fn(base_img, level)
                thumb = noisy.copy()
                thumb.thumbnail((160, 160))
                ax.imshow(np.array(thumb))
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                if r == 0:
                    ax.set_title(f"{level}%", fontsize=8)
                if c == 0:
                    ax.set_ylabel(label, fontsize=9, rotation=0, ha="right", va="center", labelpad=25)

        fig.suptitle(f"Exp 4 — Ruído (amostra {idx}/{len(instances)})", fontsize=12, fontweight="bold")
        out_path = os.path.join(out_dir, f"noise_grid_{idx:02d}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  [samples] grid de ruído salvo: {out_path}")


# ── Exp 5a / 5b (Temporal) — helpers de vídeo ───────────────────────────────────

def _render_frame(item, apply_noise, size=(224, 224)):
    img = _open_img(item.get("path"), size)
    if apply_noise:
        noise_type = item.get("noise_type")
        level = item.get("noise_level", 0)
        if noise_type and noise_type not in (None, "none") and level:
            img = NOISE_FNS[noise_type](img, int(level))
    return img


def _frame_label(item, show_degradation=False):
    label = f"Frame {item.get('frame_idx', '?')}"
    if show_degradation:
        deg = item.get("degradation_pct")
        if deg is not None:
            label += f"\n{float(deg):.0f}%"
    return label


def _save_gif(frames, apply_noise, out_path, fps=4, size=(224, 224)):
    imgs = []
    for item in frames:
        img = _render_frame(item, apply_noise, size).copy()
        draw = ImageDraw.Draw(img)
        text = f"Frame {item.get('frame_idx', '?')}"
        deg = item.get("degradation_pct")
        if apply_noise and deg is not None:
            text += f" | {float(deg):.0f}%"
        draw.text((4, 4), text, fill=(255, 255, 0))
        imgs.append(img)

    if not imgs:
        return
    imgs[0].save(out_path, save_all=True, append_images=imgs[1:],
                 duration=int(1000 / fps), loop=0)
    print(f"  [samples] GIF salvo: {out_path}")


def _save_frame_grid(rows, out_path, title, apply_noise, size=(180, 180)):
    """rows: lista de (video_id, [frame_items])."""
    rows = [(vid, frames) for vid, frames in rows if frames]
    if not rows:
        return

    n_rows = len(rows)
    n_cols = max(len(frames) for _, frames in rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.0, n_rows * 2.4))
    if n_rows == 1:
        axes = [axes]
    if n_cols == 1:
        axes = [[ax] for ax in axes]

    for r, (video_id, frames) in enumerate(rows):
        for c in range(n_cols):
            ax = axes[r][c]
            ax.axis("off")
            if c < len(frames):
                item = frames[c]
                ax.imshow(np.array(_render_frame(item, apply_noise, size)))
                ax.set_title(_frame_label(item, show_degradation=apply_noise), fontsize=8)
            if c == 0:
                ax.text(-0.1, 0.5, str(video_id), transform=ax.transAxes,
                        fontsize=9, ha="right", va="center")

    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [samples] grid salvo: {out_path}")


# ── Exp 5a — sequência amostrada sem ruído ──────────────────────────────────────

def _video_samples_no_noise(data, out_dir, n_videos=SAMPLE_N_INSTANCES, grid_frames=6):
    video_ids = _pick_video_ids(data, n_videos)
    if not video_ids:
        print("[samples] sem vídeos (exp5a), pulando.")
        return

    rows = []
    for vid in video_ids:
        frames = _frames_for_video(data, vid)
        if not frames:
            continue
        _save_gif(frames, apply_noise=False,
                  out_path=os.path.join(out_dir, f"sequence_{vid}.gif"))
        rows.append((vid, frames[-grid_frames:]))  # últimos N frames, em ordem

    _save_frame_grid(rows, os.path.join(out_dir, "frame_grid_last6.png"),
                      title="Exp 5a — últimos frames amostrados (sem ruído)",
                      apply_noise=False)


# ── Exp 5c — janela macro (vídeo inteiro, 1 fps, sem ruído) ─────────────────────

def _video_samples_macro(data, out_dir, n_videos=SAMPLE_N_INSTANCES, grid_frames=5):
    """
    Mesma ideia de _video_samples_no_noise (sem ruído), mas a sequência aqui é
    bem mais longa (o vídeo inteiro truncado, 1 frame/seg, em vez de ~24
    frames do início) — por isso o grid usa frames uniformemente espaçados ao
    longo de toda a sequência (como _video_samples_progressive já faz para o
    Exp5b), em vez de só os últimos N, pra dar uma amostra representativa do
    vídeo inteiro. O rótulo de cada frame no grid mostra o índice real dele
    na sequência (segundo do vídeo), não uma renumeração 1..5.
    """
    video_ids = _pick_video_ids(data, n_videos)
    if not video_ids:
        print("[samples] sem vídeos (exp5c), pulando.")
        return

    rows = []
    for vid in video_ids:
        frames = _frames_for_video(data, vid)
        if not frames:
            continue
        _save_gif(frames, apply_noise=False,
                  out_path=os.path.join(out_dir, f"sequence_{vid}.gif"))
        idxs = sorted(set(np.linspace(0, len(frames) - 1, grid_frames).round().astype(int)))
        rows.append((vid, [frames[i] for i in idxs]))

    _save_frame_grid(rows, os.path.join(out_dir, "frame_grid_uniform5.png"),
                      title="Exp 5c — frames uniformemente amostrados ao longo do vídeo (1 fps, sem ruído)",
                      apply_noise=False)


# ── Exp 5b — degradação progressiva ─────────────────────────────────────────────

def _video_samples_progressive(data, out_dir, n_videos=SAMPLE_N_INSTANCES, grid_frames=6):
    """
    Cada frame do Exp5b tem 3 variantes (uma por tipo de ruído) no mesmo nível de
    degradação. Gera 1 GIF por (vídeo, tipo de ruído) e um único grid combinado
    com 1 linha por (vídeo, tipo) — mesma lógica de "linha = tipo de ruído" do
    grid estático do Exp4 (_noise_grids), aplicada aqui por vídeo.
    """
    video_ids = _pick_video_ids(data, n_videos)
    if not video_ids:
        print("[samples] sem vídeos (exp5b), pulando.")
        return

    rows = []
    for vid in video_ids:
        for noise_key, label in NOISE_ROW_TYPES:
            frames = _frames_for_video(data, vid, noise_type=noise_key)
            if not frames:
                continue
            _save_gif(frames, apply_noise=True,
                      out_path=os.path.join(out_dir, f"degradation_{vid}_{noise_key}.gif"))
            idxs = sorted(set(np.linspace(0, len(frames) - 1, grid_frames).round().astype(int)))
            rows.append((f"{vid} ({label})", [frames[i] for i in idxs]))

    _save_frame_grid(rows, os.path.join(out_dir, "frame_grid_uniform6.png"),
                      title="Exp 5b — degradação progressiva (frames uniformes, por tipo de ruído)",
                      apply_noise=True)


# ── Dispatcher ───────────────────────────────────────────────────────────────────

def run_samples(cfg, data):
    if not data:
        print("[samples] sem dados, pulando amostras.")
        return

    name = cfg["experiment"]["name"]
    out_dir = os.path.join(cfg["experiment"]["output_dir"], cfg["experiment"]["name"], "samples")
    os.makedirs(out_dir, exist_ok=True)

    if name.startswith(("exp0_iccc", "exp1_apdd")):
        _panel_original_caption_generated(data, out_dir, caption_label="Descrição (Janus-7B)")
    elif name.startswith("exp2a_portinari"):
        _panel_original_caption_generated(data, out_dir, caption_label="Descrição (Janus-7B)")
    elif name.startswith("exp2b_portinari_human"):
        _panel_original_caption_generated(data, out_dir, caption_label="Descrição (Humana)")
    elif name.startswith("exp3_mnist"):
        _mnist_panel(data, out_dir)
    elif name.startswith("exp4_noise"):
        _noise_grids(data, out_dir)
    elif name.startswith("exp5a_temporal"):
        _video_samples_no_noise(data, out_dir)
    elif name.startswith("exp5b_temporal_error"):
        _video_samples_progressive(data, out_dir)
    elif name.startswith("exp5c_temporal_macro"):
        _video_samples_macro(data, out_dir)
    else:
        print(f"[samples] nenhum layout de amostra definido para '{name}', pulando.")
