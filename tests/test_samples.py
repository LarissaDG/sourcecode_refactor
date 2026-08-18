"""
Testa pipeline/samples.py (amostras visuais geradas junto com o experimento).
Usa imagens sintéticas do mini_apdd_dir — sem GPU, sem modelos reais.
"""
import os

import pipeline.samples as samples_mod


# ── _pick_instances / _pick_video_ids / _frames_for_video ──────────────────────

def test_pick_instances_is_deterministic_and_sorted():
    data = [{"filename": "c.jpg"}, {"filename": "a.jpg"}, {"filename": "b.jpg"}, {"filename": "a.jpg"}]
    chosen = samples_mod._pick_instances(data, "filename", n=3)
    assert [c["filename"] for c in chosen] == ["a.jpg", "b.jpg", "c.jpg"]


def test_pick_instances_respects_n():
    data = [{"filename": f"{i}.jpg"} for i in range(10)]
    assert len(samples_mod._pick_instances(data, "filename", n=3)) == 3


def test_pick_instances_repeated_call_same_result():
    data = [{"filename": f"{i}.jpg"} for i in reversed(range(10))]
    r1 = samples_mod._pick_instances(data, "filename", n=3)
    r2 = samples_mod._pick_instances(data, "filename", n=3)
    assert r1 == r2


def test_pick_video_ids_sorted_unique():
    data = [{"video_id": "v2"}, {"video_id": "v1"}, {"video_id": "v2"}, {"video_id": "v3"}]
    assert samples_mod._pick_video_ids(data, n=2) == ["v1", "v2"]


def test_frames_for_video_sorted_by_frame_idx():
    data = [
        {"video_id": "v1", "frame_idx": 2},
        {"video_id": "v1", "frame_idx": 0},
        {"video_id": "v2", "frame_idx": 0},
        {"video_id": "v1", "frame_idx": 1},
    ]
    frames = samples_mod._frames_for_video(data, "v1")
    assert [f["frame_idx"] for f in frames] == [0, 1, 2]


def test_frames_for_video_filters_by_noise_type():
    data = [
        {"video_id": "v1", "frame_idx": 0, "noise_type": "gaussian"},
        {"video_id": "v1", "frame_idx": 0, "noise_type": "blur"},
    ]
    frames = samples_mod._frames_for_video(data, "v1", noise_type="gaussian")
    assert len(frames) == 1
    assert frames[0]["noise_type"] == "gaussian"


# ── Painel Original/Descrição/Gerado (exp0/1/2a/2b) ─────────────────────────────

def _image_paths(mini_apdd_dir, n):
    img_dir = os.path.join(mini_apdd_dir, "images")
    files = sorted(os.listdir(img_dir))[:n]
    return [os.path.join(img_dir, f) for f in files]


def test_panel_creates_file(mini_apdd_dir, tmp_path):
    paths = _image_paths(mini_apdd_dir, 3)
    data = [
        {
            "filename": f"img_{i:03d}.jpg",
            "path": p,
            "caption": f"A painting number {i}.",
            "generated_Janus-Pro-1B": [p],
            "generated_Janus-Pro-7B": [p],
        }
        for i, p in enumerate(paths)
    ]
    samples_mod._panel_original_caption_generated(data, str(tmp_path), caption_label="Descrição (Janus-7B)")
    out_path = tmp_path / "sample_panel.png"
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_panel_handles_missing_generated_gracefully(mini_apdd_dir, tmp_path):
    paths = _image_paths(mini_apdd_dir, 2)
    data = [{"filename": "a.jpg", "path": paths[0], "caption": "", "generated_Janus-Pro-1B": []}]
    samples_mod._panel_original_caption_generated(data, str(tmp_path), caption_label="Descrição (Humana)")
    assert (tmp_path / "sample_panel.png").exists()


def test_panel_empty_data_noop(tmp_path):
    samples_mod._panel_original_caption_generated([], str(tmp_path), caption_label="x")
    assert not (tmp_path / "sample_panel.png").exists()


# ── Painel MNIST (exp3) ──────────────────────────────────────────────────────────

def test_mnist_panel_creates_file(mini_apdd_dir, tmp_path):
    paths = _image_paths(mini_apdd_dir, 3)
    data = [{"filename": f"{i}.png", "path": p, "digit": i} for i, p in enumerate(paths)]
    samples_mod._mnist_panel(data, str(tmp_path))
    out_path = tmp_path / "sample_panel.png"
    assert out_path.exists()
    assert out_path.stat().st_size > 0


# ── Grids de ruído (exp4) ────────────────────────────────────────────────────────

def test_noise_grids_creates_one_file_per_instance(mini_apdd_dir, tmp_path):
    paths = _image_paths(mini_apdd_dir, 2)
    data = [{"filename": f"img_{i:03d}.jpg", "path": p} for i, p in enumerate(paths)]
    samples_mod._noise_grids(data, str(tmp_path), n=2)
    assert (tmp_path / "noise_grid_01.png").exists()
    assert (tmp_path / "noise_grid_02.png").exists()


# ── Vídeo — Exp 5a (sem ruído) ────────────────────────────────────────────────────

def _video_data(mini_apdd_dir, n_videos=2, n_frames=5, with_noise=False, noise_types=("gaussian",)):
    paths = _image_paths(mini_apdd_dir, n_frames)
    data = []
    for v in range(n_videos):
        vid = f"v{v:02d}"
        for i, p in enumerate(paths):
            base = {
                "video_id": vid,
                "frame_idx": i,
                "path": p,
                "filename": f"{vid}_frame_{i:04d}.png",
            }
            if not with_noise:
                data.append(base)
                continue
            for nt in noise_types:
                item = dict(base)
                item["noise_type"] = nt
                item["noise_level"] = round(i / (n_frames - 1) * 100)
                item["degradation_pct"] = float(item["noise_level"])
                data.append(item)
    return data


def test_video_no_noise_creates_gif_and_grid(mini_apdd_dir, tmp_path):
    data = _video_data(mini_apdd_dir, n_videos=2, n_frames=5)
    samples_mod._video_samples_no_noise(data, str(tmp_path), n_videos=2, grid_frames=3)
    assert (tmp_path / "sequence_v00.gif").exists()
    assert (tmp_path / "sequence_v01.gif").exists()
    assert (tmp_path / "frame_grid_last6.png").exists()


def test_video_progressive_creates_gif_and_grid(mini_apdd_dir, tmp_path):
    data = _video_data(mini_apdd_dir, n_videos=2, n_frames=6, with_noise=True,
                        noise_types=("blur", "gaussian", "shapes"))
    samples_mod._video_samples_progressive(data, str(tmp_path), n_videos=2, grid_frames=3)
    for vid in ("v00", "v01"):
        for nt in ("blur", "gaussian", "shapes"):
            assert (tmp_path / f"degradation_{vid}_{nt}.gif").exists()
    assert (tmp_path / "frame_grid_uniform6.png").exists()


# ── Vídeo — Exp 5c (janela macro, 1 fps, sem ruído) ──────────────────────────────

def test_video_macro_creates_gif_and_grid(mini_apdd_dir, tmp_path):
    data = _video_data(mini_apdd_dir, n_videos=2, n_frames=5)
    samples_mod._video_samples_macro(data, str(tmp_path), n_videos=2, grid_frames=3)
    assert (tmp_path / "sequence_v00.gif").exists()
    assert (tmp_path / "sequence_v01.gif").exists()
    assert (tmp_path / "frame_grid_uniform5.png").exists()


def test_video_macro_grid_labels_show_real_frame_position():
    """O grid do Exp5c mostra a posição real do frame na sequência (não uma
    renumeração 1..N) -- checa isso direto em _frame_label, sem depender de
    inspecionar pixels do PNG gerado."""
    frames = [{"frame_idx": i} for i in range(30)]
    idxs = sorted({0, 7, 14, 22, 29})
    labels = [samples_mod._frame_label(frames[i]) for i in idxs]
    assert labels == ["Frame 0", "Frame 7", "Frame 14", "Frame 22", "Frame 29"]


# ── Vídeo — Exp 5d (janelas micro, 10x3s espalhadas, sem ruído) ──────────────────

def test_video_micro_creates_gif_and_grid_with_custom_title(mini_apdd_dir, tmp_path):
    """Exp5d reusa _video_samples_macro, só com título e nome de grid próprios
    (não pode sair rotulado "Exp 5c" por engano)."""
    data = _video_data(mini_apdd_dir, n_videos=2, n_frames=5)
    samples_mod._video_samples_macro(
        data, str(tmp_path), n_videos=2, grid_frames=3,
        title="Exp 5d — 10 sub-amostras aleatórias de 3s espalhadas pelo vídeo (sem ruído)",
        grid_filename="frame_grid_micro10.png",
    )
    assert (tmp_path / "sequence_v00.gif").exists()
    assert (tmp_path / "sequence_v01.gif").exists()
    assert (tmp_path / "frame_grid_micro10.png").exists()
    assert not (tmp_path / "frame_grid_uniform5.png").exists()


def test_video_micro_grid_labels_show_true_second_with_gaps():
    """As 10 janelas de 3s do Exp5d têm lacunas de frame_idx entre si (ex.:
    bloco 0 = segundos 0-2, bloco 10 = segundos 30-32) -- _frame_label precisa
    continuar mostrando o segundo real, não a posição na lista."""
    frame_idxs = [0, 1, 2, 30, 31, 32, 60, 61, 62]
    frames = [{"frame_idx": i} for i in frame_idxs]
    labels = [samples_mod._frame_label(f) for f in frames]
    assert labels == [f"Frame {i}" for i in frame_idxs]


def test_run_samples_dispatches_exp5d_to_macro_panel(mini_apdd_dir, tmp_path, monkeypatch):
    data = _video_data(mini_apdd_dir, n_videos=1, n_frames=5)
    cfg = {"experiment": {"name": "exp5d_temporal_micro", "output_dir": str(tmp_path)}}
    samples_mod.run_samples(cfg, data)
    out_dir = tmp_path / "exp5d_temporal_micro" / "samples"
    assert (out_dir / "frame_grid_micro10.png").exists()
    assert not (out_dir / "frame_grid_uniform5.png").exists()


def test_video_progressive_skips_missing_noise_type(mini_apdd_dir, tmp_path):
    """Se um vídeo só tem 1 dos 3 tipos de ruído nos dados, gera só o GIF
    daquele tipo, sem quebrar nos outros dois."""
    data = _video_data(mini_apdd_dir, n_videos=1, n_frames=6, with_noise=True,
                        noise_types=("gaussian",))
    samples_mod._video_samples_progressive(data, str(tmp_path), n_videos=1, grid_frames=3)
    assert (tmp_path / "degradation_v00_gaussian.gif").exists()
    assert not (tmp_path / "degradation_v00_blur.gif").exists()
    assert not (tmp_path / "degradation_v00_shapes.gif").exists()


# ── Dispatcher (run_samples) ─────────────────────────────────────────────────────

def test_run_samples_dispatches_panel_with_janus_label(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(samples_mod, "_panel_original_caption_generated",
                         lambda data, out_dir, caption_label: calls.append(caption_label))
    cfg = {"experiment": {"name": "exp1_apdd_uniform_bins", "output_dir": str(tmp_path)}}
    samples_mod.run_samples(cfg, [{"filename": "a.jpg"}])
    assert calls == ["Descrição (Janus-7B)"]


def test_run_samples_dispatches_panel_with_human_label(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(samples_mod, "_panel_original_caption_generated",
                         lambda data, out_dir, caption_label: calls.append(caption_label))
    cfg = {"experiment": {"name": "exp2b_portinari_human", "output_dir": str(tmp_path)}}
    samples_mod.run_samples(cfg, [{"filename": "a.jpg"}])
    assert calls == ["Descrição (Humana)"]


def test_run_samples_dispatches_mnist(monkeypatch, tmp_path):
    called = []
    monkeypatch.setattr(samples_mod, "_mnist_panel", lambda data, out_dir: called.append(True))
    cfg = {"experiment": {"name": "exp3_mnist", "output_dir": str(tmp_path)}}
    samples_mod.run_samples(cfg, [{"filename": "a.jpg"}])
    assert called == [True]


def test_run_samples_creates_out_dir(tmp_path):
    cfg = {"experiment": {"name": "exp_unknown", "output_dir": str(tmp_path)}}
    samples_mod.run_samples(cfg, [{"filename": "a.jpg"}])
    assert (tmp_path / "exp_unknown" / "samples").is_dir()


def test_run_samples_empty_data_noop(tmp_path):
    cfg = {"experiment": {"name": "exp1_apdd", "output_dir": str(tmp_path)}}
    samples_mod.run_samples(cfg, [])
    assert not (tmp_path / "exp1_apdd").exists()
