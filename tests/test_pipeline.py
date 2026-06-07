"""
Testa as Caixinhas 2, 3 e 4 com mocks dos modelos.
Nenhuma chamada real ao Janus ou ArtClip — roda 100% sem GPU.
"""
import os
import torch
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
from PIL import Image


# ═══════════════════════════════════════════════════════════════════════════════
# CAIXINHA 2 — Captioning
# ═══════════════════════════════════════════════════════════════════════════════

def _mock_captioning_deps(monkeypatch):
    """Substitui _load_janus e _describe_image por versões falsas."""
    import pipeline.captioning as cap

    fake_model     = MagicMock()
    fake_processor = MagicMock()
    monkeypatch.setattr(cap, "_load_janus",     lambda *a, **kw: (fake_model, fake_processor))
    monkeypatch.setattr(cap, "_describe_image", lambda *a, **kw: "A beautiful painting.")
    return fake_model, fake_processor


def _make_loader(mini_apdd_dir, n=4):
    from datasets.apdd import APDDv2Dataset
    from torch.utils.data import DataLoader
    ds = APDDv2Dataset(root=mini_apdd_dir).sample(n=n, seed=42)
    return DataLoader(ds, batch_size=2, shuffle=False)


def test_captioning_returns_list(mini_apdd_dir, base_cfg, monkeypatch):
    _mock_captioning_deps(monkeypatch)
    from pipeline.captioning import run_captioning
    loader  = _make_loader(mini_apdd_dir)
    results = run_captioning(base_cfg, loader)
    assert isinstance(results, list)


def test_captioning_output_size(mini_apdd_dir, base_cfg, monkeypatch):
    _mock_captioning_deps(monkeypatch)
    from pipeline.captioning import run_captioning
    loader  = _make_loader(mini_apdd_dir, n=4)
    results = run_captioning(base_cfg, loader)
    assert len(results) == 4


def test_captioning_output_has_caption(mini_apdd_dir, base_cfg, monkeypatch):
    _mock_captioning_deps(monkeypatch)
    from pipeline.captioning import run_captioning
    loader  = _make_loader(mini_apdd_dir)
    results = run_captioning(base_cfg, loader)
    for r in results:
        assert "caption"  in r
        assert "filename" in r
        assert r["caption"] == "A beautiful painting."


# ═══════════════════════════════════════════════════════════════════════════════
# CAIXINHA 3 — Generation
# ═══════════════════════════════════════════════════════════════════════════════

def _mock_generation_deps(monkeypatch, mini_apdd_dir):
    """Substitui _load_janus e _generate_image por versões falsas."""
    import pipeline.generation as gen

    fake_model     = MagicMock()
    fake_processor = MagicMock()

    def fake_generate(caption, model, processor, device, num_images=1):
        # Retorna imagens sintéticas
        return [Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
                for _ in range(num_images)]

    monkeypatch.setattr(gen, "_load_janus",      lambda *a, **kw: (fake_model, fake_processor))
    monkeypatch.setattr(gen, "_generate_image",  fake_generate)


def _make_sample_data(mini_apdd_dir, n=4):
    img_dir = os.path.join(mini_apdd_dir, "images")
    files   = sorted(os.listdir(img_dir))[:n]
    return [
        {
            "filename": f,
            "image":    None,
            "score":    7.0,
            "caption":  f"A painting called {f}.",
        }
        for f in files
    ]


def test_generation_adds_generated_keys(mini_apdd_dir, base_cfg, monkeypatch, tmp_path):
    _mock_generation_deps(monkeypatch, mini_apdd_dir)
    from pipeline.generation import run_generation

    cfg  = {**base_cfg, "experiment": {**base_cfg["experiment"], "output_dir": str(tmp_path)}}
    data = _make_sample_data(mini_apdd_dir)
    out  = run_generation(cfg, data)

    assert "generated_Janus-Pro-1B" in out[0]
    assert "generated_Janus-Pro-7B" in out[0]


def test_generation_saves_images(mini_apdd_dir, base_cfg, monkeypatch, tmp_path):
    _mock_generation_deps(monkeypatch, mini_apdd_dir)
    from pipeline.generation import run_generation

    cfg = {**base_cfg, "experiment": {**base_cfg["experiment"], "output_dir": str(tmp_path)}}
    run_generation(cfg, _make_sample_data(mini_apdd_dir))

    for model_name in ["Janus-Pro-1B", "Janus-Pro-7B"]:
        save_dir = os.path.join(str(tmp_path), "test_exp1", "generated", model_name)
        assert os.path.isdir(save_dir)
        assert len(os.listdir(save_dir)) > 0


def test_generation_skips_missing_caption(mini_apdd_dir, base_cfg, monkeypatch, tmp_path):
    _mock_generation_deps(monkeypatch, mini_apdd_dir)
    from pipeline.generation import run_generation

    cfg  = {**base_cfg, "experiment": {**base_cfg["experiment"], "output_dir": str(tmp_path)}}
    data = _make_sample_data(mini_apdd_dir, n=2)
    data[0]["caption"] = None   # sem caption → deve ser pulado

    out = run_generation(cfg, data)
    assert out[0]["generated_Janus-Pro-1B"] == []
    assert out[1]["generated_Janus-Pro-1B"] != []


# ═══════════════════════════════════════════════════════════════════════════════
# CAIXINHA 4 — Scoring
# ═══════════════════════════════════════════════════════════════════════════════

def _mock_scoring_deps(monkeypatch):
    import pipeline.scoring as sc

    fake_model = MagicMock()
    monkeypatch.setattr(sc, "_load_agent", lambda *a, **kw: fake_model)
    monkeypatch.setattr(sc, "_predict",    lambda model, t: np.random.uniform(0.1, 1.0))

    # Mock do clip.load → retorna (None, transform identidade)
    from torchvision import transforms
    monkeypatch.setattr(sc.clip, "load", lambda name, device: (None, transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])))


def _make_data_with_generated(mini_apdd_dir, tmp_path, n=4):
    """Cria sample_data com imagens geradas sintéticas salvas em disco."""
    img_dir  = os.path.join(mini_apdd_dir, "images")
    gen_dir  = os.path.join(str(tmp_path), "generated", "Janus-Pro-1B")
    os.makedirs(gen_dir, exist_ok=True)

    files = sorted(os.listdir(img_dir))[:n]
    data  = []
    for f in files:
        gen_path = os.path.join(gen_dir, f)
        arr = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        Image.fromarray(arr).save(gen_path)
        data.append({
            "filename":              os.path.join(img_dir, f),
            "caption":               "A painting.",
            "generated_Janus-Pro-1B": [gen_path],
        })
    return data


def test_scoring_creates_csv(mini_apdd_dir, base_cfg, monkeypatch, tmp_path):
    _mock_scoring_deps(monkeypatch)
    from pipeline.scoring import run_scoring

    cfg  = {**base_cfg, "experiment": {**base_cfg["experiment"], "output_dir": str(tmp_path)}}
    data = _make_data_with_generated(mini_apdd_dir, tmp_path)
    run_scoring(cfg, data)

    scores_dir = os.path.join(str(tmp_path), "test_exp1", "scores")
    csvs = os.listdir(scores_dir)
    assert any("original" in c for c in csvs)
    assert any("Janus-Pro-1B" in c for c in csvs)


def test_scoring_csv_has_agent_columns(mini_apdd_dir, base_cfg, monkeypatch, tmp_path):
    _mock_scoring_deps(monkeypatch)
    from pipeline.scoring import run_scoring

    cfg  = {**base_cfg, "experiment": {**base_cfg["experiment"], "output_dir": str(tmp_path)}}
    data = _make_data_with_generated(mini_apdd_dir, tmp_path)
    run_scoring(cfg, data)

    csv_path = os.path.join(str(tmp_path), "test_exp1", "scores", "scores_original.csv")
    df = pd.read_csv(csv_path)
    for agent in ["score", "color", "mood"]:
        assert agent in df.columns


def test_scoring_no_all_nan_columns(mini_apdd_dir, base_cfg, monkeypatch, tmp_path):
    _mock_scoring_deps(monkeypatch)
    from pipeline.scoring import run_scoring

    cfg  = {**base_cfg, "experiment": {**base_cfg["experiment"], "output_dir": str(tmp_path)}}
    data = _make_data_with_generated(mini_apdd_dir, tmp_path)
    run_scoring(cfg, data)

    csv_path = os.path.join(str(tmp_path), "test_exp1", "scores", "scores_original.csv")
    df = pd.read_csv(csv_path)
    assert not df.isnull().all().any(), "Nenhuma coluna deve ser 100% NaN"
