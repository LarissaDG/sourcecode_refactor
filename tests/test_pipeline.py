"""
Testa as Caixinhas 2, 3 e 4 com mocks dos modelos.
Nenhuma chamada real ao Janus ou ArtClip — roda 100% sem GPU.
"""
import os
import sys
import torch
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
from PIL import Image
from torchvision import transforms


# ═══════════════════════════════════════════════════════════════════════════════
# CAIXINHA 2 — Captioning
# ═══════════════════════════════════════════════════════════════════════════════

def _mock_captioning_deps(monkeypatch):
    import pipeline.captioning as cap
    monkeypatch.setattr(cap, "_load_janus",     lambda *a, **kw: (MagicMock(), MagicMock()))
    monkeypatch.setattr(cap, "_describe_image", lambda *a, **kw: "A beautiful painting.")


def _make_loader(mini_apdd_dir, n=4):
    from datasets.apddv2 import APDDv2Dataset          # corrigido: apddv2
    from torch.utils.data import DataLoader
    ds = APDDv2Dataset(root=mini_apdd_dir).sample(n=n, seed=42)
    return DataLoader(ds, batch_size=2, shuffle=False)


def test_captioning_returns_list(mini_apdd_dir, base_cfg, monkeypatch):
    _mock_captioning_deps(monkeypatch)
    from pipeline.captioning import run_captioning
    results = run_captioning(base_cfg, _make_loader(mini_apdd_dir))
    assert isinstance(results, list)


def test_captioning_output_size(mini_apdd_dir, base_cfg, monkeypatch):
    _mock_captioning_deps(monkeypatch)
    from pipeline.captioning import run_captioning
    results = run_captioning(base_cfg, _make_loader(mini_apdd_dir, n=4))
    assert len(results) == 4


def test_captioning_output_has_caption(mini_apdd_dir, base_cfg, monkeypatch):
    _mock_captioning_deps(monkeypatch)
    from pipeline.captioning import run_captioning
    results = run_captioning(base_cfg, _make_loader(mini_apdd_dir))
    for r in results:
        assert "caption"  in r
        assert "filename" in r
        assert r["caption"] == "A beautiful painting."


# ═══════════════════════════════════════════════════════════════════════════════
# CAIXINHA 3 — Generation
# ═══════════════════════════════════════════════════════════════════════════════

def _mock_generation_deps(monkeypatch):
    import pipeline.generation as gen
    monkeypatch.setattr(gen, "_load_janus", lambda *a, **kw: (MagicMock(), MagicMock()))
    monkeypatch.setattr(gen, "_generate_image", lambda *a, **kw: [
        Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
    ])


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
    _mock_generation_deps(monkeypatch)
    from pipeline.generation import run_generation
    cfg  = {**base_cfg, "experiment": {**base_cfg["experiment"], "output_dir": str(tmp_path)}}
    out  = run_generation(cfg, _make_sample_data(mini_apdd_dir))
    assert "generated_Janus-Pro-1B" in out[0]
    assert "generated_Janus-Pro-7B" in out[0]


def test_generation_saves_images(mini_apdd_dir, base_cfg, monkeypatch, tmp_path):
    _mock_generation_deps(monkeypatch)
    from pipeline.generation import run_generation
    cfg = {**base_cfg, "experiment": {**base_cfg["experiment"], "output_dir": str(tmp_path)}}
    run_generation(cfg, _make_sample_data(mini_apdd_dir))
    for model_name in ["Janus-Pro-1B", "Janus-Pro-7B"]:
        save_dir = os.path.join(str(tmp_path), "test_exp1", "generated", model_name)
        assert os.path.isdir(save_dir)
        assert len(os.listdir(save_dir)) > 0


def test_generation_skips_missing_caption(mini_apdd_dir, base_cfg, monkeypatch, tmp_path):
    _mock_generation_deps(monkeypatch)
    from pipeline.generation import run_generation
    cfg  = {**base_cfg, "experiment": {**base_cfg["experiment"], "output_dir": str(tmp_path)}}
    data = _make_sample_data(mini_apdd_dir, n=2)
    data[0]["caption"] = None
    out = run_generation(cfg, data)
    assert out[0]["generated_Janus-Pro-1B"] == []
    assert out[1]["generated_Janus-Pro-1B"] != []


# ═══════════════════════════════════════════════════════════════════════════════
# CAIXINHA 4 — Scoring
# ═══════════════════════════════════════════════════════════════════════════════

# Mock do módulo models.clip inteiro — evita precisar do pacote instalado
_fake_clip = MagicMock()
_fake_clip.load.return_value = (None, transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
]))


def _mock_scoring_deps(monkeypatch):
    # Injeta módulos falsos antes de importar pipeline.scoring
    monkeypatch.setitem(sys.modules, "models",          MagicMock())
    monkeypatch.setitem(sys.modules, "models.clip",     _fake_clip)
    monkeypatch.setitem(sys.modules, "models.aesclip",  MagicMock())

    import pipeline.scoring as sc
    monkeypatch.setattr(sc, "_load_agent", lambda *a, **kw: MagicMock())
    monkeypatch.setattr(sc, "_predict",    lambda model, t: np.random.uniform(0.1, 1.0))


def _make_data_with_generated(mini_apdd_dir, tmp_path, n=4):
    img_dir = os.path.join(mini_apdd_dir, "images")
    gen_dir = os.path.join(str(tmp_path), "generated", "Janus-Pro-1B")
    os.makedirs(gen_dir, exist_ok=True)
    files = sorted(os.listdir(img_dir))[:n]
    data  = []
    for f in files:
        gen_path = os.path.join(gen_dir, f)
        Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)).save(gen_path)
        data.append({
            "filename":               os.path.join(img_dir, f),
            "caption":                "A painting.",
            "generated_Janus-Pro-1B": [gen_path],
        })
    return data


def test_scoring_creates_csv(mini_apdd_dir, base_cfg, monkeypatch, tmp_path):
    _mock_scoring_deps(monkeypatch)
    from pipeline.scoring import run_scoring
    cfg  = {**base_cfg, "experiment": {**base_cfg["experiment"], "output_dir": str(tmp_path)}}
    run_scoring(cfg, _make_data_with_generated(mini_apdd_dir, tmp_path))
    scores_dir = os.path.join(str(tmp_path), "test_exp1", "scores")
    csvs = os.listdir(scores_dir)
    assert any("original"     in c for c in csvs)
    assert any("Janus-Pro-1B" in c for c in csvs)


def test_scoring_csv_has_agent_columns(mini_apdd_dir, base_cfg, monkeypatch, tmp_path):
    _mock_scoring_deps(monkeypatch)
    from pipeline.scoring import run_scoring
    cfg  = {**base_cfg, "experiment": {**base_cfg["experiment"], "output_dir": str(tmp_path)}}
    run_scoring(cfg, _make_data_with_generated(mini_apdd_dir, tmp_path))
    df = pd.read_csv(os.path.join(str(tmp_path), "test_exp1", "scores", "scores_original.csv"))
    for agent in ["Total aesthetic score", "Color", "Mood"]:
        assert agent in df.columns


def test_scoring_no_all_nan_columns(mini_apdd_dir, base_cfg, monkeypatch, tmp_path):
    _mock_scoring_deps(monkeypatch)
    from pipeline.scoring import run_scoring
    cfg  = {**base_cfg, "experiment": {**base_cfg["experiment"], "output_dir": str(tmp_path)}}
    run_scoring(cfg, _make_data_with_generated(mini_apdd_dir, tmp_path))
    df = pd.read_csv(os.path.join(str(tmp_path), "test_exp1", "scores", "scores_original.csv"))
    assert not df.isnull().all().any()
