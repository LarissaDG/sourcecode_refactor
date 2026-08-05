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
    monkeypatch.setattr(gen, "_generate_token_based", lambda *a, **kw:
        Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
    )


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


# ═══════════════════════════════════════════════════════════════════════════════
# CAIXINHA 4 — reaplicação de ruído no scoring (Exp4/Exp5b)
#
# `data` passa por _save_data/_load_data (JSON) entre etapas do pipeline, então
# o tensor de imagem já-ruidosa nunca sobrevive até run_scoring — o campo
# 'image' chega sempre None. Sem reaplicar o ruído a partir de
# noise_type/noise_level, toda amostra "ruidosa" era pontuada como a imagem
# original, produzindo o mesmo score pra todos os níveis/tipos (bug real
# encontrado analisando outputs/exp4_noise e outputs/exp5b_temporal_error).
# ═══════════════════════════════════════════════════════════════════════════════

def _make_solid_image(path, color=(120, 120, 120)):
    Image.new("RGB", (64, 64), color=color).save(path)


def test_apply_noise_if_needed_changes_pixels(tmp_path):
    from pipeline.scoring import _apply_noise_if_needed

    path = os.path.join(str(tmp_path), "img.png")
    _make_solid_image(path)
    image = Image.open(path).convert("RGB")

    noised = _apply_noise_if_needed(
        image, {"noise_type": "gaussian", "noise_level": 80}, path
    )
    assert np.array(noised).tobytes() != np.array(image).tobytes()


def test_apply_noise_if_needed_noop_without_noise(tmp_path):
    from pipeline.scoring import _apply_noise_if_needed

    path = os.path.join(str(tmp_path), "img.png")
    _make_solid_image(path)
    image = Image.open(path).convert("RGB")

    for sample in [
        {"noise_type": "none", "noise_level": 50},
        {"noise_type": None, "noise_level": 50},
        {"noise_type": "gaussian", "noise_level": 0},
        {},
    ]:
        result = _apply_noise_if_needed(image, sample, path)
        assert np.array(result).tobytes() == np.array(image).tobytes()


def test_apply_noise_if_needed_is_reproducible(tmp_path):
    from pipeline.scoring import _apply_noise_if_needed

    path = os.path.join(str(tmp_path), "img.png")
    _make_solid_image(path)
    image = Image.open(path).convert("RGB")
    sample = {"noise_type": "shapes", "noise_level": 60}

    a = _apply_noise_if_needed(image, sample, path)
    b = _apply_noise_if_needed(image, sample, path)
    assert np.array(a).tobytes() == np.array(b).tobytes()


def test_apply_noise_if_needed_differs_by_level(tmp_path):
    from pipeline.scoring import _apply_noise_if_needed

    path = os.path.join(str(tmp_path), "img.png")
    _make_solid_image(path)
    image = Image.open(path).convert("RGB")

    low  = _apply_noise_if_needed(image, {"noise_type": "gaussian", "noise_level": 10}, path)
    high = _apply_noise_if_needed(image, {"noise_type": "gaussian", "noise_level": 90}, path)
    assert np.array(low).tobytes() != np.array(high).tobytes()


def _mock_scoring_deps_pixel_aware(monkeypatch):
    """Como _mock_scoring_deps, mas o `_predict` mockado depende de verdade do
    conteúdo do tensor (média dos pixels), pra detectar se o ruído foi
    realmente aplicado antes da predição — o mock padrão (`np.random.uniform`)
    não seria sensível a isso."""
    monkeypatch.setitem(sys.modules, "models",          MagicMock())
    monkeypatch.setitem(sys.modules, "models.clip",     _fake_clip)
    monkeypatch.setitem(sys.modules, "models.aesclip",  MagicMock())

    import pipeline.scoring as sc
    monkeypatch.setattr(sc, "_load_agent", lambda *a, **kw: MagicMock())
    monkeypatch.setattr(sc, "_predict",    lambda model, t: float(t.mean()))


def test_scoring_produces_different_scores_per_noise_level(tmp_path, base_cfg, monkeypatch):
    """Regressão pro bug real: mesma imagem-base, dois níveis de ruído
    diferentes -> scores diferentes no CSV final (antes do fix, eram iguais)."""
    _mock_scoring_deps_pixel_aware(monkeypatch)
    from pipeline.scoring import run_scoring

    img_path = os.path.join(str(tmp_path), "base.png")
    _make_solid_image(img_path, color=(100, 150, 200))

    data = [
        {"filename": img_path, "path": img_path, "noise_type": "gaussian", "noise_level": 10},
        {"filename": img_path, "path": img_path, "noise_type": "gaussian", "noise_level": 90},
    ]
    cfg = {**base_cfg, "experiment": {**base_cfg["experiment"], "output_dir": str(tmp_path)}}
    run_scoring(cfg, data)

    df = pd.read_csv(os.path.join(str(tmp_path), "test_exp1", "scores", "scores_original.csv"))
    scores = df["Total aesthetic score"].tolist()
    assert len(scores) == 2
    assert scores[0] != scores[1]
