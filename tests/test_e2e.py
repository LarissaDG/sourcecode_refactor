"""
Teste end-to-end do pipeline completo do Exp 1.
Todas as dependências de modelo são mockadas — roda sem GPU.
"""
import os
import numpy as np
import pandas as pd
from PIL import Image
from unittest.mock import MagicMock
from torchvision import transforms
from torch.utils.data import DataLoader


def test_full_pipeline_exp1(mini_apdd_dir, base_cfg, monkeypatch, tmp_path):
    """
    Roda o pipeline completo: sampling → captioning → generation → scoring.
    Valida que:
      - Cada etapa passa dados para a próxima corretamente
      - CSVs de scores são gerados para 'original', '1B' e '7B'
      - Nenhum arquivo intermediário fica faltando
    """
    # ── Mocks ──────────────────────────────────────────────────────────────
    import pipeline.captioning as cap
    import pipeline.generation as gen
    import pipeline.scoring    as sc

    fake_model     = MagicMock()
    fake_processor = MagicMock()

    monkeypatch.setattr(cap, "_load_janus",     lambda *a, **kw: (fake_model, fake_processor))
    monkeypatch.setattr(cap, "_describe_image", lambda *a, **kw: "A beautiful painting.")
    monkeypatch.setattr(gen, "_load_janus",     lambda *a, **kw: (fake_model, fake_processor))
    monkeypatch.setattr(gen, "_generate_image", lambda *a, **kw: [
        Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
    ])
    import sys
    from torchvision import transforms as T
    fake_clip = MagicMock()
    fake_clip.load.return_value = (None, T.Compose([T.Resize((224, 224)), T.ToTensor()]))
    monkeypatch.setitem(sys.modules, "models",         MagicMock())
    monkeypatch.setitem(sys.modules, "models.clip",    fake_clip)
    monkeypatch.setitem(sys.modules, "models.aesclip", MagicMock())
    monkeypatch.setattr(sc, "_load_agent", lambda *a, **kw: MagicMock())
    monkeypatch.setattr(sc, "_predict",    lambda model, t: np.random.uniform(0.1, 1.0))

    # ── Config apontando para tmp_path ─────────────────────────────────────
    cfg = {
        **base_cfg,
        "experiment": {
            **base_cfg["experiment"],
            "output_dir": str(tmp_path),
        },
        "dataset": {"name": "apdd", "path": mini_apdd_dir},
    }

    # ── Rodar pipeline ─────────────────────────────────────────────────────
    from pipeline.sampling   import run_sampling
    from pipeline.captioning import run_captioning
    from pipeline.generation import run_generation
    from pipeline.scoring    import run_scoring

    loader = run_sampling(cfg)
    assert isinstance(loader, DataLoader), "Sampling deve retornar DataLoader"

    data = run_captioning(cfg, loader)
    assert len(data) == cfg["sampling"]["n_samples"], "Captioning deve preservar n_samples"
    assert all("caption" in d for d in data), "Todos os itens devem ter caption"

    data = run_generation(cfg, data)
    assert all("generated_Janus-Pro-1B" in d for d in data), "Geração 1B ausente"
    assert all("generated_Janus-Pro-7B" in d for d in data), "Geração 7B ausente"

    run_scoring(cfg, data)

    # ── Valida outputs ─────────────────────────────────────────────────────
    scores_dir = os.path.join(str(tmp_path), "test_exp1", "scores")
    assert os.path.isdir(scores_dir), "Pasta scores/ não foi criada"

    expected_csvs = ["scores_original.csv", "scores_Janus-Pro-1B.csv", "scores_Janus-Pro-7B.csv"]
    for csv_name in expected_csvs:
        csv_path = os.path.join(scores_dir, csv_name)
        assert os.path.exists(csv_path), f"{csv_name} não foi gerado"

        df = pd.read_csv(csv_path)
        assert len(df) > 0,               f"{csv_name} está vazio"
        assert "filename" in df.columns,  f"{csv_name} não tem coluna 'filename'"
        assert not df.isnull().all().any(), f"{csv_name} tem coluna 100% NaN"

    # ── Valida imagens geradas ─────────────────────────────────────────────
    for model_name in ["Janus-Pro-1B", "Janus-Pro-7B"]:
        gen_dir = os.path.join(str(tmp_path), "test_exp1", "generated", model_name)
        assert os.path.isdir(gen_dir), f"Pasta generated/{model_name} não foi criada"
        assert len(os.listdir(gen_dir)) > 0, f"Nenhuma imagem salva em {model_name}"