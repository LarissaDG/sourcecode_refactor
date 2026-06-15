"""
Fixtures compartilhadas entre todos os testes.
Cria um mini-dataset sintético em memória/disco temporário,
sem depender de GPU ou dos modelos reais.
"""
import os
import pytest
import numpy as np
import pandas as pd
from PIL import Image


# ── Mini dataset sintético ────────────────────────────────────────────────────

N_IMAGES = 10
CATEGORIES = ["oil_painting", "watercolor", "sketch"]

@pytest.fixture(scope="session")
def mini_apdd_dir(tmp_path_factory):
    """
    Cria um APDDv2 mini em disco temporário:
      <tmp>/mini_apdd/
          images/img_000.jpg ... img_009.jpg
          APDDv2-10023.csv
    """
    root = tmp_path_factory.mktemp("mini_apdd")
    img_dir = root / "images"
    img_dir.mkdir()

    rows = []
    for i in range(N_IMAGES):
        fname = f"img_{i:03d}.jpg"
        # Imagem sintética 64x64 com cor aleatória
        arr = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        Image.fromarray(arr).save(img_dir / fname)

        rows.append({
            "filename":    fname,
            "Score":       round(np.random.uniform(1, 10), 2),
            "Composition": round(np.random.uniform(1, 10), 2),
            "Color":       round(np.random.uniform(1, 10), 2),
            "Light":       round(np.random.uniform(1, 10), 2) if i % 3 != 0 else None,  # simula NaN
            "Mood":        round(np.random.uniform(1, 10), 2),
            # Atributos estéticos usados pela amostragem "uniform_bins"
            "Theme and logic":         round(np.random.uniform(1, 10), 2),
            "Creativity":              round(np.random.uniform(1, 10), 2),
            "Layout and composition":  round(np.random.uniform(1, 10), 2),
            "Space and perspective":   round(np.random.uniform(1, 10), 2),
            "The sense of order":      round(np.random.uniform(1, 10), 2),
            "Light and shadow":        round(np.random.uniform(1, 10), 2),
            "Details and texture":     round(np.random.uniform(1, 10), 2),
            "The overall":             round(np.random.uniform(1, 10), 2),
            "category":    CATEGORIES[i % len(CATEGORIES)],
            "comment":     f"A {CATEGORIES[i % len(CATEGORIES)]} painting number {i}.",
        })

    pd.DataFrame(rows).to_csv(root / "APDDv2-10023.csv", index=False)
    return str(root)


N_PORTINARI = 8

@pytest.fixture(scope="session")
def mini_portinari_dir(tmp_path_factory):
    """
    Cria um mini acervo Portinari em disco temporário:
      <tmp>/mini_portinari/
          Imagens/obra_000.jpg ... obra_007.jpg
          acervoPortinari.csv (Numero da Obra, Description_en)
    """
    root = tmp_path_factory.mktemp("mini_portinari")
    img_dir = root / "Imagens"
    img_dir.mkdir()

    rows = []
    for i in range(N_PORTINARI):
        fname = f"obra_{i:03d}.jpg"
        arr = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        Image.fromarray(arr).save(img_dir / fname)
        rows.append({
            "Numero da Obra": fname,
            "Description_en": f"A Portinari painting number {i}.",
        })

    pd.DataFrame(rows).to_csv(root / "acervoPortinari.csv", index=False)
    return str(root)


N_GIFS = 2
FRAMES_PER_GIF = 5

@pytest.fixture(scope="session")
def mini_gif_dir(tmp_path_factory):
    """
    Cria um mini conjunto de GIFs em disco temporário:
      <tmp>/mini_gifs/
          video_00.gif (5 frames)
          video_01.gif (5 frames)
    """
    root = tmp_path_factory.mktemp("mini_gifs")

    for g in range(N_GIFS):
        frames = []
        for f in range(FRAMES_PER_GIF):
            arr = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
            frames.append(Image.fromarray(arr))
        frames[0].save(
            root / f"video_{g:02d}.gif",
            save_all=True,
            append_images=frames[1:],
            duration=100,
            loop=0,
        )

    return str(root)


@pytest.fixture(scope="session")
def mini_image_dir(tmp_path_factory):
    """
    Cria um diretório com uma única imagem-alvo, para o exp5 (ruído).
      <tmp>/mini_image/target.png
    """
    root = tmp_path_factory.mktemp("mini_image")
    arr = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    Image.fromarray(arr).save(root / "target.png")
    return str(root)


@pytest.fixture(scope="session")
def base_cfg(mini_apdd_dir, tmp_path_factory):
    """Config mínima para rodar o pipeline completo sem GPU."""
    output_dir = str(tmp_path_factory.mktemp("outputs"))
    return {
        "experiment": {
            "name": "test_exp1",
            "seed": 42,
            "output_dir": output_dir,
        },
        "dataset": {
            "name": "apdd",
            "path": mini_apdd_dir,
        },
        "pipeline": {
            "steps": {
                "sampling":   True,
                "captioning": True,
                "generation": True,
                "scoring":    True,
            }
        },
        "sampling": {
            "n_samples": 6,
            "strategy":  "random",
        },
        "captioning": {
            "model":      "deepseek-ai/Janus-Pro-7B",
            "model_path": "deepseek-ai/Janus-Pro-7B",
            "batch_size": 2,
        },
        "generation": {
            "models": [
                {"name": "Janus-Pro-1B", "model_path": "deepseek-ai/Janus-Pro-1B"},
                {"name": "Janus-Pro-7B", "model_path": "deepseek-ai/Janus-Pro-7B"},
            ],
            "batch_size": 2,
            "num_images_per_prompt": 1,
        },
        "scoring": {
            "weights_dir": "/fake/weights",
            "artclip_base_weight": "0.ArtCLIP/fake_best.pth",
            "agents": [
                {"name": "score",      "weight_file": "1.Score/fake.pth"},
                {"name": "color",      "weight_file": "8.Color/fake.pth"},
                {"name": "mood",       "weight_file": "11.Mood/fake.pth"},
            ],
            "save_csv": True,
        },
    }


@pytest.fixture
def sample_data(mini_apdd_dir):
    """Lista de dicts simulando a saída da Caixinha 1 (pós-sampling)."""
    img_dir = os.path.join(mini_apdd_dir, "images")
    files = sorted(os.listdir(img_dir))[:6]
    return [
        {
            "filename": f,
            "image":    None,   # tensor omitido nos testes sem GPU
            "score":    7.5,
            "category": CATEGORIES[i % len(CATEGORIES)],
            "caption":  f"Caption for {f}",
        }
        for i, f in enumerate(files)
    ]
