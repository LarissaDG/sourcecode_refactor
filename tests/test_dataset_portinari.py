import pytest

from datasets.portinari import PortinariDataset


def test_load_dataset_length(mini_portinari_dir):
    ds = PortinariDataset(root=mini_portinari_dir)
    assert len(ds) == 8


def test_getitem_keys_without_human_captions(mini_portinari_dir):
    ds = PortinariDataset(root=mini_portinari_dir, use_human_captions=False)
    item = ds[0]
    assert "image" in item
    assert "filename" in item
    assert "path" in item
    assert "caption" not in item


def test_getitem_keys_with_human_captions(mini_portinari_dir):
    ds = PortinariDataset(root=mini_portinari_dir, use_human_captions=True)
    item = ds[0]
    assert "caption" in item
    assert item["caption"].startswith("A Portinari painting number")


def test_sample_random_size_and_reproducibility(mini_portinari_dir):
    ds = PortinariDataset(root=mini_portinari_dir)
    s1 = ds.sample(n=5, strategy="random", seed=42)
    s2 = ds.sample(n=5, strategy="random", seed=42)
    assert len(s1) == 5
    assert [s1[i]["filename"] for i in range(len(s1))] == [s2[i]["filename"] for i in range(len(s2))]


def test_sample_unknown_strategy(mini_portinari_dir):
    ds = PortinariDataset(root=mini_portinari_dir)
    with pytest.raises(ValueError):
        ds.sample(n=5, strategy="stratified")


def test_use_human_captions_without_caption_column(tmp_path):
    import pandas as pd
    import numpy as np
    from PIL import Image

    img_dir = tmp_path / "Imagens"
    img_dir.mkdir()
    arr = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    Image.fromarray(arr).save(img_dir / "obra_000.jpg")
    pd.DataFrame([{"Numero da Obra": "obra_000.jpg"}]).to_csv(tmp_path / "acervo.csv", index=False)

    with pytest.raises(ValueError):
        PortinariDataset(root=str(tmp_path), use_human_captions=True)
