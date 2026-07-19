"""
Testa o loader de imagem-alvo com corrupção progressiva (exp5).
Usa o mini diretório sintético criado pelo conftest.py.
"""
import os
import torch
import pytest

from datasets.image import ImageDataset


def test_sample_noise_levels_size(mini_image_dir):
    ds = ImageDataset(root=mini_image_dir)
    levels = [0, 10, 20, 30]
    # 1 imagem base (mini_image_dir tem 1 imagem) × 1 tipo × 4 níveis = 4
    subset = ds.sample(n=4, strategy="noise_levels", noise_levels=levels, noise_types=["gaussian"])
    assert len(subset) == 4


def test_getitem_keys(mini_image_dir):
    ds = ImageDataset(root=mini_image_dir)
    subset = ds.sample(n=4, strategy="noise_levels", noise_levels=[0, 50])
    item = subset[0]
    assert "image" in item
    assert "filename" in item
    assert "path" in item
    assert "noise_level" in item
    assert isinstance(item["image"], torch.Tensor)
    assert item["image"].shape == (3, 224, 224)


def test_getitem_saves_png(mini_image_dir):
    ds = ImageDataset(root=mini_image_dir)
    subset = ds.sample(n=4, strategy="noise_levels", noise_levels=[0, 50])
    item = subset[0]
    assert os.path.exists(item["path"])


def test_level_zero_is_original(mini_image_dir):
    """Nível 0 não deve aplicar ruído (imagem == original)."""
    import numpy as np
    from PIL import Image

    ds = ImageDataset(root=mini_image_dir)
    subset = ds.sample(n=1, strategy="noise_levels", noise_levels=[0], noise_types=["gaussian"])
    item = subset[0]

    original_path = ds.df.iloc[0]["image_path"]
    original = np.array(Image.open(original_path).convert("RGB"))
    level0 = np.array(Image.open(item["path"]).convert("RGB"))
    assert np.array_equal(original, level0)


def test_levels_present_in_subset(mini_image_dir):
    # 1 imagem × 1 tipo × 4 níveis = 4 itens; todos os níveis pedidos estão presentes
    ds = ImageDataset(root=mini_image_dir)
    levels = [0, 10, 20, 30]
    subset = ds.sample(n=4, strategy="noise_levels", noise_levels=levels, noise_types=["gaussian"])
    assert len(subset) == 4
    found_levels = sorted({subset[i]["noise_level"] for i in range(len(subset))})
    assert found_levels == levels


def test_sample_unknown_strategy(mini_image_dir):
    ds = ImageDataset(root=mini_image_dir)
    with pytest.raises(ValueError, match="Estratégia desconhecida"):
        ds.sample(n=4, strategy="random")
