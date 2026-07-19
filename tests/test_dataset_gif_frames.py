"""
Testa o loader de frames de GIFs.
Usa o mini conjunto de GIFs sintético criado pelo conftest.py.
"""
import os
import torch
import pytest

from datasets.gif_frames import GIFFramesDataset


def test_dataset_loads(mini_gif_dir):
    ds = GIFFramesDataset(root=mini_gif_dir)
    # 2 gifs x 5 frames
    assert len(ds) == 10


def test_getitem_keys(mini_gif_dir):
    ds = GIFFramesDataset(root=mini_gif_dir)
    item = ds[0]
    assert "image" in item
    assert "filename" in item
    assert "path" in item
    assert "gif_name" in item
    assert "frame_idx" in item
    assert isinstance(item["image"], torch.Tensor)
    assert item["image"].shape == (3, 224, 224)


def test_getitem_saves_png(mini_gif_dir):
    ds = GIFFramesDataset(root=mini_gif_dir)
    item = ds[0]
    assert os.path.exists(item["path"])


def test_sample_sequential_size(mini_gif_dir):
    ds = GIFFramesDataset(root=mini_gif_dir)
    subset = ds.sample(n=4, strategy="sequential", seed=42)
    assert len(subset) > 0
    assert len(subset) <= 4


def test_sample_sequential_preserves_order(mini_gif_dir):
    ds = GIFFramesDataset(root=mini_gif_dir)
    subset = ds.sample(n=4, strategy="sequential", seed=42)
    # Para cada gif presente, os frame_idx devem estar em ordem crescente
    by_gif = {}
    for i in range(len(subset)):
        item = subset[i]
        by_gif.setdefault(item["gif_name"], []).append(item["frame_idx"])
    for indices in by_gif.values():
        assert indices == sorted(indices)
        assert indices[0] == 0  # sequencial começa do primeiro frame


def test_sample_random_reproducible(mini_gif_dir):
    ds = GIFFramesDataset(root=mini_gif_dir)
    s1 = ds.sample(n=4, strategy="random", seed=42)
    s2 = ds.sample(n=4, strategy="random", seed=42)
    assert [s1[i]["filename"] for i in range(len(s1))] == [s2[i]["filename"] for i in range(len(s2))]


def test_sample_unknown_strategy(mini_gif_dir):
    ds = GIFFramesDataset(root=mini_gif_dir)
    with pytest.raises(ValueError, match="Estratégia desconhecida"):
        ds.sample(n=4, strategy="uniform_bins")
