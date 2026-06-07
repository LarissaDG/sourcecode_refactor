"""
Testa o loader do APDDv2.
Usa o mini dataset sintético criado pelo conftest.py.
"""
import os
import torch
from datasets.apdd import APDDv2Dataset


# ── Carregamento básico ───────────────────────────────────────────────────────

def test_dataset_loads(mini_apdd_dir):
    ds = APDDv2Dataset(root=mini_apdd_dir)
    assert len(ds) > 0


def test_dataset_len(mini_apdd_dir):
    ds = APDDv2Dataset(root=mini_apdd_dir)
    assert len(ds) == 10   # N_IMAGES definido no conftest


def test_getitem_keys(mini_apdd_dir):
    ds = APDDv2Dataset(root=mini_apdd_dir)
    item = ds[0]
    assert "image"    in item
    assert "filename" in item
    assert "score"    in item


def test_getitem_image_is_tensor(mini_apdd_dir):
    ds = APDDv2Dataset(root=mini_apdd_dir)
    item = ds[0]
    assert isinstance(item["image"], torch.Tensor)
    assert item["image"].shape == (3, 224, 224)


def test_getitem_score_is_float(mini_apdd_dir):
    ds = APDDv2Dataset(root=mini_apdd_dir)
    item = ds[0]
    assert isinstance(item["score"], float)
    assert 0 <= item["score"] <= 10


def test_getitem_caption_present(mini_apdd_dir):
    """O mini dataset tem coluna 'comment' → deve virar 'caption'."""
    ds = APDDv2Dataset(root=mini_apdd_dir)
    item = ds[0]
    assert "caption" in item
    assert isinstance(item["caption"], str)
    assert len(item["caption"]) > 0


def test_getitem_category_present(mini_apdd_dir):
    ds = APDDv2Dataset(root=mini_apdd_dir)
    item = ds[0]
    assert "category" in item


# ── Amostragem ────────────────────────────────────────────────────────────────

def test_sample_random_size(mini_apdd_dir):
    ds     = APDDv2Dataset(root=mini_apdd_dir)
    subset = ds.sample(n=5, strategy="random", seed=42)
    assert len(subset) == 5


def test_sample_is_reproducible(mini_apdd_dir):
    ds = APDDv2Dataset(root=mini_apdd_dir)
    s1 = ds.sample(n=5, seed=42)
    s2 = ds.sample(n=5, seed=42)
    assert [s1[i]["filename"] for i in range(5)] == [s2[i]["filename"] for i in range(5)]


def test_sample_different_seeds(mini_apdd_dir):
    ds = APDDv2Dataset(root=mini_apdd_dir)
    s1 = ds.sample(n=5, seed=0)
    s2 = ds.sample(n=5, seed=99)
    names1 = [s1[i]["filename"] for i in range(5)]
    names2 = [s2[i]["filename"] for i in range(5)]
    assert names1 != names2


def test_sample_stratified(mini_apdd_dir):
    ds     = APDDv2Dataset(root=mini_apdd_dir)
    subset = ds.sample(n=6, strategy="stratified", seed=42)
    assert len(subset) > 0
    categories = [subset[i]["category"] for i in range(len(subset))]
    # Deve ter mais de uma categoria
    assert len(set(categories)) > 1


def test_sample_unknown_strategy(mini_apdd_dir):
    import pytest
    ds = APDDv2Dataset(root=mini_apdd_dir)
    with pytest.raises(ValueError, match="Estratégia desconhecida"):
        ds.sample(n=5, strategy="inexistente")


def test_sample_returns_valid_dataset(mini_apdd_dir):
    """Subset retornado deve se comportar como Dataset normal."""
    ds     = APDDv2Dataset(root=mini_apdd_dir)
    subset = ds.sample(n=4)
    item   = subset[0]
    assert "image" in item
    assert isinstance(item["image"], torch.Tensor)
