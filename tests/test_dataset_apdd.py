"""
Testa o loader do APDDv2.
Usa o mini dataset sintético criado pelo conftest.py.
"""
import os
import pytest
import torch
from datasets.apddv2 import APDDv2Dataset


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


def test_sample_uniform_bins_size(mini_apdd_dir):
    ds = APDDv2Dataset(root=mini_apdd_dir)
    subset = ds.sample(n=6, strategy="uniform_bins", seed=42, n_bins=30)
    assert len(subset) > 0
    assert len(subset) <= 6


def test_sample_uniform_bins_reproducible(mini_apdd_dir):
    ds = APDDv2Dataset(root=mini_apdd_dir)
    s1 = ds.sample(n=6, strategy="uniform_bins", seed=42)
    s2 = ds.sample(n=6, strategy="uniform_bins", seed=42)
    assert [s1[i]["filename"] for i in range(len(s1))] == [s2[i]["filename"] for i in range(len(s2))]


def test_sample_proportional_stratified_size(mini_apdd_dir):
    ds = APDDv2Dataset(root=mini_apdd_dir)
    subset = ds.sample(n=6, strategy="proportional_stratified", seed=42, n_bins=30)
    assert len(subset) > 0
    assert len(subset) <= 6


def test_sample_proportional_stratified_reproducible(mini_apdd_dir):
    ds = APDDv2Dataset(root=mini_apdd_dir)
    s1 = ds.sample(n=6, strategy="proportional_stratified", seed=42)
    s2 = ds.sample(n=6, strategy="proportional_stratified", seed=42)
    assert [s1[i]["filename"] for i in range(len(s1))] == [s2[i]["filename"] for i in range(len(s2))]


def test_sample_proportional_stratified_has_bin_report(mini_apdd_dir):
    ds = APDDv2Dataset(root=mini_apdd_dir)
    subset = ds.sample(n=6, strategy="proportional_stratified", seed=42)
    assert subset.bin_report is not None
    assert "proportional_stratified" in subset.bin_report
    assert f"Total amostrado: {len(subset)}" in subset.bin_report


def test_sample_uniform_bins_has_no_bin_report(mini_apdd_dir):
    """bin_report é exclusivo da proportional_stratified — outras estratégias não geram."""
    ds = APDDv2Dataset(root=mini_apdd_dir)
    subset = ds.sample(n=6, strategy="uniform_bins", seed=42)
    assert subset.bin_report is None


# ── proportional_stratified com legacy_csv (reusa amostra fixa externa) ────────

def _write_legacy_csv(tmp_path, filenames, col="filename"):
    import pandas as pd
    path = tmp_path / "sampled_dataset.csv"
    pd.DataFrame({col: filenames}).to_csv(path, index=False)
    return str(path)


def test_sample_proportional_stratified_legacy_csv_filters_exact_set(mini_apdd_dir, tmp_path):
    legacy_csv = _write_legacy_csv(tmp_path, ["img_000.jpg", "img_002.jpg", "img_005.jpg"])
    ds = APDDv2Dataset(root=mini_apdd_dir)
    subset = ds.sample(n=10, strategy="proportional_stratified", legacy_csv=legacy_csv)
    filenames = {subset[i]["filename"] for i in range(len(subset))}
    assert filenames == {"img_000.jpg", "img_002.jpg", "img_005.jpg"}


def test_sample_proportional_stratified_legacy_csv_ignores_extension(mini_apdd_dir, tmp_path):
    """Casa por stem — funciona mesmo se a extensão no CSV legado for diferente."""
    legacy_csv = _write_legacy_csv(tmp_path, ["img_000.png", "img_001.png"])
    ds = APDDv2Dataset(root=mini_apdd_dir)
    subset = ds.sample(n=10, strategy="proportional_stratified", legacy_csv=legacy_csv)
    filenames = {subset[i]["filename"] for i in range(len(subset))}
    assert filenames == {"img_000.jpg", "img_001.jpg"}


def test_sample_proportional_stratified_legacy_csv_warns_on_missing(mini_apdd_dir, tmp_path):
    legacy_csv = _write_legacy_csv(tmp_path, ["img_000.jpg", "nao_existe.jpg"])
    ds = APDDv2Dataset(root=mini_apdd_dir)
    with pytest.warns(RuntimeWarning, match="não encontradas"):
        subset = ds.sample(n=10, strategy="proportional_stratified", legacy_csv=legacy_csv)
    assert len(subset) == 1
    assert subset[0]["filename"] == "img_000.jpg"


def test_sample_proportional_stratified_legacy_csv_bin_report(mini_apdd_dir, tmp_path):
    legacy_csv = _write_legacy_csv(tmp_path, ["img_000.jpg", "img_001.jpg", "img_002.jpg"])
    ds = APDDv2Dataset(root=mini_apdd_dir)
    subset = ds.sample(n=10, strategy="proportional_stratified", legacy_csv=legacy_csv)
    assert subset.bin_report is not None
    assert "legacy" in subset.bin_report.lower()
    assert legacy_csv in subset.bin_report
    assert "3/3" in subset.bin_report


def test_sample_proportional_stratified_legacy_csv_missing_column_raises(mini_apdd_dir, tmp_path):
    legacy_csv = _write_legacy_csv(tmp_path, ["img_000.jpg"], col="not_a_filename_column")
    ds = APDDv2Dataset(root=mini_apdd_dir)
    with pytest.raises(ValueError, match="Coluna de filename"):
        ds.sample(n=10, strategy="proportional_stratified", legacy_csv=legacy_csv)


def test_sample_uniform_bins_ignores_legacy_csv(mini_apdd_dir, tmp_path):
    """legacy_csv só tem efeito em proportional_stratified — uniform_bins ignora e amostra normal."""
    legacy_csv = _write_legacy_csv(tmp_path, ["img_000.jpg"])
    ds = APDDv2Dataset(root=mini_apdd_dir)
    subset = ds.sample(n=6, strategy="uniform_bins", seed=42, legacy_csv=legacy_csv)
    assert len(subset) > 1  # não ficou restrito ao único item do CSV legado


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