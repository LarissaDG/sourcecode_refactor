"""
Testa a Caixinha 1 (sampling).
Mock do dataset para rodar sem GPU.
"""
from unittest.mock import patch, MagicMock
from torch.utils.data import DataLoader
from pipeline.sampling import run_sampling


def _make_cfg(mini_apdd_dir, tmp_path, strategy="random", n=6):
    return {
        "experiment": {"name": "test", "seed": 42, "output_dir": str(tmp_path)},
        "dataset":    {"name": "apdd", "path": mini_apdd_dir},
        "sampling":   {"n_samples": n, "strategy": strategy},
        "captioning": {"batch_size": 2},
    }


def test_sampling_returns_dataloader(mini_apdd_dir, tmp_path):
    cfg    = _make_cfg(mini_apdd_dir, tmp_path)
    loader = run_sampling(cfg)
    assert isinstance(loader, DataLoader)


def test_sampling_correct_size(mini_apdd_dir, tmp_path):
    cfg    = _make_cfg(mini_apdd_dir, tmp_path, n=6)
    loader = run_sampling(cfg)
    total  = sum(len(b["filename"]) for b in loader)
    assert total == 6


def test_sampling_batch_has_required_keys(mini_apdd_dir, tmp_path):
    cfg   = _make_cfg(mini_apdd_dir, tmp_path)
    batch = next(iter(run_sampling(cfg)))
    assert "image"    in batch
    assert "filename" in batch
    assert "score"    in batch


def test_sampling_stratified(mini_apdd_dir, tmp_path):
    cfg    = _make_cfg(mini_apdd_dir, tmp_path, strategy="stratified", n=6)
    loader = run_sampling(cfg)
    total  = sum(len(b["filename"]) for b in loader)
    assert total > 0


def test_sampling_reproducible(mini_apdd_dir, tmp_path):
    cfg = _make_cfg(mini_apdd_dir, tmp_path, n=6)
    l1  = list(run_sampling(cfg))
    l2  = list(run_sampling(cfg))
    names1 = [f for b in l1 for f in b["filename"]]
    names2 = [f for b in l2 for f in b["filename"]]
    assert names1 == names2
