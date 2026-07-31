"""
Testa a orquestração multi-estratégia do run.py (sampling.strategies),
usada pelo exp1_apdd.yaml para rodar uniform_bins + proportional_stratified
em pastas de output separadas. Sampling é mockado — não depende de GPU.
"""
import os
import json

import run as run_module


def _fake_run_sampling(cfg):
    class FakeLoader:
        def __iter__(self):
            yield {"filename": ["a.jpg", "b.jpg"], "score": [7.0, 8.0]}
    return FakeLoader()


def _base_cfg(tmp_path, strategies):
    return {
        "experiment": {"name": "exp_multi", "seed": 42, "output_dir": str(tmp_path)},
        "dataset":    {"name": "apdd", "path": "unused"},
        "pipeline":   {"steps": {"sampling": True, "captioning": False,
                                  "generation": False, "scoring": False}},
        "sampling":   {"n_samples": 2, "n_bins": 5, "strategies": strategies},
    }


def test_multi_strategy_creates_separate_output_dirs(monkeypatch, tmp_path):
    monkeypatch.setattr(run_module, "run_sampling", _fake_run_sampling)
    cfg = _base_cfg(tmp_path, ["uniform_bins", "proportional_stratified"])

    run_module.run_pipeline(cfg)

    for strategy in ["uniform_bins", "proportional_stratified"]:
        data_path = os.path.join(str(tmp_path), f"exp_multi_{strategy}", "pipeline_data.json")
        assert os.path.exists(data_path)
        with open(data_path) as f:
            data = json.load(f)
        assert len(data) == 2


def test_multi_strategy_leaves_base_cfg_untouched(monkeypatch, tmp_path):
    """cfg original não deve ser mutado pelo loop (cada estratégia usa uma deepcopy)."""
    monkeypatch.setattr(run_module, "run_sampling", _fake_run_sampling)
    cfg = _base_cfg(tmp_path, ["uniform_bins", "proportional_stratified"])

    run_module.run_pipeline(cfg)

    assert cfg["experiment"]["name"] == "exp_multi"
    assert cfg["sampling"]["strategies"] == ["uniform_bins", "proportional_stratified"]
    assert "strategy" not in cfg["sampling"]


def test_single_strategy_unaffected(monkeypatch, tmp_path):
    """Sem `strategies`, comportamento de single-run é preservado."""
    monkeypatch.setattr(run_module, "run_sampling", _fake_run_sampling)
    cfg = _base_cfg(tmp_path, None)
    del cfg["sampling"]["strategies"]
    cfg["sampling"]["strategy"] = "random"

    run_module.run_pipeline(cfg)

    data_path = os.path.join(str(tmp_path), "exp_multi", "pipeline_data.json")
    assert os.path.exists(data_path)
