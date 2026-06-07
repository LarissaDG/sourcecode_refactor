"""
Testa o carregamento e merge de YAMLs.
Não precisa de GPU nem de dataset real.
"""
import pytest
from utils.config import load_config, _deep_merge


# ── _deep_merge ───────────────────────────────────────────────────────────────

def test_merge_simple_override():
    base     = {"a": 1, "b": 2}
    override = {"b": 99}
    result   = _deep_merge(base, override)
    assert result["a"] == 1
    assert result["b"] == 99


def test_merge_nested():
    base     = {"sampling": {"n_samples": 500, "strategy": "random"}}
    override = {"sampling": {"strategy": "stratified"}}
    result   = _deep_merge(base, override)
    assert result["sampling"]["n_samples"] == 500     # manteve do base
    assert result["sampling"]["strategy"] == "stratified"  # sobrescreveu


def test_merge_does_not_mutate_base():
    base     = {"a": {"x": 1}}
    override = {"a": {"x": 99}}
    _deep_merge(base, override)
    assert base["a"]["x"] == 1   # base intocado


# ── load_config ───────────────────────────────────────────────────────────────

def test_load_config_returns_dict(tmp_path):
    base_cfg = tmp_path / "base.yaml"
    exp_cfg  = tmp_path / "exp.yaml"

    base_cfg.write_text("""
experiment:
  seed: 42
sampling:
  n_samples: 500
  strategy: random
""")
    exp_cfg.write_text("""
experiment:
  name: test_exp
sampling:
  n_samples: 100
""")

    # Patch temporário do caminho do base.yaml
    import utils.config as cfg_module
    original = cfg_module.BASE_CONFIG_PATH
    cfg_module.BASE_CONFIG_PATH = base_cfg

    cfg = load_config(str(exp_cfg))

    cfg_module.BASE_CONFIG_PATH = original  # restaura

    assert isinstance(cfg, dict)
    assert cfg["experiment"]["seed"] == 42           # veio do base
    assert cfg["experiment"]["name"] == "test_exp"   # veio do exp
    assert cfg["sampling"]["n_samples"] == 100       # exp sobrescreveu
    assert cfg["sampling"]["strategy"] == "random"   # manteve do base


def test_load_config_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(str(tmp_path / "nao_existe.yaml"))
