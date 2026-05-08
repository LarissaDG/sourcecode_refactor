import yaml
from pathlib import Path


BASE_CONFIG_PATH = Path("configs/base.yaml")


def _deep_merge(base: dict, override: dict) -> dict:
    """Merge recursivo: override sobrescreve base, sem apagar chaves não mencionadas."""
    merged = base.copy()
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = _deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged


def load_config(exp_config_path: str) -> dict:
    """
    Carrega base.yaml e faz merge com o yaml do experimento.
    O yaml do experimento sempre tem precedência.
    """
    with open(BASE_CONFIG_PATH) as f:
        base_cfg = yaml.safe_load(f)

    with open(exp_config_path) as f:
        exp_cfg = yaml.safe_load(f)

    cfg = _deep_merge(base_cfg, exp_cfg)
    return cfg