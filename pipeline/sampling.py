# ── pipeline/sampling.py ──────────────────────────────────────────────────────

from datasets import load_dataset # seu loader customizado em datasets/
from torch.utils.data import DataLoader


def run_sampling(cfg) -> DataLoader:
    """
    Caixinha 1 — Amostragem.
    Carrega o dataset, amostra N itens conforme a estratégia do YAML
    e devolve um DataLoader pronto para as caixinhas seguintes.
    """
    dataset  = load_dataset(cfg)
    n        = cfg["sampling"]["n_samples"]
    strategy = cfg["sampling"].get("strategy", "random")
    seed     = cfg["experiment"].get("seed", 42)

    subset = dataset.sample(n=n, strategy=strategy, seed=seed)
    loader = DataLoader(subset, batch_size=cfg["captioning"].get("batch_size", 8), shuffle=False)
    return loader
