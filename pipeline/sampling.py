from datasets import load_dataset
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

    extra_kwargs = {}
    if "n_bins" in cfg["sampling"]:
        extra_kwargs["n_bins"] = cfg["sampling"]["n_bins"]
    if "noise_levels" in cfg["sampling"]:
        extra_kwargs["noise_levels"] = cfg["sampling"]["noise_levels"]

    subset = dataset.sample(n=n, strategy=strategy, seed=seed, **extra_kwargs)
    loader = DataLoader(subset, batch_size=cfg["sampling"].get("batch_size", 8), shuffle=False)
    return loader


def loader_to_list(loader: DataLoader) -> list:
    """
    Achata um DataLoader em uma lista de dicts (um por amostra), no mesmo
    formato produzido pela Caixinha 2 (captioning). Usado quando a
    captioning é pulada (ex: Exp 3b, com descrições humanas já no CSV).
    """
    results = []
    for batch in loader:
        n = len(batch["filename"])
        for i in range(n):
            item = {}
            for key, values in batch.items():
                item[key] = values[i]
            results.append(item)
    return results