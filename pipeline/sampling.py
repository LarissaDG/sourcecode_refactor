import os
from datasets import load_dataset
from torch.utils.data import DataLoader


def _write_bin_report(cfg, subset):
    """Se a estratégia de amostragem gerou um relatório de distribuição por
    bin (ex: proportional_stratified), grava em <output_dir>/<nome>/sampling_bin_distribution.txt."""
    bin_report = getattr(subset, "bin_report", None)
    if not bin_report:
        return
    out_dir = os.path.join(cfg["experiment"]["output_dir"], cfg["experiment"]["name"])
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "sampling_bin_distribution.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(bin_report)


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
    for key in ("n_bins", "noise_levels", "noise_types",
                "error_frame", "error_type", "error_level"):
        if key in cfg["sampling"]:
            extra_kwargs[key] = cfg["sampling"][key]

    subset = dataset.sample(n=n, strategy=strategy, seed=seed, **extra_kwargs)
    _write_bin_report(cfg, subset)
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