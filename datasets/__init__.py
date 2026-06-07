from datasets.apddv2 import APDDv2Dataset
# from datasets.mnist      import MNISTDataset      # em breve
# from datasets.portinari  import PortinariDataset   # em breve
# from datasets.gif_frames import GIFFramesDataset   # em breve

REGISTRY = {
    "apdd":      APDDv2Dataset,
    # "mnist":     MNISTDataset,
    # "portinari": PortinariDataset,
    # "gif":       GIFFramesDataset,
}


def load_dataset(cfg: dict):
    """
    Instancia o dataset correto a partir das configs do YAML.

    Uso:
        dataset = load_dataset(cfg)
        subset  = dataset.sample(n=cfg["sampling"]["n_samples"],
                                 strategy=cfg["sampling"]["strategy"])
    """
    name = cfg["dataset"]["name"]
    path = cfg["dataset"]["path"]

    if name not in REGISTRY:
        raise ValueError(f"Dataset '{name}' não registrado. Disponíveis: {list(REGISTRY.keys())}")

    return REGISTRY[name](root=path)