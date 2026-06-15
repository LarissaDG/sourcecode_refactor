from datasets.apddv2 import APDDv2Dataset
from datasets.portinari import PortinariDataset
from datasets.mnist import MNISTDataset
from datasets.gif_frames import GIFFramesDataset
from datasets.image import ImageDataset

REGISTRY = {
    "apdd":      APDDv2Dataset,
    "portinari": PortinariDataset,
    "mnist":     MNISTDataset,
    "gif":       GIFFramesDataset,
    "image":     ImageDataset,
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

    extra_kwargs = {}
    if name == "portinari":
        extra_kwargs["use_human_captions"] = cfg["dataset"].get("use_human_captions", False)

    return REGISTRY[name](root=path, **extra_kwargs)