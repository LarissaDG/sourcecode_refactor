import os
import pandas as pd
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms


class MNISTDataset(Dataset):
    """
    Loader para o MNIST.

    Usa torchvision.datasets.MNIST (baixa automaticamente para <root> se
    necessário). Cada item, ao ser acessado, é salvo como PNG em
    <root>/mnist_cache/ para que a Caixinha 4 (scoring) possa abri-lo via
    'path', assim como os demais datasets.
    """

    def __init__(self, root: str, train: bool = True, transform=None):
        self.root = root
        self._mnist = torchvision.datasets.MNIST(root=root, train=train, download=True)
        self.cache_dir = os.path.join(root, "mnist_cache")
        os.makedirs(self.cache_dir, exist_ok=True)

        self.df = pd.DataFrame({
            "index": range(len(self._mnist)),
            "digit": [int(d) for d in self._mnist.targets],
        })

        self.transform = transform or self._default_transform()

    # ------------------------------------------------------------------
    # Interface Dataset
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        mnist_idx = int(row["index"])
        digit = int(row["digit"])

        img, _ = self._mnist[mnist_idx]
        img = img.convert("RGB")

        filename = f"mnist_{mnist_idx:05d}.png"
        path = os.path.join(self.cache_dir, filename)
        if not os.path.exists(path):
            img.save(path)

        image_t = self.transform(img)

        return {
            "image":    image_t,
            "filename": filename,
            "path":     path,
            "digit":    digit,
        }

    @staticmethod
    def _default_transform() -> transforms.Compose:
        """Normalização padrão do CLIP."""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275,  0.40821073],
                std= [0.26862954, 0.26130258, 0.27577711],
            ),
        ])

    def _make_subset(self, sampled_df: pd.DataFrame) -> "MNISTDataset":
        subset = MNISTDataset.__new__(MNISTDataset)
        subset.root      = self.root
        subset._mnist    = self._mnist
        subset.cache_dir = self.cache_dir
        subset.df        = sampled_df.reset_index(drop=True)
        subset.transform = self.transform
        return subset

    # ------------------------------------------------------------------
    # Amostragem — chamada pela Caixinha 1
    # ------------------------------------------------------------------

    def sample(self, n: int, strategy: str = "stratified", seed: int = 42, **kwargs) -> "MNISTDataset":
        """
        Retorna um subconjunto do dataset.

        Args:
            n:        Número de amostras desejadas.
            strategy: "random"     — amostragem aleatória simples.
                      "stratified" — balanceia por dígito (0-9).
            seed:     Semente para reprodutibilidade.
        """
        if strategy == "random":
            sampled_df = self.df.sample(n=min(n, len(self.df)), random_state=seed)

        elif strategy == "stratified":
            n_digits = self.df["digit"].nunique()
            per_digit = max(1, n // n_digits)
            sampled_df = (
                self.df
                .groupby("digit", group_keys=False)
                .apply(lambda g: g.sample(min(len(g), per_digit), random_state=seed), include_groups=False)
                .sample(frac=1, random_state=seed)  # shuffle final
                .head(n)
            )
            sampled_df = self.df.loc[sampled_df.index]

        else:
            raise ValueError(f"Estratégia desconhecida: '{strategy}'. Use 'random' ou 'stratified'.")

        return self._make_subset(sampled_df)
