import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


# Atributos estéticos disponíveis no APDDv2
AESTHETIC_ATTRIBUTES = [
    "Theme and logic", 
    "Creativity", 
    "Layout and composition",
    "Space and perspective",
    "The sense of order", 
    "Light and shadow",
    "Color",
    "Details and texture",
    "The overall",
    "Mood"
]


class APDDv2Dataset(Dataset):
    """
    Loader para o APDDv2.

    Estrutura esperada no disco:
        <root>/
            APDDv2-10023.csv
            images/
                painting_001.jpg
                painting_002.jpg
                ...

    O CSV contém: filename, Score, atributos estéticos e comentários linguísticos.
    """

    def __init__(self, root: str, split: str = "all", transform=None):
        """
        Args:
            root:      Caminho raiz do dataset (onde está o CSV e a pasta images/).
            split:     "all" | "train" | "test" — usa a coluna 'split' do CSV se existir.
            transform: Transformações torchvision. Se None, usa o padrão para CLIP.
        """
        self.root = root
        self.images_dir = os.path.join(root, "images")

        csv_path = os.path.join(root, "APDDv2-10023.csv")
        self.df = pd.read_csv(csv_path, encoding="ISO-8859-1")
        self.df = self.df.dropna(subset=["Score"])

        if split != "all" and "split" in self.df.columns:
            self.df = self.df[self.df["split"] == split].reset_index(drop=True)

        self.transform = transform or self._default_transform()

    # ------------------------------------------------------------------
    # Interface Dataset
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        image = self._load_image(str(row["filename"]).strip())
        image_t = self.transform(image)

        sample = {
            "image":    image_t,
            "filename": str(row["filename"]).strip(),
            "score":    float(row["Score"]),
        }

        # Adiciona atributos estéticos presentes no CSV
        for attr in AESTHETIC_ATTRIBUTES[1:]:  # pula "Score"
            if attr in row and pd.notna(row[attr]):
                sample[attr.lower()] = float(row[attr])

        # Comentário linguístico (quando disponível — usado no Exp 3b)
        if "comment" in row and pd.notna(row["comment"]):
            sample["caption"] = str(row["comment"])

        # Categoria artística
        if "category" in row and pd.notna(row["category"]):
            sample["category"] = str(row["category"])

        return sample

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_image(self, filename: str) -> Image.Image:
        path = os.path.join(self.images_dir, filename)
        if not os.path.exists(path):
            # Tenta extensão alternativa
            base = os.path.splitext(filename)[0]
            for ext in (".png", ".jpg", ".jpeg"):
                alt = os.path.join(self.images_dir, base + ext)
                if os.path.exists(alt):
                    path = alt
                    break
        return Image.open(path).convert("RGB")

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

    # ------------------------------------------------------------------
    # Amostragem — chamada pela Caixinha 1
    # ------------------------------------------------------------------

    def sample(self, n: int, strategy: str = "random", seed: int = 42) -> "APDDv2Dataset":
        """
        Retorna um subconjunto do dataset.

        Args:
            n:        Número de amostras desejadas.
            strategy: "random"     — amostragem aleatória simples.
                      "stratified" — balanceia por categoria artística.
            seed:     Semente para reprodutibilidade.
        """
        if strategy == "random":
            sampled_df = self.df.sample(n=n, random_state=seed)

        elif strategy == "stratified":
            if "category" not in self.df.columns:
                raise ValueError("Coluna 'category' não encontrada para amostragem estratificada.")
            n_categories = self.df["category"].nunique()
            per_category = max(1, n // n_categories)
            sampled_df = (
                self.df
                .groupby("category", group_keys=False)
                .apply(lambda g: g.sample(min(len(g), per_category), random_state=seed))
                .sample(frac=1, random_state=seed)  # shuffle final
                .head(n)
            )
        else:
            raise ValueError(f"Estratégia desconhecida: '{strategy}'. Use 'random' ou 'stratified'.")

        subset = APDDv2Dataset.__new__(APDDv2Dataset)
        subset.root       = self.root
        subset.images_dir = self.images_dir
        subset.df         = sampled_df.reset_index(drop=True)
        subset.transform  = self.transform
        return subset