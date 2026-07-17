import os
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from datasets.image import NOISE_FNS, DEFAULT_NOISE_LEVELS, DEFAULT_NOISE_TYPES


# Os 10 atributos estéticos do APDDv2 (sem o score total)
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
    "Mood",
]

# Os 9 atributos usados para calcular o score médio na amostragem por bins
# (exclui "The overall", que já é uma síntese dos demais)
BIN_ATTRIBUTES = [a for a in AESTHETIC_ATTRIBUTES if a != "The overall"]

# Possíveis nomes de colunas no CSV (CSV oficial vs. fixtures de teste)
SCORE_COL_CANDIDATES = ["Total aesthetic score", "Score"]
CATEGORY_COL_CANDIDATES = ["Artistic Categories", "category"]
COMMENT_COL_CANDIDATES = ["Language Comment", "comment"]


def _first_present(columns, candidates):
    for c in candidates:
        if c in columns:
            return c
    return None


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

    O CSV contém: filename, score total, atributos estéticos e comentários linguísticos.
    """

    def __init__(self, root: str, split: str = "all", transform=None):
        """
        Args:
            root:      Caminho raiz do dataset (onde está o CSV e a pasta images/).
            split:     "all" | "train" | "test" — usa a coluna 'split' do CSV se existir.
            transform: Transformações torchvision. Se None, usa o padrão para CLIP.
        """
        self.root = root
        # Aceita "images" ou "APDDv2images" (nome original do dataset)
        for _candidate in ("images", "APDDv2images"):
            _candidate_path = os.path.join(root, _candidate)
            if os.path.isdir(_candidate_path):
                self.images_dir = _candidate_path
                break
        else:
            self.images_dir = os.path.join(root, "images")

        csv_path = os.path.join(root, "APDDv2-10023.csv")
        self.df = pd.read_csv(csv_path, encoding="ISO-8859-1")

        self.score_col = _first_present(self.df.columns, SCORE_COL_CANDIDATES)
        self.category_col = _first_present(self.df.columns, CATEGORY_COL_CANDIDATES)
        self.comment_col = _first_present(self.df.columns, COMMENT_COL_CANDIDATES)
        self.bin_cols = [c for c in BIN_ATTRIBUTES if c in self.df.columns]

        if self.score_col:
            self.df = self.df.dropna(subset=[self.score_col])

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
        filename = str(row["filename"]).strip()
        path = self._resolve_path(filename)
        image = Image.open(path).convert("RGB")

        noise_type  = row.get("noise_type")  if hasattr(row, "get") else None
        noise_level = int(row["noise_level"]) if "noise_level" in row.index and pd.notna(row["noise_level"]) else 0

        if noise_type and noise_level > 0:
            np.random.seed(noise_level + idx)
            image = NOISE_FNS[noise_type](image, noise_level)

        image_t = self.transform(image)

        sample = {
            "image":       image_t,
            "filename":    filename,
            "path":        path,
            "noise_type":  noise_type or "none",
            "noise_level": noise_level,
        }

        sample["score"] = float(row[self.score_col]) if self.score_col else float("nan")

        # Sempre inclui todas as chaves para manter dicts do batch consistentes
        for attr in AESTHETIC_ATTRIBUTES:
            val = row[attr] if attr in row.index else float("nan")
            sample[attr.lower()] = float(val) if pd.notna(val) else float("nan")

        sample["caption"] = (
            str(row[self.comment_col])
            if self.comment_col and pd.notna(row[self.comment_col])
            else ""
        )
        sample["category"] = (
            str(row[self.category_col])
            if self.category_col and pd.notna(row[self.category_col])
            else ""
        )

        return sample

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_path(self, filename: str) -> str:
        path = os.path.join(self.images_dir, filename)
        if not os.path.exists(path):
            # Tenta extensão alternativa
            base = os.path.splitext(filename)[0]
            for ext in (".png", ".jpg", ".jpeg"):
                alt = os.path.join(self.images_dir, base + ext)
                if os.path.exists(alt):
                    path = alt
                    break
        return path

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

    def _make_subset(self, sampled_df: pd.DataFrame) -> "APDDv2Dataset":
        subset = APDDv2Dataset.__new__(APDDv2Dataset)
        subset.root        = self.root
        subset.images_dir  = self.images_dir
        subset.df          = sampled_df.reset_index(drop=True)
        subset.transform   = self.transform
        subset.score_col   = self.score_col
        subset.category_col = self.category_col
        subset.comment_col = self.comment_col
        subset.bin_cols    = self.bin_cols
        return subset

    # ------------------------------------------------------------------
    # Amostragem — chamada pela Caixinha 1
    # ------------------------------------------------------------------

    def sample(self, n: int, strategy: str = "random", seed: int = 42, n_bins: int = 30,
               noise_levels=None, noise_types=None, **kwargs) -> "APDDv2Dataset":
        """
        Retorna um subconjunto do dataset.

        Args:
            n:        Número de amostras desejadas.
            strategy: "random"       — amostragem aleatória simples.
                      "stratified"   — balanceia por categoria artística.
                      "uniform_bins" — calcula a média aritmética dos 9 atributos
                                        estéticos (BIN_ATTRIBUTES), divide o range
                                        em `n_bins` faixas de igual largura e amostra
                                        uniformemente entre elas.
            seed:     Semente para reprodutibilidade.
            n_bins:   Número de faixas usadas pela estratégia "uniform_bins".
        """
        if strategy == "random":
            sampled_df = self.df.sample(n=n, random_state=seed)

        elif strategy == "stratified":
            if not self.category_col:
                raise ValueError("Coluna de categoria não encontrada para amostragem estratificada.")
            n_categories = self.df[self.category_col].nunique()
            per_category = max(1, n // n_categories)
            sampled_df = (
                self.df
                .groupby(self.category_col, group_keys=False)
                .apply(lambda g: g.sample(min(len(g), per_category), random_state=seed))
                .sample(frac=1, random_state=seed)  # shuffle final
                .head(n)
            )

        elif strategy == "uniform_bins":
            if not self.bin_cols:
                raise ValueError("Nenhum dos atributos estéticos usados para o binning foi encontrado no CSV.")

            mean_score = self.df[self.bin_cols].mean(axis=1)
            n_bins_eff = max(1, min(n_bins, mean_score.nunique(), len(self.df)))

            df_binned = self.df.copy()
            df_binned["_bin"] = pd.cut(mean_score, bins=n_bins_eff, labels=False, duplicates="drop")

            per_bin = max(1, n // df_binned["_bin"].nunique())
            sampled_df = (
                df_binned
                .groupby("_bin", group_keys=False)
                .apply(lambda g: g.sample(min(len(g), per_bin), random_state=seed))
                .sample(frac=1, random_state=seed)  # shuffle final
                .head(n)
            )
            sampled_df = self.df.loc[sampled_df.index]

        else:
            raise ValueError(
                f"Estratégia desconhecida: '{strategy}'. "
                "Use 'random', 'stratified' ou 'uniform_bins'."
            )

        # Expansão por ruído: cada imagem × tipo × nível
        if noise_levels is not None or noise_types is not None:
            levels = list(noise_levels) if noise_levels is not None else DEFAULT_NOISE_LEVELS
            types  = list(noise_types)  if noise_types  is not None else DEFAULT_NOISE_TYPES
            rows = []
            for _, row in sampled_df.iterrows():
                for noise_type in types:
                    for level in levels:
                        r = row.to_dict()
                        r["noise_type"]  = noise_type
                        r["noise_level"] = level
                        rows.append(r)
            sampled_df = pd.DataFrame(rows)

        return self._make_subset(sampled_df)
