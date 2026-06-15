import glob
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


# Possíveis nomes de coluna no CSV do acervo Portinari
FILENAME_COL_CANDIDATES = ["Numero da Obra"]
CAPTION_COL_CANDIDATES = ["Description_en", "Descrição", "Description", "caption"]

# Possíveis nomes da pasta de imagens dentro de <root>
IMAGES_DIR_CANDIDATES = ["Imagens", "AmostraImagens", "images"]


def _first_present(columns, candidates):
    for c in candidates:
        if c in columns:
            return c
    return None


def _find_images_dir(root: str) -> str:
    for name in IMAGES_DIR_CANDIDATES:
        candidate = os.path.join(root, name)
        if os.path.isdir(candidate):
            return candidate
    return os.path.join(root, IMAGES_DIR_CANDIDATES[0])


def _find_csv(root: str) -> str:
    csvs = sorted(glob.glob(os.path.join(root, "*.csv")))
    if not csvs:
        raise FileNotFoundError(f"Nenhum CSV encontrado em {root}")
    return csvs[0]


class PortinariDataset(Dataset):
    """
    Loader para o acervo Portinari.

    Estrutura esperada no disco:
        <root>/
            <algumAcervo>.csv     (coluna 'Numero da Obra' com o nome do arquivo da imagem)
            Imagens/ ou AmostraImagens/
                obra_001.jpg
                obra_002.jpg
                ...

    Se `use_human_captions=True` e o CSV tiver uma coluna de descrição
    (ex: 'Description_en'), cada item já vem com 'caption' preenchido —
    a Caixinha 2 (captioning) deve ser pulada no YAML deste experimento.
    """

    def __init__(self, root: str, use_human_captions: bool = False, transform=None):
        self.root = root
        self.images_dir = _find_images_dir(root)
        self.use_human_captions = use_human_captions

        csv_path = _find_csv(root)
        self.df = pd.read_csv(csv_path)

        self.filename_col = _first_present(self.df.columns, FILENAME_COL_CANDIDATES)
        if not self.filename_col:
            raise ValueError(f"CSV {csv_path} não contém coluna de identificação da obra ('Numero da Obra').")

        self.caption_col = _first_present(self.df.columns, CAPTION_COL_CANDIDATES)
        if use_human_captions and not self.caption_col:
            raise ValueError(f"CSV {csv_path} não contém coluna de descrição humana para 'use_human_captions=True'.")

        self.df = self.df.dropna(subset=[self.filename_col]).reset_index(drop=True)
        self.transform = transform or self._default_transform()

    # ------------------------------------------------------------------
    # Interface Dataset
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        filename = str(row[self.filename_col]).strip()
        path = os.path.join(self.images_dir, filename)
        image = Image.open(path).convert("RGB")
        image_t = self.transform(image)

        sample = {
            "image":    image_t,
            "filename": filename,
            "path":     path,
        }

        if self.use_human_captions and self.caption_col and pd.notna(row[self.caption_col]):
            sample["caption"] = str(row[self.caption_col])

        return sample

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

    def sample(self, n: int, strategy: str = "random", seed: int = 42) -> "PortinariDataset":
        """
        Retorna um subconjunto do dataset.

        Args:
            n:        Número de amostras desejadas (truncado para len(dataset) se necessário).
            strategy: "random" — única estratégia suportada.
            seed:     Semente para reprodutibilidade.
        """
        if strategy != "random":
            raise ValueError(f"Estratégia desconhecida: '{strategy}'. Use 'random'.")

        n = min(n, len(self.df))
        sampled_df = self.df.sample(n=n, random_state=seed)

        subset = PortinariDataset.__new__(PortinariDataset)
        subset.root               = self.root
        subset.images_dir         = self.images_dir
        subset.use_human_captions = self.use_human_captions
        subset.filename_col       = self.filename_col
        subset.caption_col        = self.caption_col
        subset.df                 = sampled_df.reset_index(drop=True)
        subset.transform          = self.transform
        return subset
