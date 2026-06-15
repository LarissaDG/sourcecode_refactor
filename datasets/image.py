import glob
import os
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp"]

DEFAULT_NOISE_LEVELS = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]


def _find_image(root: str) -> str:
    for ext in IMAGE_EXTS:
        matches = sorted(glob.glob(os.path.join(root, f"*{ext}")))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"Nenhuma imagem encontrada em {root}")


def sample_color_from_image(image: Image.Image, n_samples: int = 50) -> np.ndarray:
    """Amostra n_samples pixels e retorna a cor média (R, G, B)."""
    arr = np.array(image.convert("RGB"))
    h, w, _ = arr.shape
    ys = np.random.randint(0, h, n_samples)
    xs = np.random.randint(0, w, n_samples)
    return arr[ys, xs].mean(axis=0)


def add_gaussian_noise(image: Image.Image, intensity: int) -> Image.Image:
    """
    Aplica ruído gaussiano progressivo.
    intensity: 0 (sem ruído) a 100 (corrupção máxima) -> sigma de 0 a 1000.
    """
    sigma = 1000 * intensity / 100
    rgb = image.convert("RGB")
    arr = np.array(rgb).astype(np.int16)
    base_color = sample_color_from_image(rgb, n_samples=50)
    noise = np.random.normal(0, sigma, arr.shape)
    noisy = arr + noise + (base_color - base_color.mean())
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy)


class ImageDataset(Dataset):
    """
    Loader para uma única imagem-alvo, corrompida progressivamente.

    Estrutura esperada no disco:
        <root>/
            <qualquer_imagem>.jpg|.png|...

    A Caixinha 1 (sampling) gera uma versão corrompida da imagem para cada
    nível em `noise_levels`, salva em <root>/noise_cache/level_<NNN>.png,
    para que a Caixinha 4 (scoring) possa abri-las via 'path'.
    """

    def __init__(self, root: str, transform=None):
        self.root = root
        self.image_path = _find_image(root)
        self.cache_dir = os.path.join(root, "noise_cache")
        os.makedirs(self.cache_dir, exist_ok=True)

        self.df = pd.DataFrame({"level": [0]})
        self.transform = transform or self._default_transform()

    # ------------------------------------------------------------------
    # Interface Dataset
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        level = int(self.df.iloc[idx]["level"])

        filename = f"level_{level:03d}.png"
        path = os.path.join(self.cache_dir, filename)

        if not os.path.exists(path):
            img = Image.open(self.image_path).convert("RGB")
            if level > 0:
                np.random.seed(level)
                img = add_gaussian_noise(img, intensity=level)
            img.save(path)

        image = Image.open(path).convert("RGB")
        image_t = self.transform(image)

        return {
            "image":       image_t,
            "filename":    filename,
            "path":        path,
            "noise_level": level,
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

    def _make_subset(self, sampled_df: pd.DataFrame) -> "ImageDataset":
        subset = ImageDataset.__new__(ImageDataset)
        subset.root       = self.root
        subset.image_path = self.image_path
        subset.cache_dir  = self.cache_dir
        subset.df         = sampled_df.reset_index(drop=True)
        subset.transform  = self.transform
        return subset

    # ------------------------------------------------------------------
    # Amostragem — chamada pela Caixinha 1
    # ------------------------------------------------------------------

    def sample(self, n: int, strategy: str = "noise_levels", seed: int = 42,
               noise_levels=None, **kwargs) -> "ImageDataset":
        """
        Retorna um item por nível de ruído (versão progressivamente corrompida
        da imagem-alvo).

        Args:
            n:            Número de níveis desejados (trunca `noise_levels`).
            strategy:     "noise_levels" — única estratégia suportada.
            noise_levels: Lista de intensidades (0-100). Default: 0..100 em passos de 10.
        """
        if strategy != "noise_levels":
            raise ValueError(f"Estratégia desconhecida: '{strategy}'. Use 'noise_levels'.")

        levels = noise_levels if noise_levels is not None else DEFAULT_NOISE_LEVELS
        levels = list(levels)[:n] if n else list(levels)

        sampled_df = pd.DataFrame({"level": levels})
        return self._make_subset(sampled_df)
