import glob
import os
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter, ImageDraw
from torch.utils.data import Dataset
from torchvision import transforms

IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp"]
DEFAULT_NOISE_LEVELS = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
DEFAULT_NOISE_TYPES  = ["gaussian", "blur", "shapes"]


def _list_images(root: str) -> list[str]:
    paths = []
    for ext in IMAGE_EXTS:
        paths.extend(sorted(glob.glob(os.path.join(root, f"*{ext}"))))
    return paths


# ── Funções de corrupção ──────────────────────────────────────────────────────

def add_gaussian_noise(image: Image.Image, intensity: int) -> Image.Image:
    sigma = 128 * intensity / 100
    arr = np.array(image.convert("RGB")).astype(np.int16)
    noise = np.random.normal(0, sigma, arr.shape)
    noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy)


def add_blur(image: Image.Image, intensity: int) -> Image.Image:
    radius = 20 * intensity / 100
    return image.filter(ImageFilter.GaussianBlur(radius=radius))


def add_shapes(image: Image.Image, intensity: int) -> Image.Image:
    img = image.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size
    n_shapes = int(50 * intensity / 100)
    rng = np.random.RandomState(intensity)
    for _ in range(n_shapes):
        x0 = rng.randint(0, w)
        y0 = rng.randint(0, h)
        size = rng.randint(10, max(11, int(min(w, h) * intensity / 200)))
        color = tuple(rng.randint(0, 256, 3).tolist())
        draw.rectangle([x0, y0, x0 + size, y0 + size], fill=color)
    return img


NOISE_FNS = {
    "gaussian": add_gaussian_noise,
    "blur":     add_blur,
    "shapes":   add_shapes,
}


class ImageDataset(Dataset):
    """
    Loader para análise de robustez a ruído (Exp 4).

    Estrutura esperada:
        <root>/
            image1.jpg
            image2.jpg
            ...

    Para cada imagem × tipo de ruído × nível, gera e cacheia uma versão
    corrompida em <root>/noise_cache/<noise_type>/level_<NNN>/<filename>.png.
    """

    def __init__(self, root: str, transform=None):
        self.root      = root
        self.cache_dir = os.path.join(root, "noise_cache")
        os.makedirs(self.cache_dir, exist_ok=True)

        image_paths = _list_images(root)
        if not image_paths:
            raise FileNotFoundError(f"Nenhuma imagem encontrada em {root}")

        rows = [{"image_path": p, "filename": os.path.basename(p),
                 "noise_type": "gaussian", "noise_level": 0}
                for p in image_paths]
        self.df        = pd.DataFrame(rows)
        self.transform = transform or self._default_transform()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row         = self.df.iloc[idx]
        image_path  = row["image_path"]
        filename    = row["filename"]
        noise_type  = row["noise_type"]
        noise_level = int(row["noise_level"])

        cache_subdir = os.path.join(self.cache_dir, noise_type, f"level_{noise_level:03d}")
        os.makedirs(cache_subdir, exist_ok=True)
        cached_path = os.path.join(cache_subdir, os.path.splitext(filename)[0] + ".png")

        if not os.path.exists(cached_path):
            img = Image.open(image_path).convert("RGB")
            if noise_level > 0:
                np.random.seed(noise_level)
                img = NOISE_FNS[noise_type](img, noise_level)
            img.save(cached_path)

        image   = Image.open(cached_path).convert("RGB")
        image_t = self.transform(image)

        return {
            "image":       image_t,
            "filename":    filename,
            "path":        cached_path,
            "noise_type":  noise_type,
            "noise_level": noise_level,
        }

    @staticmethod
    def _default_transform() -> transforms.Compose:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275,  0.40821073],
                std= [0.26862954, 0.26130258, 0.27577711],
            ),
        ])

    def _make_subset(self, sampled_df: pd.DataFrame) -> "ImageDataset":
        subset            = ImageDataset.__new__(ImageDataset)
        subset.root       = self.root
        subset.cache_dir  = self.cache_dir
        subset.df         = sampled_df.reset_index(drop=True)
        subset.transform  = self.transform
        return subset

    def sample(self, n: int, strategy: str = "noise_levels", seed: int = 42,
               noise_levels=None, noise_types=None, **kwargs) -> "ImageDataset":
        """
        Gera combinações imagem × tipo de ruído × nível.

        Args:
            n:           Número de imagens base (truncado ao disponível).
            strategy:    "noise_levels" — única estratégia suportada.
            noise_levels: Lista de intensidades 0-100.
            noise_types:  Lista de tipos: "gaussian", "blur", "shapes".
        """
        if strategy != "noise_levels":
            raise ValueError(f"Estratégia desconhecida: '{strategy}'. Use 'noise_levels'.")

        levels = list(noise_levels) if noise_levels is not None else DEFAULT_NOISE_LEVELS
        types  = list(noise_types)  if noise_types  is not None else DEFAULT_NOISE_TYPES

        # Amostra n imagens base
        base_images = self.df[["image_path", "filename"]].drop_duplicates()
        n = min(n, len(base_images))
        base_images = base_images.sample(n=n, random_state=seed)

        # Expande: cada imagem × cada tipo × cada nível
        rows = []
        for _, row in base_images.iterrows():
            for noise_type in types:
                for level in levels:
                    rows.append({
                        "image_path":  row["image_path"],
                        "filename":    row["filename"],
                        "noise_type":  noise_type,
                        "noise_level": level,
                    })

        return self._make_subset(pd.DataFrame(rows))
