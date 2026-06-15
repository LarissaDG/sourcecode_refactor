import glob
import os
import pandas as pd
from PIL import Image, ImageSequence
from torch.utils.data import Dataset
from torchvision import transforms


class GIFFramesDataset(Dataset):
    """
    Loader para frames extraídos de GIFs.

    Estrutura esperada no disco:
        <root>/
            video_001.gif
            video_002.gif
            ...

    Cada frame de cada GIF é extraído e salvo em <root>/frames_cache/ como
    PNG (<gif_stem>_frame_<idx>.png), para que a Caixinha 4 (scoring) possa
    abri-lo via 'path'.
    """

    def __init__(self, root: str, transform=None):
        self.root = root
        self.cache_dir = os.path.join(root, "frames_cache")
        os.makedirs(self.cache_dir, exist_ok=True)

        gif_paths = sorted(glob.glob(os.path.join(root, "*.gif")))
        if not gif_paths:
            raise FileNotFoundError(f"Nenhum .gif encontrado em {root}")

        rows = []
        for gif_path in gif_paths:
            gif_stem = os.path.splitext(os.path.basename(gif_path))[0]
            with Image.open(gif_path) as gif:
                n_frames = sum(1 for _ in ImageSequence.Iterator(gif))
            for frame_idx in range(n_frames):
                rows.append({
                    "gif_path":   gif_path,
                    "gif_name":   gif_stem,
                    "frame_idx":  frame_idx,
                })

        self.df = pd.DataFrame(rows)
        self.transform = transform or self._default_transform()

    # ------------------------------------------------------------------
    # Interface Dataset
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        gif_path  = row["gif_path"]
        gif_name  = row["gif_name"]
        frame_idx = int(row["frame_idx"])

        filename = f"{gif_name}_frame_{frame_idx:04d}.png"
        path = os.path.join(self.cache_dir, filename)

        if not os.path.exists(path):
            with Image.open(gif_path) as gif:
                for i, frame in enumerate(ImageSequence.Iterator(gif)):
                    if i == frame_idx:
                        frame.convert("RGB").save(path)
                        break

        image = Image.open(path).convert("RGB")
        image_t = self.transform(image)

        return {
            "image":     image_t,
            "filename":  filename,
            "path":      path,
            "gif_name":  gif_name,
            "frame_idx": frame_idx,
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

    def _make_subset(self, sampled_df: pd.DataFrame) -> "GIFFramesDataset":
        subset = GIFFramesDataset.__new__(GIFFramesDataset)
        subset.root      = self.root
        subset.cache_dir = self.cache_dir
        subset.df        = sampled_df.reset_index(drop=True)
        subset.transform = self.transform
        return subset

    # ------------------------------------------------------------------
    # Amostragem — chamada pela Caixinha 1
    # ------------------------------------------------------------------

    def sample(self, n: int, strategy: str = "sequential", seed: int = 42, **kwargs) -> "GIFFramesDataset":
        """
        Retorna um subconjunto do dataset.

        Args:
            n:        Número total de frames desejados.
            strategy: "sequential" — para cada GIF, pega os primeiros frames
                                      em ordem temporal (preserva a sequência),
                                      distribuindo `n` igualmente entre os GIFs.
                      "random"     — amostragem aleatória simples.
            seed:     Semente para reprodutibilidade.
        """
        if strategy == "random":
            sampled_df = self.df.sample(n=min(n, len(self.df)), random_state=seed)
            sampled_df = sampled_df.sort_values(["gif_name", "frame_idx"])

        elif strategy == "sequential":
            n_gifs = self.df["gif_name"].nunique()
            per_gif = max(1, n // n_gifs)

            parts = []
            for gif_name, group in self.df.groupby("gif_name", sort=False):
                group_sorted = group.sort_values("frame_idx")
                parts.append(group_sorted.head(per_gif))

            sampled_df = pd.concat(parts).head(n)

        else:
            raise ValueError(f"Estratégia desconhecida: '{strategy}'. Use 'sequential' ou 'random'.")

        return self._make_subset(sampled_df)
