import os
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from datasets.image import NOISE_FNS, DEFAULT_NOISE_LEVELS, DEFAULT_NOISE_TYPES


class VideoFramesDataset(Dataset):
    """
    Loader para análise temporal de frames de vídeo (Exp 5a / 5b).

    Estrutura esperada:
        <root>/
            frames/
                0001/
                    0001_frame_0000.png
                    0001_frame_0001.png
                    ...
                0002/
                    ...
            metadata.csv
    """

    def __init__(self, root: str, transform=None):
        self.root      = Path(root)
        self.frames_dir = self.root / "frames"

        if not self.frames_dir.exists():
            raise FileNotFoundError(f"Pasta de frames não encontrada: {self.frames_dir}")

        rows = []
        for video_dir in sorted(self.frames_dir.iterdir()):
            if not video_dir.is_dir():
                continue
            for frame_path in sorted(video_dir.glob("*.png")):
                rows.append({
                    "video_id":   video_dir.name,
                    "frame_path": str(frame_path),
                    "filename":   frame_path.name,
                    "frame_idx":  self._parse_frame_idx(frame_path.name),
                    "noise_type":  None,
                    "noise_level": 0,
                    "error_applied": False,
                })

        if not rows:
            raise FileNotFoundError(f"Nenhum frame .png encontrado em {self.frames_dir}")

        self.df        = pd.DataFrame(rows)
        self.transform = transform or self._default_transform()

    @staticmethod
    def _parse_frame_idx(filename: str) -> int:
        parts = filename.replace(".png", "").split("_")
        try:
            return int(parts[-1])
        except (ValueError, IndexError):
            return 0

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row        = self.df.iloc[idx]
        frame_path = row["frame_path"]
        noise_type  = row["noise_type"]
        noise_level = int(row["noise_level"])

        image = Image.open(frame_path).convert("RGB")

        if noise_type is not None and noise_level > 0:
            np.random.seed(noise_level + idx)
            image = NOISE_FNS[noise_type](image, noise_level)

        return {
            "image":           self.transform(image),
            "filename":        row["filename"],
            "path":            frame_path,
            "video_id":        row["video_id"],
            "frame_idx":       int(row["frame_idx"]),
            "noise_type":      noise_type or "none",
            "noise_level":     noise_level,
            "degradation_pct": float(row.get("degradation_pct", 0.0)),
            "error_applied":   bool(row["error_applied"]),
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

    def _make_subset(self, sampled_df: pd.DataFrame) -> "VideoFramesDataset":
        subset             = VideoFramesDataset.__new__(VideoFramesDataset)
        subset.root        = self.root
        subset.frames_dir  = self.frames_dir
        subset.df          = sampled_df.reset_index(drop=True)
        subset.transform   = self.transform
        return subset

    def sample(self, n: int, strategy: str = "sequential", seed: int = 42,
               error_frame: int = 12, error_type: str = "gaussian",
               error_level: int = 50, noise_types=None, **kwargs) -> "VideoFramesDataset":
        """
        Args:
            n:           Número de frames por vídeo (sequential/sequential_with_error)
                         ou total de linhas desejado (progressive_degradation ignora este parâmetro
                         e usa todos os frames disponíveis × tipos de ruído).
            strategy:    "sequential"              — todos os frames em ordem, sem ruído.
                         "sequential_with_error"   — ruído fixo a partir de error_frame.
                         "progressive_degradation" — degradação crescente proporcional ao
                                                     frame_idx (frame 0 = 0 %, último = 100 %),
                                                     expandida por noise_types.
            error_frame: Frame a partir do qual o erro é aplicado (sequential_with_error).
            error_type:  Tipo de ruído para sequential_with_error.
            error_level: Intensidade para sequential_with_error (0-100).
            noise_types: Lista de tipos para progressive_degradation (padrão: todos).
        """
        valid = ("sequential", "sequential_with_error", "progressive_degradation")
        if strategy not in valid:
            raise ValueError(f"Estratégia inválida: '{strategy}'. Use {valid}.")

        if strategy == "progressive_degradation":
            return self._sample_progressive(noise_types)

        video_ids = sorted(self.df["video_id"].unique())
        frames_per_video = max(1, n // max(len(video_ids), 1))
        rows = []

        for vid in video_ids:
            vid_frames = (
                self.df[self.df["video_id"] == vid]
                .sort_values("frame_idx")
                .head(frames_per_video)
            )
            for _, row in vid_frames.iterrows():
                r = row.to_dict()
                r.setdefault("degradation_pct", 0.0)
                if strategy == "sequential_with_error" and int(row["frame_idx"]) >= error_frame:
                    r["noise_type"]    = error_type
                    r["noise_level"]   = error_level
                    r["error_applied"] = True
                rows.append(r)

            if len(rows) >= n:
                break

        df = pd.DataFrame(rows[:n]).sort_values(["video_id", "frame_idx"]).reset_index(drop=True)
        return self._make_subset(df)

    def _sample_progressive(self, noise_types=None) -> "VideoFramesDataset":
        """
        Para cada vídeo, pega todos os frames em ordem e aplica degradação
        proporcional ao índice: frame i → noise_level = round(i / (total-1) * 100).
        Expande por noise_types → n_videos × n_frames × n_tipos linhas.
        """
        from datasets.image import DEFAULT_NOISE_TYPES
        types = list(noise_types) if noise_types is not None else DEFAULT_NOISE_TYPES

        rows = []
        for vid in sorted(self.df["video_id"].unique()):
            vid_frames = (
                self.df[self.df["video_id"] == vid]
                .sort_values("frame_idx")
                .reset_index(drop=True)
            )
            n_frames = len(vid_frames)
            for i, (_, row) in enumerate(vid_frames.iterrows()):
                degradation_pct = round(i / max(n_frames - 1, 1) * 100, 2)
                noise_level     = round(degradation_pct)
                for noise_type in types:
                    r = row.to_dict()
                    r["noise_type"]      = noise_type
                    r["noise_level"]     = noise_level
                    r["degradation_pct"] = degradation_pct
                    r["error_applied"]   = noise_level > 0
                    rows.append(r)

        df = pd.DataFrame(rows).sort_values(["video_id", "frame_idx", "noise_type"]).reset_index(drop=True)
        return self._make_subset(df)
