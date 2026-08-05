import hashlib
import os
import torch
import numpy as np
import pandas as pd
from PIL import Image, ImageFile

from datasets.noise import NOISE_FNS

ImageFile.LOAD_TRUNCATED_IMAGES = True


def _stable_seed(*parts) -> int:
    """Seed determinístico (estável entre processos/máquinas, ao contrário de
    hash() nativo do Python, que é aleatorizado por padrão a cada execução)."""
    key = "|".join(str(p) for p in parts).encode("utf-8")
    return int(hashlib.md5(key).hexdigest(), 16) % (2**31)


def _apply_noise_if_needed(image: Image.Image, sample: dict, img_path: str) -> Image.Image:
    """
    Reaplica o ruído descrito em sample['noise_type']/['noise_level'] sobre a
    imagem recém-aberta do disco.

    Necessário porque `data` passa por _save_data/_load_data (JSON) entre as
    etapas do pipeline, e um tensor de imagem não sobrevive a esse round-trip
    — o campo 'image' chega sempre None em run_scoring(). Sem isso, toda
    imagem "ruidosa" era pontuada como se fosse a original sem ruído (Exp4 e
    Exp5b apresentavam o mesmo score para todos os níveis/tipos de ruído).
    """
    noise_type = sample.get("noise_type")
    if not noise_type or noise_type == "none":
        return image
    noise_level = sample.get("noise_level")
    try:
        noise_level = int(noise_level)
    except (TypeError, ValueError):
        return image
    if noise_level <= 0:
        return image
    if noise_type not in NOISE_FNS:
        return image
    seed = _stable_seed(os.path.basename(img_path), noise_type, noise_level)
    np.random.seed(seed)
    return NOISE_FNS[noise_type](image, noise_level)


# ── Carregamento de um agente ─────────────────────────────────────────────────

def _load_agent(weight_path: str, base_weight_path: str, device: torch.device):
    import models.clip as clip                    # lazy import
    from models.aesclip import AesCLIP_reg        # lazy import
    model = AesCLIP_reg(clip_name="ViT-B/16", weight=base_weight_path)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.to(device).eval()
    return model


# ── Score de uma imagem com um agente ────────────────────────────────────────

def _predict(model, image_tensor: torch.Tensor) -> float:
    with torch.no_grad():
        pred = model(image_tensor)
    return float(pred.data.cpu().numpy())


# ── Caixinha 4 ────────────────────────────────────────────────────────────────

def run_scoring(cfg, data: list) -> None:
    """
    Caixinha 4 — Avaliação com ArtClip.

    Para cada item em `data`:
      - Roda todos os agentes configurados em cfg['scoring']['agents']
      - Coleta os scores numa linha de DataFrame
      - Faz drop de NaN (replicando a lógica do código original)
      - Salva CSV em outputs/<exp_name>/scores/scores_<source>.csv

    `data` pode conter imagens originais e/ou imagens geradas (Caixinha 3).
    O campo 'generated_<model_name>' é detectado automaticamente.
    """
    device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights_dir  = cfg["scoring"]["weights_dir"]
    base_weight  = os.path.join(weights_dir, cfg["scoring"]["artclip_base_weight"])
    agents_cfg   = cfg["scoring"]["agents"]
    output_dir   = os.path.join(cfg["experiment"]["output_dir"], cfg["experiment"]["name"], "scores")
    os.makedirs(output_dir, exist_ok=True)

    import importlib
    clip = importlib.import_module("models.clip")    # lazy import
    _, preprocess = clip.load("ViT-B/16", device)

    # Carrega todos os agentes de uma vez
    agents = {}
    for agent in agents_cfg:
        w_path = os.path.join(weights_dir, agent["weight_file"])
        try:
            agents[agent["name"]] = _load_agent(w_path, base_weight, device)
        except Exception as e:
            print(f"  [!] Agente '{agent['name']}' ignorado (erro ao carregar peso): {e}")
    print(f"[scoring] {len(agents)} agentes carregados.")

    # Campos de metadados preservados da amostra no CSV de scores
    META_FIELDS = ("noise_type", "noise_level", "degradation_pct",
                   "frame_idx", "video_id", "error_applied")

    # ── Detecta quais "fontes" de imagem existem no data ─────────────────────
    # Sempre tem a imagem original; pode ter também generated_Janus-Pro-1B, etc.
    sources = {"original": lambda s: s.get("path", s["filename"])}

    gen_keys = [k for k in data[0].keys() if k.startswith("generated_")]
    for k in gen_keys:
        model_name = k.replace("generated_", "")
        sources[model_name] = lambda s, _k=k: (s[_k][0] if s[_k] else None)

    # ── Pontua cada fonte separadamente ──────────────────────────────────────
    for source_name, get_path in sources.items():
        rows = []

        for sample in data:
            img_path = get_path(sample)
            if not img_path:
                continue

            try:
                image = Image.open(img_path).convert("RGB")
                image = _apply_noise_if_needed(image, sample, img_path)
                image_t = preprocess(image).unsqueeze(0).to(device)
            except Exception as e:
                print(f"  [!] Erro ao abrir {img_path}: {e}")
                continue

            row = {"filename": os.path.basename(img_path)}

            # Para imagens geradas, guarda o filename original para matching
            if source_name != "original":
                row["original_filename"] = os.path.basename(
                    str(sample.get("filename", ""))
                )

            # Preserva metadados relevantes (ruído, frame, vídeo, etc.)
            for field in META_FIELDS:
                val = sample.get(field)
                if val is not None:
                    if hasattr(val, "item"):
                        val = val.item()
                    elif hasattr(val, "tolist"):
                        val = val.tolist()
                    row[field] = val

            # Score total (×10 para escala 0–10, como no demo.py original)
            for name, model in agents.items():
                try:
                    score = _predict(model, image_t)
                    row[name] = score * 10 if name == "Total aesthetic score" else score
                except Exception as e:
                    print(f"  [!] Agente '{name}' falhou em {img_path}: {e}")
                    row[name] = np.nan

            rows.append(row)

        if not rows:
            print(f"  [!] Nenhuma amostra pontuada para fonte '{source_name}'.")
            continue

        df = pd.DataFrame(rows)

        # Drop NaN — mantém intersecção com o que o dataset original anotou
        df = df.dropna(axis=1, how="all")   # remove colunas 100% NaN
        df = df.dropna(axis=0, how="any")   # remove linhas com qualquer NaN restante

        csv_path = os.path.join(output_dir, f"scores_{source_name}.csv")
        df.to_csv(csv_path, index=False)
        print(f"  ✓ [{source_name}] {len(df)} imagens pontuadas → {csv_path}")