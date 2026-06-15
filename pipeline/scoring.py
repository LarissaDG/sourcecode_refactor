import os
import torch
import numpy as np
import pandas as pd
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


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
        agents[agent["name"]] = _load_agent(w_path, base_weight, device)
    print(f"[scoring] {len(agents)} agentes carregados.")

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
                image_t = preprocess(image).unsqueeze(0).to(device)
            except Exception as e:
                print(f"  [!] Erro ao abrir {img_path}: {e}")
                continue

            row = {"filename": os.path.basename(img_path)}

            # Score total (×10 para escala 0–10, como no demo.py original)
            for name, model in agents.items():
                try:
                    score = _predict(model, image_t)
                    row[name] = score * 10 if name == "score" else score
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