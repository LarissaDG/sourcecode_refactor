import os
import sys
sys.path.append('/home_cerberus/disk3/larissa.gomide/APDDv2/')

import torch
import numpy as np
import warnings
import argparse
import pandas as pd

import models.clip as clip
from models.aesclip import AesCLIP_reg

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

warnings.filterwarnings("ignore")


# -------------------------------
# 1) CONFIGURAR VARIÁVEIS DE AMBIENTE
# -------------------------------
def setup_environment():
    os.environ['HOME'] = '/scratch/larissa.gomide/minha_home/'
    os.environ['TRANSFORMERS_CACHE'] = "/scratch/larissa.gomide/minha_home/.cache/huggingface"
    os.environ['CLIP_CACHE'] = "/scratch/larissa.gomide/minha_home/.cache/clip"
    os.environ['HF_HOME'] = "/scratch/larissa.gomide/minha_home/.cache/huggingface"
    os.environ['XDG_CACHE_HOME'] = "/scratch/larissa.gomide/minha_home/.cache"
    os.environ['MPLCONFIGDIR'] = '/scratch/larissa.gomide/minha_home/.matplotlib'


# -------------------------------
# 2) PARSE DE ARGUMENTOS
# -------------------------------
def init_args():
    parser = argparse.ArgumentParser(description="PyTorch Aesthetic Scoring")
    args = parser.parse_args()
    return args


# -------------------------------
# 3) EXTRAIR SCORE DO MODELO
# -------------------------------
def get_score(opt, y_pred):
    """
    Retorna a predição do modelo e seu valor numérico em numpy.
    """
    score_np = y_pred.data.cpu().numpy()
    return y_pred, score_np


# -------------------------------
# 4) CARREGAR MODELO AesCLIP
# -------------------------------
def load_model(weight_path, device):
    """
    Tenta carregar o modelo AesCLIP_reg.
    Em caso de falha, retorna None.
    """
    try:
        base_weight = (
            "/home_cerberus/disk3/larissa.gomide/APDDv2/modle_weights/"
            "0.AesCLIP_weight--e11-train2.4314-test4.0253_best.pth"
        )
        model = AesCLIP_reg(clip_name='ViT-B/16', weight=base_weight)
        model.load_state_dict(torch.load(weight_path))
        model.to(device)
        model.eval()
        print(f"Modelo carregado com sucesso: {weight_path}")
        return model
    except Exception as e:
        print(f"Falha ao carregar modelo de {weight_path}: {e}")
        return None


# -------------------------------
# 5) AVALIAR UMA IMAGEM
# -------------------------------
def evaluate_image(image_path, models_dict, preprocess, opt, device):
    """
    Processa uma imagem e retorna os scores para cada métrica.
    """
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Erro ao abrir imagem {image_path}: {e}")
        return {col: np.nan for col in models_dict.keys()}

    try:
        image_input = preprocess(image).unsqueeze(0).to(device)
    except Exception as e:
        print(f"Erro ao preprocessar imagem {image_path}: {e}")
        return {col: np.nan for col in models_dict.keys()}

    scores = {}
    for col, model in models_dict.items():
        if model is None:
            scores[col] = np.nan
            continue

        try:
            pred = model(image_input)
            _, pred_val = get_score(opt, pred)

            if isinstance(pred_val, np.ndarray) and pred_val.size == 1:
                pred_val = pred_val.item()

            if col == "Total aesthetic score":
                pred_val = pred_val * 10

            scores[col] = pred_val
        except Exception as e:
            print(f"Erro ao prever {col} para imagem {image_path}: {e}")
            scores[col] = np.nan

    return scores


# -------------------------------
# 6) PROCESSAR CSV
# -------------------------------
def process_csv(
    input_csv,
    output_csv,
    models_dict,
    preprocess,
    opt,
    device,
    cols_to_compare
):
    """
    Lê o CSV, avalia as imagens e salva o CSV atualizado.
    """
    try:
        df = pd.read_csv(input_csv, encoding="utf-8")
    except Exception:
        df = pd.read_csv(input_csv, encoding="latin1")

    for idx, row in df.iterrows():
        image_path = row.get("generated_filename")

        if not image_path or not os.path.exists(image_path):
            print(f"Imagem não encontrada na linha {idx}: {image_path}")
            for col in cols_to_compare:
                df.loc[idx, col] = np.nan
            continue

        scores = evaluate_image(image_path, models_dict, preprocess, opt, device)
        for col in cols_to_compare:
            df.loc[idx, col] = scores.get(col, np.nan)

        print(f"Linha {idx} processada.")

    df.to_csv(output_csv, index=False)
    print(f"Arquivo salvo: {output_csv}")


# -------------------------------
# 7) CARREGAR TODOS OS MODELOS
# -------------------------------
def load_all_models(device):
    model_paths = {
        "Total aesthetic score": "1.Score_reg_weight--e4-train0.4393-test0.6835_best.pth",
        "Theme and logic": "2.Theme and logic_reg_weight--e5-train0.3792-test0.5953_best.pth",
        "Creativity": "3.Creativity_reg_weight--e5-train0.4212-test0.7122_best.pth",
        "Layout and composition": "4.Layout and composition_reg_weight--e6-train0.2783-test0.6342_best.pth",
        "Space and perspective": "5.Space and perspective_reg_weight--e7-train0.2168-test0.5998_best.pth",
        "The sense of order": "Model_6.pth",
        "Light and shadow": "7.Light and shadow_reg_weight--e7-train0.1937-test0.6518_best.pth",
        "Color": "8.Color_reg_weight--e5-train0.2905-test0.5871_best.pth",
        "Details and texture": "9.Details and texture_reg_weight--e4-train0.4385-test0.7034_best.pth",
        "The overall": "10.The overall_reg_weight--e3-train0.5131-test0.6343_best.pth",
        "Mood": "11.Mood_reg_weight--e7-train0.3108-test0.7097_best.pth",
    }

    base_dir = "/home_cerberus/disk3/larissa.gomide/APDDv2/modle_weights/"
    models = {}

    for name, filename in model_paths.items():
        models[name] = load_model(os.path.join(base_dir, filename), device)

    return models


# -------------------------------
# 8) EXECUÇÃO PRINCIPAL
# -------------------------------
def main():
    setup_environment()
    opt = init_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    _, preprocess = clip.load('ViT-B/16', device)

    cols_to_compare = [
        "Total aesthetic score", "Theme and logic", "Creativity",
        "Layout and composition", "Space and perspective",
        "The sense of order", "Light and shadow", "Color",
        "Details and texture", "The overall", "Mood"
    ]

    models_dict = load_all_models(device)

    # SMALL
    print("Processando tabela SMALL...")
    process_csv(
        input_csv="/home_cerberus/disk3/larissa.gomide/oficial/amostraGauss/sampled_SMALL_with_gen.csv",
        output_csv="/home_cerberus/disk3/larissa.gomide/oficial/amostraGauss/sampled_SMALL_with_gen_scored.csv",
        models_dict=models_dict,
        preprocess=preprocess,
        opt=opt,
        device=device,
        cols_to_compare=cols_to_compare
    )

    # BIG
    print("Processando tabela BIG...")
    process_csv(
        input_csv="/home_cerberus/disk3/larissa.gomide/oficial/amostraGauss/sampled_BIG_with_gen.csv",
        output_csv="/home_cerberus/disk3/larissa.gomide/oficial/amostraGauss/sampled_BIG_with_gen_scored.csv",
        models_dict=models_dict,
        preprocess=preprocess,
        opt=opt,
        device=device,
        cols_to_compare=cols_to_compare
    )


if __name__ == "__main__":
    main()
