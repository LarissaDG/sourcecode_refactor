import os

# Cache dirs — definidos antes de qualquer import HuggingFace/CLIP
# Os scripts SLURM exportam as mesmas variáveis e sobrescrevem se necessário
os.environ.setdefault("HF_HOME",            "/snfs1/speed/larissa.gomide/hf_cache")
os.environ.setdefault("TRANSFORMERS_CACHE", "/snfs1/speed/larissa.gomide/hf_cache")
os.environ.setdefault("CLIP_CACHE",         "/snfs1/speed/larissa.gomide/hf_cache")
os.environ.setdefault("XDG_CACHE_HOME",     "/sonic_home/larissa.gomide/casa/.cache")
os.environ.setdefault("MPLCONFIGDIR",       "/sonic_home/larissa.gomide/casa/.matplotlib")

import glob
import json
import argparse
import pandas as pd
from utils.logging import setup_logger
from utils.config import load_config
from pipeline.sampling import run_sampling, loader_to_list
from pipeline.captioning import run_captioning
from pipeline.generation import run_generation
from pipeline.scoring import run_scoring

logger = setup_logger("master", "logs")


def _data_cache_path(cfg):
    return os.path.join(
        cfg["experiment"]["output_dir"],
        cfg["experiment"]["name"],
        "pipeline_data.json",
    )


def _json_serialize(obj):
    import torch
    if isinstance(obj, torch.Tensor):
        return None  # tensores de imagem não precisam ser persistidos entre fases
    if isinstance(obj, list):
        return [_json_serialize(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _json_serialize(v) for k, v in obj.items()}
    return obj


def _save_data(cfg, data):
    path = _data_cache_path(cfg)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(_json_serialize(data), f)


def _load_data(cfg):
    path = _data_cache_path(cfg)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def _reload_human_captions(cfg, data):
    """
    Substitui o campo 'caption' de cada item em `data` com a descrição humana
    (Description_en) do CSV do Portinari.  Itens sem correspondência mantêm
    o caption vazio.  Usado pelo Exp2b para reutilizar as imagens do Exp2a.
    """
    root = cfg["dataset"]["path"]
    csvs = sorted(glob.glob(os.path.join(root, "*.csv")))
    if not csvs:
        raise FileNotFoundError(f"Nenhum CSV encontrado em {root}")
    df = pd.read_csv(csvs[0])

    fn_candidates = ["Numero da Obra"]
    cap_candidates = ["Description_en", "Descrição", "Description", "caption"]

    fn_col = next((c for c in fn_candidates if c in df.columns), None)
    cap_col = next((c for c in cap_candidates if c in df.columns), None)

    if fn_col is None or cap_col is None:
        print("[!] _reload_human_captions: coluna de filename ou description não encontrada no CSV.")
        return data

    stem_to_caption = {
        os.path.splitext(str(row[fn_col]).strip())[0]: str(row[cap_col])
        for _, row in df.iterrows()
        if pd.notna(row.get(cap_col))
    }

    updated = 0
    for item in data:
        fn = item.get("filename", "")
        stem = os.path.splitext(os.path.basename(str(fn)))[0]
        if stem in stem_to_caption:
            item["caption"] = stem_to_caption[stem]
            updated += 1
        else:
            item["caption"] = ""

    print(f"[reuse_from] {updated}/{len(data)} itens com captions humanas carregadas.")
    return data


def run_pipeline(cfg):
    steps = cfg["pipeline"]["steps"]
    data = None

    if steps.get("sampling"):
        reuse_from = cfg.get("sampling", {}).get("reuse_from")
        if reuse_from:
            reuse_path = os.path.join(
                cfg["experiment"]["output_dir"], reuse_from, "pipeline_data.json"
            )
            print(f"[sampling] reuse_from='{reuse_from}' → carregando {reuse_path}")
            with open(reuse_path) as f:
                data = json.load(f)
            if cfg.get("dataset", {}).get("use_human_captions"):
                data = _reload_human_captions(cfg, data)
            # Remove caminhos de imagens geradas para forçar nova geração com as novas captions
            for item in data:
                for k in list(item.keys()):
                    if k.startswith("generated_"):
                        del item[k]
            _save_data(cfg, data)
        else:
            data = run_sampling(cfg)

    if steps.get("captioning"):
        data = run_captioning(cfg, data)
        _save_data(cfg, data)
    elif steps.get("sampling"):
        # Sem captioning, achata o DataLoader em lista de dicts para as
        # próximas caixinhas (ex: Exp 3b, com captions humanas no CSV).
        data = loader_to_list(data)
        _save_data(cfg, data)

    if steps.get("generation"):
        if data is None:
            data = _load_data(cfg)
        data = run_generation(cfg, data)
        _save_data(cfg, data)

    if steps.get("scoring"):
        if data is None:
            data = _load_data(cfg)
        if data is None:
            raise RuntimeError(
                "Scoring solicitado mas nenhum dado disponível. "
                "Rode sampling (e opcionalmente captioning/generation) antes."
            )
        run_scoring(cfg, data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Caminho para o .yaml do experimento")
    parser.add_argument("--test", action="store_true",
                        help="Modo de teste: limita n_samples a 5 e salva em outputs/test_<nome>/")
    parser.add_argument("--steps", default=None,
                        help="Sobrescreve os steps do YAML. Ex: sampling,captioning,generation")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.test:
        cfg["sampling"]["n_samples"] = 5
        cfg["experiment"]["name"] = "test_" + cfg["experiment"]["name"]
        logger.info("[MODO TESTE] n_samples=5, saída em outputs/%s/", cfg["experiment"]["name"])

    if args.steps:
        requested = {s.strip() for s in args.steps.split(",")}
        for step in cfg["pipeline"]["steps"]:
            cfg["pipeline"]["steps"][step] = step in requested

    run_pipeline(cfg)