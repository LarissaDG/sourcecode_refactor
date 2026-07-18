import os
import json
import argparse
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


def run_pipeline(cfg):
    steps = cfg["pipeline"]["steps"]
    data = None

    if steps.get("sampling"):
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