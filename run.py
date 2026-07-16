import argparse
from utils.logging import setup_logger
from utils.config import load_config
from pipeline.sampling import run_sampling, loader_to_list
from pipeline.captioning import run_captioning
from pipeline.generation import run_generation
from pipeline.scoring import run_scoring

logger = setup_logger("master", "logs")

def run_pipeline(cfg):
    steps = cfg["pipeline"]["steps"]
    data = None

    if steps.get("sampling"):
        data = run_sampling(cfg)

    if steps.get("captioning"):
        data = run_captioning(cfg, data)
    elif steps.get("sampling"):
        # Sem captioning, achata o DataLoader em lista de dicts para as
        # próximas caixinhas (ex: Exp 3b, com captions humanas no CSV).
        data = loader_to_list(data)

    if steps.get("generation"):
        data = run_generation(cfg, data)

    if steps.get("scoring"):
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