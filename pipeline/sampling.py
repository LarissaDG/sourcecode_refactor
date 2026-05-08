# ── pipeline/sampling.py ──────────────────────────────────────────────────────

from datasets import load_dataset  # seu loader customizado em datasets/

def run_sampling(cfg) -> list:
    """
    Caixinha 1 — Amostragem.
    Carrega a base de dados e retorna uma lista de amostras.
    A estratégia de amostragem varia por experimento (random, stratified,
    sequential, noise_levels).
    """
    # lógica de amostragem
    pass

