import os
import logging
from typing import Literal, Dict

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew


# =========================
# Logger
# =========================
def setup_logger(
    name: str = "sampling",
    log_dir: str = "logs",
    level=logging.INFO
):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{name}.log")

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        fh = logging.FileHandler(log_path)
        ch = logging.StreamHandler()

        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s"
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger


# =========================
# Estatísticas auxiliares
# =========================
def distribution_stats(series: pd.Series) -> Dict[str, float]:
    return {
        "mean": series.mean(),
        "std": series.std(),
        "kurtosis": kurtosis(series, fisher=True),
        "skewness": skew(series)
    }


# =========================
# Amostragem
# =========================
class Sampler:
    def __init__(
        self,
        n_samples: int,
        n_bins: int = 30,
        seed: int = 42,
        score_column: str = "Avg Score",
        method: Literal["uniform", "gaussian", "both"] = "both",
        log_dir: str = "logs"
    ):
        self.n_samples = n_samples
        self.n_bins = n_bins
        self.seed = seed
        self.score_column = score_column
        self.method = method

        np.random.seed(seed)
        self.logger = setup_logger("sampling", log_dir)

        self.logger.info("Sampler initialized")
        self.logger.info(f"method={method}, n_samples={n_samples}, bins={n_bins}, seed={seed}")

    # -------------------------
    # Uniforme por bin
    # -------------------------
    def _uniform_sampling(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Running uniform sampling")

        df = df.copy()
        df["bin"] = pd.cut(df[self.score_column], bins=self.n_bins)

        samples_per_bin = self.n_samples // self.n_bins

        sampled = (
            df.groupby("bin", group_keys=False)
              .apply(lambda x: x.sample(
                  n=min(samples_per_bin, len(x)),
                  random_state=self.seed
              ))
        )

        sampled = sampled.drop(columns=["bin"])

        self.logger.info(f"Uniform sampling result size: {len(sampled)}")
        self.logger.info(f"Uniform stats: {distribution_stats(sampled[self.score_column])}")

        return sampled

    # -------------------------
    # "Gaussiana" (estratificada proporcional)
    # -------------------------
    def _gaussian_sampling(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Running gaussian (proportional stratified) sampling")

        df = df.copy()
        df["bin"] = pd.cut(df[self.score_column], bins=self.n_bins)

        frac = self.n_samples / len(df)

        sampled = (
            df.groupby("bin", group_keys=False)
              .apply(lambda x: x.sample(
                  frac=frac,
                  random_state=self.seed
              ))
        )

        sampled = sampled.drop(columns=["bin"])

        self.logger.info(f"Gaussian sampling result size: {len(sampled)}")
        self.logger.info(f"Gaussian stats: {distribution_stats(sampled[self.score_column])}")

        return sampled

    # -------------------------
    # Pipeline principal
    # -------------------------
    def run(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        self.logger.info("Starting sampling pipeline")

        results = {}

        original_stats = distribution_stats(df[self.score_column])
        self.logger.info(f"Original distribution stats: {original_stats}")

        if self.method in ["uniform", "both"]:
            results["uniform"] = self._uniform_sampling(df)

        if self.method in ["gaussian", "both"]:
            results["gaussian"] = self._gaussian_sampling(df)

        self.logger.info("Sampling pipeline finished")

        return results
