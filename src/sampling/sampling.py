from pathlib import Path
from typing import Dict, Literal
import numpy as np
import pandas as pd

from src.utils.logging import setup_logger
from src.utils.stats import distribution_stats
from src.sampling.methods import uniform_sampling, gaussian_sampling


class Sampler:
    def __init__(
        self,
        n_samples: int,
        n_bins: int = 30,
        seed: int = 42,
        score_column: str = "Avg Score",
        method: Literal["uniform", "gaussian", "both"] = "both",
        output_dir: str = "results/sampling",
        log_dir: str = "logs"
    ):
        self.n_samples = n_samples
        self.n_bins = n_bins
        self.seed = seed
        self.score_column = score_column
        self.method = method
        self.output_dir = Path(output_dir)

        np.random.seed(seed)
        self.logger = setup_logger("sampling", log_dir)

        self.logger.info(
            f"Sampler initialized | method={method}, "
            f"samples={n_samples}, bins={n_bins}, seed={seed}"
        )

    def _save(self, df: pd.DataFrame, method: str):
        path = self.output_dir / method
        path.mkdir(parents=True, exist_ok=True)

        out_file = path / "sampled.csv"
        df.to_csv(out_file, index=False)

        self.logger.info(f"Saved {method} sample to {out_file}")

    def run(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        self.logger.info("Starting sampling pipeline")

        results = {}

        original_stats = distribution_stats(df[self.score_column])
        self.logger.info(f"Original stats: {original_stats}")

        if self.method in ["uniform", "both"]:
            sampled = uniform_sampling(
                df, self.score_column,
                self.n_samples, self.n_bins, self.seed
            )
            self.logger.info(
                f"Uniform stats: {distribution_stats(sampled[self.score_column])}"
            )
            self._save(sampled, "uniform")
            results["uniform"] = sampled

        if self.method in ["gaussian", "both"]:
            sampled = gaussian_sampling(
                df, self.score_column,
                self.n_samples, self.n_bins, self.seed
            )
            self.logger.info(
                f"Gaussian stats: {distribution_stats(sampled[self.score_column])}"
            )
            self._save(sampled, "gaussian")
            results["gaussian"] = sampled

        self.logger.info("Sampling finished")
        return results
