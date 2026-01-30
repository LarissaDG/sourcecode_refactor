import unittest
import pandas as pd
import numpy as np

from src.sampling import Sampler

class TestSampling(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Dataset sintético simples (controlado)
        np.random.seed(0)
        cls.df = pd.DataFrame({
            "Avg Score": np.random.normal(loc=5, scale=2, size=10_000)
        })

    def test_reproducibility_uniform(self):
        sampler1 = Sampler(
            n_samples=500,
            n_bins=20,
            seed=42,
            method="uniform"
        )

        sampler2 = Sampler(
            n_samples=500,
            n_bins=20,
            seed=42,
            method="uniform"
        )

        s1 = sampler1.run(self.df)["uniform"]
        s2 = sampler2.run(self.df)["uniform"]

        self.assertTrue(s1.equals(s2))

    def test_sample_size_close(self):
        sampler = Sampler(
            n_samples=500,
            n_bins=25,
            seed=42,
            method="gaussian"
        )

        sampled = sampler.run(self.df)["gaussian"]

        self.assertTrue(abs(len(sampled) - 500) < 50)

    def test_support_preservation(self):
        sampler = Sampler(
            n_samples=500,
            n_bins=30,
            seed=42,
            method="both"
        )

        results = sampler.run(self.df)

        for name, sampled in results.items():
            self.assertGreaterEqual(sampled["Avg Score"].min(), self.df["Avg Score"].min())
            self.assertLessEqual(sampled["Avg Score"].max(), self.df["Avg Score"].max())

    def test_gaussian_preserves_mean(self):
        sampler = Sampler(
            n_samples=800,
            n_bins=30,
            seed=42,
            method="gaussian"
        )

        sampled = sampler.run(self.df)["gaussian"]

        diff = abs(sampled["Avg Score"].mean() - self.df["Avg Score"].mean())

        self.assertLess(diff, 0.2)


if __name__ == "__main__":
    unittest.main()
