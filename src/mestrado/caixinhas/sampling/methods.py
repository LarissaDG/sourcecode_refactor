import pandas as pd
import numpy as np

def uniform_sampling(df, score_column, n_samples, n_bins, seed):
    df = df.copy()
    df["bin"] = pd.cut(df[score_column], bins=n_bins)

    samples_per_bin = n_samples // n_bins

    sampled = (
        df.groupby("bin", group_keys=False)
          .apply(lambda x: x.sample(
              n=min(samples_per_bin, len(x)),
              random_state=seed
          ))
    )

    return sampled.drop(columns=["bin"])


def gaussian_sampling(df, score_column, n_samples, n_bins, seed):
    df = df.copy()
    df["bin"] = pd.cut(df[score_column], bins=n_bins)

    frac = n_samples / len(df)

    sampled = (
        df.groupby("bin", group_keys=False)
          .apply(lambda x: x.sample(
              frac=frac,
              random_state=seed
          ))
    )

    return sampled.drop(columns=["bin"])
