from typing import Dict
import pandas as pd
from scipy.stats import kurtosis, skew

def distribution_stats(series: pd.Series) -> Dict[str, float]:
    return {
        "mean": series.mean(),
        "std": series.std(),
        "kurtosis": kurtosis(series, fisher=True),
        "skewness": skew(series)
    }
