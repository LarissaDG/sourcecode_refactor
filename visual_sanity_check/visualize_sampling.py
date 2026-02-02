import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sourcecode_refactor.src.preprocessing.columns import add_score_columns
from src.sampling.sampling import Sampler

# =========================
# Config
# =========================
cols_to_compare = [
    "Total aesthetic score", "Theme and logic", "Creativity",
    "Layout and composition", "Space and perspective",
    "Light and shadow", "Color", "Details and texture",
    "The overall", "Mood"
]

OUTPUT_DIR = "./results/sampling_debug"
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_PNG = os.path.join(OUTPUT_DIR, "sampling_distribution.png")

# =========================
# Load data
# =========================
df = pd.read_csv("data/APDDv2-10023.csv", encoding="ISO-8859-1")

df = add_score_columns(df, cols_to_compare)

sampler = Sampler(
    n_samples=500,
    n_bins=30,
    seed=42,
    method="both"
)

results = sampler.run(df)

# =========================
# Plot
# =========================
plt.figure(figsize=(10, 4))

sns.histplot(
    df["Avg Score"],
    bins=30,
    label="Original",
    stat="density",
    alpha=0.4
)

sns.histplot(
    results["uniform"]["Avg Score"],
    bins=30,
    label="Uniform",
    stat="density",
    alpha=0.4
)

sns.histplot(
    results["gaussian"]["Avg Score"],
    bins=30,
    label="Gaussian",
    stat="density",
    alpha=0.4
)

plt.legend()
plt.title("Sampling sanity check")
plt.xlabel("Average aesthetic attribute score")
plt.ylabel("Density")

# =========================
# Save instead of show
# =========================
plt.tight_layout()
plt.savefig(OUTPUT_PNG, dpi=300)
plt.close()

print(f"✔ Plot saved to {OUTPUT_DIR}")
