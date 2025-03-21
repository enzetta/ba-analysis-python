import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Define paths
DATA_PATH = "analysis/data/toxicity_distribution.csv"
SAVE_PATH = "analysis/visualisations/toxicity_distribution.png"

# Ensure directories exist
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_PATH)

# Convert toxicity bucket to numeric
df["toxicity_start"] = df["toxicity_bucket"].astype(float)

# Sort values
df = df.sort_values("toxicity_start")

# Define bar width
bar_width = 0.04  # Ensures bars fit within bin width

# Create figure
plt.figure(figsize=(12, 5))
sns.set(style="white")

# Align bars properly at their exact positions
plt.bar(df["toxicity_start"],
        df["tweet_count"],
        width=bar_width,
        color="steelblue",
        align="edge")

# Use actual toxicity values as labels
plt.xticks(df["toxicity_start"],
           labels=df["toxicity_start"].astype(str),
           fontsize=10)

# Set x-axis limits to fit bars well
plt.xlim(-0.02, 1.0)

# Set log scale for y-axis
plt.yscale("log")

# Ensure y-axis starts at 1 (log scale requires positive values)
plt.ylim(1, df["tweet_count"].max() * 1.1)

# Titles & Labels
plt.title("Toxicity Score Distribution (Log Scale)",
          fontsize=14,
          weight="bold")
plt.xlabel("Toxicity Score", fontsize=12)
plt.ylabel("Tweet Count (log scale)", fontsize=12)

# Remove grid and top/right borders
plt.grid(False)
sns.despine(top=True, right=True)

# Save plot
plt.tight_layout()
plt.savefig(SAVE_PATH, dpi=300)
plt.show()
