"""Plot the LSH banding S-curve P(s) = 1 - (1 - s^r)^b for various configurations."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def lsh_probability(s, b, r):
    return 1.0 - (1.0 - s**r)**b

s = np.linspace(0, 1, 1000)

configs = [
    (100, 1, "k=100, b=100, r=1"),
    (50, 2, "k=100, b=50, r=2 (default)"),
    (20, 3, "k=60, b=20, r=3"),
    (10, 5, "k=50, b=10, r=5"),
    (10, 10, "k=100, b=10, r=10"),
]

fig, ax = plt.subplots(figsize=(7, 5))

for b, r, label in configs:
    p = lsh_probability(s, b, r)
    style = "-" if "(default)" not in label else "-"
    lw = 2.5 if "(default)" in label else 1.5
    ax.plot(s, p, linewidth=lw, label=label)

ax.set_xlabel("True Jaccard Similarity $s$", fontsize=12)
ax.set_ylabel("Candidate Probability $P(s)$", fontsize=12)
ax.set_title("LSH Banding S-Curve: Candidate Selection Probability", fontsize=13)
ax.legend(fontsize=9, loc="lower right")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.02)
ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
ax.grid(True, alpha=0.3)
fig.tight_layout()

out_dir = Path("results/figures")
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "lsh_s_curve.png"
fig.savefig(out_path, dpi=150)
print(f"Saved to {out_path}")
