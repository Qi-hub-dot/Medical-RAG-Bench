"""Generate 2x2 crossover heatmap for Medical-RAG-Bench paper.
Requires: pip install matplotlib seaborn
Usage: python plot_crossover.py
Output: crossover_heatmap.png
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Data: 2x2 crossover experiment
# Rows: Embedding Model (English / Chinese)
# Cols: Document Language (Chinese Medical / English Medical)
data = np.array([[33.3, 53.3],    # English embedding
                  [46.7, 33.3]])   # Chinese embedding

row_labels = ["English Embedding\n(all-MiniLM-L6-v2)", "Chinese Embedding\n(bge-small-zh-v1.5)"]
col_labels = ["Chinese\nMedical Text", "English\nMedical Text"]

fig, ax = plt.subplots(figsize=(7, 4.5))
im = ax.imshow(data, cmap="RdYlGn", vmin=20, vmax=60, aspect="equal")

# Annotate cells
for i in range(2):
    for j in range(2):
        color = "white" if data[i, j] < 40 else "black"
        text = ax.text(j, i, f"{data[i, j]:.1f}%", ha="center", va="center",
                       fontsize=18, fontweight="bold", color=color)
        # Add delta annotation
        if (i == 0 and j == 0):  # English emb on Chinese text
            ax.text(j, i + 0.35, "(-20.0pp)", ha="center", va="center",
                   fontsize=9, color="#8b0000", fontstyle="italic")
        elif (i == 1 and j == 0):  # Chinese emb on Chinese text
            ax.text(j, i + 0.35, "(+13.4pp)", ha="center", va="center",
                   fontsize=9, color="#006400", fontstyle="italic")

ax.set_xticks(range(2))
ax.set_yticks(range(2))
ax.set_xticklabels(col_labels, fontsize=11)
ax.set_yticklabels(row_labels, fontsize=11)
ax.set_title("Semantic Retrieval P@3 by Embedding Language vs. Document Language\n"
             "(2x2 Crossover — Identical Medical Content, Translated)",
             fontsize=13, fontweight="bold", pad=15)

# Colorbar
cbar = plt.colorbar(im, ax=ax, shrink=0.85)
cbar.set_label("P@3 (%)", fontsize=11)

# Annotation box
ax.text(-0.5, -0.65,
        "Language mismatch penalty: 13-20 pp\n"
        "Domain (medical) is NOT the cause —\n"
        "English embedding performs best on\n"
        "English medical text (53.3%).",
        transform=ax.transAxes, fontsize=9, style="italic",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff9e6", edgecolor="#ccc"))

plt.tight_layout()
plt.savefig("crossover_heatmap.png", dpi=200, bbox_inches="tight")
plt.savefig("crossover_heatmap.pdf", bbox_inches="tight")
print("[OK] Saved crossover_heatmap.png and crossover_heatmap.pdf")
