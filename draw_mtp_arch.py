import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

fig, ax = plt.subplots(1, 1, figsize=(10, 14))
ax.set_xlim(0, 10)
ax.set_ylim(0, 18)
ax.axis("off")

# Colors
C_INPUT = "#E8F5E9"
C_EMBED = "#C8E6C9"
C_LAYER = "#E3F2FD"
C_ATTN = "#BBDEFB"
C_MOE = "#FFF3E0"
C_ROUTER = "#FFE0B2"
C_EXPERT = "#FFCCBC"
C_NORM = "#F3E5F5"
C_LM = "#FCE4EC"
C_MTP = "#F3E5F5"  
C_MTP_ROUTER = "#CE93D8"
C_ARROW = "#546E7A"
C_LABEL = "#37474F"
C_HIGHLIGHT = "#FF5252"

def box(ax, x, y, w, h, color, text, fc=None, ec="#333", lw=1.5, fontsize=9, style="round"):
    box = FancyBboxPatch((x, y), w, h, boxstyle=f"{style},pad=0.15",
                         facecolor=color, edgecolor=ec, lw=lw, zorder=2)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center",
            fontsize=fontsize, color=C_LABEL, fontweight="bold", zorder=3)
    return box

def arrow(ax, x1, y1, x2, y2, color=C_ARROW, lw=1.5, style="-|>"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw, shrinkA=2, shrinkB=2), zorder=1)

def label(ax, x, y, text, fontsize=8, color=C_LABEL, ha="left", style="normal"):
    ax.text(x, y, text, fontsize=fontsize, color=color, ha=ha, va="center", fontstyle=style, zorder=3)

# ===========================
# INPUT
# ===========================
box(ax, 3.5, 16.5, 3, 0.7, C_INPUT, "Input Tokens", fontsize=10)
arrow(ax, 5, 16.5, 5, 15.8)

# ===========================
# Embedding
# ===========================
box(ax, 2.5, 14.8, 5, 0.8, C_EMBED, "Word Embedding", fontsize=10)
arrow(ax, 5, 14.8, 5, 14.0)

# ===========================
# Decoder Layers × N
# ===========================
# Background for decoder layers
dec_bg = FancyBboxPatch((1.5, 4.5), 7, 9.2, boxstyle="round,pad=0.2",
                         facecolor="#F5F5F5", edgecolor="#9E9E9E", lw=2, zorder=0)
ax.add_patch(dec_bg)
label(ax, 5, 13.4, "BailingMoeV2DecoderLayer × 20", fontsize=10, ha="center", style="italic")

# Layer 1 (representative)
# Self-Attention
box(ax, 2.5, 11.8, 5, 0.9, C_ATTN, "Self-Attention (QKV)", fontsize=9)
arrow(ax, 5, 11.8, 5, 10.9)

# Add & Norm
box(ax, 3, 10.0, 4, 0.7, C_NORM, "Add & Norm", fontsize=8)
arrow(ax, 5, 10.0, 5, 9.2)

# MoE FFN
moe_bg = FancyBboxPatch((2.2, 5.8), 5.6, 3.2, boxstyle="round,pad=0.1",
                         facecolor=C_MOE, edgecolor="#FFB74D", lw=1.5, zorder=0)
ax.add_patch(moe_bg)
label(ax, 5, 8.7, "MoE FFN", fontsize=9, ha="center")

# Router
router_box = FancyBboxPatch((2.8, 7.2), 4.4, 0.8, boxstyle="round,pad=0.1",
                             facecolor=C_ROUTER, edgecolor="#FF9800", lw=1.5, zorder=2)
ax.add_patch(router_box)
ax.text(5, 7.6, "Router (Top-8 of 256 Experts)", ha="center", va="center",
        fontsize=8, color=C_LABEL, fontweight="bold", zorder=3)

# Expert boxes
for i in range(4):
    ex = FancyBboxPatch((2.5 + i*1.3, 6.1), 1.0, 0.6, boxstyle="round,pad=0.05",
                         facecolor=C_EXPERT, edgecolor="#FF7043", lw=1, zorder=2)
    ax.add_patch(ex)
    ax.text(3.0 + i*1.3, 6.4, f"Exp {i+1}", ha="center", va="center",
            fontsize=7, color=C_LABEL, zorder=3)
label(ax, 2.5, 5.9, "... (256 experts)", fontsize=6, color="#888")

arrow(ax, 5, 7.2, 5, 6.8)

# Router label - decoder router
label(ax, 7.9, 7.6, "← router_logits[-2]", fontsize=7, color=C_HIGHLIGHT, ha="left")

arrow(ax, 5, 6.1, 5, 5.4)

# Add & Norm
box(ax, 3, 4.7, 4, 0.7, C_NORM, "Add & Norm", fontsize=8)
arrow(ax, 5, 4.7, 5, 4.0)

# ===========================
# Final Norm
# ===========================
box(ax, 3.5, 3.3, 3, 0.6, C_NORM, "Final Norm", fontsize=9)
arrow(ax, 5, 3.3, 5, 2.6)

# ===========================
# Split: LM Head vs MTP Head
# ===========================
# Branch line
ax.plot([5, 3.5, 3.5], [2.6, 2.6, 1.8], color=C_ARROW, lw=1.5, zorder=1)
ax.plot([5, 6.5, 6.5], [2.6, 2.6, 1.8], color=C_ARROW, lw=1.5, zorder=1)
arrow(ax, 3.5, 1.8, 3.5, 1.2)
arrow(ax, 6.5, 1.8, 6.5, 1.2)

# ===========================
# LM Head
# ===========================
box(ax, 1.5, 0.4, 4, 0.8, C_LM, "LM Head (Vocab Softmax)", fontsize=9)
label(ax, 1.5, 0.8, "Output: next token", fontsize=7, color="#666", ha="left")

# ===========================
# MTP Head
# ===========================
mtp_bg = FancyBboxPatch((4.8, 0.0), 4.8, 1.8, boxstyle="round,pad=0.15",
                          facecolor=C_MTP, edgecolor="#AB47BC", lw=2, zorder=0)
ax.add_patch(mtp_bg)
label(ax, 7.2, 1.7, "BailingMoeV2MTPLayer (MTP Head)", fontsize=9, ha="center", color="#6A1B9A")

# MTP Router
mtp_router = FancyBboxPatch((5.2, 0.8), 4.0, 0.6, boxstyle="round,pad=0.08",
                              facecolor=C_MTP_ROUTER, edgecolor="#9C27B0", lw=1.5, zorder=2)
ax.add_patch(mtp_router)
ax.text(7.2, 1.1, "MTP Router (predicts next token's experts)", ha="center", va="center",
        fontsize=7, color="white", fontweight="bold", zorder=3)

# MTP Router label
label(ax, 9.4, 1.1, "← router_logits[-1]", fontsize=7, color=C_HIGHLIGHT, ha="left")

arrow(ax, 7.2, 0.8, 7.2, 0.4)
box(ax, 5.8, 0.1, 2.8, 0.3, "#E1BEE7", "Self-Attention + Norm", fontsize=6, ec="#CE93D8")
label(ax, 5.2, 0.1, "Output: next token routing pred", fontsize=6, color="#666", ha="left")

# ===========================
# Legend
# ===========================
legend_x, legend_y = 0.3, 0.3
leg_elements = [
    mpatches.Patch(facecolor=C_ROUTER, edgecolor="#FF9800", label="Decoder Router (actual)"),
    mpatches.Patch(facecolor=C_MTP_ROUTER, edgecolor="#9C27B0", label="MTP Router (predicted)"),
]
legend = ax.legend(handles=leg_elements, loc="lower left", fontsize=7,
                   framealpha=0.9, edgecolor="#ccc")
ax.add_artist(legend)

# ===========================
# Title
# ===========================
ax.text(5, 17.5, "BailingMoeV2 — MTP Model Architecture", ha="center", va="center",
        fontsize=13, color=C_LABEL, fontweight="bold")

# Note
ax.text(5, 0.0, "MTP predicts next token's expert routing.  Comparison: mtp_router[:, :-1, :] vs decoder_router[:, 1:, :]",
        ha="center", va="bottom", fontsize=7, color="#999", fontstyle="italic")

plt.tight_layout()
plt.savefig("mtp_architecture.png", dpi=200, bbox_inches="tight", pad_inches=0.3)
plt.close()
print("Diagram saved to mtp_architecture.png")
