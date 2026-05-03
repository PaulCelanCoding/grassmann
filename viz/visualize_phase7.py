"""Phase 7 architecture visualization.

A simple diagram showing the two rendering paths and how the adapter routes
between them. Generated as an HTML artifact via matplotlib's text rendering.
"""
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

import torch

from grassmann.fast_rasterizer import is_available


def draw_phase7_diagram():
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 8)
    ax.axis("off")

    def box(x, y, w, h, label, color, alpha=0.3, fontsize=10):
        b = FancyBboxPatch((x, y), w, h,
                           boxstyle="round,pad=0.1,rounding_size=0.15",
                           linewidth=1.5, edgecolor=color, facecolor=color, alpha=alpha)
        ax.add_patch(b)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold")

    def arrow(x1, y1, x2, y2, label="", color="black", style="-"):
        a = FancyArrowPatch((x1, y1), (x2, y2),
                             arrowstyle="->", mutation_scale=18,
                             linewidth=1.4, color=color, linestyle=style)
        ax.add_patch(a)
        if label:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mx, my + 0.1, label, fontsize=8, ha="center",
                    style="italic", color=color)

    # Title
    ax.text(6.5, 7.6, "Phase 7: Dual-Path Rasterization Architecture",
            ha="center", fontsize=14, fontweight="bold")

    # Trainer box
    box(0.3, 5.5, 2.5, 1.2, "Trainer\n(training.py)", "steelblue")
    # TrainableGaussians / forward
    box(0.3, 4.0, 2.5, 1.0, "TrainableGaussians\nforward()", "steelblue")

    # Decision diamond
    ax.plot([5, 6.5, 8, 6.5, 5], [5, 6, 5, 4, 5], "k-", linewidth=1.5)
    ax.text(6.5, 5, "use_fast_rasterizer?\n& CUDA available?\n& params on GPU?",
            ha="center", va="center", fontsize=9, fontweight="bold")

    # Path A: Toy rasterizer (fallback)
    box(3.3, 2.0, 2.8, 1.0, "toy_rasterize()\npure PyTorch CPU/GPU", "darkorange")
    box(3.3, 0.4, 2.8, 1.0, "project_to_screen()\n+ eval_2d_gaussian()\n+ composite loop", "darkorange", fontsize=8)

    # Path B: Fast rasterizer (CUDA)
    box(8.0, 2.0, 2.8, 1.0, "fast_rasterize()\nadapter", "seagreen")
    box(8.0, 0.4, 2.8, 1.0, "diff_gaussian_rasterization\n(CUDA kernel)", "seagreen", fontsize=8)

    # Inputs box
    box(10.5, 5.0, 2.2, 1.7,
        "compute_derived\n+ condition_on_time\n\nproduces\n(V_3D, Σ_3D, α_eff)",
        "mediumpurple", alpha=0.25, fontsize=8)

    # Arrows
    arrow(1.55, 5.5, 1.55, 5.05)
    arrow(2.8, 4.5, 5.0, 5.0, label="render_one()")
    arrow(2.8, 4.5, 10.5, 5.85, label="(world-independent)", color="mediumpurple", style="--")
    arrow(5.5, 4.5, 4.7, 3.05, label="fallback\n(no GPU)", color="darkorange")
    arrow(7.5, 4.5, 9.0, 3.05, label="CUDA available", color="seagreen")
    arrow(4.7, 2.0, 4.7, 1.45, color="darkorange")
    arrow(9.4, 2.0, 9.4, 1.45, color="seagreen")
    # Inputs to both
    arrow(10.5, 5.4, 9.4, 3.1, color="mediumpurple", style="--")
    arrow(10.5, 5.2, 4.7, 2.55, color="mediumpurple", style="--")

    # Legend
    legend_elems = [
        Line2D([0], [0], color="darkorange", lw=2, label="Phase 3 toy path (CPU-safe)"),
        Line2D([0], [0], color="seagreen", lw=2, label="Phase 7 CUDA path (100x+ faster)"),
        Line2D([0], [0], color="mediumpurple", lw=2, linestyle="--",
                label="view-independent (amortized across K cameras)"),
    ]
    ax.legend(handles=legend_elems, loc="lower center", ncol=3, fontsize=9,
              bbox_to_anchor=(0.5, -0.02))

    # Availability status
    avail_text = ("CUDA rasterizer detected ✓"
                  if is_available() else
                  "CPU environment: toy fallback active")
    color = "seagreen" if is_available() else "darkorange"
    ax.text(6.5, 7.15, avail_text, ha="center", fontsize=10,
            style="italic", color=color,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor=color, alpha=0.8))

    plt.savefig("docs/images/phase7_architecture.png",
                dpi=110, bbox_inches="tight")
    plt.close()
    print("Saved phase7_architecture.png")


if __name__ == "__main__":
    draw_phase7_diagram()
