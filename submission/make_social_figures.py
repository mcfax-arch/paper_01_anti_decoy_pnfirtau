"""Two 16:9 explanatory schematics for the paper / social posts.

Outputs (PNG + PDF) in figures/:
  fig19_schematic_architecture_16x9
  fig20_schematic_abstention_16x9
No data axes; these are diagrams, so they are not part of the data-figure audit.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 13,
    "mathtext.fontset": "dejavusans",
})

BLUE = "#dbe8f7"
YELLOW = "#ffe8a3"
ORANGE = "#f7c59f"
GREEN = "#d8f0d8"
RED = "#f6d0d0"
GREY = "#eeeeee"
EDGE = "#2b2b2b"


def box(ax, x, y, w, h, text, fc=BLUE, ec=EDGE, fs=13, bold=False, lw=1.6):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.006,rounding_size=0.012",
        fc=fc, ec=ec, lw=lw, transform=ax.transAxes, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, fontweight="bold" if bold else "normal",
            transform=ax.transAxes, zorder=3, linespacing=1.25)


def arrow(ax, x1, y1, x2, y2, color=EDGE, lw=2.0, style="-|>", rad=0.0):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style, mutation_scale=18,
        lw=lw, color=color, connectionstyle=f"arc3,rad={rad}",
        transform=ax.transAxes, zorder=1))


def architecture():
    fig = plt.figure(figsize=(16, 9), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.03, 0.955, "pn_fir_tau: a causal FIR on K, then a calibrated off switch",
            fontsize=22, fontweight="bold", transform=ax.transAxes, va="top")
    ax.text(0.03, 0.905, "The filter acts on the key stream only; Q and V stay sharp.",
            fontsize=15, color="#444444", transform=ax.transAxes, va="top")

    # Left: vertical pipeline
    box(ax, 0.10, 0.76, 0.38, 0.09, "hidden states  $B_t$", fc=GREY, fs=15)
    box(ax, 0.10, 0.615, 0.10, 0.085, "$Q = W_Q B_t$", fs=12)
    box(ax, 0.24, 0.615, 0.10, 0.085, "$K = W_K B_t$", fs=12)
    box(ax, 0.38, 0.615, 0.10, 0.085, "$V = W_V B_t$", fs=12)

    box(ax, 0.20, 0.465, 0.28, 0.09,
        "causal 7-tap FIR\n(low-pass on $K$ only, $\\sigma=2.0$)", fc=YELLOW, fs=13, bold=True)
    box(ax, 0.20, 0.315, 0.28, 0.09,
        "scores  $S_{t,j} = \\langle Q_t, \\tilde{K}_j \\rangle / \\sqrt{d_k}$", fc=GREEN, fs=13)
    box(ax, 0.20, 0.165, 0.28, 0.09,
        "threshold $\\tau_\\ell$ + peak-silence gate\n$g_t = \\max S^+ / (\\max S^+ + \\gamma)$", fc=ORANGE, fs=12, bold=True)
    box(ax, 0.20, 0.035, 0.28, 0.075,
        "weights  $W_{t,j} = g_t \\cdot (S^+_{t,j})^p / \\sum (S^+)^p$", fc=BLUE, fs=12)

    # arrows
    arrow(ax, 0.29, 0.76, 0.15, 0.70, rad=0.0)
    arrow(ax, 0.29, 0.76, 0.29, 0.70)
    arrow(ax, 0.29, 0.76, 0.43, 0.70, rad=0.0)
    arrow(ax, 0.29, 0.615, 0.34, 0.557)
    arrow(ax, 0.34, 0.465, 0.34, 0.405)
    arrow(ax, 0.34, 0.315, 0.34, 0.255)
    arrow(ax, 0.34, 0.165, 0.34, 0.110)

    # bypass Q and V to scores
    arrow(ax, 0.15, 0.615, 0.20, 0.405, rad=0.25, color="#6b6b6b", lw=1.3, style="-|>")
    arrow(ax, 0.43, 0.615, 0.48, 0.405, rad=-0.25, color="#6b6b6b", lw=1.3, style="-|>")
    ax.text(0.115, 0.535, "Q: unfiltered", fontsize=10, color="#555555", rotation=72,
            transform=ax.transAxes, ha="center")
    ax.text(0.505, 0.535, "V: unfiltered", fontsize=10, color="#555555", rotation=-72,
            transform=ax.transAxes, ha="center")

    # Right: why it matters
    ax.text(0.63, 0.855, "What this buys you", fontsize=19, fontweight="bold",
            transform=ax.transAxes, va="top")

    box(ax, 0.60, 0.60, 0.37, 0.19,
        "One layer binds a queried key to the\nadjacent value slot.\n\n"
        "Non-bijective recall: 44.5% $\\rightarrow$ 100.0%",
        fc=BLUE, fs=14)
    box(ax, 0.60, 0.335, 0.37, 0.19,
        "Rectification alone starves under the\nscore shift: fixed $\\tau=0$ drops to 64.4%,\n"
        "with 91.3% negative scores and 44.5% dead rows.",
        fc=RED, fs=13)
    box(ax, 0.60, 0.07, 0.37, 0.19,
        "One learned $\\tau$ per layer restores 100% recall\nand keeps unanswerable rows silent:\n"
        "row mass 0.002 vs 0.100 for virtual sinks.",
        fc=GREEN, fs=13)

    ax.text(0.03, 0.005, "Schematic; not to scale. Numbers from the controlled experiments in the paper.",
            fontsize=10, color="#666666", transform=ax.transAxes, va="bottom")

    fig.savefig("social/fig19_schematic_architecture_16x9.png", bbox_inches=None, facecolor="white")
    fig.savefig("social/fig19_schematic_architecture_16x9.pdf", bbox_inches=None, facecolor="white")
    plt.close(fig)


def token_row(ax, y, weights, colors, labels, x0=0.21, width=0.44, h=0.11):
    n = len(weights)
    gap = 0.008
    w = (width - gap * (n - 1)) / n
    for i, (wt, col, lab) in enumerate(zip(weights, colors, labels)):
        x = x0 + i * (w + gap)
        ax.add_patch(Rectangle((x, y), w, h, fc=col, ec=EDGE, lw=1.2,
                               transform=ax.transAxes, zorder=2))
        ax.text(x + w / 2, y + h / 2, lab, ha="center", va="center", fontsize=10,
                transform=ax.transAxes, zorder=3)
        ax.text(x + w / 2, y - 0.022, f"{wt:.3f}", ha="center", va="top", fontsize=9,
                color="#444444", transform=ax.transAxes)


def abstention():
    fig = plt.figure(figsize=(16, 9), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.03, 0.955, "Softmax has no off switch. pn_fir_tau does.",
            fontsize=22, fontweight="bold", transform=ax.transAxes, va="top")
    ax.text(0.03, 0.905, "Same query, no matching key: what each normalizer writes into the residual stream.",
            fontsize=14, color="#444444", transform=ax.transAxes, va="top")

    labels = ["junk", "junk", "junk", "junk", "junk", "junk", "junk", "junk"]

    # Softmax row
    y = 0.70
    ax.text(0.03, y + 0.055, "Softmax", fontsize=17, fontweight="bold", transform=ax.transAxes)
    ax.text(0.03, y + 0.015, "simplex: mass = 1.000", fontsize=11, color="#555555", transform=ax.transAxes)
    token_row(ax, y, [0.125] * 8, [BLUE] * 8, labels)
    box(ax, 0.70, y, 0.27, 0.11, "residual gets a weighted average\nof irrelevant values  →  hallucination",
        fc=RED, fs=12, bold=False)
    arrow(ax, 0.665, y + 0.055, 0.70, y + 0.055, color="#a33", lw=2)
    ax.text(0.21, y - 0.065, "Hallucination rate on absent queries: 27.2–34.4% across simplex normalizers (Exp 5).",
            fontsize=11, color="#7a1f1f", transform=ax.transAxes)

    # Virtual sink row
    y = 0.43
    ax.text(0.03, y + 0.055, "Virtual-sink Softmax", fontsize=17, fontweight="bold", transform=ax.transAxes)
    ax.text(0.03, y + 0.015, "mass < 1, but tails stay positive", fontsize=11, color="#555555", transform=ax.transAxes)
    token_row(ax, y, [0.1125] * 8, [BLUE] * 8, labels)
    ax.add_patch(Rectangle((0.665, y), 0.04, 0.11, fc=GREY, ec=EDGE, lw=1.2, transform=ax.transAxes))
    ax.text(0.685, y + 0.055, "sink", ha="center", va="center", fontsize=9, transform=ax.transAxes)
    ax.text(0.685, y - 0.022, "0.100", ha="center", va="top", fontsize=9, color="#444444", transform=ax.transAxes)
    box(ax, 0.72, y, 0.25, 0.11, "still leaks 0.100 mass\ninto the residual", fc=ORANGE, fs=12)
    arrow(ax, 0.705, y + 0.055, 0.72, y + 0.055, color="#a33", lw=2)
    ax.text(0.21, y - 0.065, "Residual leakage on absent keys: 0.100 ± 0.005 (Exp 5).",
            fontsize=11, color="#7a4a1f", transform=ax.transAxes)

    # pn_fir_tau row
    y = 0.16
    ax.text(0.03, y + 0.055, "pn_fir_tau", fontsize=17, fontweight="bold", transform=ax.transAxes)
    ax.text(0.03, y + 0.015, "sub-simplex: mass ≤ 1", fontsize=11, color="#555555", transform=ax.transAxes)
    token_row(ax, y, [0.00025] * 8, [GREEN] * 8, ["0"] * 8)
    box(ax, 0.70, y, 0.27, 0.11, "residual preserved\n(no update)", fc=GREEN, fs=13, bold=True)
    arrow(ax, 0.665, y + 0.055, 0.70, y + 0.055, color="#2e7d32", lw=2)
    ax.text(0.21, y - 0.065, "Unanswerable-row mass: 0.002 ± 0.002; peak gate closes the row (Exp 5).",
            fontsize=11, color="#1f5a24", transform=ax.transAxes)

    ax.text(0.03, 0.02, "Controlled synthetic abstention task, 50% unanswerable queries; see the paper for scope.",
            fontsize=10, color="#666666", transform=ax.transAxes, va="bottom")

    fig.savefig("social/fig20_schematic_abstention_16x9.png", bbox_inches=None, facecolor="white")
    fig.savefig("social/fig20_schematic_abstention_16x9.pdf", bbox_inches=None, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    architecture()
    abstention()
    print("wrote fig19 and fig20 (16:9, PNG+PDF)")
