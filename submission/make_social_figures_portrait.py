"""Portrait (9:16) versions of the two explanatory schematics for X/mobile.

Outputs in social/:
  fig19_schematic_architecture_9x16.png/pdf  (1800x3200)
  fig20_schematic_abstention_9x16.png/pdf
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
                     "mathtext.fontset": "dejavusans"})

BLUE = "#dbe8f7"; YELLOW = "#ffe8a3"; ORANGE = "#f7c59f"
GREEN = "#d8f0d8"; RED = "#f6d0d0"; GREY = "#eeeeee"; EDGE = "#2b2b2b"


def box(ax, x, y, w, h, text, fc=BLUE, ec=EDGE, fs=12, bold=False, lw=1.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
        boxstyle="round,pad=0.005,rounding_size=0.008",
        fc=fc, ec=ec, lw=lw, transform=ax.transAxes, zorder=2))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", transform=ax.transAxes,
            zorder=3, linespacing=1.2)


def arrow(ax, x1, y1, x2, y2, color=EDGE, lw=1.8, rad=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
        mutation_scale=15, lw=lw, color=color,
        connectionstyle=f"arc3,rad={rad}", transform=ax.transAxes, zorder=1))


def architecture_portrait():
    fig = plt.figure(figsize=(9, 16), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    ax.text(0.04, 0.975, "pn_fir_tau:\ncausal FIR on K + a calibrated off switch",
            fontsize=20, fontweight="bold", transform=ax.transAxes, va="top")
    ax.text(0.04, 0.918, "The filter acts on the key stream only; Q and V stay sharp.",
            fontsize=12, color="#444444", transform=ax.transAxes, va="top")

    box(ax, 0.14, 0.835, 0.72, 0.055, "hidden states  $B_t$", fc=GREY, fs=15)
    box(ax, 0.14, 0.735, 0.21, 0.055, "$Q = W_Q B_t$", fs=11)
    box(ax, 0.395, 0.735, 0.21, 0.055, "$K = W_K B_t$", fs=11)
    box(ax, 0.65, 0.735, 0.21, 0.055, "$V = W_V B_t$", fs=11)
    box(ax, 0.14, 0.625, 0.72, 0.065,
        "causal 7-tap FIR, low-pass on $K$ only ($\\sigma=2.0$)", fc=YELLOW, fs=13, bold=True)
    box(ax, 0.14, 0.515, 0.72, 0.065,
        "scores  $S_{t,j} = \\langle Q_t, \\tilde{K}_j \\rangle / \\sqrt{d_k}$", fc=GREEN, fs=13)
    box(ax, 0.14, 0.405, 0.72, 0.065,
        "threshold $\\tau_\\ell$ + peak-silence gate", fc=ORANGE, fs=13, bold=True)
    box(ax, 0.14, 0.295, 0.72, 0.065,
        "weights  $W_{t,j} = g_t \\cdot (S^+_{t,j})^p / \\sum (S^+)^p$", fs=12)

    arrow(ax, 0.50, 0.835, 0.245, 0.790)
    arrow(ax, 0.50, 0.835, 0.50, 0.790)
    arrow(ax, 0.50, 0.835, 0.755, 0.790)
    arrow(ax, 0.50, 0.735, 0.50, 0.690)
    arrow(ax, 0.50, 0.625, 0.50, 0.580)
    arrow(ax, 0.50, 0.515, 0.50, 0.470)
    arrow(ax, 0.50, 0.405, 0.50, 0.360)
    ax.text(0.50, 0.710, "Q and V bypass the filter (unfiltered)", fontsize=9,
            color="#555555", ha="center", transform=ax.transAxes)

    ax.text(0.04, 0.270, "What this buys you", fontsize=16, fontweight="bold",
            transform=ax.transAxes, va="top")
    box(ax, 0.05, 0.175, 0.90, 0.065,
        "One layer binds a queried key to the adjacent value slot.\n"
        "Non-bijective recall: 44.5% $\\rightarrow$ 100.0%", fc=BLUE, fs=11)
    box(ax, 0.05, 0.095, 0.90, 0.065,
        "Fixed $\\tau=0$ rectification starves: 64.4% recall,\n"
        "91.3% negative scores, 44.5% dead rows.", fc=RED, fs=11)
    box(ax, 0.05, 0.015, 0.90, 0.065,
        "One learned $\\tau$ per layer: 100% recall; unanswerable row mass\n"
        "0.002 vs 0.100 for virtual sinks.", fc=GREEN, fs=11)

    fig.savefig("social/fig19_schematic_architecture_9x16.png", bbox_inches=None, facecolor="white")
    fig.savefig("social/fig19_schematic_architecture_9x16.pdf", bbox_inches=None, facecolor="white")
    plt.close(fig)


def token_row(ax, y, weights, colors, labels, h=0.055, x0=0.05, width=0.90):
    n = len(weights); gap = 0.008
    w = (width - gap*(n-1))/n
    for i, (wt, col, lab) in enumerate(zip(weights, colors, labels)):
        x = x0 + i*(w+gap)
        ax.add_patch(Rectangle((x, y), w, h, fc=col, ec=EDGE, lw=1.0,
                               transform=ax.transAxes, zorder=2))
        ax.text(x+w/2, y+h/2, lab, ha="center", va="center", fontsize=9,
                transform=ax.transAxes, zorder=3)
        ax.text(x+w/2, y-0.012, f"{wt:.3f}", ha="center", va="top", fontsize=8,
                color="#444444", transform=ax.transAxes)


def abstention_portrait():
    fig = plt.figure(figsize=(9, 16), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    ax.text(0.04, 0.975, "Softmax has no off switch.\npn_fir_tau does.",
            fontsize=21, fontweight="bold", transform=ax.transAxes, va="top")
    ax.text(0.04, 0.918, "Same query, no matching key: what each normalizer\nwrites into the residual stream.",
            fontsize=12, color="#444444", transform=ax.transAxes, va="top")

    # Softmax
    ax.text(0.04, 0.855, "Softmax", fontsize=17, fontweight="bold", transform=ax.transAxes)
    ax.text(0.04, 0.832, "simplex: mass = 1.000", fontsize=10, color="#555555", transform=ax.transAxes)
    token_row(ax, 0.755, [0.125]*8, [BLUE]*8, ["junk"]*8)
    box(ax, 0.50, 0.665, 0.45, 0.055, "residual gets a weighted average of\nirrelevant values $\\rightarrow$ hallucination", fc=RED, fs=10)
    arrow(ax, 0.72, 0.755, 0.72, 0.720, color="#a33")
    ax.text(0.04, 0.635, "Hallucination rate on absent queries: 27.2–34.4% (Exp 5).",
            fontsize=10, color="#7a1f1f", transform=ax.transAxes)

    # Virtual sink
    ax.text(0.04, 0.575, "Virtual-sink Softmax", fontsize=17, fontweight="bold", transform=ax.transAxes)
    ax.text(0.04, 0.552, "mass < 1, but tails stay positive", fontsize=10, color="#555555", transform=ax.transAxes)
    token_row(ax, 0.475, [0.1125]*8, [BLUE]*8, ["junk"]*8, width=0.80)
    ax.add_patch(Rectangle((0.87, 0.475), 0.045, 0.055, fc=GREY, ec=EDGE, lw=1.0,
                           transform=ax.transAxes, zorder=2))
    ax.text(0.892, 0.502, "sink", ha="center", va="center", fontsize=8, transform=ax.transAxes)
    ax.text(0.892, 0.463, "0.100", ha="center", va="top", fontsize=8, color="#444444", transform=ax.transAxes)
    box(ax, 0.50, 0.385, 0.45, 0.055, "still leaks 0.100 mass into the residual", fc=ORANGE, fs=10)
    arrow(ax, 0.72, 0.475, 0.72, 0.440, color="#a33")
    ax.text(0.04, 0.355, "Residual leakage on absent keys: 0.100 ± 0.005 (Exp 5).",
            fontsize=10, color="#7a4a1f", transform=ax.transAxes)

    # pn_fir_tau
    ax.text(0.04, 0.295, "pn_fir_tau", fontsize=17, fontweight="bold", transform=ax.transAxes)
    ax.text(0.04, 0.272, "sub-simplex: mass ≤ 1", fontsize=10, color="#555555", transform=ax.transAxes)
    token_row(ax, 0.195, [0.00025]*8, [GREEN]*8, ["0"]*8)
    box(ax, 0.50, 0.105, 0.45, 0.055, "residual preserved (no update)", fc=GREEN, fs=11, bold=True)
    arrow(ax, 0.72, 0.195, 0.72, 0.160, color="#2e7d32")
    ax.text(0.04, 0.075, "Unanswerable-row mass: 0.002 ± 0.002 (Exp 5).",
            fontsize=10, color="#1f5a24", transform=ax.transAxes)

    ax.text(0.04, 0.02, "Controlled synthetic abstention task, 50% unanswerable queries.",
            fontsize=9, color="#666666", transform=ax.transAxes, va="bottom")

    fig.savefig("social/fig20_schematic_abstention_9x16.png", bbox_inches=None, facecolor="white")
    fig.savefig("social/fig20_schematic_abstention_9x16.pdf", bbox_inches=None, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    architecture_portrait()
    abstention_portrait()
    print("wrote portrait 9:16 schematics")
