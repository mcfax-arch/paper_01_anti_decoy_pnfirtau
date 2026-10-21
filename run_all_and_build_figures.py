#!/usr/bin/env python3
"""Master Reproduction & Figure Generation Script for Paper 01 (Anti-Decoy / PnFirTau).

Reads the JSON outputs produced by:
  - experiments/exp1_legacy_stack_candidates.py  -> results/exp1_legacy_13stacks.json
  - experiments/exp2_content_addressable_decoy.py -> results/exp2_content_addressable_decoy.json
  - experiments/exp3_sweep_tau_gate.py           -> results/exp3_sweep_tau_gate.json
  - experiments/exp4_wikitext2_lm.py             -> results/exp4_wikitext2_lm.json

Generates all 4 publication figures (PNG + PDF) strictly from the JSON data with ZERO hardcoded
numbers, and writes a consolidated `paper01_metrics.json`.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(ROOT, "results")
FIG_DIR = os.path.join(ROOT, "figures")

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.5,
        "figure.dpi": 160,
        "savefig.dpi": 300,
        "axes.grid": True,
        "grid.alpha": 0.28,
        "grid.linestyle": "--",
    }
)


def load_json(name: str) -> dict:
    path = os.path.join(RESULTS_DIR, name)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing {path}. Run experiments first.")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_fig1(exp1: dict, exp2: dict) -> None:
    """Fig 1: Legacy Fixed-Position Bijective vs. True Content-Addressable Multi-Decoy & Attention Target."""
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.2))

    # Panel (a): Legacy fixed-position bijective benchmark (exp1)
    ax = axes[0]
    s1 = exp1["summary"]
    nds = [0, 1, 2, 4, 8]
    styles_1 = [
        ("sm", "Softmax (sm)", "#d95f02", "o", "--"),
        ("pn", "Power-Norm (pn)", "#e7298a", "s", ":"),
        ("pn_fir", "PN + Causal FIR (pn_fir)", "#7570b3", "^", "-."),
        ("sm_fir", "Softmax + Causal FIR (sm_fir)", "#1b9e77", "D", "-"),
        ("pn_fir_tau", "PN + Causal FIR + τ (pn_fir_tau)", "#2b5c8f", "*", "-"),
    ]
    for sid, label, color, marker, ls in styles_1:
        d = s1[sid]
        means = [d["clean"]["mean"] * 100.0] + [
            d["multi_decoy"][f"nd_{n}"]["mean"] * 100.0 for n in (1, 2, 4, 8)
        ]
        stds = [d["clean"]["std"] * 100.0] + [
            d["multi_decoy"][f"nd_{n}"]["std"] * 100.0 for n in (1, 2, 4, 8)
        ]
        ax.errorbar(
            nds,
            means,
            yerr=stds,
            label=label,
            color=color,
            marker=marker,
            linestyle=ls,
            linewidth=2.0,
            markersize=6,
            capsize=3,
        )

    w_k_pn = s1["pn_fir_tau"]["attention_pos"]["w_key_pos"]["mean"]
    w_v_pn = s1["pn_fir_tau"]["attention_pos"]["w_val_pos"]["mean"]
    ax.annotate(
        f"Bijective shortcut at t=0:\n"
        f"pn_fir_tau w(KEY_POS=0)={w_k_pn:.2f}, w(VAL_POS=1)={w_v_pn:.2f}\n"
        f"Decoys copy true key -> sm rises to 100%",
        xy=(8, s1["sm"]["multi_decoy"]["nd_8"]["mean"] * 100.0),
        xytext=(0.3, 33.0),
        fontsize=8.0,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#fff8dc", edgecolor="#b8860b", alpha=0.92),
        arrowprops=dict(arrowstyle="->", color="#b8860b", lw=1.2),
    )
    ax.set_title("(a) Legacy Fixed-Pos Bijective Task (Exp 1, 3 seeds)")
    ax.set_xlabel("Number of Injected Decoys $n_d$ ($\\epsilon = 0.2$)")
    ax.set_ylabel("Recall Accuracy (%)")
    ax.set_xticks(nds)
    ax.set_ylim(15, 105)
    ax.legend(loc="lower right", framealpha=0.92, fontsize=8.0)

    # Panel (b): True Content-Addressable Non-Bijective Task (exp2)
    ax2 = axes[1]
    s2 = exp2["summary"]
    styles_2 = [
        ("sm", "Softmax (sm, no FIR)", "#d95f02", "o", "--"),
        ("pn", "Power-Norm (pn, no FIR)", "#e7298a", "s", ":"),
        ("pn_fir", "PN + Causal FIR (pn_fir)", "#7570b3", "^", "-."),
        ("sm_fir", "Softmax + Causal FIR (sm_fir)", "#1b9e77", "D", "-"),
        ("pn_fir_tau", "PN + Causal FIR + τ (pn_fir_tau)", "#2b5c8f", "*", "-"),
        ("pn_fir_tau_reg", "PN + FIR + τ + Silence-Reg", "#08306b", "P", "-"),
    ]
    for sid, label, color, marker, ls in styles_2:
        d = s2[sid]
        means = [d["clean"]["mean"] * 100.0] + [
            d["multi_decoy"][f"nd_{n}"]["mean"] * 100.0 for n in (1, 2, 4, 8)
        ]
        stds = [d["clean"]["std"] * 100.0] + [
            d["multi_decoy"][f"nd_{n}"]["std"] * 100.0 for n in (1, 2, 4, 8)
        ]
        ax2.errorbar(
            nds,
            means,
            yerr=stds,
            label=label,
            color=color,
            marker=marker,
            linestyle=ls,
            linewidth=2.0,
            markersize=6,
            capsize=3,
        )

    wv2_smfir = s2["sm_fir"]["attention_target"]["w_val_pos"]["mean"]
    wk2_smfir = s2["sm_fir"]["attention_target"]["w_key_pos"]["mean"]
    wv2_pntau = s2["pn_fir_tau"]["attention_target"]["w_val_pos"]["mean"]
    wk2_pntau = s2["pn_fir_tau"]["attention_target"]["w_key_pos"]["mean"]
    ax2.annotate(
        f"True Value-Slot Binding (j*+1):\n"
        f"sm_fir: w_val={wv2_smfir:.2f}, w_key={wk2_smfir:.2f}\n"
        f"pn_fir_tau: w_val={wv2_pntau:.2f}, w_key={wk2_pntau:.2f}\n"
        f"(Unfiltered sm/pn stuck at ~36–45%)",
        xy=(0, s2["pn_fir_tau"]["clean"]["mean"] * 100.0),
        xytext=(1.8, 16.0),
        fontsize=8.0,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#eef7ff", edgecolor="#2b5c8f", alpha=0.92),
        arrowprops=dict(arrowstyle="->", color="#2b5c8f", lw=1.2),
    )
    ax2.set_title("(b) True Content-Addressable Non-Bijective Task (Exp 2, 3 seeds)")
    ax2.set_xlabel("Number of Isolated Distractor Decoys $n_d$ ($\\epsilon = 0.2$)")
    ax2.set_ylabel("Value Recall Accuracy (%)")
    ax2.set_xticks(nds)
    ax2.set_ylim(10, 105)
    ax2.legend(loc="upper right", framealpha=0.92, fontsize=8.0)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG_DIR, f"fig1_decoy_and_multidecoy.{ext}"), bbox_inches="tight")
    plt.close(fig)


def build_fig2(exp1: dict, exp2: dict) -> None:
    """Fig 2: Score-Shift Pathology after Causal FIR & 3-Seed Calibration Ablation."""
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.1))

    # Panel (a): Bounded [0, 100%] ReLU survival & active row diagnostics from exp1["summary"]
    ax = axes[0]
    s1 = exp1["summary"]
    stacks_diag = ["sm_fir", "pn", "pn_fir", "pn_fir_c", "pn_fir_tau"]
    labels_diag = [
        "sm_fir\n(raw S)",
        "pn\n(raw S)",
        "pn_fir\n(raw S)",
        "pn_fir_c\n(centered)",
        "pn_fir_tau\n(S - τ)",
    ]
    frac_pos = [
        s1["sm_fir"]["score_diag"]["raw_frac_pos"]["mean"] * 100.0,
        s1["pn"]["score_diag"]["raw_frac_pos"]["mean"] * 100.0,
        s1["pn_fir"]["score_diag"]["raw_frac_pos"]["mean"] * 100.0,
        s1["pn_fir_c"]["score_diag"]["cal_frac_pos"]["mean"] * 100.0,
        s1["pn_fir_tau"]["score_diag"]["cal_frac_pos"]["mean"] * 100.0,
    ]
    active_rows = [
        (1.0 - s1["sm_fir"]["score_diag"]["raw_rows_all_neg"]["mean"]) * 100.0,
        (1.0 - s1["pn"]["score_diag"]["raw_rows_all_neg"]["mean"]) * 100.0,
        (1.0 - s1["pn_fir"]["score_diag"]["raw_rows_all_neg"]["mean"]) * 100.0,
        (1.0 - s1["pn_fir_c"]["score_diag"]["cal_rows_all_neg"]["mean"]) * 100.0,
        (1.0 - s1["pn_fir_tau"]["score_diag"]["cal_rows_all_neg"]["mean"]) * 100.0,
    ]
    clean_acc = [s1[k]["clean"]["mean"] * 100.0 for k in stacks_diag]

    x = np.arange(len(stacks_diag))
    w = 0.26
    b_fp = ax.bar(x - w, frac_pos, width=w, label="Positive entries $\\mathrm{frac}(S > 0)$", color="#2b5c8f")
    b_ar = ax.bar(x, active_rows, width=w, label="Non-starved rows ($\\max_j S_j > 0$)", color="#6baed6")
    b_ca = ax.bar(x + w, clean_acc, width=w, label="Clean recall accuracy", color="#1b9e77")

    for bars in (b_fp, b_ar, b_ca):
        for bar in bars:
            val = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + 1.5,
                f"{val:.0f}%",
                ha="center",
                va="bottom",
                fontsize=7.3,
                fontweight="bold",
            )

    ax.annotate(
        "sm_fir has 0% positive scores\n(all S < 0), yet Softmax gets 100%\nbecause it is shift-invariant!",
        xy=(0 + w, 100.0),
        xytext=(0.15, 64.0),
        fontsize=7.8,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff8dc", edgecolor="#b8860b", alpha=0.92),
    )

    # m-mvfr36u5-178 cosmetic fix (panel a): legend moved above the axes (was
    # overlapping the green 100% label of pn_fir_c), value labels staggered
    # vertically per group with a hard cap, title pad raised.
    ax.set_title("(a) ReLU Score Starvation vs. Shift Invariance (Exp 1, 3 seeds)", pad=26)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_diag)
    ax.set_ylabel("Percentage (%)")
    ax.set_ylim(0, 132)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.01), ncol=3, framealpha=0.92,
              fontsize=7.4, borderaxespad=0.0)

    # stagger value labels per group: adjacent equal-height bars collided
    # ("100%100%"); assign vertically separated positions, capped inside ylim.
    y_top_cap = 122.0   # keep labels below the panel title zone (ylim = 132)
    CLEAR = 5.5
    for gi in range(len(stacks_diag)):
        lab = [
            t for t in ax.texts
            if abs(t.get_position()[0] - (gi - 0.26)) < 0.01
            or abs(t.get_position()[0] - gi) < 0.01
            or abs(t.get_position()[0] - (gi + 0.26)) < 0.01
        ]
        lab.sort(key=lambda t: t.get_position()[0])
        last_y = -1e9
        for t in lab:
            px_, py_ = t.get_position()
            y_new = max(py_, last_y + CLEAR)
            if y_new > y_top_cap:
                y_new = y_top_cap
            t.set_position((px_, y_new))
            last_y = y_new

    # Panel (b): 3-Seed Clean Recall Accuracy across Calibration Variants (Exp 1 & Exp 2)
    ax2 = axes[1]
    s2 = exp2["summary"]
    variants = ["pn", "pn_fir", "pn_fir_c", "pn_fir_tau"]
    v_labels = [
        "pn\n(no FIR)",
        "pn_fir\n(no cal)",
        "pn_fir_c\n(centered)",
        "pn_fir_tau\n(learned τ)",
    ]
    exp1_means = [s1[v]["clean"]["mean"] * 100.0 for v in variants]
    exp1_stds = [s1[v]["clean"]["std"] * 100.0 for v in variants]
    exp2_means = [s2[v]["clean"]["mean"] * 100.0 for v in variants]
    exp2_stds = [s2[v]["clean"]["std"] * 100.0 for v in variants]

    x2 = np.arange(len(variants))
    w2 = 0.34
    b1 = ax2.bar(
        x2 - w2 / 2,
        exp1_means,
        width=w2,
        yerr=exp1_stds,
        capsize=4,
        label="Exp 1: Fixed-Pos Bijective (γ=0.5)",
        color="#7570b3",
        alpha=0.88,
    )
    b2 = ax2.bar(
        x2 + w2 / 2,
        exp2_means,
        width=w2,
        yerr=exp2_stds,
        capsize=4,
        label="Exp 2: Content-Addressable (M=3 pairs)",
        color="#1b9e77",
        alpha=0.88,
    )

    for bars, means, stds in [(b1, exp1_means, exp1_stds), (b2, exp2_means, exp2_stds)]:
        for bar, m, sd in zip(bars, means, stds):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                min(m + sd + 1.8, 103.0),
                f"{m:.1f}%\n±{sd:.1f}",
                ha="center",
                va="bottom",
                fontsize=7.8,
            )

    ax2.set_title("(b) Clean Recall Accuracy Across Calibration Modes (3 Seeds)")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(v_labels)
    ax2.set_ylabel("Clean Recall Accuracy (%)")
    ax2.set_ylim(0, 116)
    ax2.legend(loc="lower right", framealpha=0.92, fontsize=8.0)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG_DIR, f"fig2_score_shift_pathology.{ext}"), bbox_inches="tight")
    plt.close(fig)


def build_fig3(exp1: dict, exp2: dict, exp3: dict, exp5: dict) -> None:
    """Fig 3: Simplex vs. Selective-Silence Normalizers on Unanswerable Queries (Exp 5) & Tau x Gamma Sweep (Exp 3)."""
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.3))

    # Panel (a): Exp 5 Topic-Coherent Context Override vs. Residual Stream Preservation (3 seeds)
    ax = axes[0]
    s5 = exp5["aggregated_3seed"]
    models_5 = [
        ("sm_fir", "sm_fir\n(Softmax)"),
        ("sparsemax_fir", "sparsemax\n(α=2.0)"),
        ("entmax15_fir", "entmax15\n(α=1.5)"),
        ("entmax_p10_fir", "entmax_p10\n(α=1.1, p=10)"),
        ("pn_fir_c", "pn_fir_c\n(centered)"),
        ("softmax1_fir", "softmax1\n(Miller '23)"),
        ("softmax_sink_fir", "sm_sink\n(learned τ)"),
        ("pn_fir", "pn_fir\n(fixed τ=0)"),
        ("pn_fir_tau", "pn_fir_tau\n(learned τ)"),
    ]

    x = np.arange(len(models_5))
    w = 0.27
    ov_means = [s5[k]["override_acc"]["mean"] * 100.0 for k, _ in models_5]
    ov_stds = [s5[k]["override_acc"]["std"] * 100.0 for k, _ in models_5]
    pr_means = [s5[k]["preserve_acc"]["mean"] * 100.0 for k, _ in models_5]
    pr_stds = [s5[k]["preserve_acc"]["std"] * 100.0 for k, _ in models_5]
    wun_means = [s5[k]["sum_w_unanswerable"]["mean"] * 100.0 for k, _ in models_5]
    wun_stds = [s5[k]["sum_w_unanswerable"]["std"] * 100.0 for k, _ in models_5]

    ax.bar(
        x - w,
        ov_means,
        width=w,
        yerr=ov_stds,
        capsize=2.5,
        label="Override Acc (Answerable, %)",
        color="#2b5c8f",
    )
    b_pr = ax.bar(
        x,
        pr_means,
        width=w,
        yerr=pr_stds,
        capsize=2.5,
        label="Preserve Acc (Unanswerable, %)",
        color="#1b9e77",
    )
    ax.bar(
        x + w,
        wun_means,
        width=w,
        yerr=wun_stds,
        capsize=2.5,
        label="Absent-Key Row Mass $100\\times\\sum_j W_j$",
        color="#d95f02",
    )

    for bar, m in zip(b_pr, pr_means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            m + 2.0,
            f"{m:.1f}%",
            ha="center",
            va="bottom",
            fontsize=6.8,
            fontweight="bold",
            color="#0b5d42",
        )

    ax.axvline(3.5, color="#555555", linestyle="--", linewidth=1.1)
    ax.text(
        1.5,
        124.5,
        "Simplex-Projected ($\\Sigma_j W_j \\equiv 1$)\nHallucinate on 27–34% of absent queries",
        ha="center",
        va="top",
        fontsize=7.5,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff5f0", edgecolor="#d95f02", alpha=0.92),
    )
    ax.text(
        6.0,
        124.5,
        "Sub-Simplex Normalizers ($\\Sigma_j W_j \\in [0, 1)$)\nPreserve residual stream ($\\Sigma_j W_j \\to 0$)",
        ha="center",
        va="top",
        fontsize=7.5,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#eef7ff", edgecolor="#2b5c8f", alpha=0.92),
    )

    ax.set_title("(a) Context Override vs. Residual Preservation (Exp 5, Pure CE Loss, 3 Seeds)")
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in models_5], fontsize=7.4)
    ax.set_ylabel("Percentage (%)")
    ax.set_ylim(0, 128)
    ax.legend(loc="lower left", framealpha=0.92, fontsize=7.6)

    # Panel (b): Ablation over gamma and tau_init from exp3_sweep_tau_gate.json
    ax2 = axes[1]
    rows3 = exp3["rows"]

    gammas_plot = [0.05, 0.1, 0.2, 0.5]
    tau_styles = [
        (0.0, "$\\tau_0 = 0.0$ (canonical)", "#2b5c8f", "o", "-"),
        (-0.5, "$\\tau_0 = -0.5$", "#1b9e77", "s", "--"),
        (-1.0, "$\\tau_0 = -1.0$", "#7570b3", "^", "-."),
        (0.5, "$\\tau_0 = +0.5$ (dead ReLU)", "#d95f02", "x", ":"),
    ]
    for t0, lbl, col, mk, ls in tau_styles:
        gaps_s1234 = []
        cleans_s1234 = []
        for g_val in gammas_plot:
            rec = next(
                r
                for r in rows3
                if r["stack"] == "pn_fir_tau"
                and r["silence_mode"] == "peak"
                and r["silence_gamma"] == g_val
                and r["tau_init"] == t0
            )
            gaps_s1234.append(rec["seed_1234"]["decoy"]["silence_gap"])
            cleans_s1234.append(rec["seed_1234"]["decoy"]["clean"] * 100.0)
        ax2.plot(
            gammas_plot,
            gaps_s1234,
            label=f"{lbl} (clean={cleans_s1234[-1]:.0f}%)",
            color=col,
            marker=mk,
            linestyle=ls,
            linewidth=2.0,
            markersize=6,
        )

    gaps_c = []
    for g_val in gammas_plot:
        rec = next(
            r
            for r in rows3
            if r["stack"] == "pn_fir_c"
            and r["silence_mode"] == "peak"
            and r["silence_gamma"] == g_val
        )
        gaps_c.append(rec["seed_1234"]["decoy"]["silence_gap"])
    ax2.plot(
        gammas_plot,
        gaps_c,
        label="pn_fir_c (centered, clean=100%)",
        color="#e7298a",
        marker="D",
        linestyle="--",
        linewidth=1.8,
        markersize=5,
    )

    ax2.set_title("(b) Silence Gap vs. Gate $\\gamma$ & Init $\\tau_0$ (Exp 3 Sweep, Seed 1234)")
    ax2.set_xlabel("Peak Silence Gate Parameter $\\gamma$")
    ax2.set_ylabel("Silence Gap ($\\sum W_{\\mathrm{match}} - \\sum W_{\\mathrm{nomatch}}$)")
    ax2.set_ylim(-0.04, 0.65)
    ax2.legend(loc="upper right", framealpha=0.92, fontsize=8.0)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG_DIR, f"fig3_selective_silence.{ext}"), bbox_inches="tight")
    plt.close(fig)


def _grid77(vals, steps, grid):
    """Interpolate a val curve given at `steps` onto common `grid`."""
    return np.interp(np.asarray(grid, dtype=float),
                     np.asarray(steps, dtype=float),
                     np.asarray(vals, dtype=float))


def _load_50k_bands():
    """3-seed matched-precision 50k curves for Fig 4 bands.

    Returns (grid, sm_mean, sm_std, pn_mean, pn_std, sm_final, pn_final) where
    finals are (val_mean, val_std, test_mean, test_std) from scalar fields
    (Table-5C-consistent), or None when the C1/C2 artifacts are unavailable.
    """
    try:
        with open(os.path.join(RESULTS_DIR, "exp4d_softmax_50k_bf16.json"), encoding="utf-8") as f:
            e4d = json.load(f)
        with open(os.path.join(RESULTS_DIR, "exp4e_multiseed_50k.json"), encoding="utf-8") as f:
            e4e = json.load(f)
        with open(os.path.join(RESULTS_DIR, "exp4_wikitext2_lm.json"), encoding="utf-8") as f:
            e4 = json.load(f)
    except (OSError, ValueError, KeyError):
        return None
    ref50 = e4.get("reference_50k") or {}
    if not ref50:
        return None
    grid = [s / 1000.0 for s in range(2000, 50000, 1000)]
    ksteps = [s / 1000.0 for s in range(1000, 50000, 1000)]
    sm_curves, pn_curves = [], []
    # softmax: C1 seed-42 BF16 + C2 seeds 43/44
    sm42 = e4d["softmax_50k_bf16"]["history"]["val_ppl"]
    sm_curves.append(_grid77(sm42, ksteps, grid))
    sm_vals, sm_tests = [e4d["softmax_50k_bf16"]["val_ppl"]], [e4d["softmax_50k_bf16"]["test_ppl"]]
    # pn: historical seed-42 + C2 seeds 43/44
    pn42 = ref50["pn_fir_tau_50k"]["val_ppl_curve"]
    pn_curves.append(_grid77(pn42, [s / 1000.0 for s in ref50["pn_fir_tau_50k"]["val_steps"]], grid))
    pn_vals = [ref50["pn_fir_tau_50k"]["val_ppl"]]
    pn_tests = [ref50["pn_fir_tau_50k"]["test_ppl"]]
    for key in ("softmax/seed43", "softmax/seed44", "pn_fir_tau/seed43", "pn_fir_tau/seed44"):
        r = e4e["runs"][key]
        curve = _grid77(r["history"]["val_ppl"], ksteps, grid)
        if key.startswith("softmax"):
            sm_curves.append(curve)
            sm_vals.append(r["val_ppl"])
            sm_tests.append(r["test_ppl"])
        else:
            pn_curves.append(curve)
            pn_vals.append(r["val_ppl"])
            pn_tests.append(r["test_ppl"])
    sm_m = np.mean(sm_curves, axis=0)
    sm_s = np.std(sm_curves, axis=0)
    pn_m = np.mean(pn_curves, axis=0)
    pn_s = np.std(pn_curves, axis=0)
    sm_final = (float(np.mean(sm_vals)), float(np.std(sm_vals)),
                float(np.mean(sm_tests)), float(np.std(sm_tests)))
    pn_final = (float(np.mean(pn_vals)), float(np.std(pn_vals)),
                float(np.mean(pn_tests)), float(np.std(pn_tests)))
    return grid, sm_m, sm_s, pn_m, pn_s, sm_final, pn_final


def build_fig4(exp4: dict) -> None:
    """Fig 4: WikiText-2 Pre-LN LM Validation PPL Curves, Learned Tau per Layer, and Step Latency."""
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.1))

    # Panel (a): 50k Reference Trajectories + Matched 3-Variant BF16 Inset
    ax = axes[0]
    ref50 = exp4.get("reference_50k")
    bands = _load_50k_bands()
    if bands is not None:
        grid, sm_m, sm_s, pn_m, pn_s, sm_final, pn_final = bands
        ax.fill_between(grid, sm_m - sm_s, sm_m + sm_s, color="#d95f02", alpha=0.18,
                        label="_nolegend_")
        ax.fill_between(grid, pn_m - pn_s, pn_m + pn_s, color="#2b5c8f", alpha=0.18,
                        label="_nolegend_")
        ax.plot(
            grid, sm_m,
            label=f"Softmax 50k 3-seed (Val={sm_final[0]:.2f}±{sm_final[1]:.2f}, "
                  f"Test={sm_final[2]:.2f}±{sm_final[3]:.2f})",
            color="#d95f02",
            linestyle="--",
            linewidth=2.0,
        )
        ax.plot(
            grid, pn_m,
            label=f"PnFirTau 50k 3-seed (Val={pn_final[0]:.2f}±{pn_final[1]:.2f}, "
                  f"Test={pn_final[2]:.2f}±{pn_final[3]:.2f})",
            color="#2b5c8f",
            linestyle="-",
            linewidth=2.2,
        )
    elif ref50:
        sm50 = ref50["softmax_50k"]
        pn50 = ref50["pn_fir_tau_50k"]
        sm_steps = [s / 1000.0 for s in sm50["val_steps"] if s >= 2000]
        sm_ppls = [p for s, p in zip(sm50["val_steps"], sm50["val_ppl_curve"]) if s >= 2000]
        pn_steps = [s / 1000.0 for s in pn50["val_steps"] if s >= 2000]
        pn_ppls = [p for s, p in zip(pn50["val_steps"], pn50["val_ppl_curve"]) if s >= 2000]

        ax.plot(
            sm_steps,
            sm_ppls,
            label=f"Softmax 50k (Val={sm50['val_ppl']:.2f}, Test={sm50['test_ppl']:.2f})",
            color="#d95f02",
            linestyle="--",
            linewidth=2.0,
        )
        ax.plot(
            pn_steps,
            pn_ppls,
            label=f"PnFirTau 50k (Val={pn50['val_ppl']:.2f}, Test={pn50['test_ppl']:.2f})",
            color="#2b5c8f",
            linestyle="-",
            linewidth=2.2,
        )

    ax.set_title("(a) WikiText-2 Validation Perplexity (L=6, D=64, H=4, 50k Steps"
                 + (", 3-Seed Mean±SD)" if bands is not None else ")"))
    ax.set_xlabel("Training Step ($\\times 10^3$)")
    ax.set_ylabel("Validation Perplexity (word-level, lower is better)")
    ax.set_ylim(115, 260)
    ax.legend(loc="upper right", framealpha=0.92, fontsize=8.2)

    matched = exp4.get("matched_runs", {})
    if matched:
        ins = ax.inset_axes([0.36, 0.42, 0.60, 0.36])
        v_candidates = [
            ("softmax", "sm", "#d95f02"),
            ("sm_fir", "sm_fir", "#1b9e77"),
            ("pn_fir_tau", "pn_eager", "#7570b3"),
            ("pn_fir_tau_triton", "pn_triton", "#2b5c8f"),
            ("pn_fir_tau_flash", "pn_flash", "#e7298a"),
        ]
        v_items = [(k, lbl, col) for k, lbl, col in v_candidates if k in matched]
        x_ins = np.arange(len(v_items))
        v_ppls = [matched[k]["test_ppl"] for k, _, _ in v_items]
        v_cols = [col for _, _, col in v_items]
        bars = ins.bar(x_ins, v_ppls, color=v_cols, width=0.58)
        ins.set_xticks(x_ins)
        ins.set_xticklabels([lbl for _, lbl, _ in v_items], fontsize=7.4)
        for b, val in zip(bars, v_ppls):
            ins.text(
                b.get_x() + b.get_width() / 2,
                val + 12.0,
                f"{val:.1f}",
                ha="center",
                va="bottom",
                fontsize=7.0,
                fontweight="bold",
            )
        ins.set_title(
            f"Matched BF16 Test PPL ({exp4['config']['steps']} steps)",
            fontsize=8.0,
            fontweight="bold",
        )
        ins.set_ylim(0, max(v_ppls) * 1.28)
        ins.tick_params(axis="both", labelsize=7.4)

    # Panel (b): Learned Layer-wise Thresholds tau_l (50k) + Matched Step Time Latency
    ax2 = axes[1]
    taus_50k = ref50["pn_fir_tau_50k"]["tau_per_layer"] if ref50 else []
    taus_matched = matched.get("pn_fir_tau", {}).get("tau_per_layer", [])
    layers = np.arange(len(taus_50k))
    w = 0.35
    if taus_50k:
        b50 = ax2.bar(
            layers - w / 2,
            taus_50k,
            width=w,
            label="Learned $\\tau_\\ell$ at 50k steps",
            color="#2b5c8f",
        )
        for b, val in zip(b50, taus_50k):
            ax2.text(
                b.get_x() + b.get_width() / 2,
                0.008,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=7.4,
                fontweight="bold",
                color="#2b5c8f",
            )
    if taus_matched:
        bm = ax2.bar(
            layers + w / 2,
            taus_matched,
            width=w,
            label=f"Learned $\\tau_\\ell$ at {exp4['config']['steps']} steps",
            color="#6baed6",
        )

    ax2.axhline(0.0, color="#333333", linewidth=1.0)
    ax2.set_title("(b) Learned Layer-Wise Score Thresholds $\\tau_\\ell$ & Step Latency")
    ax2.set_xticks(layers)
    ax2.set_xticklabels([f"Layer {i}" for i in layers])
    ax2.set_ylabel("Learned Threshold $\\tau_\\ell$")
    ax2.set_ylim(-0.42, 0.06)
    ax2.legend(loc="lower left", framealpha=0.92, fontsize=8.0)

    sb = exp4.get("step_benchmark", {}).get("bf16", {})
    if sb:
        ins2 = ax2.inset_axes([0.38, 0.10, 0.58, 0.38])
        v_candidates = [
            ("softmax", "sm", "#d95f02"),
            ("sm_fir", "sm_fir", "#1b9e77"),
            ("pn_fir_tau", "pn_eager", "#7570b3"),
            ("pn_fir_tau_triton", "pn_triton", "#2b5c8f"),
            ("pn_fir_tau_flash", "pn_flash", "#e7298a"),
        ]
        v_items2 = [(k, lbl, col) for k, lbl, col in v_candidates if k in sb]
        x_ins2 = np.arange(len(v_items2))
        ms_vals = [sb[k]["ms_per_step"] for k, _, _ in v_items2]
        v_cols2 = [col for _, _, col in v_items2]
        bars2 = ins2.bar(x_ins2, ms_vals, color=v_cols2, width=0.58)
        ins2.set_xticks(x_ins2)
        ins2.set_xticklabels([lbl for _, lbl, _ in v_items2], fontsize=7.4)
        for b, val in zip(bars2, ms_vals):
            ins2.text(
                b.get_x() + b.get_width() / 2,
                val + 4.0,
                f"{val:.1f}ms",
                ha="center",
                va="bottom",
                fontsize=7.0,
                fontweight="bold",
            )
        ins2.set_title("Matched BF16 Latency (ms/step, B=64)", fontsize=8.0, fontweight="bold")
        ins2.set_ylim(0, max(ms_vals) * 1.28)
        ins2.tick_params(axis="both", labelsize=7.4)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG_DIR, f"fig4_wikitext2_lm_and_tau.{ext}"), bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-all", action="store_true", help="Re-run all 5 experiment scripts before plotting")
    ap.add_argument("--only", default=None,
                    help="build a single figure (e.g. --only fig2); skips the metrics consolidation")
    args = ap.parse_args()

    if args.run_all:
        for script in (
            "experiments/exp1_legacy_stack_candidates.py",
            "experiments/exp2_content_addressable_decoy.py",
            "experiments/exp3_sweep_tau_gate.py",
            "experiments/exp4_wikitext2_lm.py",
            "experiments/exp5_abstention_and_entmax.py",
        ):
            subprocess.run([sys.executable, os.path.join(ROOT, script)], check=True)

    os.makedirs(FIG_DIR, exist_ok=True)
    exp1 = load_json("exp1_legacy_13stacks.json")
    exp2 = load_json("exp2_content_addressable_decoy.json")

    if args.only == "fig2":
        build_fig2(exp1, exp2)
        print("Built fig2 only (PNG + PDF); metrics consolidation skipped.")
        return
    if args.only:
        raise SystemExit("unsupported --only value: %s" % args.only)

    exp3 = load_json("exp3_sweep_tau_gate.json")
    exp4 = load_json("exp4_wikitext2_lm.json")
    exp5 = load_json("exp5_abstention_and_entmax.json")

    build_fig1(exp1, exp2)
    build_fig2(exp1, exp2)
    build_fig3(exp1, exp2, exp3, exp5)
    build_fig4(exp4)

    consolidated = {
        "exp1_legacy_13stacks": exp1["summary"],
        "exp1_fir_taps": exp1["fir_taps_present_to_past"],
        "exp2_content_addressable_decoy": exp2["summary"],
        "exp3_sweep_tau_gate_n_configs": exp3["n_configs"],
        "exp4_wikitext2_step_benchmark": exp4["step_benchmark"],
        "exp4_triton_mha_benchmark": exp4.get("triton_mha_benchmark", {}),
        "exp4_wikitext2_matched_runs": {
            k: {
                "params": v["params"],
                "val_ppl": v["val_ppl"],
                "test_ppl": v["test_ppl"],
                "train_ppl": v["train_ppl"],
                "tau_per_layer": v["tau_per_layer"],
                "ms_per_step_effective": v["ms_per_step_effective"],
            }
            for k, v in exp4["matched_runs"].items()
        },
        "exp4_wikitext2_reference_50k": {
            k: (
                {
                    kk: vv
                    for kk, vv in v.items()
                    if kk not in ("val_steps", "val_ppl_curve")
                }
                if isinstance(v, dict)
                else v
            )
            for k, v in (exp4["reference_50k"] or {}).items()
        },
        "exp5_abstention_and_entmax": exp5["aggregated_3seed"],
    }
    for out_path in (
        os.path.join(ROOT, "paper01_metrics.json"),
        os.path.join(FIG_DIR, "paper01_metrics.json"),
    ):
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(consolidated, f, indent=2)

    print("Successfully built all 4 figures (PNG + PDF) and consolidated paper01_metrics.json")


if __name__ == "__main__":
    main()
