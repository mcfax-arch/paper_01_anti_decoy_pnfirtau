#!/usr/bin/env python3
"""Experiment 3: Self-Contained 31-Point Ablation Sweep of (tau_init x silence_gamma x silence_mode).

Consolidates `I:\\FFT_Attention\\experiments\\stack_candidates\\sweep_tau_gate.py` without
external path dependencies:
  1. Evaluates all 31 configurations on canonical seed 1234 (reproducing `SWEEP_TAU_GATE.md`).
  2. Evaluates all 12 core ablation configurations across all 3 seeds (1234, 2025, 3141)
     to report 3-seed mean ± std in Table 2 and Figure 3.
"""
from __future__ import annotations

import itertools
import json
import os
import sys
import time
from copy import deepcopy
from dataclasses import asdict

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from experiments.exp1_legacy_stack_candidates import (  # noqa: E402
    Cfg,
    DEVICE,
    StackModel,
    eval_decoy,
    set_seed,
    train_model,
)

RESULTS_DIR = os.path.join(ROOT, "results")
OUT_JSON = os.path.join(RESULTS_DIR, "exp3_sweep_tau_gate.json")


def run_one_seed(stack: str, cfg: Cfg, seed: int) -> dict:
    set_seed(seed)
    g = torch.Generator().manual_seed(seed)
    model = StackModel(stack, cfg).to(DEVICE)
    t0 = time.time()
    train_s = train_model(model, cfg, g)
    decoy = eval_decoy(model, cfg, seed)
    clean = decoy["clean"]
    sil_nm = 1.0 - min(decoy["sum_w_nomatch"], 1.0)
    sil_noise = 1.0 - min(decoy["sum_w_noise"], 1.0)
    decoy_ok = 1.0 - min(abs(decoy["delta"]), 1.0)
    score = clean * 2.0 + (sil_nm + sil_noise) * clean + decoy_ok * clean
    return {
        "train_s": float(train_s),
        "wall_s": float(time.time() - t0),
        "decoy": decoy,
        "score": float(score),
    }


def is_core_3seed_config(stack: str, g: float | None, mode: str, t0: float) -> bool:
    """Selects the 12 core configurations for full 3-seed evaluation."""
    if stack == "pn_fir_tau" and mode == "peak" and t0 == 0.0:
        return True  # all 5 gammas: None, 0.05, 0.1, 0.2, 0.5
    if stack == "pn_fir_tau" and mode == "peak" and g == 0.5:
        return True  # all tau_inits: 0.0, -0.5, -1.0, +0.5
    if stack == "pn_fir_tau" and mode == "mass_relu" and t0 == 0.0 and g in (0.5, 1.0):
        return True
    if stack == "pn_fir_c" and mode == "peak" and t0 == 0.0 and g in (None, 0.5):
        return True
    return False


def main() -> dict:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    base = Cfg()

    stacks = ["pn_fir_tau", "pn_fir_c"]
    gammas = [None, 0.05, 0.1, 0.2, 0.5]
    modes = ["peak"]
    tau_inits = [0.0, -0.5, -1.0, 0.5]

    jobs = []
    for stack, g, mode, t0 in itertools.product(stacks, gammas, modes, tau_inits):
        if stack.endswith("_c") and t0 != 0.0:
            continue
        jobs.append((stack, g, mode, t0))
    for g in [0.1, 0.5, 1.0]:
        jobs.append(("pn_fir_tau", g, "mass_relu", 0.0))
        jobs.append(("pn_fir_tau", g, "mass_relu", -0.5))

    print("=" * 92)
    print(f"EXP 3: 31-Configuration Sweep (tau_init x silence_gamma x mode) | jobs={len(jobs)}")
    print("=" * 92)

    rows = []
    for i, (stack, g, mode, t0) in enumerate(jobs, 1):
        cfg = deepcopy(base)
        cfg.silence_gamma = g
        cfg.silence_mode = mode
        cfg.tau_init = t0

        seeds_to_run = base.seeds if is_core_3seed_config(stack, g, mode, t0) else (1234,)
        per_seed = {}
        for seed in seeds_to_run:
            per_seed[str(seed)] = run_one_seed(stack, cfg, seed)

        r1234 = per_seed["1234"]
        d1234 = r1234["decoy"]
        mean_clean = float(np.mean([per_seed[str(s)]["decoy"]["clean"] for s in seeds_to_run]))
        mean_sw_m = float(np.mean([per_seed[str(s)]["decoy"]["sum_w_match"] for s in seeds_to_run]))
        mean_sw_nm = float(np.mean([per_seed[str(s)]["decoy"]["sum_w_nomatch"] for s in seeds_to_run]))
        mean_sw_n = float(np.mean([per_seed[str(s)]["decoy"]["sum_w_noise"] for s in seeds_to_run]))
        mean_gap = float(np.mean([per_seed[str(s)]["decoy"]["silence_gap"] for s in seeds_to_run]))
        std_gap = float(np.std([per_seed[str(s)]["decoy"]["silence_gap"] for s in seeds_to_run]))
        mean_score = float(np.mean([per_seed[str(s)]["score"] for s in seeds_to_run]))

        rec = {
            "stack": stack,
            "tau_init": t0,
            "silence_gamma": g,
            "silence_mode": mode,
            "power_p": cfg.power_p,
            "n_seeds": len(seeds_to_run),
            "seed_1234": r1234,
            "mean_seeds": {
                "clean": mean_clean,
                "sum_w_match": mean_sw_m,
                "sum_w_nomatch": mean_sw_nm,
                "sum_w_noise": mean_sw_n,
                "silence_gap": mean_gap,
                "silence_gap_std": std_gap,
                "score": mean_score,
            },
            "per_seed": per_seed,
        }
        rows.append(rec)
        print(
            f"[{i:2d}/{len(jobs)}] {stack:10s} gamma={str(g):4s} mode={mode:9s} tau0={t0:+4.1f} "
            f"(seeds={len(seeds_to_run)}) | s1234: clean={d1234['clean']:.3f} "
            f"Sw_m/nm/n={d1234['sum_w_match']:.3f}/{d1234['sum_w_nomatch']:.3f}/{d1234['sum_w_noise']:.3f} "
            f"gap={d1234['silence_gap']:+.3f} | mean_gap={mean_gap:+.3f}±{std_gap:.3f}",
            flush=True,
        )

    payload = {
        "base": asdict(base),
        "n_configs": len(rows),
        "rows": rows,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved {OUT_JSON}")
    return payload


if __name__ == "__main__":
    main()
