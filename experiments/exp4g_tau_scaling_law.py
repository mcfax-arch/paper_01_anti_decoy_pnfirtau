#!/usr/bin/env python3
"""Experiment 4g: C3c tau scaling law (PAPER_01_REWORK_PLAN.md).

Question: can the width-dependence of the score-shift pathology be turned
into a predictive law for tau, via attention-row entropy?

Ladder: widths d in {16, 24, 32, 48, 64, 96, 128}, exp1 protocol, arms
sm_fir / pn_fir / pn_fir_tau (3 seeds each). Per cell records:
  - raw score stats on match query rows: mu_S, sigma_S, frac_pos, deadrows
  - attention-row entropy H on match vs noise rows (renormalized rows;
    all-zero rows count H=0 + silent fraction): H_match, H_noise,
    gap = H_noise - H_match  (the order parameter "answer vs noise")
  - converged tau* (score_tau) for the tau arm

Fit (on widths where pn_fir_tau converges, clean mean >= 0.99):
  tau*(d) = a * (-mu_S^pnfir(d)) + b
i.e. the threshold cancels the measured uncalibrated baseline drift.
Validation: d=128 pn_fir_tau with tau_init = predicted value (3 seeds).
If it rescues d=128, the law is confirmed functionally, not correlationally.

Decision rule (for the manuscript):
  - H-gap closes with d  -> fundamental wall (no (tau, gamma) works);
  - H-gap persists + tau stalls -> optimization/recipe (this script's bet).

Idempotent: artifact accumulates per (d, arm, seed); reruns skip completed
cells. Stages: --stage ladder | validate | all.
Artifact: results/exp4g_tau_scaling_law.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
import exp1_legacy_stack_candidates as e1  # noqa: E402

DEVICE = e1.DEVICE
RESULTS_DIR = os.path.join(ROOT, "results")
OUT_JSON = os.path.join(RESULTS_DIR, "exp4g_tau_scaling_law.json")

WIDTHS = (16, 24, 32, 48, 64, 96, 128)
ARMS = ("sm_fir", "pn_fir", "pn_fir_tau")
SEEDS = (1234, 2025, 3141)
EVAL_N = 2000


def ms_of(vals) -> dict:
    arr = np.array(list(vals), dtype=np.float64)
    return {"mean": float(arr.mean()), "std": float(arr.std()),
            "per_seed": [float(v) for v in vals]}


@torch.no_grad()
def entropy_diag(model, cfg: e1.Cfg, seed: int) -> dict:
    """Row-entropy diagnostics on match vs noise contexts (same distribution
    as e1.eval_decoy: fresh generator, identical batch shapes)."""
    model.eval()
    g = torch.Generator().manual_seed(seed + 99)
    ctx, _ = e1.gen(cfg, EVAL_N, g)
    ctx = ctx.to(DEVICE)
    xe = model.emb(ctx)
    _, w, s_raw, _ = model(xe, return_scores=True)
    wq = w[:, -1, :]
    sq = s_raw[:, -1, :]

    def row_stats(ww: torch.Tensor) -> dict:
        mass = ww.sum(dim=-1)
        what = ww / mass.clamp_min(1e-12).unsqueeze(-1)
        ent = -(what * what.clamp_min(1e-12).log()).sum(dim=-1)
        silent = mass <= 1e-6
        ent = torch.where(silent, torch.zeros_like(ent), ent)
        return {
            "H_mean": float(ent.mean().item()),
            "H_std": float(ent.std().item()),
            "silent_frac": float(silent.float().mean().item()),
            "mass_mean": float(mass.mean().item()),
        }

    m = row_stats(wq)
    xn = torch.randint(11, cfg.vocab, (EVAL_N, e1.CTX_LEN), generator=g).to(DEVICE)
    w_n = model(model.emb(xn))[1][:, -1, :]
    n = row_stats(w_n)
    return {
        "raw_mu": float(sq.float().mean().item()),
        "raw_sigma": float(sq.float().std().item()),
        "H_match": m["H_mean"],
        "H_match_std": m["H_std"],
        "H_noise": n["H_mean"],
        "H_noise_std": n["H_std"],
        "H_gap": float(n["H_mean"] - m["H_mean"]),
        "silent_frac_match": m["silent_frac"],
        "silent_frac_noise": n["silent_frac"],
        "mass_match": m["mass_mean"],
        "mass_noise": n["mass_mean"],
    }


def load_payload() -> dict:
    if os.path.isfile(OUT_JSON):
        with open(OUT_JSON, encoding="utf-8") as f:
            return json.load(f)
    return {"config": {}, "ladder": {}, "fit": None, "validation": {}}


def run_ladder() -> dict:
    payload = load_payload()
    payload["config"] = {
        "widths": list(WIDTHS), "arms": list(ARMS),
        "seeds": list(SEEDS), "eval_n": EVAL_N,
        "note": "exp1 protocol verbatim; entropy_diag replicates its batch distribution",
    }
    for d in WIDTHS:
        payload["ladder"].setdefault(str(d), {})
        for arm in ARMS:
            payload["ladder"][str(d)].setdefault(arm, {})
            for seed in SEEDS:
                key = str(seed)
                if key in payload["ladder"][str(d)][arm]:
                    continue
                cfg = e1.Cfg(d=d)
                e1.set_seed(seed)
                gg = torch.Generator().manual_seed(seed)
                model = e1.StackModel(arm, cfg).to(DEVICE)
                e1.train_model(model, cfg, gg)
                res = e1.eval_decoy(model, cfg, seed)
                diag = entropy_diag(model, cfg, seed)
                sd = res["score_diag"]
                payload["ladder"][str(d)][arm][key] = {
                    "clean": res["clean"],
                    "d@1.0": res["d@1.0"],
                    "raw_frac_pos": sd["raw_frac_pos"],
                    "raw_rows_all_neg": sd["raw_rows_all_neg"],
                    "tau": res["tau"],
                    **diag,
                }
                with open(OUT_JSON, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2)
                c = payload["ladder"][str(d)][arm][key]
                print(
                    f"  d={d:3d} {arm:12s} s={seed} clean={c['clean']*100:5.1f}% "
                    f"mu={c['raw_mu']:+.3f} sig={c['raw_sigma']:.3f} "
                    f"Hm={c['H_match']:.3f} Hn={c['H_noise']:.3f} gap={c['H_gap']:+.3f}"
                    + (f" tau={c['tau']:+.3f}" if c["tau"] is not None else ""),
                    flush=True,
                )
                del model
    return payload


def fit_law(payload: dict) -> dict:
    xs, ys, ds = [], [], []
    for d in WIDTHS:
        cell_t = payload["ladder"][str(d)].get("pn_fir_tau", {})
        cell_f = payload["ladder"][str(d)].get("pn_fir", {})
        if len(cell_t) < len(SEEDS) or len(cell_f) < len(SEEDS):
            continue
        clean = float(np.mean([cell_t[str(s)]["clean"] for s in SEEDS]))
        if clean < 0.99:
            continue
        mu = float(np.mean([cell_f[str(s)]["raw_mu"] for s in SEEDS]))
        tau = float(np.mean([cell_t[str(s)]["tau"] for s in SEEDS]))
        xs.append(-mu)
        ys.append(tau)
        ds.append(d)
    if len(xs) >= 2:
        a, b = np.polyfit(xs, ys, 1)
        pred = np.polyval([a, b], xs)
        ss_res = float(np.sum((np.array(ys) - pred) ** 2))
        ss_tot = float(np.sum((np.array(ys) - np.mean(ys)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    else:
        a, b, r2 = 0.0, 0.0, 0.0
    fit = {"widths_used": ds, "slope": float(a), "intercept": float(b), "r2": float(r2),
           "law": "tau*(d) = a * (-mu_S_pnfir(d)) + b"}
    payload["fit"] = fit
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"FIT on widths {ds}: tau* = {a:+.4f} * (-mu_S) {b:+.4f}, R2={r2:.4f}", flush=True)
    return fit


def run_validation(fit: dict) -> dict:
    payload = load_payload()
    mu128 = float(np.mean([payload["ladder"]["128"]["pn_fir"][str(s)]["raw_mu"] for s in SEEDS]))
    pred = fit["slope"] * (-mu128) + fit["intercept"]
    print(f"mu_S_pnfir(128) = {mu128:+.4f} -> predicted tau_init = {pred:+.4f}", flush=True)
    payload["validation"] = {"predicted_tau_init": float(pred), "mu128": float(mu128), "runs": {}}
    for seed in SEEDS:
        key = str(seed)
        if key in payload["validation"]["runs"]:
            continue
        cfg = e1.Cfg(d=128, tau_init=float(pred))
        e1.set_seed(seed)
        gg = torch.Generator().manual_seed(seed)
        model = e1.StackModel("pn_fir_tau", cfg).to(DEVICE)
        e1.train_model(model, cfg, gg)
        res = e1.eval_decoy(model, cfg, seed)
        diag = entropy_diag(model, cfg, seed)
        sd = res["score_diag"]
        payload["validation"]["runs"][key] = {
            "clean": res["clean"], "d@1.0": res["d@1.0"],
            "raw_frac_pos": sd["raw_frac_pos"],
            "raw_rows_all_neg": sd["raw_rows_all_neg"],
            "tau": res["tau"], **diag,
        }
        with open(OUT_JSON, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        c = payload["validation"]["runs"][key]
        print(f"  VALID d=128 s={seed} clean={c['clean']*100:5.1f}% tau={c['tau']:+.3f} "
              f"H_gap={c['H_gap']:+.3f}", flush=True)
        del model
    return payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["ladder", "validate", "all"], default="all")
    args = ap.parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    t0 = time.time()
    if args.stage in ("ladder", "all"):
        print("=== C3c ladder ===", flush=True)
        payload = run_ladder()
    else:
        payload = load_payload()
    if args.stage in ("validate", "all"):
        print("=== C3c fit + validation ===", flush=True)
        fit = fit_law(payload)
        run_validation(fit)
    print(f"saved -> {OUT_JSON} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
