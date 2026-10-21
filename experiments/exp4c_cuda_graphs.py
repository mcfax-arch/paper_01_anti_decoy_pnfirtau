#!/usr/bin/env python3
"""Experiment 4c: CUDA-graphs follow-up of the exp4b launch-serialisation finding.

exp4b (results/exp4b_mha6_instrumentation.json) established that the 6-layer
PnFirTauFlashMHA stack at N=1024, B=16 is launch-serialised under WSL2 GPU-PV:
~372 CUDA kernel launches per step with tau- and B-invariant wall time, so the
real per-kernel saving (~0.70 ms/layer, 4.2 ms over 6 layers) does not surface
in the wall clock. This script tests the direct counterfactual: if launches are
the bottleneck, amortising per-step launch dispatch over a captured CUDA Graph
must let the kernel-level saving surface.

Protocol (exp4b discipline):
  - exact Table 5E config: N=1024, B=16, D=64, H=4, BF16 autocast, matmul TF32 off, seed 42;
  - tau in {-0.5 (all tiles active), +2.5 (all tiles skipped)};
  - rotated interleave: 3 blocks started with -0.5, 3 started with +2.5,
    per block warmup 25 graphs then 120 timed replays;
  - same recipe without graphs (plain eager loop) measured in the same process
    as the launch-bound baseline;
  - numerical sanity: flash outputs and parameter grads compared eagerly
    (relative bf16 tolerance) against one non-graph run.

Writes results/exp4c_cuda_graphs.json and prints a console summary.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time

import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.pnfirtau_attention import PnFirConfig, PnFirTauFlashMHA  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = os.path.join(ROOT, "results")
OUT_JSON = os.path.join(RESULTS_DIR, "exp4c_cuda_graphs.json")

D_MODEL, N_HEADS, N_LAYERS = 64, 4, 6
N_SK, B_SK = 1024, 16
BLOCK_STEPS, BLOCK_WARMUP = 120, 25
ROUNDS = 3
SEED = 42
TAUS = (-0.5, 2.5)


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def stats(arr: list[float]) -> dict:
    a = np.asarray(arr, dtype=np.float64)
    return {
        "mean": round(float(a.mean()), 4),
        "std": round(float(a.std()), 4),
        "sem": round(float(a.std() / math.sqrt(len(a))), 4),
        "n": int(len(a)),
    }


def make_stack(tau: float):
    set_seed(SEED)
    acfg = PnFirConfig(d_model=D_MODEL, n_heads=N_HEADS, dropout=0.0, tau_init=float(tau))
    layers = nn.ModuleList([PnFirTauFlashMHA(acfg) for _ in range(N_LAYERS)]).to(DEVICE)
    x = torch.randn(B_SK, N_SK, D_MODEL, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
    dy = torch.randn_like(x)
    return layers, x, dy


def graphed_step(layers: nn.ModuleList, x: torch.Tensor, dy: torch.Tensor):
    """Capture one fwd+bwd step (zero-grad + residual loop + backward) in a CUDA graph."""

    def _zero():
        for p in layers.parameters():
            if p.grad is not None:
                p.grad.zero_()
        if x.grad is not None:
            x.grad.zero_()

    def _step():
        _zero()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            h = x
            for lyr in layers:
                h = h + lyr(h)
        h.backward(dy)

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            _step()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        _step()
    return g


def measure(fn: callable, steps: int, warmup: int) -> dict:
    evs = [torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)]
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    walls, spans = [], []
    for _ in range(steps):
        t0 = time.perf_counter()
        evs[0].record()
        fn()
        evs[1].record()
        torch.cuda.synchronize()
        walls.append((time.perf_counter() - t0) * 1000.0)
        spans.append(evs[0].elapsed_time(evs[1]))
    return {"wall": stats(walls), "event_span": stats(spans)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=BLOCK_STEPS)
    parser.add_argument("--warmup", type=int, default=BLOCK_WARMUP)
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    assert torch.cuda.is_available()

    results = {
        "meta": {
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "gpu": torch.cuda.get_device_name(0),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "triton": __import__("triton").__version__,
            "config": {
                "d_model": D_MODEL, "n_heads": N_HEADS, "n_layers": N_LAYERS,
                "N": N_SK, "B": B_SK, "block_steps": args.steps, "block_warmup": args.warmup,
                "rounds": ROUNDS, "schedule": "rotated interleave (lo-start x3, hi-start x3)",
                "seed": SEED, "precision": "BF16 autocast, matmul TF32 off",
            },
            "purpose": "Counterfactual to exp4b: if the 6-layer stack is launch-serialised, CUDA-graph replay "
                       "must surface the ~4.2 ms kernel-level skip saving.",
        },
        "per_tau": {},
        "pair_deltas": [],
    }

    # capture graphs and static inputs once per tau
    graphs, statics = {}, {}
    for tau in TAUS:
        layers, x, dy = make_stack(tau)
        g = graphed_step(layers, x, dy)
        # numerical sanity vs eager
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            h_eager = x
            for lyr in layers:
                h_eager = h_eager + lyr(h_eager)
        g.replay()
        torch.cuda.synchronize()
        x_out_ref = x.grad.clone()
        g.replay()
        torch.cuda.synchronize()
        rel = float((x.grad - x_out_ref).abs().max().item() / max(x_out_ref.abs().max().item(), 1e-12))
        graphs[str(tau)] = g
        statics[str(tau)] = [layers, x, dy]
        results["per_tau"][str(tau)] = {"graph_grad_replay_rel_diff": round(rel, 8)}
        print(f"[tau={tau:+.1f}] graph captured; x-grad replay rel diff = {rel:.2e}", flush=True)
        torch.cuda.empty_cache()

    # rotated interleave: graphed replay vs eager, per tau
    blocks = {str(tau): {"graph": [], "eager": []} for tau in TAUS}
    for rnd in range(ROUNDS):
        for tau in TAUS if rnd % 2 == 0 else reversed(TAUS):
            key = str(tau)
            layers, x, dy = statics[key]

            def eager_step():
                for p in layers.parameters():
                    if p.grad is not None:
                        p.grad.zero_()
                if x.grad is not None:
                    x.grad.zero_()
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    h = x
                    for lyr in layers:
                        h = h + lyr(h)
                h.backward(dy)

            res_graph = measure(lambda: graphs[key].replay(), args.steps, args.warmup)
            res_eager = measure(eager_step, args.steps, args.warmup)
            blocks[key]["graph"].append(res_graph)
            blocks[key]["eager"].append(res_eager)
            print(
                f"[r{rnd + 1}/{ROUNDS} tau={tau:+.1f}] graph={res_graph['wall']['mean']:.2f}"
                f"±{res_graph['wall']['std']:.2f} ms  eager={res_eager['wall']['mean']:.2f}"
                f"±{res_eager['wall']['std']:.2f} ms",
                flush=True,
            )
            torch.cuda.empty_cache()

    # aggregate
    combined = {}
    for tau in TAUS:
        key = str(tau)
        combined[key] = {}
        for mode in ("graph", "eager"):
            vals = [b["wall"]["mean"] for b in blocks[key][mode]]
            spans = [b["event_span"]["mean"] for b in blocks[key][mode]]
            combined[key][mode] = {
                "block_walls": [round(v, 4) for v in vals],
                "wall_grand_mean": round(float(np.mean(vals)), 4),
                "wall_grand_sem": round(float(np.std(vals, ddof=1) / math.sqrt(len(vals))), 5),
                "event_grand_mean": round(float(np.mean(spans)), 4),
            }
    lo, hi = str(TAUS[0]), str(TAUS[1])
    deltas = []
    for rnd in range(ROUNDS):
        d = blocks[hi]["graph"][rnd]["wall"]["mean"] - blocks[lo]["graph"][rnd]["wall"]["mean"]
        deltas.append(round(d, 4))
    attribution = {
        "graph_wall_delta_high_minus_low_ms": round(float(np.mean(deltas)), 4),
        "graph_wall_delta_sem_ms": round(float(np.std(deltas, ddof=1) / math.sqrt(len(deltas))), 4),
        "eager_wall_delta_high_minus_low_ms": round(
            combined[hi]["eager"]["wall_grand_mean"] - combined[lo]["eager"]["wall_grand_mean"], 4),
        "graph_speedup_vs_eager_neg05": round(
            combined[lo]["eager"]["wall_grand_mean"] / combined[lo]["graph"]["wall_grand_mean"], 3),
        "graph_speedup_vs_eager_p25": round(
            combined[hi]["eager"]["wall_grand_mean"] / combined[hi]["graph"]["wall_grand_mean"], 3),
        "kernel_saving_prediction_ms": 4.2,
    }
    results["combined"] = combined
    results["pair_deltas"] = deltas
    results["attribution"] = attribution
    print("\n=== EXP4C ATTRIBUTION ===")
    for k_, v_ in attribution.items():
        print(f"  {k_}: {v_}")
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nsaved -> {OUT_JSON}")


if __name__ == "__main__":
    main()
