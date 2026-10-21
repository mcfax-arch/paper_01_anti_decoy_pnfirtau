#!/usr/bin/env python3
"""Experiment 4b: R2 closure instrumentation for the 6-Layer PnFirTauFlashMHA stack (v2).

Closes audit finding R2 (AUDIT_paper01_r5.md): the isolated 1-layer Flash kernel
(`kern1`) saving of 0.86 ms/layer predicts ~5.1 ms over 6 layers, while the full
stack realised only 1.60 ms (Table 5E). v2 hardens the measurement protocol:

  A. Per-layer silent-row and FULLY-SILENT 64x64-tile fractions inside the exact
     stack execution path, derived from raw score blocks (the flash kernel's skip
     condition), plus the Table-5E-convention `w`-based row mass.
  B. Fine-grained CUDA-event spans per region of the step (forward residual loop
     with per-layer sub-spans, backward pass, total), plus total wall time. This
     separates host dispatch gaps (wall - event) from GPU execution spans without
     relying on CUPTI, whose self-time records proved unreliable under WSL2
     GPU-PV (self-busy sums can exceed wall time).
  C. Interleaved ABAB-block schedule over tau ([-0.5, +2.5] x 3 rounds, warmup +
     120 steps per block) so that environment drift cannot masquerade as a tau
     effect; kern1 isolated kernels measured on the same interleaved schedule.
  D. One auxiliary torch.profiler pass per tau (kernel launch counts and
     per-kernel names) recorded as supporting evidence only.

Writes results/exp4b_mha6_instrumentation.json (overwrites v1) and prints a
console summary. Elevated wall jitter in this environment (per-step std 2-4 ms)
is reported alongside SEM so the significance level stays honest.
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
from src.triton_pnfirtau import flash_pn_fir_tau_attention, triton_causal_fir  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = os.path.join(ROOT, "results")
OUT_JSON = os.path.join(RESULTS_DIR, "exp4b_mha6_instrumentation.json")

D_MODEL, N_HEADS, N_LAYERS = 64, 4, 6
N_SK, B_SK = 1024, 16
TAUS = (-0.5, 0.0, 2.5)          # 0.0 participates in diagnostics only
ABBA_TAUS = (-0.5, 2.5)          # rotated interleave pair
BLOCK_STEPS, BLOCK_WARMUP = 120, 25
ROUNDS = 6                       # 3 pairs: (lo,hi), (hi,lo), ... rotation cancels within-round drift
PROF_STEPS = 20
SEED = 42


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
        "min": round(float(a.min()), 4),
        "max": round(float(a.max()), 4),
        "n": int(len(a)),
    }


# ------------------------------------------------------------------ A. per-layer tile diagnostics
@torch.no_grad()
def per_layer_diagnostics(layers: nn.ModuleList, x: torch.Tensor, tau: float) -> list[dict]:
    """Per layer, on the exact stack residual stream:
      - silent_rows_u: fraction of query rows with no positive (S - tau) entry
      - fully_silent_tiles_u: fraction of 64x64 (row x col) tiles with no positive
        entry inside their causal column range (flash kernel skip condition)
      - silent_rows_w / mean_row_mass_w: Table-5E convention via the row-fused path
    """
    B, N, _ = x.shape
    H, hd = N_HEADS, D_MODEL // N_HEADS
    scale = 1.0 / math.sqrt(hd)
    rows_per_tile, cols_per_tile = 64, 64
    n_rt, n_ct = N // rows_per_tile, N // cols_per_tile
    causal = torch.arange(N, device=x.device)[:, None] >= torch.arange(N, device=x.device)[None, :]
    records = []
    h = x
    for li, lyr in enumerate(layers):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            # (a) kernel-view diagnostics from raw scores (identical tensors to the flash path)
            q_raw, k_raw, v_raw = lyr.qkv(h).chunk(3, dim=-1)
            k_s = triton_causal_fir(k_raw, lyr.fir_taps).view(B, N, H, hd).transpose(1, 2)
            q_s = q_raw.view(B, N, H, hd).transpose(1, 2)
            raw = torch.matmul(q_s, k_s.transpose(-2, -1)) * scale  # [B, H, N, N]
            present = (raw > tau) & causal
            blk = (
                (~present)
                .reshape(B, H, n_rt, rows_per_tile, n_ct, cols_per_tile)
                .permute(0, 1, 2, 4, 3, 5)  # -> [B, H, n_rt, n_ct, rows, cols]
            )
            tile_present = (~blk).any(dim=(-2, -1))  # any positive score in the 64x64 tile
            silent_rows_u = 1.0 - present.any(dim=-1).float().mean().item()
            fully_silent_tiles_u = 1.0 - tile_present.float().mean().item()
            # (b) Table-5E convention via the row-fused Triton path (returns (out, w))
            _, w = PnFirTauFlashMHA.forward(lyr, h, return_weights=True)
            row_mass = w.sum(dim=-1)
            silent_rows_w = (row_mass <= 1e-6).float().mean().item()
            mean_row_mass_w = float(row_mass.float().mean().item())
            # exact stack residual progression under the same autocast state
            h = h + PnFirTauFlashMHA.forward(lyr, h)
        records.append(
            {
                "layer": li,
                "silent_rows_u": round(silent_rows_u, 6),
                "fully_silent_tiles_u": round(fully_silent_tiles_u, 6),
                "silent_rows_w": round(silent_rows_w, 6),
                "mean_row_mass_w": round(mean_row_mass_w, 6),
            }
        )
    return records


def _mk_events() -> tuple[torch.cuda.Event, torch.cuda.Event]:
    return (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )


def stack_blocks(layers: nn.ModuleList, x: torch.Tensor, dy: torch.Tensor, steps: int, warmup: int) -> dict:
    """One timed block: stats for wall, total event, forward-loop and backward spans,
    plus per-layer forward sub-spans (outer events around each layer call)."""
    ev_layer = [_mk_events() for _ in range(N_LAYERS)]
    ev_fwd_start, ev_fwd_end = _mk_events()
    ev_bwd = _mk_events()
    ev_tot = _mk_events()

    def step() -> None:
        for lyr in layers:
            lyr.zero_grad(set_to_none=True)
        x.grad = None
        ev_tot[0].record()
        ev_fwd_start.record()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            h = x
            for i, lyr in enumerate(layers):
                ev_layer[i][0].record()
                h = h + lyr(h)
                ev_layer[i][1].record()
        ev_fwd_end.record()
        ev_bwd[0].record()
        h.backward(dy)
        ev_bwd[1].record()
        ev_tot[1].record()
        torch.cuda.synchronize()

    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    walls, fwd_ms, bwd_ms, tot_ms = [], [], [], []
    per_layer_ms = [[] for _ in range(N_LAYERS)]
    for _ in range(steps):
        t0 = time.perf_counter()
        step()
        walls.append((time.perf_counter() - t0) * 1000.0)
        torch.cuda.synchronize()
        for i in range(N_LAYERS):
            per_layer_ms[i].append(ev_layer[i][0].elapsed_time(ev_layer[i][1]))
        fwd_ms.append(ev_fwd_start.elapsed_time(ev_fwd_end))
        bwd_ms.append(ev_bwd[0].elapsed_time(ev_bwd[1]))
        tot_ms.append(ev_tot[0].elapsed_time(ev_tot[1]))
    torch.cuda.synchronize()
    return {
        "wall": stats(walls),
        "event_total": stats(tot_ms),
        "event_fwd_loop": stats(fwd_ms),
        "event_bwd": stats(bwd_ms),
        "event_fwd_per_layer": [stats(v) for v in per_layer_ms],
    }


def kern_blocks(q, k, v, tau_s, scale, do, steps: int, warmup: int) -> dict:
    fwd = _mk_events()
    bwd = _mk_events()
    tot = _mk_events()

    def step() -> None:
        q.grad = k.grad = v.grad = tau_s.grad = None
        tot[0].record()
        fwd[0].record()
        out = flash_pn_fir_tau_attention(q, k, v, tau_s, scale=scale)
        fwd[1].record()
        bwd[0].record()
        out.backward(do)
        bwd[1].record()
        tot[1].record()
        torch.cuda.synchronize()

    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    wall, fwd_ms, bwd_ms, tot_ms = [], [], [], []
    for _ in range(steps):
        t0 = time.perf_counter()
        step()
        wall.append((time.perf_counter() - t0) * 1000.0)
        fwd_ms.append(fwd[0].elapsed_time(fwd[1]))
        bwd_ms.append(bwd[0].elapsed_time(bwd[1]))
        tot_ms.append(tot[0].elapsed_time(tot[1]))
    torch.cuda.synchronize()
    return {
        "wall": stats(wall),
        "event_total": stats(tot_ms),
        "event_fwd": stats(fwd_ms),
        "event_bwd": stats(bwd_ms),
    }


def interleaved(build_fn: callable, blocks_fn: callable, label: str, rounds: int = ROUNDS) -> dict:
    """Rotated interleave: even rounds run (lo, hi), odd rounds (hi, lo), so any
    within-round warm-up drift appears symmetrically in both taus and cannot
    masquerade as a tau effect. Round-level deltas are recorded per pair."""
    per_block = {t: [] for t in ABBA_TAUS}
    pair_deltas = []
    for rnd in range(rounds):
        order = (ABBA_TAUS if rnd % 2 == 0 else tuple(reversed(ABBA_TAUS)))
        block_means = {}
        for tau in order:
            parts = build_fn(tau)
            res = blocks_fn(parts, BLOCK_STEPS, BLOCK_WARMUP)
            per_block[tau].append(res)
            block_means[tau] = res["wall"]["mean"]
            del parts
            torch.cuda.empty_cache()
            fwd_key = "event_fwd_loop" if "event_fwd_loop" in res else "event_fwd"
            print(
                f"  [{label}] round {rnd + 1}/{rounds} tau={tau:+.1f}: "
                f"wall={res['wall']['mean']:.2f}±{res['wall']['std']:.2f} ms "
                f"fwd={res[fwd_key]['mean']:.2f} bwd={res['event_bwd']['mean']:.2f} ms",
                flush=True,
            )
        pair_deltas.append(
            {"round": rnd + 1, "order": order[0], "wall_delta_high_minus_low": round(block_means[ABBA_TAUS[1]] - block_means[ABBA_TAUS[0]], 4)}
        )
    out = {}
    for tau, blocks in per_block.items():
        combine = {}
        for section in blocks[0]:
            if section == "event_fwd_per_layer":
                combine[section] = []
                for li in range(N_LAYERS):
                    layer_block_means = [b[section][li]["mean"] for b in blocks]
                    combine[section].append(
                        {"layer": li, "block_means": [round(m, 4) for m in layer_block_means],
                         "mean": round(float(np.mean(layer_block_means)), 4),
                         "block_std": round(float(np.std(layer_block_means)), 4)}
                    )
            else:
                vals = [b[section]["mean"] for b in blocks]
                stds = [b[section]["std"] for b in blocks]
                sems = [b[section]["sem"] for b in blocks]
                combine[section] = {
                    "block_means": [round(v, 4) for v in vals],
                    "block_stds": [round(v, 4) for v in stds],
                    "block_sems": [round(v, 4) for v in sems],
                    "grand_mean": round(float(np.mean(vals)), 4),
                    "grand_sem": round(float(np.std(vals, ddof=1) / math.sqrt(len(vals))), 5),
                }
        out[str(tau)] = {"per_block": blocks, "combined": combine, "pair_deltas": pair_deltas}
    return out


def repro_sequential_protocol() -> None:
    """Reproduce exp4's ORIGINAL sequential protocol verbatim (taus ascending,
    per-tau diagnostics then warmup 25 + 200 timed steps, perf_counter timing),
    then repeat the same six blocks in REVERSE order. If wall declines monotonically
    with block index in both orders, the Table 5E `mha6` 'saving' is identified as a
    block-order drift artifact rather than a tau effect."""
    taus_seq = (-0.5, 0.0, 0.5, 1.0, 1.5, 2.5)
    res_rows = []

    def one_block(tau: float, order_start: float) -> dict:
        torch.cuda.empty_cache()
        set_seed(SEED)
        acfg = PnFirConfig(d_model=D_MODEL, n_heads=N_HEADS, dropout=0.0, tau_init=float(tau))
        layers = nn.ModuleList([PnFirTauFlashMHA(acfg) for _ in range(N_LAYERS)]).to(DEVICE)
        x = torch.randn(B_SK, N_SK, D_MODEL, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
        dy = torch.randn_like(x)
        # exp4's silent_row_frac step (TritonMHA path)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, w0 = layers[0](x, return_weights=True)
        row_mass = w0.sum(dim=-1)

        def step():
            for lyr in layers:
                lyr.zero_grad(set_to_none=True)
            x.grad = None
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                h = x
                for lyr in layers:
                    h = h + lyr(h)
            h.backward(dy)

        for _ in range(BLOCK_WARMUP):
            step()
        torch.cuda.synchronize()
        times = []
        for _ in range(BLOCK_STEPS):
            t0 = time.perf_counter()
            step()
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000.0)
        out = {
            "tau": float(tau),
            "first_block_was": float(order_start),
            "ms_mean": round(float(np.mean(times)), 4),
            "ms_std": round(float(np.std(times)), 4),
            "ms_sem": round(float(np.std(times) / math.sqrt(len(times))), 4),
            "silent_row_frac_layer0": round(float((row_mass <= 1e-6).float().mean().item()), 6),
        }
        del layers, x, dy
        torch.cuda.empty_cache()
        return out

    res = {"config": {"taus_order": list(taus_seq), "block_steps": BLOCK_STEPS, "block_warmup": BLOCK_WARMUP, "seed": SEED}, "runs": []}
    for order_name, order in (("ascending (exp4 protocol)", list(taus_seq)), ("descending (reversed)", list(reversed(taus_seq)))):
        for tau in order:
            r = one_block(tau, order[0])
            r["run"] = order_name
            res["runs"].append(r)
            print(
                f"  [{order_name[:5]}] tau={tau:+.1f}: wall={r['ms_mean']:.2f}±{r['ms_std']:.2f} ms",
                flush=True,
            )
    with open(os.path.join(RESULTS_DIR, "exp4b_sequential_repro.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"saved -> results/exp4b_sequential_repro.json", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-diagnostics", action="store_true")
    parser.add_argument("--repro-sequential", action="store_true",
                        help="run only the exp4-protocol drift reproduction, then exit")
    args = parser.parse_args()
    if args.repro_sequential:
        repro_sequential_protocol()
        return
    torch.backends.cuda.matmul.allow_tf32 = False  # exact exp4 bench_triton_mha_scaling state
    assert torch.cuda.is_available()

    results = {
        "meta": {
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "gpu": torch.cuda.get_device_name(0),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "triton": __import__("triton").__version__,
            "config": {
                "d_model": D_MODEL,
                "n_heads": N_HEADS,
                "n_layers": N_LAYERS,
                "N": N_SK,
                "B": B_SK,
                "block_steps": BLOCK_STEPS,
                "block_warmup": BLOCK_WARMUP,
                "rounds": ROUNDS,
                "schedule": "ABAB interleaved (tau -0.5 / +2.5 x 3 rounds), diagnostics at tau in {-0.5, 0.0, +2.5}",
                "seed": SEED,
                "precision": "BF16 autocast, matmul TF32 off (exp4 bench_triton_mha_scaling state)",
                "tile": "64x64 (B_M=B_N=64)",
            },
            "purpose": "Close audit R2 (AUDIT_paper01_r5.md): attribute the kern1 0.86 ms/layer vs "
                       "in-stack saving gap with per-layer tile/score instrumentation and CUDA-event "
                       "region spans under an interleave-controlled schedule.",
        },
        "diagnostics_per_tau": {},
        "stack_timing": {},
        "kern1_timing": {},
        "gemm_reference_note": (
            "Attention region = flash module stack (qkv gemm + FIR + flash kernels + out gemm); "
            "backward includes autograd of the whole residual graph. Host dispatch gap per section "
            "= wall - event_total."
        ),
    }

    # ---- A. diagnostics (sequential tau blocks; score-statistics are deterministic)
    if not args.skip_diagnostics:
        for tau_val in TAUS:
            print(f"[diag tau={tau_val:+.1f}]", flush=True)
            torch.cuda.empty_cache()
            set_seed(SEED)  # identical tensors to exp4 silent-skip block
            acfg = PnFirConfig(d_model=D_MODEL, n_heads=N_HEADS, dropout=0.0, tau_init=float(tau_val))
            layers = nn.ModuleList([PnFirTauFlashMHA(acfg) for _ in range(N_LAYERS)]).to(DEVICE)
            x = torch.randn(B_SK, N_SK, D_MODEL, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
            recs = per_layer_diagnostics(layers, x, tau=tau_val)
            results["diagnostics_per_tau"][str(tau_val)] = recs
            for rec in recs:
                print(
                    f"    L{rec['layer']}: silent_rows_u={rec['silent_rows_u'] * 100:6.2f}%  "
                    f"fully_silent_tiles_u={rec['fully_silent_tiles_u'] * 100:6.2f}%  "
                    f"silent_rows_w={rec['silent_rows_w'] * 100:6.2f}%",
                    flush=True,
                )
            del layers, x
            torch.cuda.empty_cache()

    # ---- B/C. interleaved stack timing: build once per tau, alternate blocks
    d_h = D_MODEL // N_HEADS
    scale = 1.0 / math.sqrt(d_h)

    def build_stack(tau: float):
        set_seed(SEED)
        acfg = PnFirConfig(d_model=D_MODEL, n_heads=N_HEADS, dropout=0.0, tau_init=float(tau))
        layers = nn.ModuleList([PnFirTauFlashMHA(acfg) for _ in range(N_LAYERS)]).to(DEVICE)
        x = torch.randn(B_SK, N_SK, D_MODEL, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
        dy = torch.randn_like(x)
        return layers, x, dy

    def run_stack(parts, steps: int, warmup: int):
        layers, x, dy = parts
        return stack_blocks(layers, x, dy, steps, warmup)

    print("[stack ABAB interleaved measurement]", flush=True)
    results["stack_timing"] = interleaved(build_stack, run_stack, "stack")

    # ---- isolared kern1 on the same interleaved schedule
    def build_kern(tau: float):
        set_seed(SEED)
        acfg = PnFirConfig(d_model=D_MODEL, n_heads=N_HEADS, dropout=0.0, tau_init=float(tau))
        layers = nn.ModuleList([PnFirTauFlashMHA(acfg) for _ in range(1)]).to(DEVICE)
        x = torch.randn(B_SK, N_SK, D_MODEL, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            q_raw, k_raw, v_raw = layers[0].qkv(x).chunk(3, dim=-1)
            k_s = (
                triton_causal_fir(k_raw, layers[0].fir_taps)
                .view(B_SK, N_SK, N_HEADS, d_h)
                .transpose(1, 2)
                .contiguous()
                .detach()
                .requires_grad_(True)
            )
            q_s = q_raw.view(B_SK, N_SK, N_HEADS, d_h).transpose(1, 2).contiguous().detach().requires_grad_(True)
            v_s = v_raw.view(B_SK, N_SK, N_HEADS, d_h).transpose(1, 2).contiguous().detach().requires_grad_(True)
        tau_s = torch.tensor(float(tau), device=DEVICE, dtype=torch.float32, requires_grad=True)
        do_s = torch.randn_like(q_s)
        return q_s, k_s, v_s, tau_s, do_s

    def run_kern(parts, steps: int, warmup: int):
        q_s, k_s, v_s, tau_s, do_s = parts
        return kern_blocks(q_s, k_s, v_s, tau_s, scale, do_s, steps, warmup)

    print("[kern1 ABAB interleaved measurement]", flush=True)
    results["kern1_timing"] = interleaved(build_kern, run_kern, "kern1")

    # ---- R2 attribution arithmetic
    def c(sec: str, tau: str) -> float:
        return results["stack_timing"][tau]["combined"][sec]["grand_mean"]

    def kc(sec: str, tau: str) -> float:
        return results["kern1_timing"][tau]["combined"][sec]["grand_mean"]

    pred_wall = 6 * (kc("wall", "-0.5") - kc("wall", "2.5"))
    pred_attn = 6 * (kc("event_total", "-0.5") - kc("event_total", "2.5"))
    realized_wall = c("wall", "-0.5") - c("wall", "2.5")
    realized_event = c("event_total", "-0.5") - c("event_total", "2.5")
    realized_fwd = c("event_fwd_loop", "-0.5") - c("event_fwd_loop", "2.5")
    realized_bwd = c("event_bwd", "-0.5") - c("event_bwd", "2.5")
    # statistical test of the stack delta at the BLOCK level (the correct unit:
    # cross-block drift, not per-step jitter): rotated pair deltas
    pair_deltas = [p["wall_delta_high_minus_low"] for p in results["stack_timing"]["-0.5"]["pair_deltas"]]
    block_delta_mean = float(np.mean(pair_deltas))
    block_delta_sd = float(np.std(pair_deltas, ddof=1))
    block_delta_sem = block_delta_sd / math.sqrt(len(pair_deltas))
    attribution = {
        "pred_kern1_wall_saving_x6_ms": round(pred_wall, 3),
        "pred_kern1_event_saving_x6_ms": round(pred_attn, 3),
        "realized_stack_wall_saving_ms": round(realized_wall, 3),
        "realized_stack_event_saving_ms": round(realized_event, 3),
        "realized_fwd_loop_saving_ms": round(realized_fwd, 3),
        "realized_bwd_saving_ms": round(realized_bwd, 3),
        "block_level_pair_deltas_wall_ms": pair_deltas,
        "block_level_delta_mean_ms": round(block_delta_mean, 4),
        "block_level_delta_sem_ms": round(block_delta_sem, 4),
        "host_gap_wall_minus_event_neg05_ms": round(c("wall", "-0.5") - c("event_total", "-0.5"), 3),
        "host_gap_wall_minus_event_p25_ms": round(c("wall", "2.5") - c("event_total", "2.5"), 3),
    }
    results["r2_attribution"] = attribution
    print("\n=== R2 ATTRIBUTION (interleave-controlled) ===")
    for k_, v_ in attribution.items():
        print(f"  {k_}: {v_}")

    # ---- decisive discriminator: batch scaling at fixed launch count
    # If the stack step is host/launch-serialised, shrinking per-kernel GPU work
    # 4x (B 16 -> 4, same ~370 launches) must shrink wall only slightly; if it were
    # GPU-bound, wall would shrink proportionally with the workload.
    print("\n[batch-scaling suite B in {16, 4} x tau in {-0.5, +2.5}, ABAB-interleaved]", flush=True)
    results["batch_scaling"] = {}
    scale_combos = ((16, -0.5), (4, -0.5), (16, 2.5), (4, 2.5))
    per_combo = []

    SC_STEPS, SC_WARMUP = 100, 25

    for rnd in range(2):
        for Bx, taux in scale_combos:
            set_seed(SEED)
            acfg = PnFirConfig(d_model=D_MODEL, n_heads=N_HEADS, dropout=0.0, tau_init=float(taux))
            layers = nn.ModuleList([PnFirTauFlashMHA(acfg) for _ in range(N_LAYERS)]).to(DEVICE)
            x = torch.randn(Bx, N_SK, D_MODEL, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
            dy = torch.randn_like(x)
            res = stack_blocks(layers, x, dy, SC_STEPS, SC_WARMUP)
            key = f"B={Bx}_tau={taux:+.1f}"
            per_combo.append(
                {
                    "round": rnd + 1,
                    "B": Bx,
                    "tau": taux,
                    "wall_mean": res["wall"]["mean"],
                    "wall_std": res["wall"]["std"],
                    "event_total_mean": res["event_total"]["mean"],
                    "fwd_mean": res["event_fwd_loop"]["mean"],
                    "bwd_mean": res["event_bwd"]["mean"],
                }
            )
            del layers, x, dy
            torch.cuda.empty_cache()
            print(
                f"  [{key}] rnd {rnd + 1}: wall={res['wall']['mean']:.2f}±{res['wall']['std']:.2f} "
                f"fwd={res['event_fwd_loop']['mean']:.2f} bwd={res['event_bwd']['mean']:.2f}",
                flush=True,
            )
    results["batch_scaling"]["blocks"] = per_combo
    # launch counts per step via one short profiler pass per representative config
    launches = {}
    from torch.profiler import ProfilerActivity, profile

    for Bx, taux in ((16, -0.5), (16, 2.5), (4, 2.5), (4, -0.5)):
        set_seed(SEED)
        acfg = PnFirConfig(d_model=D_MODEL, n_heads=N_HEADS, dropout=0.0, tau_init=float(taux))
        layers = nn.ModuleList([PnFirTauFlashMHA(acfg) for _ in range(N_LAYERS)]).to(DEVICE)
        x = torch.randn(Bx, N_SK, D_MODEL, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
        dy = torch.randn_like(x)

        def step_once():
            for lyr in layers:
                lyr.zero_grad(set_to_none=True)
            x.grad = None
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                h = x
                for lyr in layers:
                    h = h + lyr(h)
            h.backward(dy)
            torch.cuda.synchronize()

        for _ in range(3):
            step_once()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(3):
                step_once()
        ka = prof.key_averages()
        nker = 0
        for e in ka:
            try:
                d = float(getattr(e, "self_device_time_total", 0.0) or 0.0) or float(
                    getattr(e, "self_cuda_time_total", 0.0) or 0.0
                )
            except Exception:
                d = 0.0
            if d > 0:
                nker += int(getattr(e, "count", 0))
        nker /= 3
        launches[f"B={Bx}_tau={taux:+.1f}"] = round(nker, 1)
        print(f"  launches/step @ B={Bx}, tau={taux:+.1f}: {nker:.1f}", flush=True)
        del layers, x, dy
        torch.cuda.empty_cache()
    results["batch_scaling"]["cuda_launches_per_step"] = launches

    # GPU state snapshot for provenance
    try:
        import subprocess

        q = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.sm,clocks.mem,power.draw,pstate",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30,
        )
        results["batch_scaling"]["gpu_state"] = q.stdout.strip()
    except Exception as exc:  # noqa: BLE001
        results["batch_scaling"]["gpu_state"] = f"unavailable: {exc}"

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nsaved -> {OUT_JSON}")


if __name__ == "__main__":
    main()
