#!/usr/bin/env python3
"""Experiment 4f: C3 scale-out (PAPER_01_REWORK_PLAN.md).

Question: is the score-shift pathology (and its tau fix) a D=64 artifact,
or does it replicate at width? Two parts:

Part A (fast, no checkpoints): exp1 legacy protocol at embedding width
d in {32, 64, 128} for the diagnostic arms sm / sm_fir / pn_fir / pn_fir_tau
(3 seeds each). Reports clean recall, dead-row fraction, positive-score
fraction and learned tau. Reuses the exp1 StackModel/training/eval verbatim
via import (both files have __main__ guards).

Part B (slow, checkpointed): WikiText-2 matched short run, Table-5B-style,
at D=256 / d_ff=1024 / H=4 / L=6 / seq=512 / batch=64 (OOM fallback 32),
2400 steps, eval_every=600, seed 42, BF16+TF32, variants
softmax / sm_fir / pn_fir_tau (eager only; Triton-at-scale out of scope).
The training loop mirrors exp4.train_matched_variant line-for-line, plus
checkpoint save/load every eval (600 steps) under
results/checkpoints_c3/<variant>/ for resume under the ~1h background limit.
One invocation trains ONE variant (idempotent skip when complete).

Usage:
  python3 experiments/exp4f_scaleout.py --part a
  python3 experiments/exp4e... no:
  python3 experiments/exp4f_scaleout.py --part b --variant softmax
Artifacts: results/exp4f_scaleout.json (partA + partB accumulated).
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
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
import exp1_legacy_stack_candidates as e1  # noqa: E402
import exp4_wikitext2_lm as e4  # noqa: E402

DEVICE = e4.DEVICE
RESULTS_DIR = os.path.join(ROOT, "results")
OUT_JSON = os.path.join(RESULTS_DIR, "exp4f_scaleout.json")
CKPT_C3_ROOT = os.path.join(RESULTS_DIR, "checkpoints_c3")

PARTA_WIDTHS = (32, 64, 128)
PARTA_ARMS = [
    ("sm", "Softmax, no filter"),
    ("sm_fir", "Softmax + 7-tap FIR"),
    ("pn_fir", "Power-norm + 7-tap FIR, no cal"),
    ("pn_fir_tau", "Power-norm + 7-tap FIR + tau"),
]
PARTA_SEEDS = (1234, 2025, 3141)
# Follow-up: d=128 with lower lr to separate "recipe" from "fundamental"
PARTA_LR_PROBE = {"d": 128, "lr": 1e-3, "arms": ("sm_fir", "pn_fir_tau")}

PARTB_VARIANTS = ("softmax", "sm_fir", "pn_fir_tau")


def ms_of(vals) -> dict:
    arr = np.array(list(vals), dtype=np.float64)
    return {"mean": float(arr.mean()), "std": float(arr.std()),
            "per_seed": [float(v) for v in vals]}


# ---------------------------------------------------------------- Part A
def part_a() -> dict:
    print("=" * 92)
    print(f"EXP 4f-A: exp1 protocol at widths {PARTA_WIDTHS} | device={DEVICE}")
    print("=" * 92)
    out = {"widths": list(PARTA_WIDTHS), "arms": {}, "per_seed": {}}
    for d in PARTA_WIDTHS:
        cfg = e1.Cfg(d=d)
        out["per_seed"][str(d)] = {}
        for sid, desc in PARTA_ARMS:
            per_seed = {}
            for seed in PARTA_SEEDS:
                e1.set_seed(seed)
                g = torch.Generator().manual_seed(seed)
                model = e1.StackModel(sid, cfg).to(DEVICE)
                e1.train_model(model, cfg, g)
                res = e1.eval_decoy(model, cfg, seed)
                sd = res["score_diag"]
                per_seed[str(seed)] = {
                    "clean": res["clean"],
                    "d@1.0": res["d@1.0"],
                    "raw_frac_pos": sd["raw_frac_pos"],
                    "raw_rows_all_neg": sd["raw_rows_all_neg"],
                    "tau": res["tau"],
                }
                del model
            agg = {
                "clean": ms_of([per_seed[s]["clean"] for s in per_seed]),
                "d@1.0": ms_of([per_seed[s]["d@1.0"] for s in per_seed]),
                "raw_frac_pos": ms_of([per_seed[s]["raw_frac_pos"] for s in per_seed]),
                "raw_rows_all_neg": ms_of([per_seed[s]["raw_rows_all_neg"] for s in per_seed]),
            }
            if per_seed[str(PARTA_SEEDS[0])]["tau"] is not None:
                agg["tau"] = ms_of([per_seed[s]["tau"] for s in per_seed])
            out["arms"].setdefault(sid, {"desc": desc, "by_width": {}})["by_width"][str(d)] = agg
            out["per_seed"][str(d)][sid] = per_seed
            a = agg
            print(
                f"  d={d:3d} {sid:12s} clean={a['clean']['mean']*100:5.1f}±{a['clean']['std']*100:4.1f} "
                f"frac_pos={a['raw_frac_pos']['mean']*100:5.1f}±{a['raw_frac_pos']['std']*100:4.1f} "
                f"deadrows={a['raw_rows_all_neg']['mean']*100:5.1f}±{a['raw_rows_all_neg']['std']*100:4.1f}"
                + (f" tau={a['tau']['mean']:+.3f}±{a['tau']['std']:.3f}" if "tau" in a else ""),
                flush=True,
            )
    # LR probe at d=128
    print("  --- lr probe d=128 lr=1e-3 ---", flush=True)
    cfg_lr = e1.Cfg(d=PARTA_LR_PROBE["d"], lr=PARTA_LR_PROBE["lr"])
    out["lr_probe"] = {"d": PARTA_LR_PROBE["d"], "lr": PARTA_LR_PROBE["lr"], "arms": {}}
    for sid in PARTA_LR_PROBE["arms"]:
        per_seed = {}
        for seed in PARTA_SEEDS:
            e1.set_seed(seed)
            g = torch.Generator().manual_seed(seed)
            model = e1.StackModel(sid, cfg_lr).to(DEVICE)
            e1.train_model(model, cfg_lr, g)
            res = e1.eval_decoy(model, cfg_lr, seed)
            sd = res["score_diag"]
            per_seed[str(seed)] = {
                "clean": res["clean"], "d@1.0": res["d@1.0"],
                "raw_frac_pos": sd["raw_frac_pos"],
                "raw_rows_all_neg": sd["raw_rows_all_neg"], "tau": res["tau"],
            }
            del model
        agg = {
            "clean": ms_of([per_seed[s]["clean"] for s in per_seed]),
            "raw_frac_pos": ms_of([per_seed[s]["raw_frac_pos"] for s in per_seed]),
            "raw_rows_all_neg": ms_of([per_seed[s]["raw_rows_all_neg"] for s in per_seed]),
        }
        if per_seed[str(PARTA_SEEDS[0])]["tau"] is not None:
            agg["tau"] = ms_of([per_seed[s]["tau"] for s in per_seed])
        out["lr_probe"]["arms"][sid] = agg
        print(
            f"  d=128 lr=1e-3 {sid:12s} clean={agg['clean']['mean']*100:5.1f}±{agg['clean']['std']*100:4.1f}"
            + (f" tau={agg['tau']['mean']:+.3f}±{agg['tau']['std']:.3f}" if "tau" in agg else ""),
            flush=True,
        )
    return out


# ---------------------------------------------------------------- Part B
def train_scaleout_variant(
    vocab: int,
    train_c: torch.Tensor,
    val_c: torch.Tensor,
    test_c: torch.Tensor,
    cfg,
    variant: str,
    ckpt_dir: str,
    resume_path: str | None = None,
) -> dict:
    """Mirror of e4.train_matched_variant, plus checkpoint save/load each eval."""
    e4.set_seed(cfg.seed)
    model = e4.WikiLM(vocab, cfg, variant).to(DEVICE)
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.95),
    )
    loader = e4.Loader(train_c, cfg.batch_size, True, cfg.seed)
    use_amp = cfg.amp == "bf16" and DEVICE.type == "cuda"
    cg_pre = e4.causal_gate(model)
    hist = {"eval_steps": [], "val_ppl": [], "train_loss": []}
    best_val = float("inf")
    best_state = None
    step = 0
    t0 = time.perf_counter()
    os.makedirs(ckpt_dir, exist_ok=True)
    latest_path = os.path.join(ckpt_dir, "latest.pt")

    def save(step_, best_val_):
        torch.save({
            "variant": variant, "step": step_, "best_val_ppl": best_val_,
            "hist": hist, "model": model.state_dict(), "optimizer": opt.state_dict(),
        }, latest_path)

    if resume_path and os.path.isfile(resume_path):
        ck = torch.load(resume_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        step = int(ck["step"]) + 1
        best_val = float(ck.get("best_val_ppl", float("inf")))
        hist = ck.get("hist") or hist
        best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
        print(f"  RESUME-MIDRUN from {resume_path} at step={step}", flush=True)

    while step < cfg.steps:
        for batch in loader.epoch():
            if step >= cfg.steps:
                break
            if step < cfg.warmup:
                lr = cfg.lr * (step + 1) / max(cfg.warmup, 1)
            else:
                prog = min((step - cfg.warmup) / max(cfg.lr_horizon - cfg.warmup, 1), 1.0)
                lr = cfg.lr * 0.5 * (1.0 + math.cos(math.pi * prog))
            for g in opt.param_groups:
                g["lr"] = lr
            x = batch.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model(x)
                    loss = F.cross_entropy(
                        logits[:, :-1].reshape(-1, logits.size(-1)), x[:, 1:].reshape(-1))
            else:
                logits = model(x)
                loss = F.cross_entropy(
                    logits[:, :-1].reshape(-1, logits.size(-1)), x[:, 1:].reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip)
            opt.step()
            step += 1
            # Step-based checkpoint (independent of eval cadence): a 50-min link
            # must always persist most of its work even if the next eval is far.
            if step % 300 == 0 and step < cfg.steps:
                save(step, best_val)
            if step % cfg.eval_every == 0 or step == cfg.steps:
                v = e4.eval_ppl(model, val_c, min(cfg.batch_size, 32), amp=cfg.amp)
                loss_f = float(loss.item())
                hist["eval_steps"].append(step)
                hist["val_ppl"].append(v["ppl"])
                hist["train_loss"].append(loss_f)
                print(
                    f"  [{variant:10s}] step={step:4d}/{cfg.steps} "
                    f"loss={loss_f:.4f} val_ppl={v['ppl']:.2f} t={time.perf_counter()-t0:.0f}s",
                    flush=True)
                if v["ppl"] < best_val:
                    best_val = v["ppl"]
                    best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
                save(step, best_val)
    wall_sec = time.perf_counter() - t0
    if best_state is not None:
        model.load_state_dict(best_state)
    val_res = e4.eval_ppl(model, val_c, min(cfg.batch_size, 32), amp=cfg.amp)
    test_res = e4.eval_ppl(model, test_c, min(cfg.batch_size, 32), amp=cfg.amp)
    train_res = e4.eval_ppl(model, train_c, min(cfg.batch_size, 32), amp=cfg.amp)
    cg_post = e4.causal_gate(model)
    taus = [float(b.attn.score_tau.detach().item()) for b in model.blocks
            if hasattr(b.attn, "score_tau")]
    peak_mb = torch.cuda.max_memory_allocated() / 1e6 if DEVICE.type == "cuda" else 0.0
    return {
        "variant": variant, "params": model.param_count(),
        "val_ppl": float(val_res["ppl"]), "test_ppl": float(test_res["ppl"]),
        "train_ppl": float(train_res["ppl"]), "causal_pre": cg_pre, "causal_post": cg_post,
        "tau_per_layer": taus, "history": hist, "wall_sec": float(wall_sec),
        "ms_per_step_effective": float(wall_sec / cfg.steps * 1000.0),
        "peak_mb": float(peak_mb), "batch_size": cfg.batch_size,
    }


def part_b_variant(variant: str) -> dict:
    e4.configure_fast_cuda()
    base = e4.WikiLMConfig()
    cfg = e4.WikiLMConfig(
        d_model=256, d_ff=1024, n_heads=4, n_layers=6, max_seq=512,
        batch_size=64, steps=2400, eval_every=600, seed=42,
        lr=base.lr, weight_decay=base.weight_decay, warmup=base.warmup,
        lr_horizon=base.lr_horizon, clip=base.clip, dropout=base.dropout,
        vocab_min_freq=base.vocab_min_freq, p=base.p,
        silence_gamma=base.silence_gamma, tau_init=base.tau_init,
        fir_k=base.fir_k, amp="bf16",
    )
    tr = e4.load_tokens("train")
    va = e4.load_tokens("validation")
    te = e4.load_tokens("test")
    vocab, stoi = e4.build_vocab(tr, cfg.vocab_min_freq)
    V = len(vocab)
    train_c = e4.make_chunks(torch.tensor(e4.encode(tr, stoi), dtype=torch.long), cfg.max_seq)
    val_c = e4.make_chunks(torch.tensor(e4.encode(va, stoi), dtype=torch.long), cfg.max_seq)
    test_c = e4.make_chunks(torch.tensor(e4.encode(te, stoi), dtype=torch.long), cfg.max_seq)
    print(f"data D=256/N=512: vocab={V} chunks={len(train_c)}/{len(val_c)}/{len(test_c)}", flush=True)
    ckpt_dir = os.path.join(CKPT_C3_ROOT, variant)
    os.makedirs(ckpt_dir, exist_ok=True)
    latest = os.path.join(ckpt_dir, "latest.pt")
    try:
        return train_scaleout_variant(V, train_c, val_c, test_c, cfg, variant, ckpt_dir,
                                      resume_path=latest if os.path.isfile(latest) else None)
    except (torch.cuda.OutOfMemoryError, torch.AcceleratorError) as e:
        # torch>=2.5 raises AcceleratorError for CUDA OOM; match by message
        if "out of memory" not in str(e).lower():
            raise
        print("  OOM at batch 64 — retry batch_size=32", flush=True)
        torch.cuda.empty_cache()
        cfg.batch_size = 32
        ckpt_dir_b = os.path.join(CKPT_C3_ROOT, variant + "_b32")
        os.makedirs(ckpt_dir_b, exist_ok=True)
        # carry over batch-64 progress when it exists (documented switch point)
        carry = os.path.join(ckpt_dir, "latest.pt")
        latest_b = os.path.join(ckpt_dir_b, "latest.pt")
        if os.path.isfile(latest_b):
            resume_b = latest_b
            switch_note = "fresh start at batch 32"
        elif os.path.isfile(carry):
            resume_b = carry
            ck_carry = torch.load(carry, map_location="cpu", weights_only=False)
            switch_note = f"batch 64->32 switch at step {int(ck_carry.get('step', -1)) + 1}"
        else:
            resume_b = None
            switch_note = "fresh start at batch 32"
        res = train_scaleout_variant(V, train_c, val_c, test_c, cfg, variant, ckpt_dir_b,
                                     resume_path=resume_b)
        res["note"] = f"OOM fallback batch=32 ({switch_note})"
        return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["a", "b"], required=True)
    ap.add_argument("--variant", default=None)
    args = ap.parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    payload = {"partA": None, "partB": {"runs": {}}}
    if os.path.isfile(OUT_JSON):
        with open(OUT_JSON, encoding="utf-8") as f:
            payload = json.load(f)
    t0 = time.time()
    if args.part == "a":
        payload["partA"] = part_a()
    else:
        assert args.variant in PARTB_VARIANTS, args.variant
        if args.variant in payload["partB"]["runs"]:
            print(f"SKIP {args.variant}: already complete (idempotent relaunch)", flush=True)
            return
        print(f"--- part B: {args.variant} D=256/N=512 ---", flush=True)
        payload["partB"]["runs"][args.variant] = part_b_variant(args.variant)
        r = payload["partB"]["runs"][args.variant]
        print(f"DONE {args.variant} train={r['train_ppl']:.2f} val={r['val_ppl']:.2f} "
              f"test={r['test_ppl']:.2f} ms/step={r['ms_per_step_effective']:.1f} "
              f"peak={r['peak_mb']:.0f}MB causal={r['causal_post']['passed']}", flush=True)
    payload["wall_s"] = round(time.time() - t0, 1)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"saved -> {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
