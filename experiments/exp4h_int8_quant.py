#!/usr/bin/env python3
"""Experiment 4h: C5 INT8 quantization check (PAPER_01_REWORK_PLAN.md).

Turns the quantization motivation of section 1.2 (attention sinks ->
massive activation outliers -> INT8 PTQ degradation, documented at scale
by Softpick [12]) into a measured result at our scales.

Per trained checkpoint (D=64 50k best.pt x2 upstream; D=256 2.4k latest x3):
  1. Outlier stats over test chunks: per-Linear input absmax (max) and
     kurtosis (mean), grouped by role (attn.qkv / attn.out / ffn-in /
     ffn-out / head), plus residual-stream (ln_f input) absmax/kurtosis.
  2. Attention-sink mass: mean weight on position 0 over queries/heads/batch,
     per layer + model mean, via return_weights block walk (eval mode).
  3. INT8 PTQ simulation: per-token symmetric INT8 quantize-dequantize on
     every nn.Linear input (forward_pre_hook, fp emulation), then test PPL;
     exact FP test PPL in the same process for the degradation delta.

Idempotent per model (accumulates into results/exp4h_int8_quant.json).
Honest bounds: small models may show no sinks at all (reported as-is);
PTQ-sim is activation-only (weights stay fp) and emulates INT8 matmul
numerics, not a kernel.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
import exp4_wikitext2_lm as e4  # noqa: E402

DEVICE = e4.DEVICE
RESULTS_DIR = os.path.join(ROOT, "results")
OUT_JSON = os.path.join(RESULTS_DIR, "exp4h_int8_quant.json")

def _upstream_dir() -> str:
    cands = [
        r"I:\FFT_Attention\experiments\wikitext_pn_fir_lm",
        "/mnt/i/FFT_Attention/experiments/wikitext_pn_fir_lm",
    ]
    for c in cands:
        if os.path.isdir(c):
            return c
    raise FileNotFoundError("upstream wikitext_pn_fir_lm dir not found")


UPSTREAM_CKPT = os.path.join(_upstream_dir(), "checkpoints")

MODELS = [
    # (run_id, variant, d_model, max_seq, ckpt_path)
    ("d64_softmax_50k", "softmax", 64, 256,
     os.path.join(UPSTREAM_CKPT, "softmax", "best.pt")),
    ("d64_pn_50k", "pn_fir_tau", 64, 256,
     os.path.join(UPSTREAM_CKPT, "pn_fir_tau", "best.pt")),
    ("d256_softmax_2k4", "softmax", 256, 512,
     os.path.join(RESULTS_DIR, "checkpoints_c3", "softmax", "latest.pt")),
    ("d256_smfir_2k4", "sm_fir", 256, 512,
     os.path.join(RESULTS_DIR, "checkpoints_c3", "sm_fir", "latest.pt")),
    ("d256_pn_2k4", "pn_fir_tau", 256, 512,
     os.path.join(RESULTS_DIR, "checkpoints_c3", "pn_fir_tau", "latest.pt")),
]


def build_cfg(d_model: int, max_seq: int):
    base = e4.WikiLMConfig()
    return e4.WikiLMConfig(
        d_model=d_model, d_ff=4 * d_model,
        n_heads=base.n_heads, n_layers=base.n_layers, max_seq=max_seq,
        batch_size=base.batch_size, steps=base.steps, eval_every=base.eval_every,
        seed=base.seed, lr=base.lr, weight_decay=base.weight_decay,
        warmup=base.warmup, lr_horizon=base.lr_horizon, clip=base.clip,
        dropout=base.dropout, vocab_min_freq=base.vocab_min_freq, p=base.p,
        silence_gamma=base.silence_gamma, tau_init=base.tau_init,
        fir_k=base.fir_k, amp="bf16",
    )


def load_model(variant: str, cfg, ckpt_path: str) -> nn.Module:
    tr = e4.load_tokens("train")
    vocab, stoi = e4.build_vocab(tr, cfg.vocab_min_freq)
    model = e4.WikiLM(len(vocab), cfg, variant).to(DEVICE)
    ck = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    benign = [k for k in list(missing) + list(unexpected) if "fir_taps" in k]
    rest_missing = [k for k in missing if "fir_taps" not in k]
    rest_unexpected = [k for k in unexpected if "fir_taps" not in k]
    assert not rest_missing and not rest_unexpected, (rest_missing, rest_unexpected)
    print(f"  loaded {ckpt_path} (ckpt step={ck.get('step')}); "
          f"benign fir_taps keys={len(benign)}", flush=True)
    model.eval()
    return model, (vocab, stoi)


def role_of(name: str) -> str:
    if "attn.qkv" in name:
        return "attn.qkv"
    if "attn.out" in name:
        return "attn.out"
    if "ffn.0" in name:
        return "ffn-in"
    if "ffn.2" in name:
        return "ffn-out"
    if name == "head":
        return "head"
    return "other"


@torch.no_grad()
def outlier_stats(model, test_chunks, n_chunks: int, batch: int) -> dict:
    agg: dict[str, dict] = {}
    res_abs, res_k = [], []

    def kurt(x: torch.Tensor) -> float:
        xf = x.float()
        mu = xf.mean()
        var = xf.var(unbiased=False)
        if float(var) < 1e-12:
            return 0.0
        return float((((xf - mu) ** 4).mean() / (var ** 2)).item())

    hooks = []

    def mk_hook(role: str):
        def fn(mod, inp):
            x = inp[0].detach()
            a = agg.setdefault(role, {"absmax": 0.0, "kurt": []})
            a["absmax"] = max(a["absmax"], float(x.abs().max().item()))
            a["kurt"].append(kurt(x))
        return fn

    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            hooks.append(mod.register_forward_pre_hook(mk_hook(role_of(name))))

    def res_hook(mod, inp):
        x = inp[0].detach()
        res_abs.append(float(x.abs().max().item()))
        res_k.append(kurt(x))

    rh = model.ln_f.register_forward_pre_hook(res_hook)
    xs = test_chunks[:n_chunks]
    for i in range(0, len(xs), batch):
        xb = xs[i:i + batch].to(DEVICE)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            model(xb)
    for h in hooks:
        h.remove()
    rh.remove()
    out = {role: {"absmax_max": v["absmax"],
                  "kurtosis_mean": float(np.mean(v["kurt"]))} for role, v in agg.items()}
    out["residual"] = {"absmax_max": max(res_abs), "kurtosis_mean": float(np.mean(res_k))}
    return out


@torch.no_grad()
def sink_mass(model, test_chunks, n_chunks: int, batch: int) -> dict:
    per_layer = []
    xs = test_chunks[:n_chunks]
    for i in range(0, len(xs), batch):
        xb = xs[i:i + batch].to(DEVICE)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            B, N = xb.shape
            pos = torch.arange(N, device=DEVICE).unsqueeze(0)
            h = model.drop(model.embed(xb) + model.pos(pos))
            for li, blk in enumerate(model.blocks):
                hn = blk.ln1(h)
                out, w = blk.attn(hn, return_weights=True)
                # mean mass on position 0 over batch/heads/queries
                m = float(w[..., 0].mean().item())
                if len(per_layer) <= li:
                    per_layer.append([])
                per_layer[li].append(m)
                h = h + out  # eval: dropout is identity; mirrors Block.forward
                h = h + blk.ffn(blk.ln2(h))
    means = [float(np.mean(v)) for v in per_layer]
    return {"per_layer": means, "mean": float(np.mean(means)), "max_layer": float(np.max(means))}


def int8_sim_hook(mod, inp):
    x = inp[0]
    s = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / 127.0
    xq = (x / s).round().clamp(-128, 127) * s
    return (xq,)


@torch.no_grad()
def run_one(run_id: str, variant: str, d_model: int, max_seq: int, ckpt_path: str) -> dict:
    e4.configure_fast_cuda()
    cfg = build_cfg(d_model, max_seq)
    model, (_vocab, stoi) = load_model(variant, cfg, ckpt_path)
    te = e4.load_tokens("test")
    test_c = e4.make_chunks(torch.tensor(e4.encode(te, stoi), dtype=torch.long), max_seq)
    n_stat = 256 if d_model <= 64 else 128
    b_stat = 16
    b_ppl = 32 if d_model <= 64 else 16
    print(f"--- {run_id} outlier stats ({n_stat} chunks) ---", flush=True)
    out = outlier_stats(model, test_c, n_stat, b_stat)
    print(f"--- {run_id} sink mass ---", flush=True)
    sink = sink_mass(model, test_c, 64, b_stat)
    print(f"--- {run_id} FP test PPL ---", flush=True)
    fp = e4.eval_ppl(model, test_c, b_ppl, amp="bf16")
    print(f"--- {run_id} INT8-sim test PPL ---", flush=True)
    handles = [m.register_forward_pre_hook(int8_sim_hook)
               for m in model.modules() if isinstance(m, nn.Linear)]
    q = e4.eval_ppl(model, test_c, b_ppl, amp="bf16")
    for h in handles:
        h.remove()
    print(f"  {run_id}: fp={fp['ppl']:.2f} int8sim={q['ppl']:.2f} "
          f"sink0={sink['mean']:.4f}", flush=True)
    return {
        "run_id": run_id, "variant": variant, "d_model": d_model,
        "ckpt": ckpt_path, "outliers": out, "sink": sink,
        "test_ppl_fp": float(fp["ppl"]), "test_ppl_int8sim": float(q["ppl"]),
        "int8_degradation": float(q["ppl"] / max(fp["ppl"], 1e-9) - 1.0),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    args = ap.parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    payload = {"protocol": (
        "C5: per-Linear input absmax/kurtosis + residual stats; sink mass on pos 0; "
        "per-token symmetric INT8 quantize-dequantize on every nn.Linear input "
        "(activation-only emulation) then test PPL. FP and INT8-sim in one process."
    ), "runs": {}}
    if os.path.isfile(OUT_JSON):
        with open(OUT_JSON, encoding="utf-8") as f:
            payload = json.load(f)
    t0 = time.time()
    for run_id, variant, d_model, max_seq, ckpt in MODELS:
        if args.only and run_id != args.only:
            continue
        if run_id in payload["runs"]:
            print(f"SKIP {run_id}: already complete", flush=True)
            continue
        payload["runs"][run_id] = run_one(run_id, variant, d_model, max_seq, ckpt)
        with open(OUT_JSON, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"saved {run_id} -> {OUT_JSON}", flush=True)
    print(f"done ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
