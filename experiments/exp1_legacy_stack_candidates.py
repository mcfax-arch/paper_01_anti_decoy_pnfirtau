#!/usr/bin/env python3
"""Experiment 1: Self-Contained 13-Stack Benchmark + Diagnostic Audit.

Consolidates `I:\\FFT_Attention\\experiments\\stack_candidates\\experiment.py` and
`AUDIT_PN_FIR_FFT.md` diagnostics into a single deterministic script across 3 seeds
(1234, 2025, 3141).

Evaluates all 13 upstream stacks:
  sm, sm_fft, sm_fir, pn, pn_c, pn_tau, pn_fft, pn_fft_c,
  pn_fir, pn_fir_c, pn_fir_tau, qe_pn, qe_pn_fft
plus explicit un-gated variants (`pn_ungated`, `pn_fir_ungated`, `pn_fir_c_ungated`,
`pn_fir_tau_ungated` with `silence_gamma=None`) to separate score calibration from
peak-gate attenuation (Audit M7).

Also records:
  - Attention weight concentration at `KEY_POS=0` vs `VALUE_POS=1` (Audit B3)
  - Multi-decoy accuracy for n_d in {1, 2, 4, 8} (Audit H2)
  - Pre-calibration score statistics (`mean_S`, `max_S`, `min_S`, `frac_pos`, `rows_all_neg`)
    for Figure 2 (Audit B2)
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.pnfirtau_attention import (  # noqa: E402
    causal_fir,
    causal_gaussian_taps,
    fft_lowpass_noncausal,
    power_norm_peak_gate,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = os.path.join(ROOT, "results")
OUT_JSON = os.path.join(RESULTS_DIR, "exp1_legacy_13stacks.json")

KEY_POS, VALUE_POS, DECOY_POS = 0, 1, 5
KEYS = [1, 2, 3, 4, 5]
VALS = [6, 7, 8, 9, 10]
CTX_LEN = 14


@dataclass
class Cfg:
    vocab: int = 24
    d: int = 32
    seq_len: int = 15
    epochs: int = 2000
    batch: int = 128
    lr: float = 5e-3
    eval_n: int = 2000
    power_p: float = 10.0
    qe_alpha: float = 2.0
    qe_rank: int = 8
    fir_k: int = 7
    fft_sigma_div: float = 6.0
    silence_gamma: float | None = 0.5
    silence_mode: str = "peak"
    tau_init: float = 0.0
    perf_batch: int = 64
    perf_seq: int = 128
    perf_warmup: int = 20
    perf_iters: int = 50
    seeds: tuple[int, ...] = (1234, 2025, 3141)


STACKS = [
    ("sm", "Softmax + KV, no filter"),
    ("sm_fft", "Softmax + FFT low-pass on K (non-causal)"),
    ("sm_fir", "Softmax + causal FIR on K"),
    ("pn", "Power-norm + QK, no filter, no cal (gamma=0.5)"),
    ("pn_c", "Power-norm + QK, center cal"),
    ("pn_tau", "Power-norm + QK, learned tau"),
    ("pn_fft", "Power-norm + FFT on K"),
    ("pn_fft_c", "Power-norm + FFT + center"),
    ("pn_fir", "Power-norm + causal FIR (no cal)"),
    ("pn_fir_c", "Power-norm + causal FIR + center"),
    ("pn_fir_tau", "Power-norm + causal FIR + learned tau"),
    ("qe_pn", "QE + Power-norm, no filter"),
    ("qe_pn_fft", "QE + Power-norm + FFT"),
    # Un-gated ablations (gamma=None) to isolate calibration from silence gate
    ("pn_ungated", "Power-norm + QK, no filter, gamma=None"),
    ("pn_fir_ungated", "Power-norm + causal FIR, no cal, gamma=None"),
    ("pn_fir_c_ungated", "Power-norm + causal FIR + center, gamma=None"),
    ("pn_fir_tau_ungated", "Power-norm + causal FIR + learned tau, gamma=None"),
]


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_stack(stack_id: str) -> tuple[str, str, str, bool]:
    """Return (norm, filter, cal, ungated)."""
    sid = stack_id
    ungated = False
    if sid.endswith("_ungated"):
        ungated = True
        sid = sid[: -len("_ungated")]

    cal = "none"
    if sid.endswith("_tau"):
        cal = "tau"
        sid = sid[: -len("_tau")]
    elif sid.endswith("_c"):
        cal = "center"
        sid = sid[: -len("_c")]

    filt = "none"
    if sid.endswith("_fft"):
        filt = "fft"
        sid = sid[: -len("_fft")]
    elif sid.endswith("_fir"):
        filt = "fir"
        sid = sid[: -len("_fir")]

    if sid in ("sm", "softmax"):
        norm = "sm"
    elif sid == "pn":
        norm = "pn"
    elif sid in ("qe_pn", "qe"):
        norm = "qe"
    else:
        raise ValueError(f"cannot parse stack_id={stack_id!r}")
    return norm, filt, cal, ungated


def qe_scores(q: torch.Tensor, k: torch.Tensor, alpha: float, rank: int) -> torch.Tensor:
    dots = torch.bmm(q, k.transpose(1, 2))
    qn = q.square().sum(-1, keepdim=True)
    kn = k.square().sum(-1, keepdim=True)
    return (2.0 * alpha) * dots - alpha * qn - alpha * kn.transpose(1, 2) + rank


class StackModel(nn.Module):
    """Exact upstream StackModel from stack_candidates/experiment.py."""

    def __init__(self, stack_id: str, cfg: Cfg):
        super().__init__()
        self.stack_id = stack_id
        self.cfg = cfg
        self.norm, self.filt, self.cal, self.ungated = parse_stack(stack_id)
        self.emb = nn.Embedding(cfg.vocab, cfg.d)
        self.query = nn.Parameter(torch.randn(1, cfg.d) * 0.1)
        self.head = nn.Linear(cfg.d, cfg.vocab)

        if self.norm == "qe":
            r = cfg.qe_rank
            self.q_proj = nn.Linear(cfg.d, r, bias=False)
            self.k_proj = nn.Linear(cfg.d, r, bias=False)
            self.v_proj = nn.Linear(cfg.d, cfg.d, bias=False)
        else:
            self.qkv = nn.Linear(cfg.d, 3 * cfg.d, bias=False)

        self.score_tau = nn.Parameter(torch.tensor(float(cfg.tau_init)))
        self.register_buffer(
            "fir_taps",
            causal_gaussian_taps(cfg.fir_k, torch.device("cpu"), torch.float32),
            persistent=False,
        )

    def _filter_k(self, k: torch.Tensor) -> torch.Tensor:
        if self.filt == "fft":
            return fft_lowpass_noncausal(k, self.cfg.fft_sigma_div)
        if self.filt == "fir":
            taps = self.fir_taps.to(device=k.device, dtype=k.dtype)
            return causal_fir(k, taps)
        return k

    def _raw_scores(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        if self.norm == "qe":
            return qe_scores(q, k, self.cfg.qe_alpha, self.cfg.qe_rank)
        scale = 1.0 / math.sqrt(q.shape[-1])
        return torch.bmm(q, k.transpose(1, 2)) * scale

    def _effective_gamma(self) -> float | None:
        return None if self.ungated else self.cfg.silence_gamma

    def _weights(self, q: torch.Tensor, k: torch.Tensor, return_scores: bool = False):
        s_raw = self._raw_scores(q, k)
        if self.norm == "sm":
            w = F.softmax(s_raw, dim=-1)
            return (w, s_raw, s_raw) if return_scores else w
        s = s_raw
        if self.cal == "tau":
            s = s - self.score_tau
        elif self.cal == "center":
            s = s - s.mean(dim=-1, keepdim=True)
        w = power_norm_peak_gate(
            s, self.cfg.power_p, self._effective_gamma(), self.cfg.silence_mode
        )
        return (w, s_raw, s) if return_scores else w

    def forward(self, ctx_e: torch.Tensor, return_scores: bool = False):
        B = ctx_e.shape[0]
        full = torch.cat([ctx_e, self.query.expand(B, -1, -1)], dim=1)
        if self.norm == "qe":
            q = torch.sigmoid(self.q_proj(full))
            k = torch.sigmoid(self.k_proj(full))
            v = self.v_proj(full)
        else:
            q, k, v = self.qkv(full).chunk(3, dim=-1)
        k = self._filter_k(k)
        if return_scores:
            w, s_raw, s_cal = self._weights(q, k, return_scores=True)
            out = torch.bmm(w, v)
            logits = self.head(out[:, -1])
            return logits, w, s_raw, s_cal
        w = self._weights(q, k)
        out = torch.bmm(w, v)
        logits = self.head(out[:, -1])
        return logits, w


def gen(cfg: Cfg, n: int, g: torch.Generator):
    x = torch.randint(1, cfg.vocab, (n, CTX_LEN), generator=g)
    idx = torch.randint(0, len(KEYS), (n,), generator=g)
    keys_t = torch.tensor([KEYS[i] for i in idx])
    vals_t = torch.tensor([VALS[i] for i in idx])
    x[:, KEY_POS] = keys_t
    x[:, VALUE_POS] = vals_t
    return x, vals_t


def train_model(model: StackModel, cfg: Cfg, g: torch.Generator) -> float:
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.CrossEntropyLoss()
    t0 = time.time()
    for _ in range(cfg.epochs):
        ctx, tgt = gen(cfg, cfg.batch, g)
        ctx = ctx.to(DEVICE)
        tgt = tgt.to(DEVICE)
        logits, _ = model(model.emb(ctx))
        loss = loss_fn(logits, tgt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    return time.time() - t0


@torch.no_grad()
def decoy_acc(
    model: StackModel,
    ctx: torch.Tensor,
    tgt: torch.Tensor,
    eps: float,
    n_decoys: int = 1,
    g: torch.Generator | None = None,
) -> float:
    model.eval()
    xe = model.emb(ctx).clone()
    ke = xe[:, KEY_POS].clone()
    decoy_positions = [5] if n_decoys == 1 else [min(2 + d, CTX_LEN - 1) for d in range(n_decoys)]
    for pos in decoy_positions:
        noise = torch.randn(ke.shape, device=ke.device, dtype=ke.dtype, generator=g)
        d = ke + noise * eps
        d = d / d.norm(dim=-1, keepdim=True).clamp_min(1e-8) * ke.norm(dim=-1, keepdim=True)
        xe[:, pos] = d
    pred = model(xe)[0].argmax(-1)
    return float((pred == tgt).float().mean().item())


@torch.no_grad()
def eval_decoy(model: StackModel, cfg: Cfg, seed: int) -> dict:
    model.eval()
    g = torch.Generator().manual_seed(seed + 99)
    ctx, tgt = gen(cfg, cfg.eval_n, g)
    ctx, tgt = ctx.to(DEVICE), tgt.to(DEVICE)
    xe = model.emb(ctx)
    logits, w, s_raw, s_cal = model(xe, return_scores=True)
    clean = float((logits.argmax(-1) == tgt).float().mean().item())
    wq = w[:, -1, :]  # [B, 15] attention weights on last (query) row
    sum_w_match = float(wq.sum(dim=-1).mean().item())

    # Attention position diagnostics (Audit B3)
    w_key_pos = float(wq[:, KEY_POS].mean().item())
    w_val_pos = float(wq[:, VALUE_POS].mean().item())
    argmax_pos = wq.argmax(dim=-1)
    argmax_at_key_frac = float((argmax_pos == KEY_POS).float().mean().item())
    argmax_at_val_frac = float((argmax_pos == VALUE_POS).float().mean().item())

    # Score diagnostics on last query row (Audit B2 & AUDIT_PN_FIR_FFT.md Section 2)
    sq_raw = s_raw[:, -1, :]
    sq_cal = s_cal[:, -1, :]
    score_diag = {
        "raw_max_mean": float(sq_raw.max(dim=-1).values.mean().item()),
        "raw_mean_mean": float(sq_raw.mean(dim=-1).mean().item()),
        "raw_min_mean": float(sq_raw.min(dim=-1).values.mean().item()),
        "raw_frac_pos": float((sq_raw > 0).float().mean().item()),
        "raw_rows_all_neg": float((sq_raw.max(dim=-1).values <= 0).float().mean().item()),
        "cal_max_mean": float(sq_cal.max(dim=-1).values.mean().item()),
        "cal_mean_mean": float(sq_cal.mean(dim=-1).mean().item()),
        "cal_min_mean": float(sq_cal.min(dim=-1).values.mean().item()),
        "cal_frac_pos": float((sq_cal > 0).float().mean().item()),
        "cal_rows_all_neg": float((sq_cal.max(dim=-1).values <= 0).float().mean().item()),
    }

    # Key-replace test (upstream line 367-371: replace KEY_POS with random filler token)
    xe2 = xe.clone()
    rand_tok = torch.randint(11, cfg.vocab, (cfg.eval_n,), generator=g)
    xe2[:, KEY_POS] = model.emb(rand_tok.to(DEVICE))
    kr = float((model(xe2)[0].argmax(-1) == tgt).float().mean().item())

    # Single-decoy sweep
    d005 = decoy_acc(model, ctx, tgt, 0.05, n_decoys=1)
    d02 = decoy_acc(model, ctx, tgt, 0.2, n_decoys=1)
    d05 = decoy_acc(model, ctx, tgt, 0.5, n_decoys=1)
    d10 = decoy_acc(model, ctx, tgt, 1.0, n_decoys=1)

    # Multi-decoy sweep at eps=0.2 (Audit H2)
    multi = {}
    for nd in (1, 2, 4, 8):
        multi[f"nd_{nd}"] = decoy_acc(model, ctx, tgt, 0.2, n_decoys=nd)

    # No-match: destroy KEY+VALUE pair
    xe_nm = xe.clone()
    junk = torch.randint(11, cfg.vocab, (cfg.eval_n, 2), generator=g)
    xe_nm[:, KEY_POS] = model.emb(junk[:, 0].to(DEVICE))
    xe_nm[:, VALUE_POS] = model.emb(junk[:, 1].to(DEVICE))
    sum_w_nomatch = float(model(xe_nm)[1][:, -1, :].sum(dim=-1).mean().item())

    # Pure noise context
    x_noise = torch.randint(11, cfg.vocab, (cfg.eval_n, CTX_LEN), generator=g).to(DEVICE)
    sum_w_noise = float(model(model.emb(x_noise))[1][:, -1, :].sum(dim=-1).mean().item())

    tau_val = float(model.score_tau.detach().item()) if model.cal == "tau" else None
    return {
        "clean": clean,
        "kr": kr,
        "d@0.05": d005,
        "d@0.2": d02,
        "d@0.5": d05,
        "d@1.0": d10,
        "delta": d10 - d005,
        "multi_decoy": multi,
        "sum_w_match": sum_w_match,
        "sum_w_nomatch": sum_w_nomatch,
        "sum_w_noise": sum_w_noise,
        "silence_gap": sum_w_match - sum_w_nomatch,
        "silence_ratio_noise": sum_w_match / max(sum_w_noise, 1e-9),
        "tau": tau_val,
        "attention_pos": {
            "w_key_pos": w_key_pos,
            "w_val_pos": w_val_pos,
            "argmax_at_key_frac": argmax_at_key_frac,
            "argmax_at_val_frac": argmax_at_val_frac,
        },
        "score_diag": score_diag,
    }


@torch.no_grad()
def causal_probe(model: StackModel, cfg: Cfg) -> dict:
    """Measure key-filter causal leakage on past positions when perturbing last context token."""
    model.eval()
    B, N = 8, CTX_LEN
    x = torch.randint(1, cfg.vocab, (B, N), device=DEVICE)
    xp = x.clone()
    xp[:, -1] = (xp[:, -1] + 3) % (cfg.vocab - 1) + 1

    e1 = model.emb(x)
    e2 = model.emb(xp)
    if model.norm == "qe":
        k1 = torch.sigmoid(model.k_proj(e1))
        k2 = torch.sigmoid(model.k_proj(e2))
    else:
        _, k1, _ = model.qkv(e1).chunk(3, dim=-1)
        _, k2, _ = model.qkv(e2).chunk(3, dim=-1)
    kf1 = model._filter_k(k1)
    kf2 = model._filter_k(k2)
    past_key_leak = float((kf1[:, :-1, :] - kf2[:, :-1, :]).abs().max().item())
    return {
        "past_key_leak": past_key_leak,
        "is_causal": past_key_leak <= 1e-6,
    }


def perf_bench(model: StackModel, cfg: Cfg) -> dict:
    model.train()
    B, S = cfg.perf_batch, cfg.perf_seq
    ctx_len = min(S, 127)
    x = torch.randint(1, cfg.vocab, (B, ctx_len), device=DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    tgt = torch.randint(0, cfg.vocab, (B,), device=DEVICE)

    def fwd():
        return model(model.emb(x))[0]

    def step():
        opt.zero_grad(set_to_none=True)
        logits, _ = model(model.emb(x))
        loss = F.cross_entropy(logits, tgt)
        loss.backward()
        opt.step()
        return loss

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    for _ in range(cfg.perf_warmup):
        fwd()
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(cfg.perf_iters):
        fwd()
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    fwd_ms = (time.perf_counter() - t0) * 1000 / cfg.perf_iters

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    for _ in range(cfg.perf_warmup):
        step()
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(cfg.perf_iters):
        step()
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    step_ms = (time.perf_counter() - t0) * 1000 / cfg.perf_iters
    peak_mb = torch.cuda.max_memory_allocated() / 1e6 if DEVICE.type == "cuda" else 0.0
    tok = B * (ctx_len + 1)
    return {
        "fwd_ms": float(fwd_ms),
        "step_ms": float(step_ms),
        "fwd_tok_s": float(tok / (fwd_ms / 1000)),
        "step_tok_s": float(tok / (step_ms / 1000)),
        "peak_mb": float(peak_mb),
    }


def aggregate_seeds(per_seed: dict[str, dict]) -> dict:
    seeds = list(per_seed.keys())

    def ms(vals):
        arr = np.array(vals, dtype=np.float64)
        return {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "per_seed": [float(v) for v in vals],
        }

    scalar_keys = [
        "clean",
        "kr",
        "d@0.05",
        "d@0.2",
        "d@0.5",
        "d@1.0",
        "delta",
        "sum_w_match",
        "sum_w_nomatch",
        "sum_w_noise",
        "silence_gap",
        "silence_ratio_noise",
    ]
    out = {k: ms([per_seed[s][k] for s in seeds]) for k in scalar_keys}
    out["multi_decoy"] = {
        k: ms([per_seed[s]["multi_decoy"][k] for s in seeds])
        for k in per_seed[seeds[0]]["multi_decoy"]
    }
    out["attention_pos"] = {
        k: ms([per_seed[s]["attention_pos"][k] for s in seeds])
        for k in per_seed[seeds[0]]["attention_pos"]
    }
    out["score_diag"] = {
        k: ms([per_seed[s]["score_diag"][k] for s in seeds])
        for k in per_seed[seeds[0]]["score_diag"]
    }
    if per_seed[seeds[0]]["tau"] is not None:
        out["tau"] = ms([per_seed[s]["tau"] for s in seeds])
    else:
        out["tau"] = None
    return out


def main() -> dict:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    cfg = Cfg()
    taps = causal_gaussian_taps(cfg.fir_k, torch.device("cpu"), torch.float32).flip(0).tolist()
    print("=" * 92)
    print(f"EXP 1: 13-Stack Canonical Benchmark + Diagnostics | device={DEVICE} | p={cfg.power_p}")
    print(f"FIR taps (present->past, K={cfg.fir_k}): {[round(t, 4) for t in taps]}")
    print("=" * 92)

    perf = {}
    causal = {}
    per_seed = {str(s): {} for s in cfg.seeds}

    for sid, desc in STACKS:
        set_seed(1234)
        m_perf = StackModel(sid, cfg).to(DEVICE)
        set_seed(1235)
        perf[sid] = perf_bench(m_perf, cfg)
        causal[sid] = causal_probe(m_perf, cfg)
        del m_perf

    for seed in cfg.seeds:
        print(f"\n--- Seed {seed} ---")
        for sid, desc in STACKS:
            set_seed(seed)
            g = torch.Generator().manual_seed(seed)
            model = StackModel(sid, cfg).to(DEVICE)
            train_s = train_model(model, cfg, g)
            res = eval_decoy(model, cfg, seed)
            res["train_s"] = float(train_s)
            per_seed[str(seed)][sid] = res
            print(
                f"  {sid:18s} clean={res['clean']*100:5.1f}% "
                f"d@0.5={res['d@0.5']*100:5.1f}% d@1.0={res['d@1.0']*100:5.1f}% "
                f"nd8={res['multi_decoy']['nd_8']*100:5.1f}% "
                f"kr={res['kr']*100:4.1f}% "
                f"Sw_m={res['sum_w_match']:.3f} Sw_n={res['sum_w_noise']:.3f} "
                f"gap={res['silence_gap']:+.3f} "
                f"w_k={res['attention_pos']['w_key_pos']:.3f} "
                f"w_v={res['attention_pos']['w_val_pos']:.3f}"
                + (f" tau={res['tau']:+.3f}" if res["tau"] is not None else ""),
                flush=True,
            )
            del model

    summary = {}
    for sid, desc in STACKS:
        summary[sid] = {
            "desc": desc,
            **aggregate_seeds({str(s): per_seed[str(s)][sid] for s in cfg.seeds}),
            "causal": causal[sid],
            "perf": perf[sid],
        }

    payload = {
        "config": asdict(cfg),
        "device": str(DEVICE)
        + (f" ({torch.cuda.get_device_name(0)})" if DEVICE.type == "cuda" else ""),
        "fir_taps_present_to_past": taps,
        "summary": summary,
        "per_seed": per_seed,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved {OUT_JSON}")
    return payload


if __name__ == "__main__":
    main()
