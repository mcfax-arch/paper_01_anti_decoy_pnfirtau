#!/usr/bin/env python3
"""Experiment 5: Topic-Coherent Context Override vs. Residual Stream Preservation
(Unanswerable Queries, Hallucination Rate, and Comparison Against Sparsemax, Entmax,
and Softmax Sinks under Pure Cross-Entropy Loss).

Directly tests the Shift-Invariance vs. Selective Silence theorem on an end-to-end
task where `sm_fir` and simplex-projected `entmax` fail:
  - Each sequence b (length T=48) represents a topic-coherent document containing
    M=3 non-bijective (key, val) pairs at random interior positions (k_m in 0..7,
    v_m in 0..7), with document background tokens drawn from the document's topic
    values v_sub. Every token in the context carries a unit-norm value representation
    from the 8 value classes (so no context token has V=0).
  - The query at t=T-1 carries a queried key k* in 0..7 AND a prior/default value
    v_default not in v_sub already present in its residual stream:
      y_query = V(v_default) + alpha_ctx * sum_{j=0}^{T-2} W[query, j] V[j]
      (alpha_ctx = 2.5).
  - In 50% of examples (Answerable / Context Override): k* is one of the 3 context
    keys k_m; the attention head must retrieve and override the residual stream to
    predict the context value v_m.
  - In 50% of examples (Unanswerable / Residual Preservation): k* is absent from the
    context; the attention head must stay SILENT (sum_j W[query, j] -> 0) so that
    y_query preserves v_default without being overwritten by a hallucinated document
    topic value from v_sub!
  - Trained with 100% pure Cross-Entropy loss on the output logits (ZERO auxiliary
    penalty on attention weights).

Compares 9 causal FIR attention normalizers across 3 random seeds (1234, 2025, 3141):
  1. `sm_fir`           : Softmax + causal FIR on K (So et al., 2021 Primer-style)
  2. `sparsemax_fir`    : Sparsemax (alpha=2, p=1 simplex projection, Martins & Astudillo 2016) + FIR
  3. `entmax15_fir`     : 1.5-Entmax (alpha=1.5, p=2 simplex projection, Peters et al. 2019) + FIR
  4. `entmax_p10_fir`   : 1.1-Entmax (alpha=1.1, p=10 simplex projection) + FIR
  5. `softmax1_fir`     : Off-by-one Softmax_1 with fixed tau=0 (Miller 2023) + FIR
  6. `softmax_sink_fir` : Quiet Softmax with learnable sink logit tau (Bondarenko et al. 2023) + FIR
  7. `pn_fir`           : Uncalibrated Power-Norm (p=10, fixed tau=0, gamma=0.5) + FIR
  8. `pn_fir_c`         : Row-mean centered Power-Norm (p=10, gamma=0.5) + FIR
  9. `pn_fir_tau`       : Learnable threshold tau + Power-Norm (p=10, gamma=0.5) + FIR
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.pnfirtau_attention import causal_fir, causal_gaussian_taps, power_norm_peak_gate


def sparsemax(scores: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Exact projection onto the probability simplex (Martins & Astudillo, ICML 2016)."""
    z = scores - scores.amax(dim=dim, keepdim=True)
    z_sorted, _ = torch.sort(z, dim=dim, descending=True)
    n = z.shape[dim]
    k = torch.arange(1, n + 1, device=z.device, dtype=z.dtype)
    view_shape = [1] * z.ndim
    view_shape[dim] = n
    k = k.view(view_shape)
    cssv = torch.cumsum(z_sorted, dim=dim) - 1.0
    cond = z_sorted - cssv / k > 0
    k_z = cond.sum(dim=dim, keepdim=True).to(z.dtype).clamp(min=1.0)
    tau_z = torch.gather(cssv, dim, k_z.long() - 1) / k_z
    return torch.clamp(z - tau_z, min=0.0)


class _EntmaxPowerFunction(torch.autograd.Function):
    """Exact simplex-projected power-norm: w_j = max(s_j - tau(s), 0)^p with sum_j w_j = 1.

    Corresponds to Tsallis alpha-entmax with alpha = 1 + 1/p (Peters et al., ACL 2019):
      - p = 2.0  -> alpha = 1.5 (1.5-entmax)
      - p = 10.0 -> alpha = 1.1 (simplex-projected p=10 power-norm)
    """

    @staticmethod
    def forward(ctx, scores: torch.Tensor, p: float = 2.0) -> torch.Tensor:
        z = scores - scores.amax(dim=-1, keepdim=True)
        n = z.shape[-1]
        lo = torch.full_like(z[..., :1], -1.0)
        hi = torch.full_like(z[..., :1], -(n ** (-1.0 / p)))
        for _ in range(35):
            mid = 0.5 * (lo + hi)
            pos = torch.clamp(z - mid, min=0.0)
            s = pos.pow(p).sum(dim=-1, keepdim=True)
            lo = torch.where(s > 1.0, mid, lo)
            hi = torch.where(s <= 1.0, mid, hi)
        tau = 0.5 * (lo + hi)
        w = torch.clamp(z - tau, min=0.0).pow(p)
        w = w / (w.sum(dim=-1, keepdim=True) + 1e-12)
        ctx.save_for_backward(w)
        ctx.p = p
        return w

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (w,) = ctx.saved_tensors
        p = ctx.p
        supp = w > 1e-12
        g = torch.where(supp, w.clamp(min=1e-12).pow(1.0 - 1.0 / p), torch.zeros_like(w))
        g_sum = g.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        dot = (grad_output * g).sum(dim=-1, keepdim=True)
        grad_scores = p * g * (grad_output - dot / g_sum)
        return grad_scores, None


def entmax_power(scores: torch.Tensor, p: float = 2.0) -> torch.Tensor:
    return _EntmaxPowerFunction.apply(scores, p)


N_CLASSES = 8
VOCAB_SIZE = 24
SEQ_LEN = 48
M_PAIRS = 3


def sample_batch(
    batch_size: int,
    device: torch.device,
    rng: np.random.Generator,
    mode: str = "mixed",  # "mixed" (50/50), "answerable", "unanswerable"
):
    seq = np.zeros((batch_size, SEQ_LEN), dtype=np.int64)
    default_val = np.zeros((batch_size,), dtype=np.int64)
    target = np.zeros((batch_size,), dtype=np.int64)
    val_pos = np.full((batch_size,), -1, dtype=np.int64)
    key_pos = np.full((batch_size,), -1, dtype=np.int64)

    for b in range(batch_size):
        k_sub = rng.choice(range(0, 8), size=M_PAIRS, replace=False)
        v_sub = rng.choice(range(0, 8), size=M_PAIRS, replace=False)
        # Topic-coherent document background: filler tokens carry values from v_sub
        seq[b] = 16 + rng.choice(v_sub, size=SEQ_LEN, replace=True)

        slots = []
        cands = list(range(2, 39))
        rng.shuffle(cands)
        for c in cands:
            if all(abs(c - s) >= 3 for s in slots):
                slots.append(c)
                if len(slots) == M_PAIRS:
                    break
        for m in range(M_PAIRS):
            seq[b, slots[m]] = k_sub[m]          # Key token in 0..7
            seq[b, slots[m] + 1] = v_sub[m] + 8  # Value token in 8..15

        rem_vals = [v for v in range(0, 8) if v not in v_sub]
        d_val = int(rng.choice(rem_vals))
        default_val[b] = d_val

        ans = (
            True
            if mode == "answerable"
            else (False if mode == "unanswerable" else bool(rng.integers(0, 2) == 1))
        )
        if ans:
            idx = int(rng.integers(0, M_PAIRS))
            q_key = int(k_sub[idx])
            target[b] = int(v_sub[idx])
            key_pos[b] = slots[idx]
            val_pos[b] = slots[idx] + 1
        else:
            absent_keys = [k for k in range(0, 8) if k not in k_sub]
            q_key = int(rng.choice(absent_keys))
            target[b] = d_val

        seq[b, SEQ_LEN - 1] = q_key

    return (
        torch.tensor(seq, device=device, dtype=torch.long),
        torch.tensor(default_val, device=device, dtype=torch.long),
        torch.tensor(target, device=device, dtype=torch.long),
        torch.tensor(key_pos, device=device, dtype=torch.long),
        torch.tensor(val_pos, device=device, dtype=torch.long),
    )


class NoSinkResidualModel(nn.Module):
    def __init__(
        self,
        variant: str,
        d_model: int = 32,
        fir_k: int = 7,
        power_p: float = 10.0,
        silence_gamma: float = 0.5,
    ):
        super().__init__()
        self.variant = variant
        self.d_model = d_model
        self.power_p = power_p
        self.silence_gamma = silence_gamma

        self.qk_emb = nn.Embedding(VOCAB_SIZE, d_model)
        self.val_emb = nn.Embedding(N_CLASSES, d_model)
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)

        taps = causal_gaussian_taps(fir_k, torch.device("cpu"), torch.float32, 3.0)
        self.register_buffer("fir_taps", taps, persistent=True)

        if variant in ("pn_fir_tau", "softmax_sink_fir"):
            self.tau = nn.Parameter(torch.tensor(0.0))
        else:
            self.register_parameter("tau", None)

    def forward(self, seq: torch.Tensor, default_val: torch.Tensor, return_diag: bool = False):
        x_qk = self.qk_emb(seq)
        q = self.wq(x_qk)
        # Causal FIR applied strictly to K (matching src/pnfirtau_attention.py)
        k = causal_fir(self.wk(x_qk), self.fir_taps)

        # Every token in the context carries a unit-norm value from the 8 classes (seq % 8)
        v_ids = seq % N_CLASSES
        v = F.normalize(self.val_emb(v_ids), dim=-1)

        q_last = q[:, -1:, :]
        k_ctx = k[:, :-1, :]
        v_ctx = v[:, :-1, :]

        raw_scores = torch.matmul(q_last, k_ctx.transpose(-2, -1)) / math.sqrt(self.d_model)
        base_var = self.variant

        if base_var == "sm_fir":
            w = F.softmax(raw_scores, dim=-1)
        elif base_var == "sparsemax_fir":
            w = sparsemax(raw_scores, dim=-1)
        elif base_var == "entmax15_fir":
            w = entmax_power(raw_scores, p=2.0)
        elif base_var == "entmax_p10_fir":
            w = entmax_power(raw_scores, p=self.power_p)
        elif base_var == "softmax1_fir":
            zeros = torch.zeros_like(raw_scores[..., :1])
            w_aug = F.softmax(torch.cat([raw_scores, zeros], dim=-1), dim=-1)
            w = w_aug[..., :-1]
        elif base_var == "softmax_sink_fir":
            sink = self.tau.view(1, 1, 1).expand_as(raw_scores[..., :1])
            w_aug = F.softmax(torch.cat([raw_scores, sink], dim=-1), dim=-1)
            w = w_aug[..., :-1]
        elif base_var == "pn_fir":
            w = power_norm_peak_gate(raw_scores, p=self.power_p, gamma=self.silence_gamma, silence_mode="peak")
        elif base_var == "pn_fir_c":
            cal_scores = raw_scores - raw_scores.mean(dim=-1, keepdim=True)
            w = power_norm_peak_gate(cal_scores, p=self.power_p, gamma=self.silence_gamma, silence_mode="peak")
        elif base_var == "pn_fir_tau":
            cal_scores = raw_scores - self.tau
            w = power_norm_peak_gate(cal_scores, p=self.power_p, gamma=self.silence_gamma, silence_mode="peak")
        else:
            raise ValueError(f"Unknown variant: {self.variant}")

        attn_out = torch.matmul(w, v_ctx).squeeze(1)
        res_default = F.normalize(self.val_emb(default_val), dim=-1)

        y = res_default + 2.5 * attn_out
        v_bank = F.normalize(self.val_emb.weight, dim=-1)
        logits = 12.0 * torch.matmul(y, v_bank.t())

        if return_diag:
            return logits, w.squeeze(1), raw_scores.squeeze(1)
        return logits


def run_single(variant: str, seed: int, device: torch.device, epochs: int = 2000, batch_size: int = 128):
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    model = NoSinkResidualModel(variant=variant).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-3, weight_decay=1e-4)

    t0 = time.perf_counter()
    model.train()
    for ep in range(1, epochs + 1):
        mode = "answerable" if ep <= 400 else "mixed"
        seq, dval, target, _, _ = sample_batch(batch_size, device, rng, mode=mode)
        logits = model(seq, dval)
        loss = F.cross_entropy(logits, target)
        opt.zero_grad()
        loss.backward()
        opt.step()
    train_s = time.perf_counter() - t0

    model.eval()
    eval_rng = np.random.default_rng(seed + 99999)
    with torch.no_grad():
        seq_ans, dval_ans, tgt_ans, kpos_ans, vpos_ans = sample_batch(2000, device, eval_rng, mode="answerable")
        logits_ans, w_ans, s_ans = model(seq_ans, dval_ans, return_diag=True)
        pred_ans = logits_ans.argmax(dim=-1)
        override_acc = (pred_ans == tgt_ans).float().mean().item()
        sum_w_ans = w_ans.sum(dim=-1).mean().item()
        b_idx = torch.arange(2000, device=device)
        w_val_slot = w_ans[b_idx, vpos_ans].mean().item()
        w_key_slot = w_ans[b_idx, kpos_ans].mean().item()

        seq_un, dval_un, tgt_un, _, _ = sample_batch(2000, device, eval_rng, mode="unanswerable")
        logits_un, w_un, s_un = model(seq_un, dval_un, return_diag=True)
        pred_un = logits_un.argmax(dim=-1)
        preserve_acc = (pred_un == tgt_un).float().mean().item()
        hallucination_rate = (pred_un != tgt_un).float().mean().item()
        sum_w_un = w_un.sum(dim=-1).mean().item()
        max_w_un = w_un.amax(dim=-1).mean().item()

        overall_acc = 0.5 * (override_acc + preserve_acc)
        silence_gap = sum_w_ans - sum_w_un
        tau_val = float(model.tau.item()) if model.tau is not None else None

    return {
        "variant": variant,
        "seed": seed,
        "train_s": train_s,
        "overall_acc": overall_acc,
        "override_acc": override_acc,
        "preserve_acc": preserve_acc,
        "hallucination_rate": hallucination_rate,
        "sum_w_answerable": sum_w_ans,
        "sum_w_unanswerable": sum_w_un,
        "max_w_unanswerable": max_w_un,
        "silence_gap": silence_gap,
        "w_val_slot": w_val_slot,
        "w_key_slot": w_key_slot,
        "raw_s_max_ans": s_ans.amax(dim=-1).mean().item(),
        "raw_s_max_un": s_un.amax(dim=-1).mean().item(),
        "tau": tau_val,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    variants = [
        "sm_fir",
        "sparsemax_fir",
        "entmax15_fir",
        "entmax_p10_fir",
        "softmax1_fir",
        "softmax_sink_fir",
        "pn_fir",
        "pn_fir_c",
        "pn_fir_tau",
    ]
    seeds = [1234, 2025, 3141]
    all_runs = {v: [] for v in variants}

    print(f"Running Experiment 5 on device={device} across seeds={seeds}...")
    for v in variants:
        for s in seeds:
            r = run_single(v, s, device)
            all_runs[v].append(r)
            print(
                f"  {v:23s} [seed={s}] | overall={r['overall_acc']*100:5.1f}% "
                f"| override={r['override_acc']*100:5.1f}% | preserve={r['preserve_acc']*100:5.1f}% "
                f"(halluc={r['hallucination_rate']*100:5.1f}%) | Wm={r['sum_w_answerable']:.3f} "
                f"Wun={r['sum_w_unanswerable']:.3f} gap={r['silence_gap']:+.3f} | w_val={r['w_val_slot']:.3f} "
                f"| tau={r['tau']}"
            )

    agg = {}
    metrics = [
        "overall_acc",
        "override_acc",
        "preserve_acc",
        "hallucination_rate",
        "sum_w_answerable",
        "sum_w_unanswerable",
        "max_w_unanswerable",
        "silence_gap",
        "w_val_slot",
        "w_key_slot",
        "raw_s_max_ans",
        "raw_s_max_un",
    ]
    for v in variants:
        runs = all_runs[v]
        entry = {}
        for m in metrics:
            vals = [float(x[m]) for x in runs]
            entry[m] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "per_seed": vals,
            }
        if runs[0]["tau"] is not None:
            tvals = [float(x["tau"]) for x in runs]
            entry["tau"] = {
                "mean": float(np.mean(tvals)),
                "std": float(np.std(tvals)),
                "per_seed": tvals,
            }
        else:
            entry["tau"] = None
        agg[v] = entry

    out_path = ROOT / "results" / "exp5_abstention_and_entmax.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"seeds": seeds, "aggregated_3seed": agg, "per_run": all_runs}, f, indent=2)
    print(f"\nSaved Experiment 5 results to {out_path}")


if __name__ == "__main__":
    main()
