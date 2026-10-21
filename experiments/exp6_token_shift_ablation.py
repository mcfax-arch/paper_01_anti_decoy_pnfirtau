#!/usr/bin/env python3
"""Experiment 6: Token-Shift Ablation (C4 of PAPER_01_REWORK_PLAN.md).

Question: is the 7-tap causal FIR doing something a 2-tap RWKV-style token
shift cannot? If a 2-tap shift concedes the decoys, the 7-tap low-pass is
essential rather than ornamental.

Arms (filter varies, everything else fixed at the exp1/exp2 protocols):
  fixed 2-tap  [0.5, 0.5]                     ("shift2", parameter-free)
  learnable 2-tap mu (RWKV form):             ("shiftmu")
      k'_t = mu * k_t + (1 - mu) * k_{t-1}, mu = sigmoid(logit) per channel,
      logit initialised at 0 (mu = 0.5). Applied to K after projection, i.e. at
      the exact position of the 7-tap FIR, so the ablation isolates tap length
      and learnability, NOT placement. (True RWKV applies the shift before the
      R/K/V projections; that placement variant is out of scope here.)
  7-tap half-Gaussian FIR (K_f=7, sigma=2.0)   (control, rerun in-script)
  no filter                                    (control, rerun in-script)

Part A reuses the exp1 legacy protocol (KEY@0/VAl@1, decoy eps sweep,
multi-decoy n_d in {1,2,4,8}) with arms:
  sm, sm_fir, sm_shift2, sm_shiftmu, pn_fir_tau, pn_shift2_tau, pn_shiftmu_tau
Part B reuses the exp2 content-addressable protocol (M=3 random pairs,
eps sweep, multi-decoy) with arms:
  sm, sm_fir, sm_shiftmu, pn_fir_tau, pn_shiftmu_tau

Seeds (1234, 2025, 3141), same as exp1/exp2. Writes
results/exp6_token_shift_ablation.json.
"""
from __future__ import annotations

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
import exp1_legacy_stack_candidates as e1  # noqa: E402  (has __main__ guard)
import exp2_content_addressable_decoy as e2  # noqa: E402  (has __main__ guard)

DEVICE = e1.DEVICE
RESULTS_DIR = os.path.join(ROOT, "results")
OUT_JSON = os.path.join(RESULTS_DIR, "exp6_token_shift_ablation.json")
SEEDS = (1234, 2025, 3141)


def shift2_fixed(k: torch.Tensor) -> torch.Tensor:
    """Parameter-free 2-tap causal shift: k'_t = 0.5*k_t + 0.5*k_{t-1}."""
    kp = torch.zeros_like(k)
    kp[:, 1:, :] = k[:, :-1, :]
    return 0.5 * k + 0.5 * kp


def shiftmu(k: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
    """RWKV-form learnable shift: k'_t = mu*k_t + (1-mu)*k_{t-1}."""
    kp = torch.zeros_like(k)
    kp[:, 1:, :] = k[:, :-1, :]
    return mu * k + (1.0 - mu) * kp


# ---------------------------------------------------------------- Part A model
class ShiftStackModel(e1.StackModel):
    """exp1 StackModel with the K-filter swapped for 2-tap shift variants."""

    def __init__(self, stack_id: str, cfg: e1.Cfg):
        base = "sm_fir" if stack_id.startswith("sm") else "pn_fir_tau"
        super().__init__(base, cfg)
        self.stack_id = stack_id
        kind = "shiftmu" if "shiftmu" in stack_id else "shift2"
        self.filt = kind  # override dispatch set by parse_stack("..._fir")
        self.shift_kind = kind
        if kind == "shiftmu":
            self.shift_mu_logit = nn.Parameter(torch.zeros(cfg.d))
        else:
            self.register_buffer("shift_mu_logit", torch.zeros(cfg.d), persistent=False)

    def _filter_k(self, k: torch.Tensor) -> torch.Tensor:
        if self.shift_kind == "shift2":
            return shift2_fixed(k)
        mu = torch.sigmoid(self.shift_mu_logit).to(dtype=k.dtype, device=k.device)
        return shiftmu(k, mu.view(1, 1, -1))

    def mu_stats(self) -> dict:
        mu = torch.sigmoid(self.shift_mu_logit).detach().float()
        return {
            "mu_mean": float(mu.mean().item()),
            "mu_min": float(mu.min().item()),
            "mu_max": float(mu.max().item()),
        }


PARTA_STACKS = [
    ("sm", "Softmax, no filter (control)"),
    ("sm_fir", "Softmax + 7-tap FIR (control)"),
    ("sm_shift2", "Softmax + fixed 2-tap [0.5,0.5] shift"),
    ("sm_shiftmu", "Softmax + learnable RWKV-style mu shift"),
    ("pn_fir_tau", "Power-norm + 7-tap FIR + tau (control)"),
    ("pn_shift2_tau", "Power-norm + fixed 2-tap shift + tau"),
    ("pn_shiftmu_tau", "Power-norm + learnable mu shift + tau"),
]


def make_model_a(stack_id: str, cfg: e1.Cfg):
    if "shift" in stack_id:
        return ShiftStackModel(stack_id, cfg).to(DEVICE)
    return e1.StackModel(stack_id, cfg).to(DEVICE)


# ---------------------------------------------------------------- Part B model
class ShiftContentModel(e2.ContentAssocModel):
    """exp2 ContentAssocModel with the K-filter swapped for 2-tap shift variants."""

    def __init__(self, cfg: e2.ContentAssocConfig, stack: str):
        base = "sm_fir" if stack.startswith("sm") else "pn_fir_tau"
        super().__init__(cfg, base)
        self.stack = stack
        kind = "shiftmu" if "shiftmu" in stack else "shift2"
        self.shift_kind = kind
        if kind == "shiftmu":
            self.shift_mu_logit = nn.Parameter(torch.zeros(cfg.d_model))
        else:
            self.register_buffer(
                "shift_mu_logit", torch.zeros(cfg.d_model), persistent=False
            )

    def _filter_k(self, k: torch.Tensor) -> torch.Tensor:
        if self.shift_kind == "shift2":
            return shift2_fixed(k)
        mu = torch.sigmoid(self.shift_mu_logit).to(dtype=k.dtype, device=k.device)
        return shiftmu(k, mu.view(1, 1, -1))

    def forward_from_emb(self, e, q_tok, return_scores=False):
        B, T, D = e.shape
        q = self.q_proj(self.emb(q_tok))
        k = self.k_proj(e)
        k_f = self._filter_k(k)
        scores = torch.einsum("bd,btd->bt", q, k_f) / math.sqrt(D)
        if self._use_center():
            scores = scores - scores.mean(dim=-1, keepdim=True)
        if self._use_tau():
            scores = scores - self.score_tau
        if self.stack.startswith("sm"):
            w = F.softmax(scores, dim=-1)
        else:
            w = e2.power_norm_peak_gate(
                scores, p=self.cfg.p, gamma=self._silence_gamma(),
                silence_mode=self.cfg.silence_mode,
            )
        v = self.v_proj(e)
        ctx = torch.einsum("bt,btd->bd", w, v)
        logits = self.head(ctx)
        if return_scores:
            return logits, w, scores
        return logits, w

    def mu_stats(self) -> dict:
        mu = torch.sigmoid(self.shift_mu_logit).detach().float()
        return {
            "mu_mean": float(mu.mean().item()),
            "mu_min": float(mu.min().item()),
            "mu_max": float(mu.max().item()),
        }


PARTB_STACKS = [
    ("sm", "Softmax, no filter (control)"),
    ("sm_fir", "Softmax + 7-tap FIR (control)"),
    ("sm_shiftmu", "Softmax + learnable RWKV-style mu shift"),
    ("pn_fir_tau", "Power-norm + 7-tap FIR + tau (control)"),
    ("pn_shiftmu_tau", "Power-norm + learnable mu shift + tau"),
]


def train_one_b(cfg: e2.ContentAssocConfig, model: e2.ContentAssocModel) -> e2.ContentAssocModel:
    """Mirror of e2.train_one but with a caller-supplied model instance."""
    model = model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    for step in range(cfg.steps):
        x, q_tok, y, _, _, _ = e2.gen_content_batch(cfg, cfg.batch_size)
        logits, w = model(x, q_tok)
        loss = F.cross_entropy(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


@torch.no_grad()
def causal_probe_b(cfg: e2.ContentAssocConfig, model: e2.ContentAssocModel) -> dict:
    """Same idea as e1.causal_probe for the exp2 model layout."""
    model.eval()
    B, T = 8, cfg.seq_len
    x = torch.randint(17, cfg.vocab_size, (B, T), device=DEVICE)
    xp = x.clone()
    xp[:, -1] = (xp[:, -1] + 3) % (cfg.vocab_size - 1) + 1
    e1_ = model.emb(x)
    e2_ = model.emb(xp)
    k1 = model.k_proj(e1_)
    k2 = model.k_proj(e2_)
    kf1 = model._filter_k(k1) if hasattr(model, "_filter_k") else k1
    kf2 = model._filter_k(k2) if hasattr(model, "_filter_k") else k2
    leak = float((kf1[:, :-1, :] - kf2[:, :-1, :]).abs().max().item())
    return {"past_key_leak": leak, "is_causal": leak <= 1e-6}


def ms_of(vals: list[float]) -> dict:
    arr = np.array(vals, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "per_seed": [float(v) for v in vals],
    }


def main() -> dict:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    t_start = time.time()

    # ---------------- Part A ----------------
    print("=" * 92)
    print(f"EXP 6A: Token-Shift Ablation on exp1 legacy protocol | device={DEVICE}")
    print("=" * 92)
    cfg_a = e1.Cfg()
    per_seed_a = {str(s): {} for s in SEEDS}
    causal_a = {}
    for sid, desc in PARTA_STACKS:
        e1.set_seed(1234)
        m_probe = make_model_a(sid, cfg_a)
        causal_a[sid] = e1.causal_probe(m_probe, cfg_a)
        del m_probe
    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---")
        for sid, desc in PARTA_STACKS:
            e1.set_seed(seed)
            g = torch.Generator().manual_seed(seed)
            model = make_model_a(sid, cfg_a)
            train_s = e1.train_model(model, cfg_a, g)
            res = e1.eval_decoy(model, cfg_a, seed)
            res["train_s"] = float(train_s)
            if hasattr(model, "mu_stats"):
                res["mu"] = model.mu_stats()
            per_seed_a[str(seed)][sid] = res
            mu_str = (
                f" mu={res['mu']['mu_mean']:.3f}[{res['mu']['mu_min']:.2f},{res['mu']['mu_max']:.2f}]"
                if "mu" in res and "shiftmu" in sid
                else ""
            )
            print(
                f"  {sid:15s} clean={res['clean']*100:5.1f}% "
                f"d@0.05={res['d@0.05']*100:5.1f}% d@0.5={res['d@0.5']*100:5.1f}% "
                f"d@1.0={res['d@1.0']*100:5.1f}% nd8={res['multi_decoy']['nd_8']*100:5.1f}% "
                f"kr={res['kr']*100:4.1f}% leak={causal_a[sid]['past_key_leak']:.1e}{mu_str}",
                flush=True,
            )
            del model
    summary_a = {}
    for sid, desc in PARTA_STACKS:
        agg = e1.aggregate_seeds({str(s): per_seed_a[str(s)][sid] for s in SEEDS})
        entry = {"desc": desc, **agg, "causal": causal_a[sid]}
        if "mu" in per_seed_a[str(SEEDS[0])][sid]:
            entry["mu"] = {
                k: ms_of([per_seed_a[str(s)][sid]["mu"][k] for s in SEEDS])
                for k in ("mu_mean", "mu_min", "mu_max")
            }
        summary_a[sid] = entry

    # ---------------- Part B ----------------
    print("\n" + "=" * 92)
    print(f"EXP 6B: Token-Shift Ablation on exp2 content-addressable protocol | device={DEVICE}")
    print("=" * 92)
    cfg_b = e2.ContentAssocConfig()
    per_seed_b = {str(s): {} for s in SEEDS}
    causal_b = {}
    for sid, desc in PARTB_STACKS:
        e2.set_seed(1234)
        if "shift" in sid:
            m_probe = ShiftContentModel(cfg_b, sid).to(DEVICE)
        else:
            m_probe = e2.ContentAssocModel(cfg_b, sid).to(DEVICE)
        causal_b[sid] = causal_probe_b(cfg_b, m_probe)
        del m_probe
    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---")
        for sid, desc in PARTB_STACKS:
            e2.set_seed(seed)
            if "shift" in sid:
                model = ShiftContentModel(cfg_b, sid)
            else:
                model = e2.ContentAssocModel(cfg_b, sid)
            model = train_one_b(cfg_b, model)
            res = e2.eval_model(cfg_b, model)
            if hasattr(model, "mu_stats"):
                res["mu"] = model.mu_stats()
            per_seed_b[str(seed)][sid] = res
            mu_str = (
                f" mu={res['mu']['mu_mean']:.3f}[{res['mu']['mu_min']:.2f},{res['mu']['mu_max']:.2f}]"
                if "mu" in res and "shiftmu" in sid
                else ""
            )
            print(
                f"  {sid:15s} clean={res['clean']*100:5.1f}% "
                f"eps0.2={res['decoy']['eps_0.2']*100:5.1f}% "
                f"nd4={res['multi_decoy']['nd_4']*100:5.1f}% "
                f"nd8={res['multi_decoy']['nd_8']*100:5.1f}% "
                f"w_key={res['attention_target']['w_key_pos']:.3f} "
                f"w_val={res['attention_target']['w_val_pos']:.3f} "
                f"leak={causal_b[sid]['past_key_leak']:.1e}{mu_str}",
                flush=True,
            )
            del model
    summary_b = {}
    for sid, desc in PARTB_STACKS:
        agg = e2.aggregate_seeds({str(s): per_seed_b[str(s)][sid] for s in SEEDS})
        entry = {"desc": desc, **agg, "causal": causal_b[sid]}
        if "mu" in per_seed_b[str(SEEDS[0])][sid]:
            entry["mu"] = {
                k: ms_of([per_seed_b[str(s)][sid]["mu"][k] for s in SEEDS])
                for k in ("mu_mean", "mu_min", "mu_max")
            }
        summary_b[sid] = entry

    payload = {
        "config": {
            "seeds": list(SEEDS),
            "partA": "exp1 legacy protocol (KEY@0/VAL@1, eps sweep, multi-decoy)",
            "partB": "exp2 content-addressable protocol (M=3 pairs, eps sweep, multi-decoy)",
            "shift2": "fixed 2-tap [0.5, 0.5] on K after projection",
            "shiftmu": "learnable RWKV-form mu shift on K after projection, mu=sigmoid(logit), logit0=0",
            "scope_note": "placement fixed at K-after-projection; true-RWKV pre-projection placement out of scope",
        },
        "device": str(DEVICE)
        + (f" ({torch.cuda.get_device_name(0)})" if DEVICE.type == "cuda" else ""),
        "partA_legacy": {"summary": summary_a, "per_seed": per_seed_a},
        "partB_content_addressable": {"summary": summary_b, "per_seed": per_seed_b},
        "wall_s": round(time.time() - t_start, 1),
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved {OUT_JSON} (wall {payload['wall_s']} s)")
    return payload


if __name__ == "__main__":
    main()
