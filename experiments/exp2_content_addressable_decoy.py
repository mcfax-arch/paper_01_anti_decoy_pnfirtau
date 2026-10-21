#!/usr/bin/env python3
"""Experiment 2: True Content-Addressable Associative Recall & Adversarial Decoy Benchmark.

Addresses the structural limitations discovered in the legacy fixed-position bijective benchmark
(Audit Issues B1, B3, H2, M7, M11):
  1. Random Key-Value Pairing (No Bijection): In every sequence, M=3 distinct keys from {1..8}
     are randomly paired with M=3 independently sampled values from {9..16}. Attending to
     KEY_POS (j*) yields zero information about the target value v* (chance = 1/8 = 12.5%).
     To succeed, a 1-layer attention mechanism MUST attend to VAL_POS (j* + 1).
  2. Variable Interior Positions: Key-value pairs (k_m, v_m) are placed at random non-overlapping
     interior positions j_m in [4, T-6] (T=32). Boundary position t=0 is always filler.
  3. Conditioned Query: The query vector q = W_Q(emb(k*)) is conditioned on the target key k*.
  4. Constructive Temporal Interference via Causal FIR:
     - Without FIR (`sm`, `pn`), K[j*+1] depends only on v*, which is independent of k*.
       Thus 1-layer unfiltered attention can at best guess among the M=3 value slots (~33.3%).
     - With Causal FIR (`sm_fir`, `pn_fir`, `pn_fir_tau`), K_tilde[j*+1] = h[0]*K(v*) + h[1]*K(k*) + ...
       combines the value-slot type signature (h[0]*K(v*)) and the preceding key identity (h[1]*K(k*)),
       creating constructive interference at VAL_POS (j* + 1).
  5. True Adversarial Decoys: Injects n_d in {1, 2, 4, 8} perturbed copies of the queried key
     emb(k*) + eps * noise at random filler positions (unpaired false triggers).
  6. Silence Calibration: Includes a variant `pn_fir_tau_reg` trained with absent-key queries
     and a lightweight peak-silence regularizer to eliminate inter-seed tau drift.
"""
from __future__ import annotations

import json
import math
import os
import random
import sys
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
    power_norm_peak_gate,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = os.path.join(ROOT, "results")
OUT_JSON = os.path.join(RESULTS_DIR, "exp2_content_addressable_decoy.json")

STACKS = [
    "sm",
    "pn",
    "sm_fir",
    "pn_fir",
    "pn_fir_c",
    "pn_fir_tau",
    "pn_fir_tau_reg",
]


@dataclass
class ContentAssocConfig:
    vocab_size: int = 48
    n_keys: int = 8       # tokens 1..8
    n_vals: int = 8       # tokens 9..16 (targets 0..7)
    n_pairs: int = 3      # M=3 random (key, val) pairs per sequence
    d_model: int = 64
    seq_len: int = 48     # 14 non-overlapping 3-token slots in [4, 44]
    batch_size: int = 128
    steps: int = 1800
    lr: float = 3e-3
    p: float = 10.0
    fir_kernel: int = 7
    silence_gamma: float = 0.5
    silence_mode: str = "peak"
    tau_init: float = 0.0
    seeds: tuple[int, ...] = (1234, 2025, 3141)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ContentAssocModel(nn.Module):
    def __init__(self, cfg: ContentAssocConfig, stack: str):
        super().__init__()
        self.cfg = cfg
        self.stack = stack
        D = cfg.d_model
        self.emb = nn.Embedding(cfg.vocab_size, D)
        self.q_proj = nn.Linear(D, D, bias=False)
        self.k_proj = nn.Linear(D, D, bias=False)
        self.v_proj = nn.Linear(D, D, bias=False)
        self.head = nn.Linear(D, cfg.n_vals)
        self.score_tau = nn.Parameter(torch.tensor(float(cfg.tau_init)))
        self.register_buffer(
            "fir_taps",
            causal_gaussian_taps(cfg.fir_kernel, torch.device("cpu"), torch.float32),
            persistent=False,
        )
        with torch.no_grad():
            self.emb.weight.mul_(0.5)

    def _use_fir(self) -> bool:
        return "fir" in self.stack

    def _use_tau(self) -> bool:
        return "tau" in self.stack

    def _use_center(self) -> bool:
        return self.stack.endswith("_c")

    def _silence_gamma(self) -> float | None:
        if self.stack in ("pn", "pn_fir", "pn_fir_c"):
            return None
        return self.cfg.silence_gamma

    def forward_from_emb(
        self,
        e: torch.Tensor,
        q_tok: torch.Tensor,
        return_scores: bool = False,
    ):
        B, T, D = e.shape
        q = self.q_proj(self.emb(q_tok))  # [B, D] conditioned on queried key
        k = self.k_proj(e)
        k_f = (
            causal_fir(k, self.fir_taps.to(device=k.device, dtype=k.dtype))
            if self._use_fir()
            else k
        )
        scores = torch.einsum("bd,btd->bt", q, k_f) / math.sqrt(D)
        if self._use_center():
            scores = scores - scores.mean(dim=-1, keepdim=True)
        if self._use_tau():
            scores = scores - self.score_tau

        if self.stack.startswith("sm"):
            w = F.softmax(scores, dim=-1)
        else:
            w = power_norm_peak_gate(
                scores,
                p=self.cfg.p,
                gamma=self._silence_gamma(),
                silence_mode=self.cfg.silence_mode,
            )
        v = self.v_proj(e)
        ctx = torch.einsum("bt,btd->bd", w, v)
        logits = self.head(ctx)
        if return_scores:
            return logits, w, scores
        return logits, w

    def forward(
        self,
        x: torch.Tensor,
        q_tok: torch.Tensor,
        return_scores: bool = False,
    ):
        return self.forward_from_emb(
            self.emb(x), q_tok, return_scores=return_scores
        )


def gen_content_batch(
    cfg: ContentAssocConfig,
    batch_size: int,
    decoy_eps: float | None = None,
    n_decoys: int = 0,
    model: ContentAssocModel | None = None,
):
    """Generate vectorized random-pairing associative recall batch with interior positions."""
    T = cfg.seq_len
    x = torch.randint(17, cfg.vocab_size, (batch_size, T), device=DEVICE)

    slot_candidates = torch.tensor(list(range(4, T - 4, 3)), device=DEVICE)  # [14] for T=48
    n_slots = slot_candidates.numel()
    assert n_slots >= cfg.n_pairs + 8

    all_keys = torch.arange(1, 1 + cfg.n_keys, device=DEVICE)
    all_vals = torch.arange(1 + cfg.n_keys, 1 + cfg.n_keys + cfg.n_vals, device=DEVICE)

    perm_k = torch.rand(batch_size, cfg.n_keys, device=DEVICE).argsort(dim=-1)[:, : cfg.n_pairs]
    perm_v = torch.randint(0, cfg.n_vals, (batch_size, cfg.n_pairs), device=DEVICE)
    keys_b = all_keys[perm_k]  # [B, M]
    vals_b = all_vals[perm_v]  # [B, M]

    perm_s = torch.rand(batch_size, n_slots, device=DEVICE).argsort(dim=-1)
    pair_slots = slot_candidates[perm_s[:, : cfg.n_pairs]]  # [B, M]

    batch_idx = torch.arange(batch_size, device=DEVICE).unsqueeze(1)
    x[batch_idx, pair_slots] = keys_b
    x[batch_idx, pair_slots + 1] = vals_b

    q_tok = keys_b[:, 0]
    y_val = perm_v[:, 0]
    key_pos_target = pair_slots[:, 0]
    val_pos_target = pair_slots[:, 0] + 1

    if decoy_eps is None or n_decoys == 0 or model is None:
        return x, q_tok, y_val, key_pos_target, val_pos_target, None

    # Remaining non-overlapping 3-token slots for isolated single-token decoys
    decoy_pool = slot_candidates[perm_s[:, cfg.n_pairs : cfg.n_pairs + n_decoys]]

    with torch.no_grad():
        e = model.emb(x).clone()
        ke = model.emb(q_tok)  # [B, D]
        b_arange = torch.arange(batch_size, device=DEVICE)
        for d in range(n_decoys):
            pos_d = decoy_pool[:, d]
            noise = torch.randn_like(ke)
            noise = noise / (noise.norm(dim=-1, keepdim=True) + 1e-6)
            k_norm = ke.norm(dim=-1, keepdim=True)
            e[b_arange, pos_d, :] = ke + decoy_eps * k_norm * noise

    return x, q_tok, y_val, key_pos_target, val_pos_target, e


def gen_absent_key_batch(cfg: ContentAssocConfig, batch_size: int):
    """Generate sequences containing M random (key, val) pairs where queried key k* is ABSENT."""
    x, q_tok, _, k_pos, v_pos, _ = gen_content_batch(cfg, batch_size)
    b_arange = torch.arange(batch_size, device=DEVICE)
    x[b_arange, k_pos] = 20
    x[b_arange, v_pos] = 21
    return x, q_tok


def train_one(cfg: ContentAssocConfig, stack: str) -> ContentAssocModel:
    model = ContentAssocModel(cfg, stack).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    use_reg = stack == "pn_fir_tau_reg"

    for step in range(cfg.steps):
        x, q_tok, y, _, _, _ = gen_content_batch(cfg, cfg.batch_size)
        logits, w = model(x, q_tok)
        loss = F.cross_entropy(logits, y)

        if use_reg and step >= cfg.steps // 3 and step % 2 == 0:
            x_abs, q_abs = gen_absent_key_batch(cfg, cfg.batch_size // 2)
            _, w_abs = model(x_abs, q_abs)
            loss = loss + 0.10 * w_abs.sum(dim=-1).mean()

        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


@torch.no_grad()
def eval_model(cfg: ContentAssocConfig, model: ContentAssocModel) -> dict:
    model.eval()
    B_eval = 1024
    x, q_tok, y, k_pos, v_pos, _ = gen_content_batch(cfg, B_eval)
    logits, w, scores = model(x, q_tok, return_scores=True)
    clean_acc = float((logits.argmax(-1) == y).float().mean().item())

    batch_idx = torch.arange(B_eval, device=DEVICE)
    w_at_key = float(w[batch_idx, k_pos].mean().item())
    w_at_val = float(w[batch_idx, v_pos].mean().item())
    argmax_pos = w.argmax(dim=-1)
    argmax_at_key_frac = float((argmax_pos == k_pos).float().mean().item())
    argmax_at_val_frac = float((argmax_pos == v_pos).float().mean().item())

    decoy = {}
    for eps in (0.05, 0.2, 0.5, 1.0):
        _, q_d, y_d, _, _, e_d = gen_content_batch(
            cfg, B_eval, decoy_eps=eps, n_decoys=1, model=model
        )
        logits_d, _ = model.forward_from_emb(e_d, q_d)
        decoy[f"eps_{eps}"] = float((logits_d.argmax(-1) == y_d).float().mean().item())

    multi = {}
    for nd in (1, 2, 4, 8):
        _, q_d, y_d, _, _, e_d = gen_content_batch(
            cfg, B_eval, decoy_eps=0.2, n_decoys=nd, model=model
        )
        logits_d, _ = model.forward_from_emb(e_d, q_d)
        multi[f"nd_{nd}"] = float((logits_d.argmax(-1) == y_d).float().mean().item())

    x_noise = torch.randint(17, cfg.vocab_size, (B_eval, cfg.seq_len), device=DEVICE)
    _, w_noise, _ = model(x_noise, q_tok, return_scores=True)
    sum_w_noise = float(w_noise.sum(dim=-1).mean().item())

    x_abs, q_abs = gen_absent_key_batch(cfg, B_eval)
    _, w_abs, _ = model(x_abs, q_abs, return_scores=True)
    sum_w_absent_key = float(w_abs.sum(dim=-1).mean().item())

    sum_w_match = float(w.sum(dim=-1).mean().item())

    out = {
        "clean": clean_acc,
        "decoy": decoy,
        "multi_decoy": multi,
        "attention_target": {
            "w_key_pos": w_at_key,
            "w_val_pos": w_at_val,
            "argmax_at_key_frac": argmax_at_key_frac,
            "argmax_at_val_frac": argmax_at_val_frac,
        },
        "silence": {
            "sum_w_match": sum_w_match,
            "sum_w_noise": sum_w_noise,
            "sum_w_absent_key": sum_w_absent_key,
            "gap_noise": float(sum_w_match - sum_w_noise),
            "gap_absent_key": float(sum_w_match - sum_w_absent_key),
            "ratio_noise": float(sum_w_match / max(sum_w_noise, 1e-9)),
        },
        "scores": {
            "s_max": float(scores.max(dim=-1).values.mean().item()),
            "s_mean": float(scores.mean(dim=-1).mean().item()),
            "s_min": float(scores.min(dim=-1).values.mean().item()),
        },
    }
    if model._use_tau():
        out["tau"] = float(model.score_tau.item())
    return out


def aggregate_seeds(per_seed: dict[str, dict]) -> dict:
    seeds = list(per_seed.keys())

    def ms(vals):
        arr = np.array(vals, dtype=np.float64)
        return {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "per_seed": [float(v) for v in vals],
        }

    out = {
        "clean": ms([per_seed[s]["clean"] for s in seeds]),
        "decoy": {
            k: ms([per_seed[s]["decoy"][k] for s in seeds])
            for k in per_seed[seeds[0]]["decoy"]
        },
        "multi_decoy": {
            k: ms([per_seed[s]["multi_decoy"][k] for s in seeds])
            for k in per_seed[seeds[0]]["multi_decoy"]
        },
        "attention_target": {
            k: ms([per_seed[s]["attention_target"][k] for s in seeds])
            for k in per_seed[seeds[0]]["attention_target"]
        },
        "silence": {
            k: ms([per_seed[s]["silence"][k] for s in seeds])
            for k in per_seed[seeds[0]]["silence"]
        },
        "scores": {
            k: ms([per_seed[s]["scores"][k] for s in seeds])
            for k in per_seed[seeds[0]]["scores"]
        },
    }
    if "tau" in per_seed[seeds[0]]:
        out["tau"] = ms([per_seed[s]["tau"] for s in seeds])
    return out


def main() -> dict:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    cfg = ContentAssocConfig()
    print("=" * 92)
    print(f"EXP 2: True Content-Addressable Associative Recall & Decoy | device={DEVICE} | p={cfg.p}")
    print("=" * 92)

    per_seed = {str(s): {} for s in cfg.seeds}
    for seed in cfg.seeds:
        print(f"\n--- Seed {seed} ---")
        for stack in STACKS:
            set_seed(seed)
            model = train_one(cfg, stack)
            res = eval_model(cfg, model)
            per_seed[str(seed)][stack] = res
            print(
                f"  {stack:15s} clean={res['clean']*100:5.1f}% "
                f"eps0.2={res['decoy']['eps_0.2']*100:5.1f}% "
                f"nd4={res['multi_decoy']['nd_4']*100:5.1f}% "
                f"nd8={res['multi_decoy']['nd_8']*100:5.1f}% "
                f"w_key={res['attention_target']['w_key_pos']:.3f} "
                f"w_val={res['attention_target']['w_val_pos']:.3f} "
                f"Sw_m={res['silence']['sum_w_match']:.3f} "
                f"Sw_n={res['silence']['sum_w_noise']:.3f} "
                f"gap={res['silence']['gap_noise']:+.3f}"
                + (f" tau={res['tau']:+.3f}" if "tau" in res else ""),
                flush=True,
            )

    summary = {
        stack: aggregate_seeds({str(s): per_seed[str(s)][stack] for s in cfg.seeds})
        for stack in STACKS
    }
    payload = {
        "config": asdict(cfg),
        "device": str(DEVICE)
        + (f" ({torch.cuda.get_device_name(0)})" if DEVICE.type == "cuda" else ""),
        "summary": summary,
        "per_seed": per_seed,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved {OUT_JSON}")
    return payload


if __name__ == "__main__":
    main()
