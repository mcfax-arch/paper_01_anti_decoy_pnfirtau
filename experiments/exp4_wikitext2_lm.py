#!/usr/bin/env python3
"""Experiment 4: Self-Contained WikiText-2 Pre-LN Transformer LM Benchmark.

Addresses Audit Issues B4, B5, H4, M4, M9:
  1. Exact model hyperparameters and parameter counts:
     D=64, H=4 (d_h=16), L=6, d_ff=256, max_seq=256, batch_size=64, vocab=23,168
     (word-level WikiText-2, min_freq=5, tied input/output embeddings).
     Total parameters:
       - `softmax`:    1,797,632
       - `sm_fir`:     1,797,632 (causal FIR K=7 on keys + Softmax MHA)
       - `pn_fir_tau`: 1,797,638 (+6 learnable scalar thresholds tau_l, one per layer)
  2. Matched Hardware Step-Time & VRAM Microbenchmark (`bench_step_time`):
     Measures exact forward+backward+AdamW step latency (ms/step) and peak allocated VRAM
     under identical settings (both FP32/TF32 and BF16) for `softmax`, `sm_fir`, and `pn_fir_tau`.
  3. Matched Same-Process Training Comparison (`softmax`, `sm_fir`, `pn_fir_tau`):
     Trains all three variants under identical precision (`bf16` + `tf32`), seed, and data loader,
     plus includes the full 50,000-step reference curves and checkpoint verification.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import random
import re
import sys
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.pnfirtau_attention import (  # noqa: E402
    PnFirConfig,
    PnFirTauFlashMHA,
    PnFirTauMHA,
    PnFirTauTritonMHA,
    SmFirMHA,
    SoftmaxMHA,
)
from src.triton_pnfirtau import flash_pn_fir_tau_attention  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = os.path.join(ROOT, "results")
OUT_JSON = os.path.join(RESULTS_DIR, "exp4_wikitext2_lm.json")


def configure_fast_cuda() -> None:
    if not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


@dataclass
class WikiLMConfig:
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 6
    d_ff: int = 256
    max_seq: int = 256
    batch_size: int = 64
    lr: float = 5e-4
    weight_decay: float = 0.1
    warmup: int = 400
    lr_horizon: int = 50_000
    steps: int = 2400
    eval_every: int = 600
    clip: float = 1.0
    dropout: float = 0.1
    vocab_min_freq: int = 5
    seed: int = 42
    p: float = 10.0
    silence_gamma: float = 0.5
    tau_init: float = 0.0
    fir_k: int = 7
    amp: str = "bf16"
    variants: list[str] = field(
        default_factory=lambda: [
            "softmax",
            "sm_fir",
            "pn_fir_tau",
            "pn_fir_tau_triton",
            "pn_fir_tau_flash",
        ]
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _find_hf_cache() -> str:
    candidates = [
        os.path.join(ROOT, "hf_cache"),
        r"I:\FFT_Attention\experiments\wikitext_pn_fir_lm\hf_cache",
        "/mnt/i/FFT_Attention/experiments/wikitext_pn_fir_lm/hf_cache",
        r"I:\FFT_Attention\experiments\poc_phase5\hf_cache",
        "/mnt/i/FFT_Attention/experiments/poc_phase5/hf_cache",
        r"I:\FFT_Attention\experiments\full_stack_lm\hf_cache",
        "/mnt/i/FFT_Attention/experiments/full_stack_lm/hf_cache",
    ]
    for p in candidates:
        if os.path.isdir(p):
            return p
    return os.path.join(ROOT, "hf_cache")


def load_tokens(split: str) -> list[str]:
    local_cache_dir = os.path.join(ROOT, "hf_cache")
    local_txt = os.path.join(local_cache_dir, f"wikitext2_{split}_tokens.txt")
    if os.path.isfile(local_txt):
        with open(local_txt, "r", encoding="utf-8") as f:
            return f.read().strip().split()

    from datasets import load_dataset

    ds = load_dataset(
        "Salesforce/wikitext",
        "wikitext-2-v1",
        split=split,
        cache_dir=_find_hf_cache(),
    )
    text = "\n".join(ds["text"])
    text = re.sub(r"^ = .+ = $", "", text, flags=re.MULTILINE)
    toks = text.strip().split()
    os.makedirs(local_cache_dir, exist_ok=True)
    with open(local_txt, "w", encoding="utf-8") as f:
        f.write(" ".join(toks))
    return toks


def build_vocab(tokens: list[str], min_freq: int):
    cnt = collections.Counter(tokens)
    vocab = ["<pad>", "<unk>", "<sos>", "<eos>"] + sorted(
        t for t, c in cnt.items() if c >= min_freq
    )
    stoi = {t: i for i, t in enumerate(vocab)}
    return vocab, stoi


def encode(tokens: list[str], stoi: dict[str, int]) -> list[int]:
    unk = stoi["<unk>"]
    return [stoi.get(t, unk) for t in tokens]


def make_chunks(ids: list[int], seq: int) -> torch.Tensor:
    n = (len(ids) // seq) * seq
    return torch.tensor(ids[:n], dtype=torch.long).view(-1, seq)


class Loader:
    def __init__(self, chunks: torch.Tensor, batch: int, shuffle: bool, seed: int):
        self.chunks = chunks
        self.batch = batch
        self.shuffle = shuffle
        self.rng = random.Random(seed)
        m = (len(chunks) // batch) * batch
        self.idx = list(range(m))

    def epoch(self):
        ix = self.idx[:]
        if self.shuffle:
            self.rng.shuffle(ix)
        for i in range(0, len(ix), self.batch):
            yield self.chunks[ix[i : i + self.batch]]


class Block(nn.Module):
    def __init__(self, d_model: int, attn: nn.Module, dropout: float, d_ff: int = 256):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = attn
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop(self.attn(self.ln1(x)))
        x = x + self.ffn(self.ln2(x))
        return x


class WikiLM(nn.Module):
    def __init__(self, vocab: int, cfg: WikiLMConfig, variant: str):
        super().__init__()
        self.cfg = cfg
        self.variant = variant
        self.embed = nn.Embedding(vocab, cfg.d_model)
        self.pos = nn.Embedding(cfg.max_seq, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        acfg = PnFirConfig(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            p=cfg.p,
            silence_gamma=cfg.silence_gamma,
            tau_init=cfg.tau_init,
            fir_k=cfg.fir_k,
            dropout=cfg.dropout,
            causal_mask=True,
            bias=False,
        )
        self.blocks = nn.ModuleList()
        for _ in range(cfg.n_layers):
            if variant == "softmax":
                attn = SoftmaxMHA(acfg)
            elif variant == "sm_fir":
                attn = SmFirMHA(acfg)
            elif variant == "pn_fir_tau":
                attn = PnFirTauMHA(acfg)
            elif variant == "pn_fir_tau_triton":
                attn = PnFirTauTritonMHA(acfg)
            elif variant == "pn_fir_tau_flash":
                attn = PnFirTauFlashMHA(acfg)
            else:
                raise ValueError(f"Unknown variant: {variant}")
            self.blocks.append(Block(cfg.d_model, attn, cfg.dropout, cfg.d_ff))
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, vocab, bias=False)
        self.head.weight = self.embed.weight

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, N = idx.shape
        pos = torch.arange(N, device=idx.device).unsqueeze(0)
        h = self.drop(self.embed(idx) + self.pos(pos))
        for b in self.blocks:
            h = b(h)
        return self.head(self.ln_f(h))

    def param_count(self) -> dict[str, int]:
        seen = set()
        total = attn = 0
        for n, p in self.named_parameters():
            if id(p) in seen:
                continue
            seen.add(id(p))
            total += p.numel()
            if ".attn." in n:
                attn += p.numel()
        return {"total": total, "attention": attn}


@torch.no_grad()
def eval_ppl(model: WikiLM, chunks: torch.Tensor, batch: int, amp: str = "bf16") -> dict:
    model.eval()
    tot_nll = 0.0
    tot_tok = 0
    use_amp = amp in ("bf16", "fp16") and DEVICE.type == "cuda"
    amp_dtype = torch.bfloat16 if amp == "bf16" else torch.float16
    for x in Loader(chunks, batch, False, 0).epoch():
        x = x.to(DEVICE, non_blocking=True)
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                logits = model(x)
                loss = F.cross_entropy(
                    logits[:, :-1].reshape(-1, logits.size(-1)),
                    x[:, 1:].reshape(-1),
                    reduction="sum",
                )
        else:
            logits = model(x)
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                x[:, 1:].reshape(-1),
                reduction="sum",
            )
        tot_nll += loss.item()
        tot_tok += x[:, 1:].numel()
    model.train()
    nll = tot_nll / max(tot_tok, 1)
    return {"nll": nll, "ppl": math.exp(min(nll, 40.0)), "tokens": tot_tok}


@torch.no_grad()
def causal_gate(model: WikiLM, tol: float = 1e-4) -> dict:
    model.eval()
    N = min(64, model.cfg.max_seq)
    V = model.head.out_features
    max_d = 0.0
    for _ in range(2):
        x = torch.randint(4, min(1000, V - 1), (2, N), device=DEVICE)
        base = model(x)
        xp = x.clone()
        xp[:, -1] = (xp[:, -1] + 17) % (V - 4) + 4
        yp = model(xp)
        max_d = max(max_d, (yp[:, :-1] - base[:, :-1]).abs().max().item())
    model.train()
    return {"passed": max_d <= tol, "max_past_delta": max_d, "tol": tol}


def bench_step_time(vocab: int, cfg: WikiLMConfig, steps: int = 30, warmup: int = 8) -> dict:
    """Measure exact ms/step and peak VRAM across variants under identical hardware settings."""
    results = {}
    all_variants = (
        "softmax",
        "sm_fir",
        "pn_fir_tau",
        "pn_fir_tau_triton",
        "pn_fir_tau_flash",
    )
    for amp_mode in ("off", "bf16"):
        results[amp_mode] = {}
        for variant in all_variants:
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            set_seed(cfg.seed)
            model = WikiLM(vocab, cfg, variant).to(DEVICE)
            opt = torch.optim.AdamW(
                model.parameters(), lr=cfg.lr, betas=(0.9, 0.95), weight_decay=cfg.weight_decay
            )
            x = torch.randint(0, vocab, (cfg.batch_size, cfg.max_seq), device=DEVICE)
            use_amp = amp_mode == "bf16" and DEVICE.type == "cuda"

            def _step():
                opt.zero_grad(set_to_none=True)
                if use_amp:
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        logits = model(x)
                        loss = F.cross_entropy(
                            logits[:, :-1].reshape(-1, vocab), x[:, 1:].reshape(-1)
                        )
                else:
                    logits = model(x)
                    loss = F.cross_entropy(
                        logits[:, :-1].reshape(-1, vocab), x[:, 1:].reshape(-1)
                    )
                loss.backward()
                opt.step()

            for _ in range(warmup):
                _step()
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(steps):
                _step()
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            ms_per_step = (time.perf_counter() - t0) / steps * 1000.0
            vram_gb = (
                torch.cuda.max_memory_allocated() / (1024 ** 3)
                if DEVICE.type == "cuda"
                else 0.0
            )
            results[amp_mode][variant] = {
                "ms_per_step": float(ms_per_step),
                "vram_allocated_gb": float(vram_gb),
                "params": model.param_count(),
            }
            print(
                f"  [bench amp={amp_mode:4s}] {variant:18s} {ms_per_step:6.2f} ms/step  "
                f"VRAM={vram_gb:.3f} GB  params={model.param_count()['total']:,}"
            )
            del model, opt, x
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()
    return results


def bench_triton_mha_scaling(cfg: WikiLMConfig, steps: int = 200, warmup: int = 25) -> dict:
    """Measure IEEE FP32 parity (abs + rel), isolated 6-layer MHA scaling (fixed B=64 + B=16 at N=1024), and silent-tile skipping."""
    if DEVICE.type != "cuda":
        return {}

    # 1. Exact IEEE FP32 numerical parity (with TF32 temporarily disabled)
    prev_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    set_seed(cfg.seed)
    acfg_p = PnFirConfig(d_model=cfg.d_model, n_heads=cfg.n_heads, dropout=0.0)
    m_ref = PnFirTauMHA(acfg_p).to(DEVICE)
    m_tri = PnFirTauTritonMHA(acfg_p).to(DEVICE)
    m_fls = PnFirTauFlashMHA(acfg_p).to(DEVICE)
    m_tri.load_state_dict(m_ref.state_dict())
    m_fls.load_state_dict(m_ref.state_dict())

    x1 = torch.randn(8, 128, cfg.d_model, device=DEVICE, dtype=torch.float32, requires_grad=True)
    x2 = x1.detach().clone().requires_grad_(True)
    x3 = x1.detach().clone().requires_grad_(True)
    dy = torch.randn_like(x1)

    y1 = m_ref(x1)
    y1.backward(dy)
    y2 = m_tri(x2)
    y2.backward(dy)
    y3 = m_fls(x3)
    y3.backward(dy)

    out_ref_norm = float(y1.abs().max().item())
    dx_ref_norm = float(x1.grad.abs().max().item())
    dtau_ref_abs = float(m_ref.score_tau.grad.abs().item())

    out_diff_tri = float((y1 - y2).abs().max().item())
    dx_diff_tri = float((x1.grad - x2.grad).abs().max().item())
    dtau_diff_tri = float((m_ref.score_tau.grad - m_tri.score_tau.grad).abs().item())

    out_diff_fls = float((y1 - y3).abs().max().item())
    dx_diff_fls = float((x1.grad - x3.grad).abs().max().item())
    dtau_diff_fls = float((m_ref.score_tau.grad - m_fls.score_tau.grad).abs().item())

    parity = {
        "reference_norms": {
            "out_ref_max_abs": out_ref_norm,
            "dx_ref_max_abs": dx_ref_norm,
            "dtau_ref_abs": dtau_ref_abs,
        },
        "pn_fir_tau_triton": {
            "out_max_diff": out_diff_tri,
            "out_rel_diff": float(out_diff_tri / max(out_ref_norm, 1e-12)),
            "dx_max_diff": dx_diff_tri,
            "dx_rel_diff": float(dx_diff_tri / max(dx_ref_norm, 1e-12)),
            "dtau_diff": dtau_diff_tri,
            "dtau_rel_diff": float(dtau_diff_tri / max(dtau_ref_abs, 1e-12)),
        },
        "pn_fir_tau_flash": {
            "out_max_diff": out_diff_fls,
            "out_rel_diff": float(out_diff_fls / max(out_ref_norm, 1e-12)),
            "dx_max_diff": dx_diff_fls,
            "dx_rel_diff": float(dx_diff_fls / max(dx_ref_norm, 1e-12)),
            "dtau_diff": dtau_diff_fls,
            "dtau_rel_diff": float(dtau_diff_fls / max(dtau_ref_abs, 1e-12)),
        },
    }
    print(
        f"  [parity fp32] Row-Triton:   out={out_diff_tri:.2e} (rel {parity['pn_fir_tau_triton']['out_rel_diff']:.2e}) "
        f"dx={dx_diff_tri:.2e} (rel {parity['pn_fir_tau_triton']['dx_rel_diff']:.2e}) "
        f"dtau={dtau_diff_tri:.2e} (rel {parity['pn_fir_tau_triton']['dtau_rel_diff']:.2e})"
    )
    print(
        f"  [parity fp32] Flash-Triton: out={out_diff_fls:.2e} (rel {parity['pn_fir_tau_flash']['out_rel_diff']:.2e}) "
        f"dx={dx_diff_fls:.2e} (rel {parity['pn_fir_tau_flash']['dx_rel_diff']:.2e}) "
        f"dtau={dtau_diff_fls:.2e} (rel {parity['pn_fir_tau_flash']['dtau_rel_diff']:.2e})"
    )
    del m_ref, m_tri, m_fls, x1, x2, x3, dy, y1, y2, y3
    torch.backends.cuda.matmul.allow_tf32 = prev_tf32
    torch.cuda.empty_cache()

    # 2. Isolated 6-Layer MHA Stack Scaling:
    #    - Fixed B=64 across N in (256, 512, 1024) ("256", "512", "1024")
    #    - Reduced B=16 at N=1024 ("1024_B16") so Eager O(N^2) baselines also fit in 12 GB VRAM
    all_variants = (
        "softmax",
        "sm_fir",
        "pn_fir_tau",
        "pn_fir_tau_triton",
        "pn_fir_tau_flash",
    )
    seq_scaling = {}
    scaling_configs = [
        ("256", 256, 64),
        ("512", 512, 64),
        ("1024", 1024, 64),
        ("1024_B16", 1024, 16),
    ]
    for key_label, N, B_sc in scaling_configs:
        seq_scaling[key_label] = {"batch_size": B_sc, "seq_len": N, "variants": {}}
        for variant in all_variants:
            # At N=1024, B=64, Eager O(N^2) variants require 12.05 GB (softmax/sm_fir) to 28.7 GB (pn_fir_tau),
            # exceeding the 12 GB physical VRAM of the RTX 3060 (triggering WDDM host paging / hard OOM).
            if N == 1024 and B_sc == 64 and variant in ("softmax", "sm_fir", "pn_fir_tau"):
                seq_scaling[key_label]["variants"][variant] = {
                    "ms_per_step": None,
                    "ms_std": None,
                    "vram_allocated_mb": None,
                    "oom": True,
                }
                print(f"  [mha6 N={N:4d} B={B_sc:2d}] {variant:18s} OOM (>12 GB physical VRAM)")
                continue
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            set_seed(cfg.seed)
            acfg = PnFirConfig(d_model=cfg.d_model, n_heads=cfg.n_heads, dropout=0.0)
            cls_map = {
                "softmax": SoftmaxMHA,
                "sm_fir": SmFirMHA,
                "pn_fir_tau": PnFirTauMHA,
                "pn_fir_tau_triton": PnFirTauTritonMHA,
                "pn_fir_tau_flash": PnFirTauFlashMHA,
            }
            layers = nn.ModuleList([cls_map[variant](acfg) for _ in range(cfg.n_layers)]).to(DEVICE)
            x = torch.randn(B_sc, N, cfg.d_model, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
            dy = torch.randn_like(x)

            def _mha_step():
                for lyr in layers:
                    lyr.zero_grad(set_to_none=True)
                x.grad = None
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    h = x
                    for lyr in layers:
                        h = h + lyr(h)
                h.backward(dy)

            for _ in range(warmup):
                _mha_step()
            torch.cuda.synchronize()
            step_times = []
            for _ in range(steps):
                t0 = time.perf_counter()
                _mha_step()
                torch.cuda.synchronize()
                step_times.append((time.perf_counter() - t0) * 1000.0)
            ms = float(np.mean(step_times))
            ms_std = float(np.std(step_times))
            vram_mb = float(torch.cuda.max_memory_allocated() / (1024 ** 2))
            seq_scaling[key_label]["variants"][variant] = {
                "ms_per_step": ms,
                "ms_std": ms_std,
                "vram_allocated_mb": vram_mb,
                "oom": False,
            }
            print(
                f"  [mha6 N={N:4d} B={B_sc:2d}] {variant:18s} {ms:6.2f} ± {ms_std:4.2f} ms/step  "
                f"VRAM={vram_mb:7.1f} MB"
            )
            del layers, x, dy
            torch.cuda.empty_cache()

        # Also measure isolated 1-layer Flash kernel (fwd+bwd) at (B_sc, H=4, N, d_h=16)
        torch.cuda.empty_cache()
        set_seed(cfg.seed)
        d_h = cfg.d_model // cfg.n_heads
        scale = 1.0 / math.sqrt(d_h)
        q_k = torch.randn(B_sc, cfg.n_heads, N, d_h, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
        k_k = torch.randn(B_sc, cfg.n_heads, N, d_h, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
        v_k = torch.randn(B_sc, cfg.n_heads, N, d_h, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
        tau_k = torch.tensor(0.0, device=DEVICE, dtype=torch.float32, requires_grad=True)
        do_k = torch.randn_like(q_k)

        def _kern_step():
            q_k.grad = k_k.grad = v_k.grad = tau_k.grad = None
            out_k = flash_pn_fir_tau_attention(q_k, k_k, v_k, tau_k, scale=scale)
            out_k.backward(do_k)

        for _ in range(warmup):
            _kern_step()
        torch.cuda.synchronize()
        ktimes = []
        for _ in range(steps):
            t0 = time.perf_counter()
            _kern_step()
            torch.cuda.synchronize()
            ktimes.append((time.perf_counter() - t0) * 1000.0)
        seq_scaling[key_label]["isolated_1layer_flash_kernel"] = {
            "ms_per_step": float(np.mean(ktimes)),
            "ms_std": float(np.std(ktimes)),
        }
        print(
            f"  [kern1 N={N:4d} B={B_sc:2d}] flash_kernel_only  "
            f"{np.mean(ktimes):6.2f} ± {np.std(ktimes):4.2f} ms"
        )
        del q_k, k_k, v_k, tau_k, do_k
        torch.cuda.empty_cache()

    # 3. Hardware conditional silent-tile skipping benchmark on PnFirTauFlashMHA (N=1024, B=16)
    silent_skip = []
    N_sk, B_sk = 1024, 16
    d_h = cfg.d_model // cfg.n_heads
    scale = 1.0 / math.sqrt(d_h)
    for tau_val in (-0.5, 0.0, 0.5, 1.0, 1.5, 2.5):
        torch.cuda.empty_cache()
        set_seed(cfg.seed)
        acfg_s = PnFirConfig(
            d_model=cfg.d_model, n_heads=cfg.n_heads, dropout=0.0, tau_init=float(tau_val)
        )
        layers = nn.ModuleList([PnFirTauFlashMHA(acfg_s) for _ in range(cfg.n_layers)]).to(DEVICE)
        x = torch.randn(B_sk, N_sk, cfg.d_model, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
        dy = torch.randn_like(x)

        # Measure silent row fraction on layer 0
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, w0 = layers[0](x, return_weights=True)
            row_mass = w0.sum(dim=-1)
            silent_row_frac = float((row_mass <= 1e-6).float().mean().item())
            mean_row_mass = float(row_mass.float().mean().item())

        def _skip_step():
            for lyr in layers:
                lyr.zero_grad(set_to_none=True)
            x.grad = None
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                h = x
                for lyr in layers:
                    h = h + lyr(h)
            h.backward(dy)

        for _ in range(warmup):
            _skip_step()
        torch.cuda.synchronize()
        stimes = []
        for _ in range(steps):
            t0 = time.perf_counter()
            _skip_step()
            torch.cuda.synchronize()
            stimes.append((time.perf_counter() - t0) * 1000.0)
        ms = float(np.mean(stimes))
        ms_std = float(np.std(stimes))
        # Extract the exact (q, k_tilde, v) tensors from layers[0] on x so the isolated 1-layer
        # kernel benchmark runs on the identical score distribution as silent_row_frac
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            from src.triton_pnfirtau import triton_causal_fir
            q_raw, k_raw, v_raw = layers[0].qkv(x).chunk(3, dim=-1)
            k_s = triton_causal_fir(k_raw, layers[0].fir_taps).view(B_sk, N_sk, cfg.n_heads, d_h).transpose(1, 2).contiguous().detach().requires_grad_(True)
            q_s = q_raw.view(B_sk, N_sk, cfg.n_heads, d_h).transpose(1, 2).contiguous().detach().requires_grad_(True)
            v_s = v_raw.view(B_sk, N_sk, cfg.n_heads, d_h).transpose(1, 2).contiguous().detach().requires_grad_(True)
        tau_s = torch.tensor(float(tau_val), device=DEVICE, dtype=torch.float32, requires_grad=True)
        do_s = torch.randn_like(q_s)

        del layers, x, dy
        torch.cuda.empty_cache()

        def _kern_skip_step():
            q_s.grad = k_s.grad = v_s.grad = tau_s.grad = None
            out_s = flash_pn_fir_tau_attention(q_s, k_s, v_s, tau_s, scale=scale)
            out_s.backward(do_s)

        for _ in range(warmup):
            _kern_skip_step()
        torch.cuda.synchronize()
        kstimes = []
        for _ in range(steps):
            t0 = time.perf_counter()
            _kern_skip_step()
            torch.cuda.synchronize()
            kstimes.append((time.perf_counter() - t0) * 1000.0)
        k_ms = float(np.mean(kstimes))
        k_std = float(np.std(kstimes))
        del q_s, k_s, v_s, tau_s, do_s
        torch.cuda.empty_cache()

        silent_skip.append(
            {
                "tau": float(tau_val),
                "silent_row_frac": silent_row_frac,
                "mean_row_mass": mean_row_mass,
                "ms_per_step": ms,
                "ms_std": ms_std,
                "isolated_1layer_kernel_ms": k_ms,
                "isolated_1layer_kernel_std": k_std,
            }
        )
        print(
            f"  [flash-skip N={N_sk} B={B_sk}] tau={tau_val:+4.1f}  "
            f"silent_rows={silent_row_frac*100:5.1f}%  mean_mass={mean_row_mass:.3f}  "
            f"mha6={ms:6.2f}±{ms_std:4.2f} ms  kern1={k_ms:5.2f}±{k_std:4.2f} ms"
        )

    return {
        "fp32_parity": parity,
        "seq_scaling_bf16": seq_scaling,
        "silent_tile_skipping_bf16": silent_skip,
    }


def train_matched_variant(
    vocab: int,
    train_c: torch.Tensor,
    val_c: torch.Tensor,
    test_c: torch.Tensor,
    cfg: WikiLMConfig,
    variant: str,
) -> dict:
    set_seed(cfg.seed)
    model = WikiLM(vocab, cfg, variant).to(DEVICE)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.95),
    )
    loader = Loader(train_c, cfg.batch_size, True, cfg.seed)
    use_amp = cfg.amp == "bf16" and DEVICE.type == "cuda"
    cg_pre = causal_gate(model)

    hist = {"eval_steps": [], "val_ppl": [], "train_loss": []}
    best_val = float("inf")
    best_state = None
    step = 0
    t0 = time.perf_counter()

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
                        logits[:, :-1].reshape(-1, logits.size(-1)),
                        x[:, 1:].reshape(-1),
                    )
            else:
                logits = model(x)
                loss = F.cross_entropy(
                    logits[:, :-1].reshape(-1, logits.size(-1)),
                    x[:, 1:].reshape(-1),
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip)
            opt.step()
            step += 1

            if step % cfg.eval_every == 0 or step == cfg.steps:
                v = eval_ppl(model, val_c, min(cfg.batch_size, 32), amp=cfg.amp)
                loss_f = float(loss.item())
                hist["eval_steps"].append(step)
                hist["val_ppl"].append(v["ppl"])
                hist["train_loss"].append(loss_f)
                elapsed = time.perf_counter() - t0
                print(
                    f"  [{variant:18s}] step={step:4d}/{cfg.steps} "
                    f"loss={loss_f:.4f} val_ppl={v['ppl']:.2f} t={elapsed:.1f}s",
                    flush=True,
                )
                if v["ppl"] < best_val:
                    best_val = v["ppl"]
                    best_state = {
                        k: t.detach().cpu().clone() for k, t in model.state_dict().items()
                    }

    wall_sec = time.perf_counter() - t0
    if best_state is not None:
        model.load_state_dict(best_state)
    val_res = eval_ppl(model, val_c, min(cfg.batch_size, 32), amp=cfg.amp)
    test_res = eval_ppl(model, test_c, min(cfg.batch_size, 32), amp=cfg.amp)
    train_res = eval_ppl(model, train_c, min(cfg.batch_size, 32), amp=cfg.amp)
    cg_post = causal_gate(model)
    taus = [
        float(b.attn.score_tau.detach().item())
        for b in model.blocks
        if hasattr(b.attn, "score_tau")
    ]
    return {
        "variant": variant,
        "params": model.param_count(),
        "val_ppl": float(val_res["ppl"]),
        "test_ppl": float(test_res["ppl"]),
        "train_ppl": float(train_res["ppl"]),
        "causal_pre": cg_pre,
        "causal_post": cg_post,
        "tau_per_layer": taus,
        "history": hist,
        "wall_sec": float(wall_sec),
        "ms_per_step_effective": float(wall_sec / cfg.steps * 1000.0),
    }


def load_50k_reference() -> dict | None:
    """Load the 50,000-step WikiText-2 results and parse full Softmax + PnFirTau trajectories."""
    ref_candidates = [
        r"I:\FFT_Attention\experiments\wikitext_pn_fir_lm\results.json",
        "/mnt/i/FFT_Attention/experiments/wikitext_pn_fir_lm/results.json",
    ]
    log_candidates = [
        r"I:\FFT_Attention\experiments\wikitext_pn_fir_lm\experiment.log",
        "/mnt/i/FFT_Attention/experiments/wikitext_pn_fir_lm/experiment.log",
    ]
    ref_json = next((p for p in ref_candidates if os.path.isfile(p)), None)
    sm_log = next((p for p in log_candidates if os.path.isfile(p)), None)
    if ref_json is None:
        if os.path.isfile(OUT_JSON):
            with open(OUT_JSON, "r", encoding="utf-8") as f:
                old = json.load(f)
            if old.get("reference_50k") is not None:
                return old["reference_50k"]
        return None
    with open(ref_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Extract Softmax val_ppl curve at 1000..50000 from experiment.log
    sm_val_steps = []
    sm_val_ppl = []
    if sm_log is not None and os.path.isfile(sm_log):
        with open(sm_log, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                m = re.search(
                    r"\[softmax\]\s+step=(\d+)\s+loss=([0-9.]+)\s+val_ppl=([0-9.]+)",
                    line,
                )
                if m:
                    s = int(m.group(1))
                    if s > 0:
                        sm_val_steps.append(s)
                        sm_val_ppl.append(float(m.group(3)))

    pn_run = next((r for r in data.get("runs", []) if r["variant"] == "pn_fir_tau"), None)
    sm_run = next((r for r in data.get("runs", []) if r["variant"] == "softmax"), None)
    if pn_run is None or sm_run is None:
        return None

    pn_val_ppl = pn_run["history"]["val_ppl"]
    pn_val_steps = [1000 * (i + 1) for i in range(len(pn_val_ppl))]

    return {
        "protocol_note": (
            "50,000-step full training runs on word-level WikiText-2 (vocab=23,168, D=64, H=4, L=6, "
            "d_ff=256, max_seq=256, batch_size=64, seed=42). Note: in the historical 50k log, "
            "`softmax` ran in FP32 without BF16 autocast, whereas `pn_fir_tau` was resumed at step 1001 "
            "with TF32 + BF16 autocast. See `step_benchmark` and `matched_3k_runs` for strictly "
            "same-process, identical-precision comparisons including `sm_fir`."
        ),
        "softmax_50k": {
            "params": sm_run["params"],
            "val_ppl": sm_run["val_ppl"],
            "test_ppl": sm_run["test_ppl"],
            "train_ppl": sm_run["train_ppl"],
            "val_steps": sm_val_steps,
            "val_ppl_curve": sm_val_ppl,
        },
        "pn_fir_tau_50k": {
            "params": pn_run["params"],
            "val_ppl": pn_run["val_ppl"],
            "test_ppl": pn_run["test_ppl"],
            "train_ppl": pn_run["train_ppl"],
            "tau_per_layer": pn_run["tau_per_layer"],
            "causal": pn_run["causal"],
            "val_steps": pn_val_steps,
            "val_ppl_curve": pn_val_ppl,
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2400)
    ap.add_argument("--eval-every", type=int, default=600)
    ap.add_argument("--retrain-all", action="store_true")
    ap.add_argument("--only-triton-mha-bench", action="store_true")
    args = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    configure_fast_cuda()
    cfg = WikiLMConfig(steps=args.steps, eval_every=args.eval_every)

    print("=" * 88)
    print(f"EXP 4: WikiText-2 Pre-LN LM & Triton Benchmark | device={DEVICE}")
    print("=" * 88)

    if args.only_triton_mha_bench and os.path.isfile(OUT_JSON):
        with open(OUT_JSON, "r", encoding="utf-8") as f:
            payload = json.load(f)
        print("\n--- Part A2: Isolated 6-Layer MHA Triton Scaling & Silent-Tile Skipping ---")
        payload["triton_mha_benchmark"] = bench_triton_mha_scaling(cfg)
        with open(OUT_JSON, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\nUpdated triton_mha_benchmark in {OUT_JSON}")
        return

    existing_runs = {}
    if os.path.isfile(OUT_JSON) and not args.retrain_all:
        with open(OUT_JSON, "r", encoding="utf-8") as f:
            old_payload = json.load(f)
        existing_runs = old_payload.get("matched_runs", {})

    tr = load_tokens("train")
    va = load_tokens("validation")
    te = load_tokens("test")
    vocab, stoi = build_vocab(tr, cfg.vocab_min_freq)
    V = len(vocab)
    train_c = make_chunks(encode(tr, stoi), cfg.max_seq)
    val_c = make_chunks(encode(va, stoi), cfg.max_seq)
    test_c = make_chunks(encode(te, stoi), cfg.max_seq)
    print(f"Loaded WikiText-2: vocab={V}, chunks train/val/test={len(train_c)}/{len(val_c)}/{len(test_c)}")

    print("\n--- Part A1: Matched Full-LM Step-Time & VRAM Microbenchmark ---")
    step_bench = bench_step_time(V, cfg)

    print("\n--- Part A2: Isolated 6-Layer MHA Triton Scaling & Silent-Tile Skipping ---")
    triton_mha_bench = bench_triton_mha_scaling(cfg)

    print(f"\n--- Part B: Matched Same-Process Training ({cfg.steps} steps, amp={cfg.amp}) ---")
    matched_runs = {}
    for variant in cfg.variants:
        if (
            variant in existing_runs
            and existing_runs[variant].get("history", {}).get("eval_steps", [0])[-1] == cfg.steps
        ):
            print(f"\nReusing existing {cfg.steps}-step run for {variant} (val_ppl={existing_runs[variant]['val_ppl']:.2f})...")
            matched_runs[variant] = existing_runs[variant]
        else:
            print(f"\nTraining {variant}...")
            matched_runs[variant] = train_matched_variant(
                V, train_c, val_c, test_c, cfg, variant
            )

    ref_50k = load_50k_reference()

    payload = {
        "config": asdict(cfg),
        "device": str(DEVICE)
        + (f" ({torch.cuda.get_device_name(0)})" if DEVICE.type == "cuda" else ""),
        "vocab_size": V,
        "train_tokens": len(tr),
        "step_benchmark": step_bench,
        "triton_mha_benchmark": triton_mha_bench,
        "matched_runs": matched_runs,
        "reference_50k": ref_50k,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved {OUT_JSON}")


if __name__ == "__main__":
    main()
