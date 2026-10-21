#!/usr/bin/env python3
"""Canonical implementation of Causal FIR + Power-Norm + Learnable Tau + Peak Silence Gate.

Consolidates the exact production modules and primitives from:
  - I:\\FFT_Attention\\experiments\\pn_fir_hw\\attention.py
  - I:\\FFT_Attention\\experiments\\pn_fir_hw\\bench.py
  - I:\\FFT_Attention\\experiments\\stack_candidates\\experiment.py

Mathematical composition per head:
  1. Causal FIR key filtering along sequence axis N:
       K_tilde[t] = sum_{u=0}^{K_fir - 1} h[u] * K[t - u],   K[t - u] = 0 for t - u < 0
     where h[u] = exp(-0.5 * (u / sigma_t)^2) / Z_h, sigma_t = (K_fir - 1) / 3.0.
     For K_fir = 7, sigma_t = 2.0:
       h = [0.3329, 0.2938, 0.2019, 0.1081, 0.0451, 0.0146, 0.0037].
  2. Learnable thresholded dot-product scores:
       S[t, j] = <Q[t], K_tilde[j]> / sqrt(d_h) - tau
  3. ReLU + p-th power normalization (default p = 10.0):
       pos[t, j] = max(0, S[t, j])
       w_hat[t, j] = pos[t, j]^p / (sum_m pos[t, m]^p + 1e-12)
  4. Peak-gated selective silence (default gamma = 0.5, mode = "peak"):
       peak[t] = max_m pos[t, m]          (in linear score units before .pow(p))
       gate[t] = peak[t] / (peak[t] + gamma)
       W[t, j] = gate[t] * w_hat[t, j]    so that sum_j W[t, j] = gate[t] in [0, 1).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class PnFirConfig:
    d_model: int = 64
    n_heads: int = 4
    p: float = 10.0
    silence_gamma: float | None = 0.5
    silence_mode: str = "peak"
    tau_init: float = 0.0
    fir_k: int = 7
    fir_sigma_frac: float = 3.0
    dropout: float = 0.0
    causal_mask: bool = True
    bias: bool = False


def causal_gaussian_taps(
    k: int,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    sigma_frac: float = 3.0,
) -> torch.Tensor:
    """Causal FIR half-Gaussian taps [k] where index k-1 is present (lag 0) and 0 is oldest lag."""
    assert k % 2 == 1 and k >= 1
    t = torch.arange(k, device=device, dtype=dtype)
    dist = (k - 1) - t  # 0 at present (t = k-1), k-1 at oldest lag (t = 0)
    sigma = max((k - 1) / sigma_frac, 0.5)
    w = torch.exp(-0.5 * (dist / sigma) ** 2)
    return w / w.sum()


def causal_fir(x: torch.Tensor, taps: torch.Tensor) -> torch.Tensor:
    """Depthwise causal 1D FIR along sequence axis N for 3D tensor x: [B, N, C]."""
    B, N, C = x.shape
    K = taps.numel()
    if K <= 1:
        return x
    xt = x.transpose(1, 2).contiguous()  # [B, C, N]
    w = taps.to(dtype=x.dtype, device=x.device).view(1, 1, K).expand(C, 1, K).contiguous()
    yt = F.conv1d(F.pad(xt, (K - 1, 0)), w, groups=C)
    return yt.transpose(1, 2).contiguous()


def fft_lowpass_noncausal(k: torch.Tensor, sigma_div: float = 6.0) -> torch.Tensor:
    """Non-causal Gaussian rFFT low-pass along sequence axis N for k: [B, N, D]."""
    kf = torch.fft.rfft(k, dim=1)
    nf = kf.shape[1]
    sigma = max(nf / sigma_div, 1.0)
    f = torch.arange(nf, device=k.device, dtype=k.dtype).view(1, nf, 1)
    kf = kf * torch.exp(-0.5 * (f / sigma) ** 2)
    return torch.fft.irfft(kf, n=k.shape[1], dim=1)


def power_norm_peak_gate(
    scores: torch.Tensor,
    p: float = 10.0,
    gamma: float | None = 0.5,
    silence_mode: str = "peak",
) -> torch.Tensor:
    """Power-norm with optional soft silence gate on scores [..., Nk].

    Returns attention weights W with row sum sum_k W[..., k] = gate <= 1.0.
    """
    pos = F.relu(scores)
    if p == 10.0:
        x2 = pos * pos
        x4 = x2 * x2
        pw = x4 * x4 * x2
    elif p == 2.0:
        pw = pos * pos
    else:
        pw = torch.where(pos > 0, pos.pow(p), torch.zeros_like(pos))
    mass = pw.sum(dim=-1, keepdim=True)
    w = pw / (mass + 1e-12)
    if gamma is not None and gamma > 0:
        if silence_mode == "peak":
            strength = pos.amax(dim=-1, keepdim=True)
        elif silence_mode == "mass_relu":
            strength = pos.sum(dim=-1, keepdim=True)
        elif silence_mode == "mass_pow":
            strength = mass
        else:
            raise ValueError(f"unknown silence_mode={silence_mode}")
        gate = strength / (strength + gamma)
        w = w * gate
    return w


class SoftmaxMHA(nn.Module):
    """Baseline multi-head Softmax attention (standard KV)."""

    def __init__(self, cfg: PnFirConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.cfg = cfg
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=cfg.bias)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.bias)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, return_weights: bool = False):
        B, N, _ = x.shape
        H, hd = self.n_heads, self.head_dim
        qkv = self.qkv(x).reshape(B, N, 3, H, hd)
        q, k, v = qkv.unbind(2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        scale = 1.0 / math.sqrt(hd)
        att = torch.matmul(q, k.transpose(-2, -1)) * scale
        if self.cfg.causal_mask:
            causal = torch.triu(
                torch.ones(N, N, device=x.device, dtype=torch.bool), diagonal=1
            )
            att = att.masked_fill(causal, -1e4)
        w = self.drop(F.softmax(att, dim=-1))
        out = torch.matmul(w, v).transpose(1, 2).reshape(B, N, -1)
        out = self.out(out)
        if return_weights:
            return out, w
        return out


class SmFirMHA(nn.Module):
    """Softmax + causal FIR on K (causal hardware dual with Softmax normalization)."""

    def __init__(self, cfg: PnFirConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.cfg = cfg
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=cfg.bias)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.bias)
        self.drop = nn.Dropout(cfg.dropout)
        taps = causal_gaussian_taps(
            cfg.fir_k, torch.device("cpu"), torch.float32, cfg.fir_sigma_frac
        )
        self.register_buffer("fir_taps", taps, persistent=True)

    def forward(self, x: torch.Tensor, return_weights: bool = False):
        B, N, _ = x.shape
        H, hd = self.n_heads, self.head_dim
        qkv = self.qkv(x).reshape(B, N, 3, H, hd)
        q, k, v = qkv.unbind(2)
        k = causal_fir(k.reshape(B, N, H * hd), self.fir_taps).view(B, N, H, hd)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        scale = 1.0 / math.sqrt(hd)
        att = torch.matmul(q, k.transpose(-2, -1)) * scale
        if self.cfg.causal_mask:
            causal = torch.triu(
                torch.ones(N, N, device=x.device, dtype=torch.bool), diagonal=1
            )
            att = att.masked_fill(causal, -1e4)
        w = self.drop(F.softmax(att, dim=-1))
        out = torch.matmul(w, v).transpose(1, 2).reshape(B, N, -1)
        out = self.out(out)
        if return_weights:
            return out, w
        return out


class PnFirTauMHA(nn.Module):
    """Canonical PnFirTau multi-head attention (hardware path)."""

    def __init__(self, cfg: PnFirConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        assert cfg.fir_k % 2 == 1
        self.cfg = cfg
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=cfg.bias)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.bias)
        self.drop = nn.Dropout(cfg.dropout)
        self.score_tau = nn.Parameter(torch.tensor(float(cfg.tau_init)))
        taps = causal_gaussian_taps(
            cfg.fir_k, torch.device("cpu"), torch.float32, cfg.fir_sigma_frac
        )
        self.register_buffer("fir_taps", taps, persistent=True)

    def forward(
        self, x: torch.Tensor, return_weights: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        cfg = self.cfg
        B, N, _ = x.shape
        H, hd = self.n_heads, self.head_dim

        qkv = self.qkv(x).reshape(B, N, 3, H, hd)
        q, k, v = qkv.unbind(2)
        k_bn = k.reshape(B, N, H * hd)
        k_bn = causal_fir(k_bn, self.fir_taps)
        k = k_bn.view(B, N, H, hd)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scale = 1.0 / math.sqrt(hd)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale - self.score_tau
        if cfg.causal_mask:
            causal = torch.triu(
                torch.ones(N, N, device=x.device, dtype=torch.bool), diagonal=1
            )
            scores = scores.masked_fill(causal, -1e4)

        w = power_norm_peak_gate(
            scores, cfg.p, cfg.silence_gamma, cfg.silence_mode
        )
        w = self.drop(w)
        out = torch.matmul(w, v).transpose(1, 2).reshape(B, N, -1)
        out = self.out(out)
        if return_weights:
            return out, w
        return out


class PnFirTauTritonMHA(PnFirTauMHA):
    """Fused Triton row-wise PnFirTau multi-head attention (1-kernel FIR + 1-kernel score-to-weight)."""

    def forward(
        self, x: torch.Tensor, return_weights: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        from src.triton_pnfirtau import triton_causal_fir, triton_pn_fir_tau_scores

        cfg = self.cfg
        B, N, _ = x.shape
        H, hd = self.n_heads, self.head_dim

        q_raw, k_raw, v_raw = self.qkv(x).chunk(3, dim=-1)
        k = triton_causal_fir(k_raw, self.fir_taps).view(B, N, H, hd).transpose(1, 2)
        q = q_raw.view(B, N, H, hd).transpose(1, 2)
        v = v_raw.view(B, N, H, hd).transpose(1, 2)

        scale = 1.0 / math.sqrt(hd)
        raw_scores = torch.matmul(q, k.transpose(-2, -1))
        w = triton_pn_fir_tau_scores(
            raw_scores,
            self.score_tau,
            scale=scale,
            p=cfg.p,
            gamma=cfg.silence_gamma,
            causal_mask=cfg.causal_mask,
        )
        w = self.drop(w)
        out = torch.matmul(w, v).transpose(1, 2).reshape(B, N, -1)
        out = self.out(out)
        if return_weights:
            return out, w
        return out


class PnFirTauFlashMHA(PnFirTauMHA):
    """FlashAttention-style O(N)-memory tiled Triton PnFirTau MHA with silent-tile skipping."""

    def forward(
        self, x: torch.Tensor, return_weights: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if return_weights:
            return PnFirTauTritonMHA.forward(self, x, return_weights=True)
        from src.triton_pnfirtau import flash_pn_fir_tau_attention, triton_causal_fir

        cfg = self.cfg
        B, N, _ = x.shape
        H, hd = self.n_heads, self.head_dim

        q_raw, k_raw, v_raw = self.qkv(x).chunk(3, dim=-1)
        k = triton_causal_fir(k_raw, self.fir_taps).view(B, N, H, hd).transpose(1, 2)
        q = q_raw.view(B, N, H, hd).transpose(1, 2)
        v = v_raw.view(B, N, H, hd).transpose(1, 2)

        scale = 1.0 / math.sqrt(hd)
        out = flash_pn_fir_tau_attention(
            q,
            k,
            v,
            self.score_tau,
            scale=scale,
            gamma=cfg.silence_gamma,
            causal_mask=cfg.causal_mask,
        )
        out = out.transpose(1, 2).reshape(B, N, -1)
        out = self.out(out)
        return out

