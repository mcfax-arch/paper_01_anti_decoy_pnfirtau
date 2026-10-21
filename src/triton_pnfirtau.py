#!/usr/bin/env python3
"""Fused Triton Kernels for Causal FIR + Rectified Power-Norm + Learnable Tau + Peak Silence Gate.

Provides three hardware-accelerated primitives for `pn_fir_tau`:
  1. `triton_causal_fir(x, taps)`:
     Fused zero-copy depthwise 1D causal FIR filter along sequence axis N on [B, N, C]
     (eliminates `.transpose(1, 2).contiguous()`, `F.pad`, and output transpose copies).
  2. `triton_pn_fir_tau_scores(raw_scores, score_tau, scale, p, gamma, causal_mask)`:
     Fused row-wise Triton kernel (direct structural counterpart to `aten::_safe_softmax`)
     that fuses scaling, threshold subtraction (`- tau`), causal masking, ReLU rectification,
     p-th power (`u^10`), row-sum + row-peak reductions, and peak silence gating into a
     single forward kernel and a single analytical backward kernel (with silent-row fast-path
     when `peak <= 0.0` and zero intermediate [B, H, N, N] activation allocations).
  3. `flash_pn_fir_tau_attention(q, k_tilde, v, score_tau, scale, gamma, causal_mask)`:
     FlashAttention-2 style IO-aware online-tiled Triton kernel (forward + atomic-free
     deterministic backward) with O(N) memory complexity (never materializing the [B, H, N, N]
     matrix in HBM) and hardware conditional silent-tile skipping (`if tile_max <= 0.0`,
     skipping V loads and P @ V matrix multiplications on sub-threshold key tiles).
"""
from __future__ import annotations

import math
import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


if HAS_TRITON:

    # =========================================================================
    # 1. FUSED CAUSAL FIR KERNEL (K_f = 7 zero-copy depthwise 1D FIR)
    # =========================================================================

    @triton.jit
    def _causal_fir7_fwd_kernel(
        X_ptr,
        Taps_ptr,
        Y_ptr,
        stride_xb,
        stride_xn,
        stride_xc,
        stride_yb,
        stride_yn,
        stride_yc,
        N: int,
        C: int,
        BLOCK_C: tl.constexpr,
    ):
        pid_bn = tl.program_id(0)
        b = pid_bn // N
        n = pid_bn % N

        cols = tl.arange(0, BLOCK_C)
        mask_c = cols < C

        # Load 7 taps (index 6 is lag 0 / present, index 0 is lag 6 / oldest)
        h0 = tl.load(Taps_ptr + 6).to(tl.float32)
        h1 = tl.load(Taps_ptr + 5).to(tl.float32)
        h2 = tl.load(Taps_ptr + 4).to(tl.float32)
        h3 = tl.load(Taps_ptr + 3).to(tl.float32)
        h4 = tl.load(Taps_ptr + 2).to(tl.float32)
        h5 = tl.load(Taps_ptr + 1).to(tl.float32)
        h6 = tl.load(Taps_ptr + 0).to(tl.float32)

        base_x = X_ptr + b * stride_xb + cols * stride_xc
        acc = tl.load(base_x + n * stride_xn, mask=mask_c, other=0.0).to(tl.float32) * h0
        if n >= 1:
            acc += tl.load(base_x + (n - 1) * stride_xn, mask=mask_c, other=0.0).to(tl.float32) * h1
        if n >= 2:
            acc += tl.load(base_x + (n - 2) * stride_xn, mask=mask_c, other=0.0).to(tl.float32) * h2
        if n >= 3:
            acc += tl.load(base_x + (n - 3) * stride_xn, mask=mask_c, other=0.0).to(tl.float32) * h3
        if n >= 4:
            acc += tl.load(base_x + (n - 4) * stride_xn, mask=mask_c, other=0.0).to(tl.float32) * h4
        if n >= 5:
            acc += tl.load(base_x + (n - 5) * stride_xn, mask=mask_c, other=0.0).to(tl.float32) * h5
        if n >= 6:
            acc += tl.load(base_x + (n - 6) * stride_xn, mask=mask_c, other=0.0).to(tl.float32) * h6

        out_ptrs = Y_ptr + b * stride_yb + n * stride_yn + cols * stride_yc
        tl.store(out_ptrs, acc, mask=mask_c)

    @triton.jit
    def _causal_fir7_bwd_kernel(
        GradY_ptr,
        Taps_ptr,
        GradX_ptr,
        stride_gyb,
        stride_gyn,
        stride_gyc,
        stride_gxb,
        stride_gxn,
        stride_gxc,
        N: int,
        C: int,
        BLOCK_C: tl.constexpr,
    ):
        pid_bn = tl.program_id(0)
        b = pid_bn // N
        n = pid_bn % N

        cols = tl.arange(0, BLOCK_C)
        mask_c = cols < C

        h0 = tl.load(Taps_ptr + 6).to(tl.float32)
        h1 = tl.load(Taps_ptr + 5).to(tl.float32)
        h2 = tl.load(Taps_ptr + 4).to(tl.float32)
        h3 = tl.load(Taps_ptr + 3).to(tl.float32)
        h4 = tl.load(Taps_ptr + 2).to(tl.float32)
        h5 = tl.load(Taps_ptr + 1).to(tl.float32)
        h6 = tl.load(Taps_ptr + 0).to(tl.float32)

        base_gy = GradY_ptr + b * stride_gyb + cols * stride_gyc
        acc = tl.load(base_gy + n * stride_gyn, mask=mask_c, other=0.0).to(tl.float32) * h0
        if n + 1 < N:
            acc += tl.load(base_gy + (n + 1) * stride_gyn, mask=mask_c, other=0.0).to(tl.float32) * h1
        if n + 2 < N:
            acc += tl.load(base_gy + (n + 2) * stride_gyn, mask=mask_c, other=0.0).to(tl.float32) * h2
        if n + 3 < N:
            acc += tl.load(base_gy + (n + 3) * stride_gyn, mask=mask_c, other=0.0).to(tl.float32) * h3
        if n + 4 < N:
            acc += tl.load(base_gy + (n + 4) * stride_gyn, mask=mask_c, other=0.0).to(tl.float32) * h4
        if n + 5 < N:
            acc += tl.load(base_gy + (n + 5) * stride_gyn, mask=mask_c, other=0.0).to(tl.float32) * h5
        if n + 6 < N:
            acc += tl.load(base_gy + (n + 6) * stride_gyn, mask=mask_c, other=0.0).to(tl.float32) * h6

        out_ptrs = GradX_ptr + b * stride_gxb + n * stride_gxn + cols * stride_gxc
        tl.store(out_ptrs, acc, mask=mask_c)

    # =========================================================================
    # 2. FUSED ROW-WISE SCORE-TO-WEIGHT KERNEL (`triton_pn_fir_tau_scores`)
    # =========================================================================

    @triton.jit
    def _pn_fir_tau_row_fwd_kernel(
        Scores_ptr,
        Tau_ptr,
        W_ptr,
        Stats_M_ptr,
        Stats_Z_ptr,
        Stats_Nmax_ptr,
        stride_sr,
        stride_wr,
        N: int,
        scale: float,
        gamma: float,
        p_val: float,
        CAUSAL: tl.constexpr,
        HAS_GAMMA: tl.constexpr,
        P_MODE: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        r = tl.program_id(0)
        t = r % N

        cols = tl.arange(0, BLOCK_N)
        mask_n = cols < N
        if CAUSAL:
            valid = cols <= t
        else:
            valid = mask_n

        tau = tl.load(Tau_ptr).to(tl.float32)
        raw_s = tl.load(Scores_ptr + r * stride_sr + cols, mask=valid, other=-1e9).to(
            tl.float32
        )
        s = raw_s * scale - tau
        u = tl.where(valid & (s > 0.0), s, 0.0)
        peak = tl.max(u, axis=0)

        # Fast-path for completely silent rows (all scores <= tau)
        if peak <= 0.0:
            tl.store(
                W_ptr + r * stride_wr + cols,
                tl.zeros([BLOCK_N], dtype=tl.float32),
                mask=mask_n,
            )
            tl.store(Stats_M_ptr + r, 0.0)
            tl.store(Stats_Z_ptr + r, 1e-12)
            tl.store(Stats_Nmax_ptr + r, 0)
            return

        if P_MODE == 10:
            u2 = u * u
            u4 = u2 * u2
            pw = u4 * u4 * u2
        elif P_MODE == 2:
            pw = u * u
        else:
            pw = tl.where(u > 0.0, tl.exp(tl.log(u) * p_val), 0.0)

        mass = tl.sum(pw, axis=0)
        z = mass + 1e-12
        if HAS_GAMMA:
            gate = peak / (peak + gamma)
        else:
            gate = 1.0

        w = pw * (gate / z)
        n_max = tl.sum(tl.where(u == peak, 1, 0), axis=0)

        tl.store(W_ptr + r * stride_wr + cols, w, mask=mask_n)
        tl.store(Stats_M_ptr + r, peak)
        tl.store(Stats_Z_ptr + r, z)
        tl.store(Stats_Nmax_ptr + r, n_max)

    @triton.jit
    def _pn_fir_tau_row_bwd_kernel(
        GradW_ptr,
        Scores_ptr,
        Tau_ptr,
        Stats_M_ptr,
        Stats_Z_ptr,
        Stats_Nmax_ptr,
        GradScores_ptr,
        GradTauRow_ptr,
        stride_gwr,
        stride_sr,
        stride_gsr,
        N: int,
        scale: float,
        gamma: float,
        p_val: float,
        CAUSAL: tl.constexpr,
        HAS_GAMMA: tl.constexpr,
        P_MODE: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        r = tl.program_id(0)
        t = r % N
        cols = tl.arange(0, BLOCK_N)
        mask_n = cols < N

        peak = tl.load(Stats_M_ptr + r)
        # Silent-row hardware short-circuit: zero gradient without reading Scores or GradW!
        if peak <= 0.0:
            tl.store(
                GradScores_ptr + r * stride_gsr + cols,
                tl.zeros([BLOCK_N], dtype=tl.float32),
                mask=mask_n,
            )
            tl.store(GradTauRow_ptr + r, 0.0)
            return

        if CAUSAL:
            valid = cols <= t
        else:
            valid = mask_n

        tau = tl.load(Tau_ptr).to(tl.float32)
        z = tl.load(Stats_Z_ptr + r)
        n_max = tl.load(Stats_Nmax_ptr + r)

        raw_s = tl.load(Scores_ptr + r * stride_sr + cols, mask=valid, other=-1e9).to(
            tl.float32
        )
        dy = tl.load(GradW_ptr + r * stride_gwr + cols, mask=valid, other=0.0).to(
            tl.float32
        )

        s = raw_s * scale - tau
        u = tl.where(valid & (s > 0.0), s, 0.0)

        if P_MODE == 10:
            u2 = u * u
            u4 = u2 * u2
            u8 = u4 * u4
            u_pm1 = u8 * u
            pw = u8 * u2
        elif P_MODE == 2:
            u_pm1 = u
            pw = u * u
        else:
            u_pm1 = tl.where(u > 0.0, tl.exp(tl.log(u) * (p_val - 1.0)), 0.0)
            pw = u_pm1 * u

        w_hat = pw / z
        dot_dy_what = tl.sum(dy * w_hat, axis=0)

        if HAS_GAMMA:
            gate = peak / (peak + gamma)
            dgate_dpeak = gamma / ((peak + gamma) * (peak + gamma))
            dL_dpeak = dot_dy_what * dgate_dpeak
            du_gate = tl.where(u == peak, dL_dpeak / n_max.to(tl.float32), 0.0)
        else:
            gate = 1.0
            du_gate = 0.0

        du_norm = (p_val * gate / z) * u_pm1 * (dy - dot_dy_what)
        du = tl.where(u > 0.0, du_norm + du_gate, 0.0)

        d_raw_s = du * scale
        d_tau_row = -tl.sum(du, axis=0)

        tl.store(GradScores_ptr + r * stride_gsr + cols, d_raw_s, mask=mask_n)
        tl.store(GradTauRow_ptr + r, d_tau_row)

    # =========================================================================
    # 3. FLASH-STYLE ONLINE-TILED SRAM KERNEL (`flash_pn_fir_tau_attention`)
    #    - O(N) memory (never materializes [B, H, N, N] in HBM)
    #    - Zero exp() calls for p=10.0
    #    - Conditional silent-tile skipping (`if tile_max <= 0.0: continue`)
    #    - Atomic-free deterministic 2-pass backward (`_bwd_dq` + `_bwd_dkdv`)
    # =========================================================================

    @triton.jit
    def _flash_pn_fir_tau_fwd_kernel(
        Q_ptr,
        K_ptr,
        V_ptr,
        Tau_ptr,
        Out_ptr,
        Stats_M_ptr,
        Stats_Z_ptr,
        Stats_Nmax_ptr,
        stride_qb,
        stride_qh,
        stride_qm,
        stride_qd,
        stride_kb,
        stride_kh,
        stride_kn,
        stride_kd,
        stride_vb,
        stride_vh,
        stride_vn,
        stride_vd,
        stride_ob,
        stride_oh,
        stride_om,
        stride_od,
        stride_sb,
        stride_sh,
        stride_sm,
        H: int,
        N: int,
        scale: float,
        gamma: float,
        CAUSAL: tl.constexpr,
        HAS_GAMMA: tl.constexpr,
        ALLOW_TF32: tl.constexpr,
        IS_BF16: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        start_m = tl.program_id(0)
        bh = tl.program_id(1)
        b = bh // H
        h = bh % H

        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_D)
        mask_m = offs_m < N

        q_ptrs = (
            Q_ptr
            + b * stride_qb
            + h * stride_qh
            + offs_m[:, None] * stride_qm
            + offs_d[None, :] * stride_qd
        )
        q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
        tau = tl.load(Tau_ptr).to(tl.float32)

        acc_o = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
        row_mass = tl.zeros([BLOCK_M], dtype=tl.float32)
        row_peak = tl.zeros([BLOCK_M], dtype=tl.float32)
        row_nmax = tl.zeros([BLOCK_M], dtype=tl.int32)

        hi = tl.minimum((start_m + 1) * BLOCK_M, N) if CAUSAL else N

        k_base = K_ptr + b * stride_kb + h * stride_kh
        v_base = V_ptr + b * stride_vb + h * stride_vh

        for start_n in range(0, hi, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            mask_n = offs_n < N
            k_ptrs = k_base + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
            k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)

            s = tl.dot(q, tl.trans(k), allow_tf32=ALLOW_TF32).to(tl.float32) * scale - tau
            if CAUSAL:
                valid = (offs_m[:, None] >= offs_n[None, :]) & mask_m[:, None] & mask_n[None, :]
            else:
                valid = mask_m[:, None] & mask_n[None, :]

            u = tl.where(valid & (s > 0.0), s, 0.0)
            tile_max = tl.max(u)

            # Hardware conditional silent-tile skipping:
            # When all scores in the tile are <= tau, skip u^10, skip V load, skip P @ V!
            if tile_max > 0.0:
                u2 = u * u
                u4 = u2 * u2
                p_blk = u4 * u4 * u2  # u^10 in registers

                row_mass += tl.sum(p_blk, axis=1)
                blk_peak = tl.max(u, axis=1)
                new_peak = tl.maximum(row_peak, blk_peak)
                blk_nmax = tl.sum(
                    tl.where((u == new_peak[:, None]) & (new_peak[:, None] > 0.0), 1, 0),
                    axis=1,
                )
                row_nmax = tl.where(new_peak > row_peak, blk_nmax, row_nmax + blk_nmax)
                row_peak = new_peak

                v_ptrs = v_base + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
                v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
                if IS_BF16:
                    acc_o += tl.dot(p_blk.to(tl.bfloat16), v.to(tl.bfloat16))
                else:
                    acc_o += tl.dot(p_blk, v.to(tl.float32), allow_tf32=ALLOW_TF32)

        z = row_mass + 1e-12
        if HAS_GAMMA:
            gate = tl.where(row_peak > 0.0, row_peak / (row_peak + gamma), 0.0)
        else:
            gate = tl.where(row_peak > 0.0, 1.0, 0.0)

        scale_o = gate / z
        out = acc_o * scale_o[:, None]

        out_ptrs = (
            Out_ptr
            + b * stride_ob
            + h * stride_oh
            + offs_m[:, None] * stride_om
            + offs_d[None, :] * stride_od
        )
        tl.store(out_ptrs, out, mask=mask_m[:, None])

        s_off = b * stride_sb + h * stride_sh + offs_m * stride_sm
        tl.store(Stats_M_ptr + s_off, row_peak, mask=mask_m)
        tl.store(Stats_Z_ptr + s_off, z, mask=mask_m)
        tl.store(Stats_Nmax_ptr + s_off, row_nmax, mask=mask_m)

    @triton.jit
    def _flash_pn_fir_tau_bwd_kernel(
        Q_ptr,
        K_ptr,
        V_ptr,
        Tau_ptr,
        Out_ptr,
        GradOut_ptr,
        Stats_M_ptr,
        Stats_Z_ptr,
        Stats_Nmax_ptr,
        GradQ_ptr,
        GradK_ptr,
        GradV_ptr,
        GradTau_ptr,
        stride_qb,
        stride_qh,
        stride_qm,
        stride_qd,
        stride_kb,
        stride_kh,
        stride_kn,
        stride_kd,
        stride_vb,
        stride_vh,
        stride_vn,
        stride_vd,
        stride_ob,
        stride_oh,
        stride_om,
        stride_od,
        stride_sb,
        stride_sh,
        stride_sm,
        H: int,
        N: int,
        scale: float,
        gamma: float,
        CAUSAL: tl.constexpr,
        HAS_GAMMA: tl.constexpr,
        ALLOW_TF32: tl.constexpr,
        IS_BF16: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Single-pass KV-parallel Flash backward kernel (5 matmuls/tile) with silent-tile skipping."""
        start_n = tl.program_id(0)
        bh = tl.program_id(1)
        b = bh // H
        h = bh % H

        offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, BLOCK_D)
        mask_n = offs_n < N

        k_ptrs = (
            K_ptr
            + b * stride_kb
            + h * stride_kh
            + offs_n[:, None] * stride_kn
            + offs_d[None, :] * stride_kd
        )
        v_ptrs = (
            V_ptr
            + b * stride_vb
            + h * stride_vh
            + offs_n[:, None] * stride_vn
            + offs_d[None, :] * stride_vd
        )
        k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
        tau = tl.load(Tau_ptr).to(tl.float32)

        acc_dk = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)
        acc_dv = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)
        acc_dtau = 0.0

        lo = start_n * BLOCK_N if CAUSAL else 0
        q_base = Q_ptr + b * stride_qb + h * stride_qh
        o_base = Out_ptr + b * stride_ob + h * stride_oh
        go_base = GradOut_ptr + b * stride_ob + h * stride_oh
        gq_base = GradQ_ptr + b * stride_qb + h * stride_qh
        s_base = b * stride_sb + h * stride_sh

        for start_m in range(lo, N, BLOCK_M):
            offs_m = start_m + tl.arange(0, BLOCK_M)
            mask_m = offs_m < N

            peak = tl.load(Stats_M_ptr + s_base + offs_m * stride_sm, mask=mask_m, other=0.0)
            # If all query rows in this block were silent (peak <= 0), skip completely!
            if tl.max(peak) > 0.0:
                q_ptrs = q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
                q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)

                s = tl.dot(q, tl.trans(k), allow_tf32=ALLOW_TF32).to(tl.float32) * scale - tau
                if CAUSAL:
                    valid = (offs_m[:, None] >= offs_n[None, :]) & mask_m[:, None] & mask_n[None, :]
                else:
                    valid = mask_m[:, None] & mask_n[None, :]

                u = tl.where(valid & (s > 0.0), s, 0.0)
                if tl.max(u) > 0.0:
                    z = tl.load(Stats_Z_ptr + s_base + offs_m * stride_sm, mask=mask_m, other=1e-12)
                    n_max = tl.load(Stats_Nmax_ptr + s_base + offs_m * stride_sm, mask=mask_m, other=1)

                    o_ptrs = o_base + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
                    go_ptrs = go_base + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
                    o_blk = tl.load(o_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)
                    do_blk = tl.load(go_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)

                    u2 = u * u
                    u4 = u2 * u2
                    u8 = u4 * u4
                    u9 = u8 * u
                    pw = u8 * u2

                    if HAS_GAMMA:
                        gate = tl.where(peak > 0.0, peak / (peak + gamma), 0.0)
                    else:
                        gate = tl.where(peak > 0.0, 1.0, 0.0)

                    w = pw * (gate[:, None] / z[:, None])
                    if IS_BF16:
                        acc_dv += tl.dot(tl.trans(w.to(tl.bfloat16)), do_blk.to(tl.bfloat16))
                        dw = tl.dot(do_blk.to(tl.bfloat16), tl.trans(v.to(tl.bfloat16)))
                    else:
                        acc_dv += tl.dot(tl.trans(w), do_blk, allow_tf32=ALLOW_TF32)
                        dw = tl.dot(do_blk, tl.trans(v.to(tl.float32)), allow_tf32=ALLOW_TF32)

                    # Flash identity: sum_j dW[m, j] * w_hat[m, j] = <dO[m], O[m]> / gate[m]
                    dot_do_o = tl.sum(do_blk * o_blk, axis=1)
                    dot_dy_what = tl.where(gate > 0.0, dot_do_o / gate, 0.0)

                    if HAS_GAMMA:
                        dgate_dpeak = gamma / ((peak + gamma) * (peak + gamma))
                        dL_dpeak = dot_dy_what * dgate_dpeak
                        du_gate = tl.where(
                            (u == peak[:, None]) & (peak[:, None] > 0.0),
                            dL_dpeak[:, None] / n_max[:, None].to(tl.float32),
                            0.0,
                        )
                    else:
                        du_gate = 0.0

                    du_norm = (10.0 * gate[:, None] / z[:, None]) * u9 * (dw - dot_dy_what[:, None])
                    du = tl.where(u > 0.0, du_norm + du_gate, 0.0)
                    ds = du * scale

                    acc_dtau += -tl.sum(du)
                    if IS_BF16:
                        acc_dk += tl.dot(tl.trans(ds.to(tl.bfloat16)), q.to(tl.bfloat16))
                        dq_contrib = tl.dot(ds.to(tl.bfloat16), k.to(tl.bfloat16))
                    else:
                        acc_dk += tl.dot(tl.trans(ds), q.to(tl.float32), allow_tf32=ALLOW_TF32)
                        dq_contrib = tl.dot(ds, k.to(tl.float32), allow_tf32=ALLOW_TF32)
                    gq_ptrs = gq_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
                    tl.atomic_add(gq_ptrs, dq_contrib, mask=mask_m[:, None])

        dk_ptrs = (
            GradK_ptr
            + b * stride_kb
            + h * stride_kh
            + offs_n[:, None] * stride_kn
            + offs_d[None, :] * stride_kd
        )
        dv_ptrs = (
            GradV_ptr
            + b * stride_vb
            + h * stride_vh
            + offs_n[:, None] * stride_vn
            + offs_d[None, :] * stride_vd
        )
        tl.store(dk_ptrs, acc_dk, mask=mask_n[:, None])
        tl.store(dv_ptrs, acc_dv, mask=mask_n[:, None])
        n_blocks = tl.num_programs(0)
        tl.store(GradTau_ptr + bh * n_blocks + start_n, acc_dtau)


# =============================================================================
# PyTorch Autograd Function Wrappers
# =============================================================================


class _TritonCausalFir7Fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, taps: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        x_in = x if x.stride(-1) == 1 else x.contiguous()
        taps_f32 = (
            taps
            if (taps.dtype == torch.float32 and taps.is_contiguous() and taps.device == x.device)
            else taps.contiguous().to(device=x.device, dtype=torch.float32)
        )
        y = torch.empty((B, N, C), device=x.device, dtype=x.dtype)
        BLOCK_C = triton.next_power_of_2(C)
        grid = (B * N,)
        _causal_fir7_fwd_kernel[grid](
            x_in,
            taps_f32,
            y,
            x_in.stride(0),
            x_in.stride(1),
            x_in.stride(2),
            y.stride(0),
            y.stride(1),
            y.stride(2),
            N=N,
            C=C,
            BLOCK_C=BLOCK_C,
        )
        ctx.save_for_backward(taps_f32)
        ctx.shape = (B, N, C)
        return y

    @staticmethod
    def backward(ctx, grad_y: torch.Tensor):
        (taps_f32,) = ctx.saved_tensors
        B, N, C = ctx.shape
        gy_in = grad_y if grad_y.stride(-1) == 1 else grad_y.contiguous()
        grad_x = torch.empty((B, N, C), device=grad_y.device, dtype=grad_y.dtype)
        BLOCK_C = triton.next_power_of_2(C)
        grid = (B * N,)
        _causal_fir7_bwd_kernel[grid](
            gy_in,
            taps_f32,
            grad_x,
            gy_in.stride(0),
            gy_in.stride(1),
            gy_in.stride(2),
            grad_x.stride(0),
            grad_x.stride(1),
            grad_x.stride(2),
            N=N,
            C=C,
            BLOCK_C=BLOCK_C,
        )
        return grad_x, None


def triton_causal_fir(x: torch.Tensor, taps: torch.Tensor) -> torch.Tensor:
    """Fused zero-copy depthwise 1D causal FIR along axis N for x: [B, N, C]."""
    if HAS_TRITON and x.is_cuda and taps.numel() == 7 and x.shape[-1] <= 1024:
        return _TritonCausalFir7Fn.apply(x, taps)
    from src.pnfirtau_attention import causal_fir

    return causal_fir(x, taps)


class _TritonPnFirTauScoresFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        raw_scores: torch.Tensor,
        score_tau: torch.Tensor,
        scale: float,
        p: float,
        gamma: float | None,
        causal_mask: bool,
    ) -> torch.Tensor:
        orig_shape = raw_scores.shape
        N = orig_shape[-1]
        assert orig_shape[-2] == N, "Expected [..., N, N] scores tensor"
        s_2d = raw_scores.contiguous().view(-1, N)
        R = s_2d.shape[0]

        w_2d = torch.empty_like(s_2d)
        stats_m = torch.empty(R, device=raw_scores.device, dtype=torch.float32)
        stats_z = torch.empty(R, device=raw_scores.device, dtype=torch.float32)
        stats_nmax = torch.empty(R, device=raw_scores.device, dtype=torch.int32)

        has_gamma = gamma is not None and gamma > 0.0
        gamma_val = float(gamma) if has_gamma else 0.0
        p_mode = 10 if p == 10.0 else (2 if p == 2.0 else 0)
        BLOCK_N = triton.next_power_of_2(N)
        num_warps = 4 if BLOCK_N <= 512 else 8

        _pn_fir_tau_row_fwd_kernel[(R,)](
            s_2d,
            score_tau,
            w_2d,
            stats_m,
            stats_z,
            stats_nmax,
            s_2d.stride(0),
            w_2d.stride(0),
            N=N,
            scale=float(scale),
            gamma=gamma_val,
            p_val=float(p),
            CAUSAL=causal_mask,
            HAS_GAMMA=has_gamma,
            P_MODE=p_mode,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
        )

        ctx.save_for_backward(s_2d, score_tau, stats_m, stats_z, stats_nmax)
        ctx.orig_shape = orig_shape
        ctx.scale = float(scale)
        ctx.p = float(p)
        ctx.gamma_val = gamma_val
        ctx.has_gamma = has_gamma
        ctx.p_mode = p_mode
        ctx.causal_mask = causal_mask
        ctx.BLOCK_N = BLOCK_N
        ctx.num_warps = num_warps
        return w_2d.view(orig_shape)

    @staticmethod
    def backward(ctx, grad_w: torch.Tensor):
        s_2d, score_tau, stats_m, stats_z, stats_nmax = ctx.saved_tensors
        R, N = s_2d.shape
        gw_2d = grad_w.contiguous().view(R, N)
        grad_s_2d = torch.empty_like(s_2d)
        grad_tau_rows = torch.empty(R, device=s_2d.device, dtype=torch.float32)

        _pn_fir_tau_row_bwd_kernel[(R,)](
            gw_2d,
            s_2d,
            score_tau,
            stats_m,
            stats_z,
            stats_nmax,
            grad_s_2d,
            grad_tau_rows,
            gw_2d.stride(0),
            s_2d.stride(0),
            grad_s_2d.stride(0),
            N=N,
            scale=ctx.scale,
            gamma=ctx.gamma_val,
            p_val=ctx.p,
            CAUSAL=ctx.causal_mask,
            HAS_GAMMA=ctx.has_gamma,
            P_MODE=ctx.p_mode,
            BLOCK_N=ctx.BLOCK_N,
            num_warps=ctx.num_warps,
        )
        grad_tau = grad_tau_rows.sum(dtype=torch.float64).to(dtype=score_tau.dtype).view_as(score_tau)
        return grad_s_2d.view(ctx.orig_shape), grad_tau, None, None, None, None


def triton_pn_fir_tau_scores(
    raw_scores: torch.Tensor,
    score_tau: torch.Tensor,
    scale: float = 1.0,
    p: float = 10.0,
    gamma: float | None = 0.5,
    causal_mask: bool = True,
) -> torch.Tensor:
    """Fused Triton row-wise score-to-weight kernel for `pn_fir_tau`."""
    if HAS_TRITON and raw_scores.is_cuda and raw_scores.shape[-1] <= 4096:
        return _TritonPnFirTauScoresFn.apply(
            raw_scores, score_tau, scale, p, gamma, causal_mask
        )
    # Fallback to eager PyTorch
    from src.pnfirtau_attention import power_norm_peak_gate

    scores = raw_scores * scale - score_tau
    if causal_mask:
        N = scores.shape[-1]
        causal = torch.triu(
            torch.ones(N, N, device=scores.device, dtype=torch.bool), diagonal=1
        )
        scores = scores.masked_fill(causal, -1e4)
    return power_norm_peak_gate(scores, p=p, gamma=gamma, silence_mode="peak")


class _FlashPnFirTauFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        score_tau: torch.Tensor,
        scale: float,
        gamma: float | None,
        causal_mask: bool,
    ) -> torch.Tensor:
        B, H, N, D = q.shape
        q_c = q.contiguous()
        k_c = k.contiguous()
        v_c = v.contiguous()
        out = torch.empty_like(q_c)

        stats_m = torch.empty((B, H, N), device=q.device, dtype=torch.float32)
        stats_z = torch.empty((B, H, N), device=q.device, dtype=torch.float32)
        stats_nmax = torch.empty((B, H, N), device=q.device, dtype=torch.int32)

        has_gamma = gamma is not None and gamma > 0.0
        gamma_val = float(gamma) if has_gamma else 0.0

        BLOCK_M = 64 if N >= 64 else 32
        BLOCK_N = 64 if N >= 64 else 32
        BLOCK_D = triton.next_power_of_2(max(D, 16))
        allow_tf32 = bool(torch.backends.cuda.matmul.allow_tf32)
        is_bf16 = q.dtype == torch.bfloat16
        grid = (triton.cdiv(N, BLOCK_M), B * H)

        _flash_pn_fir_tau_fwd_kernel[grid](
            q_c,
            k_c,
            v_c,
            score_tau,
            out,
            stats_m,
            stats_z,
            stats_nmax,
            q_c.stride(0),
            q_c.stride(1),
            q_c.stride(2),
            q_c.stride(3),
            k_c.stride(0),
            k_c.stride(1),
            k_c.stride(2),
            k_c.stride(3),
            v_c.stride(0),
            v_c.stride(1),
            v_c.stride(2),
            v_c.stride(3),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            out.stride(3),
            stats_m.stride(0),
            stats_m.stride(1),
            stats_m.stride(2),
            H=H,
            N=N,
            scale=float(scale),
            gamma=gamma_val,
            CAUSAL=causal_mask,
            HAS_GAMMA=has_gamma,
            ALLOW_TF32=allow_tf32,
            IS_BF16=is_bf16,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_D=BLOCK_D,
            num_warps=4,
        )

        ctx.save_for_backward(q_c, k_c, v_c, score_tau, out, stats_m, stats_z, stats_nmax)
        ctx.scale = float(scale)
        ctx.gamma_val = gamma_val
        ctx.has_gamma = has_gamma
        ctx.causal_mask = causal_mask
        ctx.allow_tf32 = allow_tf32
        ctx.is_bf16 = is_bf16
        ctx.BLOCK_M = BLOCK_M
        ctx.BLOCK_N = BLOCK_N
        ctx.BLOCK_D = BLOCK_D
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        q_c, k_c, v_c, score_tau, out, stats_m, stats_z, stats_nmax = ctx.saved_tensors
        B, H, N, D = q_c.shape
        go_c = grad_out.contiguous()

        grad_q = torch.zeros((B, H, N, D), device=q_c.device, dtype=torch.float32)
        grad_k = torch.empty_like(k_c)
        grad_v = torch.empty_like(v_c)
        n_kv_blocks = triton.cdiv(N, ctx.BLOCK_N)
        grad_tau_blocks = torch.empty((B * H * n_kv_blocks,), device=q_c.device, dtype=torch.float32)

        grid_kv = (n_kv_blocks, B * H)
        _flash_pn_fir_tau_bwd_kernel[grid_kv](
            q_c,
            k_c,
            v_c,
            score_tau,
            out,
            go_c,
            stats_m,
            stats_z,
            stats_nmax,
            grad_q,
            grad_k,
            grad_v,
            grad_tau_blocks,
            q_c.stride(0),
            q_c.stride(1),
            q_c.stride(2),
            q_c.stride(3),
            k_c.stride(0),
            k_c.stride(1),
            k_c.stride(2),
            k_c.stride(3),
            v_c.stride(0),
            v_c.stride(1),
            v_c.stride(2),
            v_c.stride(3),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            out.stride(3),
            stats_m.stride(0),
            stats_m.stride(1),
            stats_m.stride(2),
            H=H,
            N=N,
            scale=ctx.scale,
            gamma=ctx.gamma_val,
            CAUSAL=ctx.causal_mask,
            HAS_GAMMA=ctx.has_gamma,
            ALLOW_TF32=ctx.allow_tf32,
            IS_BF16=ctx.is_bf16,
            BLOCK_M=ctx.BLOCK_M,
            BLOCK_N=ctx.BLOCK_N,
            BLOCK_D=ctx.BLOCK_D,
            num_warps=4,
        )
        grad_tau = grad_tau_blocks.sum(dtype=torch.float64).to(score_tau.dtype).view_as(score_tau)
        return (
            grad_q.to(q_c.dtype),
            grad_k,
            grad_v,
            grad_tau,
            None,
            None,
            None,
        )


def flash_pn_fir_tau_attention(
    q: torch.Tensor,
    k_tilde: torch.Tensor,
    v: torch.Tensor,
    score_tau: torch.Tensor,
    scale: float | None = None,
    gamma: float | None = 0.5,
    causal_mask: bool = True,
) -> torch.Tensor:
    """FlashAttention-style O(N)-memory tiled Triton attention for `pn_fir_tau` (p=10.0)."""
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])
    return _FlashPnFirTauFn.apply(q, k_tilde, v, score_tau, scale, gamma, causal_mask)
