#!/usr/bin/env python3
"""Experiment 4e: C2 multi-seed 50k replication (PAPER_01_REWORK_PLAN.md).

One invocation trains ONE (variant, seed) 50k run under the exact C1/pn-history
protocol: FP32 steps 0-1000, then resume with BF16+TF32 steps 1001-50000
(upstream script I:/FFT_Attention/experiments/wikitext_pn_fir_lm/experiment.py,
same machine/code/data/cache). Seeds 43/44 complement the existing seed-42
points (softmax-BF16 from C1, pn_fir_tau historical) to n=3 per variant.

Isolation: checkpoints go to an absolute per-run dir
(<upstream>/checkpoints_c2/<variant>_seed<seed>/); upstream training_log.json
is NOT touched (LOG redirected per run); upstream results.json/REPORT.md are
never written by this driver (it only READS them for the seed-42 reference).

Resume-safety: if <run-ckpt-dir>/latest.pt exists, phase 1 is skipped and phase 2
resumes directly — relaunching a killed invocation loses at most ~10k steps.

Usage:
  python3 experiments/exp4e_multiseed_50k.py --variant softmax --seed 43
Chain: softmax/43 && softmax/44 && pn_fir_tau/43 && pn_fir_tau/44
Artifact (accumulated across invocations):
  results/exp4e_multiseed_50k.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def _upstream_dir() -> str:
    cands = [
        r"I:\FFT_Attention\experiments\wikitext_pn_fir_lm",
        "/mnt/i/FFT_Attention/experiments/wikitext_pn_fir_lm",
    ]
    for c in cands:
        if os.path.isdir(c):
            return c
    raise FileNotFoundError("upstream wikitext_pn_fir_lm dir not found")


UPSTREAM = _upstream_dir()
sys.path.insert(0, UPSTREAM)
import experiment as U  # noqa: E402  (upstream 50k script; main() guarded)

RESULTS_DIR = os.path.join(ROOT, "results")
OUT_JSON = os.path.join(RESULTS_DIR, "exp4e_multiseed_50k.json")
CKPT_C2_ROOT = os.path.join(UPSTREAM, "checkpoints_c2")


def run_one(variant: str, seed: int, ckpt_every: int = 10000) -> dict:
    run_ckpt = os.path.join(CKPT_C2_ROOT, f"{variant}_seed{seed}")
    os.makedirs(run_ckpt, exist_ok=True)
    U.LOG = os.path.join(run_ckpt, "training_log.json")  # do not touch upstream log
    U.configure_fast_cuda()

    base = U.Cfg()
    cfg1 = replace(base, steps=1001, amp="off", seed=seed, ckpt_dir=run_ckpt)
    cfg2 = replace(base, steps=50000, amp="bf16", seed=seed, ckpt_dir=run_ckpt,
                   ckpt_every=ckpt_every)

    # Data pipeline: identical to upstream main()
    tr = U.load_tokens("train")
    va = U.load_tokens("validation")
    te = U.load_tokens("test")
    vocab, stoi = U.build_vocab(tr, cfg2.vocab_min_freq)
    V = len(vocab)
    train_c = U.make_chunks(U.encode(tr, stoi), cfg2.max_seq)
    val_c = U.make_chunks(U.encode(va, stoi), cfg2.max_seq)
    test_c = U.make_chunks(U.encode(te, stoi), cfg2.max_seq)
    print(f"data: vocab={V} chunks={len(train_c)}/{len(val_c)}/{len(test_c)}", flush=True)

    t_all = time.time()
    latest = os.path.join(U._ckpt_dir(cfg2, variant), "latest.pt")
    resumed_midrun = False
    if os.path.isfile(latest):
        print(f"RESUME-MIDRUN from {latest} (phase 1 skipped)", flush=True)
        resumed_midrun = True
    else:
        print(f"--- phase 1: {variant} seed={seed} FP32 0-1000 ---", flush=True)
        U.set_seed(seed)
        m1 = U.LM(V, cfg1, variant)
        r1 = U.train_variant(m1, train_c, val_c, test_c, cfg1, resume_path=None)
        print(f"phase1 done val={r1['val_ppl']:.2f}", flush=True)
        del m1
        import torch as _t
        if _t.cuda.is_available():
            _t.cuda.empty_cache()

    print(f"--- phase 2: {variant} seed={seed} resume BF16+TF32 -> 50000 ---", flush=True)
    U.set_seed(seed)
    m2 = U.LM(V, cfg2, variant)
    r2 = U.train_variant(m2, train_c, val_c, test_c, cfg2, resume_path=latest)
    print(
        f"DONE {variant} seed={seed} val={r2['val_ppl']:.2f} "
        f"test={r2['test_ppl']:.2f} train={r2['train_ppl']:.2f} "
        f"time={r2['time_sec']/3600:.2f}h causal={r2['causal']['passed']}",
        flush=True,
    )
    r2["run_ckpt_dir"] = run_ckpt
    r2["resumed_midrun"] = resumed_midrun
    r2["wall_sec_total"] = round(time.time() - t_all, 1)
    return r2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--ckpt-every", type=int, default=10000)
    args = ap.parse_args()

    key = f"{args.variant}/seed{args.seed}"
    if os.path.isfile(OUT_JSON):
        with open(OUT_JSON, encoding="utf-8") as f:
            existing = json.load(f)
        if key in existing.get("runs", {}):
            print(f"SKIP {key}: already complete in {OUT_JSON} (idempotent relaunch)", flush=True)
            return
    res = run_one(args.variant, args.seed, args.ckpt_every)
    payload = {"config_note": (
        "C2: FP32 0-1000 then resume BF16+TF32 to 50000; eval_every=1000; "
        "same script/machine/data/cache as C1 and the historical pn run. "
        "Seed-42 points live in exp4d_softmax_50k_bf16.json (softmax) and the "
        "upstream historical results.json (pn_fir_tau)."
    ), "runs": {}}
    if os.path.isfile(OUT_JSON):
        with open(OUT_JSON, encoding="utf-8") as f:
            payload = json.load(f)
    payload["runs"][key] = {
        "variant": res["variant"],
        "params": res["params"],
        "train_ppl": res["train_ppl"],
        "val_ppl": res["val_ppl"],
        "test_ppl": res["test_ppl"],
        "time_sec": res["time_sec"],
        "causal": res["causal"],
        "tau_per_layer": res.get("tau_per_layer", []),
        "history": res["history"],
        "run_ckpt_dir": res["run_ckpt_dir"],
        "resumed_midrun": res["resumed_midrun"],
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"saved {key} -> {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
