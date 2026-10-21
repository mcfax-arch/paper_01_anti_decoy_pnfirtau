# Attention with a Calibrated Off Switch: Causal FIR Key Smoothing, the Score-Shift Pathology, and Selective Silence in Rectified Attention (`pn_fir_tau`)

[![Preprint DOI](https://img.shields.io/badge/DOI-10.21203%2Frs.3.rs--11435949%2Fv1-blue)](https://doi.org/10.21203/rs.3.rs-11435949/v1)
[![Software DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23545301.svg)](https://doi.org/10.5281/zenodo.23545302)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Self-contained codebase, benchmark suite, raw experimental outputs (`results/*.json`), and figure-generation pipeline for the preprint **"Attention with a Calibrated Off Switch: Causal FIR Key Smoothing, the Score-Shift Pathology, and Selective Silence in Rectified Attention (`pn_fir_tau`)"** by **Maxim Zhukov** (ORCID [0009-0005-9238-3778](https://orcid.org/0009-0005-9238-3778), Independent Research).

Current manuscript: **22 pages, 18 figures, 12 tables**; built from `paper_en.tex` with Tectonic (see "Build the PDF" below). **Preprint DOI:** [10.21203/rs.3.rs-11435949/v1](https://doi.org/10.21203/rs.3.rs-11435949/v1) (Research Square, 2026). **Reproducibility package DOI:** [10.5281/zenodo.23545302](https://doi.org/10.5281/zenodo.23545302) (Zenodo, v1.0.0; concept DOI [10.5281/zenodo.23545301](https://doi.org/10.5281/zenodo.23545301)). The Scope paragraph in §1 frames the work as a controlled mechanistic study on deliberately small sandboxes, not an LLM-scale claim.

---

## Repository Structure

```text
paper_01_anti_decoy_pnfirtau/
├── src/
│   ├── __init__.py
│   ├── pnfirtau_attention.py                # Core modules: PnFirConfig, PnFirTauMHA, PnFirTauTritonMHA,
│   │                                        # PnFirTauFlashMHA, SoftmaxMHA, SmFirMHA, causal_fir, power_norm_peak_gate
│   └── triton_pnfirtau.py                   # Fused Triton kernels: triton_causal_fir (zero-copy K_f=7 FIR),
│                                            # triton_pn_fir_tau_scores (row-fused u^10 + peak_gate, 0 exp()),
│                                            # flash_pn_fir_tau_attention (O(N)-VRAM tiled Flash + silent-tile skipping)
├── experiments/
│   ├── exp1_legacy_stack_candidates.py      # Exp 1: 13-stack benchmark (3 seeds) + un-gated ablations +
│   │                                        #        attention position & score-shift diagnostics
│   ├── exp2_content_addressable_decoy.py    # Exp 2: Non-bijective random-pairing associative recall (M=3 pairs
│   │                                        #        at interior positions) + distractor decoys + silence reg
│   │   └── exp4b_mha6_instrumentation.py     # R2 close-out: drift-controlled re-measurement of the
│   │                                         #        6-layer Flash stack (rotated interleave over tau,
│   │                                         #        per-layer 64x64 tile diagnostics, CUDA-event
│   │                                         #        spans, launch counts + B-scaling discriminator)
│   │   └── exp4c_cuda_graphs.py              # CUDA-graph counterfactual: launch amortisation surfaces
│   │                                         #        the silent-tile saving (replay 12.10 -> 4.91 ms)
│   │   └── exp4e_multiseed_50k.py            # C2: multi-seed 50k replication (one (variant,seed) per call)
│   │   └── exp4f_scaleout.py                 # C3: width sweep (exp1 d=32/64/128) + WikiText-2 D=256/N=512 matched run
│   │   └── exp4g_tau_scaling_law.py          # C3c: width ladder d=16..128 + entropy gap + tau*(d) fit + predicted-init validation
│   │   └── exp4h_int8_quant.py               # C5: INT8 activation-noise PTQ-sim + sink/outlier stats on trained checkpoints
│   │   └── exp6_token_shift_ablation.py      # C4: 2-tap RWKV-style shift vs 7-tap FIR on K (exp1+exp2 decoy protocols)
│   ├── exp3_sweep_tau_gate.py               # Exp 3: 31-configuration sweep of (tau_init x gamma x silence_mode)
│   ├── exp4_wikitext2_lm.py                 # Exp 4: WikiText-2 6-layer Pre-LN LM step-time benchmark, fused Triton
│   │                                        #        & Flash scaling/silent-tile benchmarks, matched 5-variant runs, & 50k trajectory
│   └── exp5_abstention_and_entmax.py        # Exp 5: Context Override vs. Residual Preservation under Pure CE Loss
│                                            #        (sm_fir, sparsemax_fir, entmax15_fir, entmax_p10_fir, softmax1_fir,
│                                            #         softmax_sink_fir, pn_fir, pn_fir_c, pn_fir_tau)
├── results/
│   ├── exp1_legacy_13stacks.json            # Raw per-seed & aggregated metrics for Experiment 1
│   ├── exp2_content_addressable_decoy.json  # Raw per-seed & aggregated metrics for Experiment 2
│   ├── exp3_sweep_tau_gate.json             # Raw 31-configuration sweep metrics for Experiment 3
│   ├── exp4_wikitext2_lm.json               # Raw WikiText-2 step benchmark, Triton MHA benchmarks, matched runs, & 50k curves
│   ├── exp4b_mha6_instrumentation.json      # R2 close-out: drift-controlled 6-layer stack re-measurement (rotated interleave + B-scaling)
│   ├── exp4b_sequential_repro.json          # Sequential-protocol reproduction (ascending & descending block order)
│   ├── exp4c_cuda_graphs.json               # CUDA-graph counterfactual: launch amortisation (rotated interleave)
│   ├── exp6_token_shift_ablation.json       # C4: 2-tap shift vs 7-tap FIR on K (legacy + content-addressable decoy protocols)
│   ├── exp4d_softmax_50k_bf16.json           # C1: matched-precision softmax 50k (BF16+TF32, same schedule as pn 50k)
│   ├── exp4e_multiseed_50k.json              # C2: multi-seed 50k replication (accumulated per (variant,seed) run)
│   ├── exp4f_scaleout.json                  # C3: width sweep + D=256/N=512 matched run
│   ├── exp4g_tau_scaling_law.json           # C3c: width ladder + H-gap + tau*(d) fit + validation
│   ├── exp4h_int8_quant.json               # C5: INT8-sim PPL + sink/outlier stats per trained checkpoint
│   └── exp5_abstention_and_entmax.json      # Raw 3-seed metrics for Experiment 5 (9 normalizers)
├── figures/                                  # 18 figures (fig1–fig18, PNG+PDF) + paper01_metrics.json
├── run_all_and_build_figures.py             # Builds fig1–fig4 from results/*.json (0 hardcoded arrays)
├── paper_en.tex                             # English LaTeX manuscript (arXiv-ready)
├── paper_en.md                              # English Markdown manuscript
├── paper_ru.md                              # Russian Markdown manuscript
├── requirements.txt                         # Python dependencies
└── README.md
```

---

## Reproducing All Experiments and Figures

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Individual Experiments

Each script is 100% self-contained (importing only from `src/`) and writes its complete per-seed and aggregated results to `results/`:

```bash
# Experiment 1: 13-stack benchmark + un-gated ablations + position/score diagnostics (3 seeds: 1234, 2025, 3141)
python experiments/exp1_legacy_stack_candidates.py

# Experiment 2: True Content-Addressable Non-Bijective Associative Recall & Distractor Decoys (3 seeds)
python experiments/exp2_content_addressable_decoy.py

# Experiment 3: 31-configuration (tau_init x silence_gamma x silence_mode) sweep
python experiments/exp3_sweep_tau_gate.py

# Experiment 4: WikiText-2 Pre-LN Transformer LM step-time benchmark + Triton/Flash benchmarks + matched 5-variant training
python experiments/exp4_wikitext2_lm.py --steps 2400 --eval-every 400

# Experiment 4b (R2 close-out): drift-controlled 6-layer Flash stack re-measurement (Linux/WSL2, CUDA GPU).
# Requires exactly the paper environment (torch + triton + CUDA); writes results/exp4b_mha6_instrumentation.json
# (rotated-interleave stack & kern1 timing, per-layer 64x64 tile diagnostics, B-scaling discriminator)
# and results/exp4b_sequential_repro.json (block-order artifact reproduction of the Table 5E mha6 column).
python experiments/exp4b_mha6_instrumentation.py
python experiments/exp4b_mha6_instrumentation.py --repro-sequential

# Experiment 4c (CUDA-graph counterfactual): capture the 6-layer Flash stack step in a CUDA graph
# (launch amortisation) and measure the tau delta under rotated interleave. Writes results/exp4c_cuda_graphs.json.
python experiments/exp4c_cuda_graphs.py

# Experiment 6 (C4 token-shift ablation): fixed 2-tap vs learnable RWKV-style mu shift vs 7-tap FIR
# on K, under the exp1 legacy and exp2 content-addressable decoy protocols (3 seeds).
# Writes results/exp6_token_shift_ablation.json.
python experiments/exp6_token_shift_ablation.py

# Experiment 4e (C2 multi-seed 50k): one (variant, seed) 50k run per call under the exact
# C1 protocol (FP32 0-1000, resume BF16+TF32 to 50000). Seeds 43/44 complement seed 42.
# Accumulates into results/exp4e_multiseed_50k.json. Each run ~0.7h (softmax) / ~2.4h (pn).
python experiments/exp4e_multiseed_50k.py --variant softmax --seed 43
python experiments/exp4e_multiseed_50k.py --variant softmax --seed 44
python experiments/exp4e_multiseed_50k.py --variant pn_fir_tau --seed 43
python experiments/exp4e_multiseed_50k.py --variant pn_fir_tau --seed 44

# Experiment 4f (C3 scale-out): Part A = exp1 protocol at widths d=32/64/128 (+lr probe);
# Part B = WikiText-2 matched run at D=256/N=512 per variant (checkpointed, resumable).
# Writes results/exp4f_scaleout.json.
python experiments/exp4f_scaleout.py --part a
python experiments/exp4f_scaleout.py --part b --variant softmax
python experiments/exp4f_scaleout.py --part b --variant sm_fir
python experiments/exp4f_scaleout.py --part b --variant pn_fir_tau

# Experiment 4g (C3c tau scaling law): width ladder d=16..128 (entropy gap diagnostics),
# tau*(d) fit, and predicted-tau_init validation at d=128. Idempotent/resumable.
# Writes results/exp4g_tau_scaling_law.json.
python experiments/exp4g_tau_scaling_law.py --stage all

# Experiment 4h (C5 INT8 quantization): outlier/sink stats + per-token INT8 activation-noise
# PTQ simulation on trained checkpoints (D=64 50k best x2, D=256 2.4k latest x3).
# Idempotent per model. Writes results/exp4h_int8_quant.json.
python experiments/exp4h_int8_quant.py

# Experiment 5: Context Override vs. Residual Stream Preservation (Sparsemax, Entmax, Sinks, PnFirTau; 3 seeds)
python experiments/exp5_abstention_and_entmax.py
```

### 3. Regenerate All Figures and Consolidated Metrics

`run_all_and_build_figures.py` reads `results/exp1..5_*.json` directly (with zero hardcoded data arrays) and regenerates fig1–fig4 in `figures/` (`PNG` and `PDF`) plus `paper01_metrics.json`. The additional 14 figures (fig5–fig18, data-driven and schematic) were produced by `build_new_figures.py`, preserved in `.agency/missions/m-mvfhqboy-5rq/` together with its manifest and per-figure array sidecars.

```bash
python run_all_and_build_figures.py
```

To re-run all five experiments from scratch and rebuild all figures in one command:

```bash
python run_all_and_build_figures.py --run-all
```

---

## Build the PDF (Tectonic, no admin rights)

The manuscript is built with Tectonic 0.17.0 (windows-msvc). The binary is not committed to the repository: download it from the Tectonic GitHub releases and place it at `tools/tectonic-msvc/tectonic.exe` (TeX Live bundle is fetched on first run):

```powershell
$env:TECTONIC_CACHE_DIR = "$PWD\.agency\tectonic-cache"
.\tools\tectonic-msvc\tectonic.exe -X compile --keep-logs --outdir .agency\build paper_en.tex
Copy-Item .agency\build\paper_en.pdf .\paper_en.pdf -Force
```

The final build has **0 errors, 0 undefined references/citations, 22 pages**. The root `paper_en.pdf` must always be copied after the build; a build left in `.agency/build` is not the deliverable.

## Submission metadata

`submission/` contains ready-to-paste metadata for Zenodo/Research Square (`metadata.md`, `zenodo.json`, `researchsquare_metadata.md`), the plain-text abstract, cover letters, and `paper01_reproducibility_package.zip` (source, figures, results, scripts; checkpoints excluded). The preprint is published at [doi:10.21203/rs.3.rs-11435949/v1](https://doi.org/10.21203/rs.3.rs-11435949/v1).

## License

- **Code** (`src/`, `experiments/`, `*.py`) — MIT License, see [LICENSE](LICENSE).
- **Manuscript and figures** (`paper_en.tex`, `paper_en.md`, `paper_en.pdf`, `paper_en.docx`, `paper_ru.md`, `figures/`) — CC BY 4.0, matching the published preprint and the Zenodo record.
