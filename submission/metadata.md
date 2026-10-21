# Метаданные для публикации препринта

## 1. Базовые поля (общие для TechRxiv и Zenodo)

**Title (EN):**
Attention with a Calibrated Off Switch: Causal FIR Key Smoothing, the Score-Shift Pathology, and Selective Silence in Rectified Attention (pn_fir_tau)

**DOI (preprint):**
10.21203/rs.3.rs-11435949/v1

**Author:**
Maxim Zhukov

**ORCID:**
0009-0005-9238-3778

**Affiliation:**
Independent Research

**Language:**
English

**Version:**
1.0

**Publication date:**
2026-10-20

**License (recommended):**
CC BY 4.0 — свободное переиспользование с указанием автора. Альтернатива, если не хотите коммерческого переиспользования: CC BY-NC-ND 4.0.

**Keywords:**
attention normalization; sub-simplex attention; selective silence; attention sinks; causal FIR filtering; induction heads; Triton kernels; FlashAttention; language modeling; quantization outliers

**Subject area (TechRxiv):**
Primary: Computer Science / Machine Learning.
Secondary (если доступно): Computer Science / Artificial Intelligence; Computation and Language.

**Comments / Notes:**
22 pages, 18 figures, 12 tables. Reproducibility package (source code, 12 experiment suites, raw JSON results, figure-generation pipeline) included. Preprint. Code and data: see the attached reproducibility archive.

---

## 2. Abstract — полная версия (для Zenodo и площадок без жёсткого лимита)

Softmax attention cannot say "nothing found": a simplex-constrained head must distribute 1.000 units of context weight even when no key matches the query, and pre-trained models pay with attention sinks, massive outliers that degrade low-bit quantization, and RAG-style hallucinations on unanswerable queries. The off-switch mechanism itself is established prior art: ReLA replaces Softmax with ReLU and its heads "attend to nothing"; Softpick scales rectified normalization to 340M–1.8B with a 0% sink rate. The open question is the interaction layer audited here. A parameter-free 7-tap causal FIR low-pass filter (σ = 2.0) applied strictly to the key stream K lets one layer bind a queried key onto the adjacent value slot, lifting non-bijective single-layer associative recall from 44.5% ± 0.9% to 100.0% ± 0.0% and removing exact-decoy vulnerability on the legacy benchmark (Δ = 0.000 at decoy noise ε ∈ {0.05, 0.2, 0.5, 1.0}). The same smoothing shifts scores downward and starves rectification: fixed-τ=0 power-norm falls to 64.4% ± 6.8% with 91.3% ± 8.6% negative scores and 44.5% ± 23.0% dead rows; Softmax masks this shift (translation-invariant), rectification does not (Proposition 1). One learnable threshold τ per layer (pn_fir_tau) restores 100.0% recall and keeps unanswerable rows silent under pure cross-entropy (ΣW_unans = 0.002 ± 0.002 vs. 0.100 ± 0.005 for virtual sinks), reaches matched-precision WikiText-2 parity on 2,400 steps (within 0.3%) and a matched-precision 50k win consistent across 3 seeds (−2.22 ± 0.46 val / −2.33 ± 0.49 test PPL). We release fused Triton kernels (no exp(), O(N) attention VRAM) and a full measurement record, including an instrumented retraction of our own earlier latency claim.

---

## 3. Abstract — короткая версия (если площадка ограничивает 200–250 слов)

Softmax attention cannot say "nothing found": a simplex-constrained head must distribute all context weight even when no key matches the query, causing attention sinks, quantization outliers, and RAG-style hallucinations. Rectified normalizers remove this constraint (ReLA, Softpick), so the off switch itself is prior art; the open question is whether rectification survives causal key smoothing. We show that a parameter-free 7-tap causal FIR low-pass filter on the key stream K lets a single layer bind a queried key onto the adjacent value slot, lifting non-bijective recall from 44.5% ± 0.9% to 100.0% ± 0.0% and removing exact-decoy vulnerability. The same smoothing shifts scores downward and starves rectification: fixed-τ=0 power-norm drops to 64.4% ± 6.8%, with 91.3% negative scores and 44.5% dead rows. Softmax masks the shift; rectification does not. One learnable threshold τ per layer (pn_fir_tau) restores 100% recall, keeps unanswerable rows silent under pure cross-entropy (residual mass 0.002 vs. 0.100 for virtual sinks), and reaches WikiText-2 parity at 2,400 steps with a matched-precision 50k gain of −2.22 ± 0.46 validation / −2.33 ± 0.49 test perplexity across three seeds. We release fused Triton kernels and a full measurement record.

---

## 4. Что прикладывать к подаче

1. `paper_en.pdf` — основной PDF (22 стр., собран Tectonic, 0 errors/undefined refs).
2. `submission/paper01_reproducibility_package.zip` — исходники, фигуры, результаты, скрипты.
3. `submission/abstract_full.txt` — абстракт в plain text.
4. `submission/techrxiv_cover_note.md` — сопроводительная записка для TechRxiv.

## 5. Чего НЕ делать

- Не указывать DOI, которого ещё нет.
- Не выбирать лицензию, противоречащую требованиям площадки; CC BY 4.0 — безопасный дефолт.
- Не заявлять «peer-reviewed»: это препринт.
- Не загружать `.agency/`, `hf_cache/`, `checkpoints_c3/` — это внутренние артефакты.
