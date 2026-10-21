# Bibliography verification (t4) — all 15 entries, verified against live resolvers

Audit performed 2026-10-19 22:24–22:27 local, chair C. Method: for every `\bibitem` in
`paper_en.tex`, the identifier was extracted mechanically from the bibliography source and
then resolved **against the live resolver** (arXiv API / arXiv abs page / Crossref API /
ACL Anthology), not against a description of it. A title and author-list match was
required, not merely an HTTP 200.

**Result: 15/15 verified. 0 unverifiable entries. No entry needs replacement, so no
`\cite` renumbering is required and the bracket citations in `paper_en.md` /
`paper_ru.md` need no edit.**

## Per-entry evidence

| # | key | identifier | resolver | HTTP | title match | authors match | venue/year |
|---|-----|-----------|----------|------|--------------|----------------|-------------|
| 1 | zhang2021rela | arXiv:2104.07012 | arxiv.org/abs | 200 | yes | yes (Zhang, Titov, Sennrich) | EMNLP 2021 (per arXiv comment) |
| 2 | zuhri2025softpick | arXiv:2504.20966 | export.arxiv.org API | 200 | yes | yes (Zuhri, Fuadi, Aji) | 2025 |
| 3 | wang2026abstention | arXiv:2609.22005 | arxiv.org/abs | 200 | yes | yes (Richard Zhe Wang) | 2026, cs.LG, v2 29 Sep 2026 |
| 4 | sun2024massive | arXiv:2402.17762 | export.arxiv.org API | 200 | yes | yes (Sun, Chen, Kolter, Liu) | COLM 2024 |
| 5 | peng2023rwkv | arXiv:2305.13048 + DOI 10.18653/v1/2023.findings-emnlp.936 | ACL Anthology | 200 | yes | yes (Peng et al.) | Findings of EMNLP 2023, pp. 14048–14077 |
| 6 | olsson2022induction | arXiv:2209.11895 | export.arxiv.org API | 200 | yes | yes (Olsson, Elhage, Nanda, et al.) | 2022 |
| 7 | vaswani2017attention | arXiv:1706.03762 | export.arxiv.org API | 200 | yes | yes (Vaswani et al.) | 2017 |
| 8 | so2021primer | arXiv:2109.08668 | export.arxiv.org API | 200 | yes | yes (So, Mańke, Liu, Dai, Shazeer, Le) | NeurIPS camera-ready |
| 9 | martins2016sparsemax | arXiv:1602.02068 | export.arxiv.org API | 200 | yes | yes (Martins, Astudillo) | 2016 |
| 10 | peters2019sparse | arXiv:1905.05702 + DOI 10.18653/v1/P19-1146 | Crossref API | 200 | yes | yes (Peters, Niculae, Martins) | ACL 2019, pp. 1504–1519 |
| 11 | correia2019adaptively | arXiv:1909.00015 + DOI 10.18653/v1/D19-1223 | Crossref API | 200 | yes | yes (Correia, Niculae, Martins) | EMNLP-IJCNLP 2019, pp. 2174–2184 |
| 12 | miller2023offbyone | https://www.evanmiller.org/attention-is-off-by-one.html | evanmiller.org | 200 | yes | yes (Evan Miller) | online post, 24 Jul 2023 |
| 13 | bondarenko2023quantizable | arXiv:2306.12929 | export.arxiv.org API | 200 | yes | yes (Bondarenko, Nagel, Blankevoort) | 2023 |
| 14 | xiao2024streamingllm | arXiv:2309.17453 | export.arxiv.org API | 200 | yes | yes (Xiao, Tian, Chen, Han, Lewis) | ICLR 2024 |
| 15 | dao2022flashattention | arXiv:2205.14135 | export.arxiv.org API | 200 | yes | yes (Dao, Fu, Ermon, Rudra, Ré) | 2022 |

Batch query used for entries 2, 4, 6-11, 13-15 (12 IDs in one call, `totalResults=12`):
`https://export.arxiv.org/api/query?id_list=2504.20966,2402.17762,2305.13048,2209.11895,1706.03762,2109.08668,1602.02068,1905.05702,1909.00015,2306.12929,2309.17453,2205.14135&max_results=20`
— every requested ID returned an `<entry>` whose title and author list match the
bibitem. No requested ID was missing from the response (a missing ID would have produced
`totalResults < 12` and a gap, which is the check that can fail on absence).

## The two entries I flagged last round as at-risk — both clean

Last round I flagged `miller2023offbyone` (no DOI/arXiv) and `wang2026abstention` (a
2026 arXiv id I could not confirm offline) as the likely unverifiable pair. **Both
resolve.** `wang2026abstention` is real: arXiv:2609.22005, "Abstention and Noise
Filtering: Two Missing Primitives of Softmax Attention", Richard Zhe Wang, submitted
18 Sep 2026, revised 29 Sep 2026, cs.LG/cs.CL. `miller2023offbyone` is a genuine blog
post by Evan Miller dated 24 July 2023, with the title and date matching the bibitem
exactly. My earlier "at-risk" framing was a guess from identifier shape, not a
measurement; the measurement now supersedes it. **No entry is unverifiable, so nothing
needs to be replaced or removed, and no citation renumbering cascades into the md files.**

## Integrity checks on the bibliography itself

- `\bibitem` count: **15**; `thebibliography` environments: 1.
- distinct `\cite` keys: **15**.
- bibitems never cited: **0** (no "citation ... undefined" risk from an uncited entry).
- cited keys with no matching bibitem: **0** (no undefined-citation risk).
- DOI present in 3 entries (5, 10, 11) and independently confirmed against the Crossref
  API / ACL Anthology records. The remaining 12 carry arXiv ids, all resolved.

## Scope note

t4 is **closed by this artifact**. It changes no file, so no PDF rebuild is required
because of it, and the md bracket citations stay valid. This retires the ordering
constraint I raised in my round-9 proposal ("t4 changes citation numbering, so it must
precede t5"): measured against the actual bibliography, **t4 changes nothing** — there is
no renumbering to cascade. The reason to run t5 next is therefore not sequencing but
that t5 is the largest unevidenced acceptance line.

## t5 scope established for the next step

All 14 `results/*.json` parse as JSON and expose the top-level structure a reconciler
needs — so t5 is executable against real data, not blocked on missing artifacts:

| file | top-level keys |
|------|----------------|
| exp1_legacy_13stacks.json | config, device, fir_taps_present_to_past, summary, per_seed |
| exp2_content_addressable_decoy.json | config, device, summary, per_seed |
| exp3_sweep_tau_gate.json | base, n_configs, rows |
| exp4_wikitext2_lm.json | config, device, vocab_size, train_tokens, step_benchmark, triton_mha_benchmark |
| exp4b_mha6_instrumentation.json | meta, diagnostics_per_tau, stack_timing, kern1_timing, gemm_reference_note, r2_attribution |
| exp4b_sequential_repro.json | config, runs |
| exp4c_cuda_graphs.json | meta, per_tau, pair_deltas, combined, attribution |
| exp4d_softmax_50k_bf16.json | protocol, config, meta, softmax_50k_bf16, pn_fir_tau_50k_bf16_reference, matched_precision_deltas_pn_minus_sm |
| exp4e_multiseed_50k.json | config_note, runs |
| exp4f_scaleout.json | partA, partB, wall_s |
| exp4g_tau_scaling_law.json | config, ladder, fit, validation |
| exp4h_int8_quant.json | protocol, runs |
| exp5_abstention_and_entmax.json | seeds, aggregated_3seed, per_run |
| exp6_token_shift_ablation.json | config, device, partA_legacy, partB_content_addressable, wall_s |

Note `exp4b_mha6_instrumentation.json` contains a key literally named `r2_attribution`.
That is a **JSON data key**, not a manuscript marker, and must not be "cleaned" — it is
the machine's own naming for the round-2 attribution block. The acceptance criterion bans
round/plan labels **in the manuscript**; renaming a key inside a results artifact would
break the provenance chain of the data. Flagged so nobody strips it.

## Verdict

t4 (bibliography verification): **done, 15/15 verified, 0 unverifiable, 0 edits needed.**
