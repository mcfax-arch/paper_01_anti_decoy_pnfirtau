# Numbers-vs-Results Audit (t5) — machine-checkable gate

Producer: `.agency/missions/m-mvfhqboy-5rq/t5_gate.py`.
Read-only on `results/*.json`; the only file this script writes is this report.

## GATE RESULT: **PASS**

`t5_gate.py` exits **0** on PASS and **1** on FAIL. It re-runs the three
worker checkers against the live manuscript every time, so the numbers below
are re-measured, not copied from a report.

## 1. Denominator (N)

| scope | N | source |
|---|---|---|
| table cells, `tab:exp1`, `tab:exp2`, `tab:exp6` | 299 | C-w1 checker |
| table cells, `tab:exp4`, `tab:int8-hygiene`, `tab:exp4_mha`, `tab:exp4_skip`, `tab:drift-ctrl`, `tab:cuda-graph` | 214 | C-w2 checker |
| table cells, `tab:width-envelope`, `tab:exp3`, `tab:exp5` | 307 | C-w3 checker |
| **N cells (total)** | **820** | |

Each checker also audits non-numeric "assertion" cells separately
(`OOM`, `---`, `---^\ddagger`): 12 such cells, all traced, 0 unmapped.

**Not counted in N, and why** (the unaudited denominator):

| item | count | reason |
|---|---|---|
| `results/*.json` numeric leaves not reachable from any manuscript claim | 206 421 | not claims; no statement is expected to use them |
| bibliography entries | 15 | audited in `results/bibliography_audit.md` (t4): 15/15 resolved, 0 unverifiable |
| numeric citation indices `[12]`, `\cite{}`/`\ref{}` targets | 29 `\cite`, 44 `\ref` | not numeric claims; t4 confirmed the bibliography is stable, so no renumbering cascades |
| markdown list markers, heading numbers, `Table 3` / `Figure 2` cross-references | — | structural, not claims |
| prose numbers | 182 (C-w1 58, C-w3 63, C-w2 61) | traced in prose reports, not re-derived here |
| figure captions (18) | 18 | audited separately by the chair: 18/18 carry a source line, 0 untraceable numbers |

## 2. Cell verdicts

| verdict | count |
|---|---|
| MATCH | 814 |
| PLACEHOLDER | 6 |
| **UNMAPPED (no JSON key path)** | 0 |

## 3. MISMATCHES (machine-greppable)

| src | table | row | column | tex value | json value | json path |
|---|---|---|---|---|---|---|

## 4. UNMAPPED cells (no JSON key path — counted as mismatches)

None. Every audited cell carries a `results/*.json` key path.

## 5. Three-way equality (EN md vs RU md — the acceptance criterion)

Tables: tex 12, EN md 12, RU md 12.

The md files are deliberately **richer** renderings than the tex (extra `dW`
column, the full `nd` sweep, raw-pos and all-neg-row columns, per-cell stds),
so md-vs-tex column counts are listed for information and are NOT failures.
The gate is EN md vs RU md.

| # | tex label | tex rows x cols (info) | EN md rows x cols | RU md rows x cols | EN==RU |
|---|---|---|---|---|---|
| 0 | `tab:exp1` | (17, 13) | (17, 16) | (17, 16) | yes |
| 1 | `tab:exp2` | (7, 12) | (7, 14) | (7, 14) | yes |
| 2 | `tab:exp6` | (5, 8) | (7, 10) | (7, 10) | yes |
| 3 | `tab:width-envelope` | (9, 7) | (9, 7) | (9, 7) | yes |
| 4 | `tab:exp3` | (11, 11) | (12, 13) | (12, 13) | yes |
| 5 | `tab:exp5` | (9, 10) | (9, 11) | (9, 11) | yes |
| 6 | `tab:exp4` | (10, 6) | (18, 11) | (18, 11) | yes |
| 7 | `tab:int8-hygiene` | (5, 7) | (5, 7) | (5, 7) | yes |
| 8 | `tab:exp4_mha` | (4, 6) | (4, 6) | (4, 6) | yes |
| 9 | `tab:exp4_skip` | (6, 7) | (6, 5) | (6, 5) | yes |
| 10 | `tab:drift-ctrl` | (7, 4) | (7, 6) | (7, 6) | yes |
| 11 | `tab:cuda-graph` | (3, 4) | (3, 4) | (3, 4) | yes |

### Number equality (EN md vs RU md, per table)

Matched within tolerance (0.004 relative, 0.0051 absolute). Numbers that
differ only in display precision are listed separately and are NOT failures.

**Every table carries the same numbers in EN and RU.**

Display-precision-only differences (same value, different rounding):

| table # | tex label | pairs |
|---|---|---|
| 6 | `tab:exp4` | [(68.83, 68.8), (75.74, 75.7), (77.42, 77.4), (99.21, 99.2), (100.15, 100.2), (101.35, 101.4), (105.84, 105.8), (120.43, 120.4), (140.92, 140.9), (143.76, 143.8), (164.44, 164.4), (167.19, 167.2), (244.17, 244.2)] |

## 6. Negative control

PASS(corrupted tab:exp1 $45.1$ -> $45.13$ in C-w1: mismatches 0 -> 1)

The control corrupts a live table cell in `paper_en.tex` (a temporary copy;
the original is restored), re-runs the whole gate, and requires the gate to
FAIL. A gate that cannot fail is not a gate.

## 7. Checker exit codes in this run

| checker | rc |
|---|---|
| C-w1 | 0 |
| C-w2 | 0 |
| C-w3 | 0 |

## 8. How to re-run

```
python .agency/missions/m-mvfhqboy-5rq/t5_gate.py
```
Exit 0 = gate PASS, exit 1 = gate FAIL. Re-run after every manuscript edit.
