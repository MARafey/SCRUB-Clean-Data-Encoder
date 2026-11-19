# Supplementary results

Full tables moved out of the paper to save space. The paper's primary metric
is duplicate-aware Recall@1 (top-1 text equals gold text), computed from the
top-1 row for every system. Sections S1, S2 and S3 below keep the rank-based
Recall@1 originally logged (credits exact score ties) and are comparable only
within each table. Correction F1 uses a validation-selected threshold.
Paired numbers come from `code/paired_analysis.py`
(`runs/paired_analysis.json`).

## S1. Architecture development, poi_library (1,656 records, 249 test queries)

No-instruction variant.

| Approach | Recall@1 | Correction F1 | Best variant |
|---|---|---|---|
| TF-IDF (char 3-5 n-gram) | 97.99 | 99.20 | n/a |
| SCRUB v1 (4L/128d, single-kernel char-CNN) | 74.30 | 86.01 | Terse Noise (F1 only) |
| SCRUB v2 (6L/256d, multi-kernel char-CNN) | 79.12 | 89.37 | Terse Noise (Recall@1 and F1) |

On russia_065, tuning batch size and early stopping raised SCRUB v2's
no-instruction Recall@1 from 53.27 to 60.80; SCRUB v3 reached 62.20 under the
same budget.

## S2. Fixed-string seven-variant instruction sweep, russia_065 (1,500 test queries)

| Variant | v3 from scratch: Recall@1 | Corr. F1 | Pretrained: Recall@1 | Corr. F1 |
|---|---|---|---|---|
| No Instruction | 60.47 | 87.63 | 81.80 | 96.13 |
| Generic Retrieval | 63.87 | 88.67 | 82.00 | 95.63 |
| Generic Noisy | 62.80 | 88.69 | 81.73 | 95.86 |
| Terse Noise | 63.67 | 89.14 | 80.33 | 95.50 |
| Specific Noise | 61.00 | 88.30 | 81.07 | 95.38 |
| Domain-Grounded Noise | 61.47 | 88.53 | 80.00 | 94.78 |
| Verbose Noise | 62.20 | 87.57 | 49.53 | 75.88 |

The highest Recall@1 in each column is a test-selected maximum; the paper uses
the No Instruction row as the primary comparison.

## S3. Paired intervals for Record-Specific vs. No Instruction (SCRUB v3, russia_065)

Wald interval for the paired difference in Recall@1 from the discordant
counts `b` (only Record-Specific correct) and `c` (only No Instruction
correct) over `n = 1500` queries:
`d = (b - c) / n`, half-width `1.96 * sqrt(b + c - (b - c)^2 / n) / n`.
McNemar uses the continuity-corrected `chi2 = (|b - c| - 1)^2 / (b + c)`.

| Seed | correct (Specific / None) | b / c | d (points) | 95% interval | chi2 | p |
|---|---|---|---|---|---|---|
| 42 | 967 / 961 | 133 / 127 | +0.40 | [-1.71, +2.51] | 0.096 | 0.756 |
| 123 | 969 / 966 | 128 / 125 | +0.20 | [-1.88, +2.28] | 0.016 | 0.900 |

## S4. Overlap of the Record-Shuffled control with the true labels

`code/shuffle_overlap.py` compares each test row's corruption-type set with
the set of the row whose instruction it receives (russia_065, 1,500 test rows).

| Seed | identical set | partial overlap | disjoint |
|---|---|---|---|
| 42 | 12.3% | 87.3% | 0.4% |
| 123 | 12.2% | 87.8% | 0.0% |

## S5. Strict row-index Recall@1 (top-1 row must be the gold row), russia_065

Same runs and test queries as the paper's Table 5. Paired lead of the
pretrained encoder over TF-IDF, with 95% Wald intervals (all McNemar p < 0.001).

| Protocol | TF-IDF | Frozen | SCRUB v3 | Pretrained | Pretrained - TF-IDF |
|---|---|---|---|---|---|
| random, seed 42 | 66.87 | 45.13 | 61.00 | 74.60 | +7.73 [+5.50, +9.97] |
| random, seed 123 | 69.00 | 44.80 | 61.27 | 73.73 | +4.73 [+2.42, +7.05] |
| grouped, seed 42 | 68.33 | 44.93 | 58.20 | 74.53 | +6.20 [+3.79, +8.61] |
| grouped, seed 123 | 67.91 | 43.76 | 60.64 | 73.52 | +5.60 [+3.20, +8.01] |

## S6. Record-specific family, duplicate-aware paired differences (random split)

Difference in duplicate-aware Recall@1 (points), 95% interval, McNemar p.

| Comparison | seed 42 | seed 123 |
|---|---|---|
| Generic Retrieval - None | -0.53 [-2.25, +1.18], p=0.59 | +0.07 [-1.63, +1.77], p=1.00 |
| Record-Specific - None | -0.33 [-2.11, +1.44], p=0.77 | -0.20 [-1.92, +1.52], p=0.88 |
| Record-Shuffled - None | -0.20 [-1.96, +1.56], p=0.88 | -1.20 [-2.89, +0.49], p=0.19 |
| Record-Nonsense - None | +0.87 [-0.96, +2.69], p=0.39 | +0.40 [-1.25, +2.05], p=0.69 |
| Record-Shuffled - Record-Specific | +0.13 [-1.71, +1.97], p=0.94 | -1.00 [-2.80, +0.80], p=0.31 |
| Record-Nonsense - Record-Specific | +1.20 [-0.66, +3.06], p=0.23 | +0.60 [-1.03, +2.23], p=0.52 |

## S7. Matched initialization pair and frozen baseline (random split, seed 42)

| Comparison | duplicate-aware difference (points) | 95% interval | p |
|---|---|---|---|
| Pretrained matched (batch 4) - random-init matched | +12.87 | [+10.46, +15.27] | <0.001 |
| Pretrained matched (batch 4) - TF-IDF | +7.13 | [+4.79, +9.48] | <0.001 |
| Pretrained (batch 8) - frozen DistilBERT | +35.40 | [+32.72, +38.08] | <0.001 |

Split leakage (`code/check_split.py`): random split, test rows whose clean text
is in train = 333 (seed 42), 326 (seed 123); grouped split = 0 for both seeds.
Seeds give different test sets (206 of 1,500 test queries coincide).
