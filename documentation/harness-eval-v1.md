# Eval set harness-eval-v1

Built by `scripts/harness_evalset.py` for rlm-harness v1 (step 1). The set lives outside git, in `RLM_DATA_DIR\harness-eval\`; this page holds numbers and ids only.

- created 2026-11-02T20:02:39Z, builder commit `72903bd347b7` (dirty)
- validated: true; permission mode `bypass`; kernel track validated on: hip
- tasks_sha256 `2e3e600f56cb15531359540d318df6935d92a95309aea4728c0bd5fd7a3949fc`
- grades_sha256 `347f53ba43b2f1e1409d9c543b58a0632b7ec001615af12c1414eac987ecd82c`

## Counts

| Track | Split | R | S | M | L |
|---|---|---|---|---|---|
| python | tune | 0 | 3 | 3 | 2 |
| python | heldout | 0 | 3 | 3 | 2 |
| web | tune | 0 | 2 | 2 | 1 |
| web | heldout | 0 | 2 | 2 | 1 |
| kernel | tune | 0 | 2 | 2 | 1 |
| kernel | heldout | 0 | 2 | 2 | 1 |
| research | tune | 4 | 0 | 0 | 0 |
| research | heldout | 4 | 0 | 0 | 0 |
| search | tune | 2 | 0 | 0 | 0 |
| search | heldout | 2 | 0 | 0 | 0 |

## Tasks

| Id | Split | f2p | p2p | Gold files | Gold lines | Gold test s | timeout s | Mutants caught |
|---|---|---|---|---|---|---|---|---|
| harness-eval-v1-research-R-01 | tune | | | | | | | |
| harness-eval-v1-research-R-02 | tune | | | | | | | |
| harness-eval-v1-research-R-03 | tune | | | | | | | |
| harness-eval-v1-research-R-04 | tune | | | | | | | |
| harness-eval-v1-research-R-05 | heldout | | | | | | | |
| harness-eval-v1-research-R-06 | heldout | | | | | | | |
| harness-eval-v1-research-R-07 | heldout | | | | | | | |
| harness-eval-v1-research-R-08 | heldout | | | | | | | |
| harness-eval-v1-search-R-01 | tune | | | | | | | |
| harness-eval-v1-search-R-02 | tune | | | | | | | |
| harness-eval-v1-search-R-03 | heldout | | | | | | | |
| harness-eval-v1-search-R-04 | heldout | | | | | | | |
| harness-eval-v1-python-S-01 | tune | 1 | 4 | 1 | 14 | 2.47 | 120 |  |
| harness-eval-v1-python-S-02 | tune | 16 | 32 | 1 | 148 | 4.17 | 120 |  |
| harness-eval-v1-python-S-03 | tune | 7 | 145 | 1 | 27 | 3.24 | 120 |  |
| harness-eval-v1-python-S-04 | heldout | 24 | 1712 | 1 | 21 | 22.69 | 120 |  |
| harness-eval-v1-python-S-05 | heldout | 4 | 204 | 1 | 10 | 1.88 | 120 |  |
| harness-eval-v1-python-S-06 | heldout | 3 | 365 | 1 | 4 | 5.8 | 120 |  |
| harness-eval-v1-python-M-01 | tune | 375 | 76 | 2 | 51 | 3.67 | 120 |  |
| harness-eval-v1-python-M-02 | tune | 6 | 57 | 2 | 17 | 4.22 | 120 |  |
| harness-eval-v1-python-M-03 | tune | 92 | 51 | 2 | 116 | 56.67 | 227 |  |
| harness-eval-v1-python-M-04 | heldout | 18 | 571 | 2 | 18 | 6.89 | 120 |  |
| harness-eval-v1-python-M-05 | heldout | 20 | 439 | 2 | 14 | 5.73 | 120 |  |
| harness-eval-v1-python-M-06 | heldout | 3 | 74 | 3 | 12 | 2.28 | 120 |  |
| harness-eval-v1-python-L-01 | tune | 139 | 4 | 4 | 1251 | 2.89 | 120 |  |
| harness-eval-v1-python-L-02 | tune | 417 | 35 | 4 | 406 | 3.16 | 120 |  |
| harness-eval-v1-python-L-03 | heldout | 911 | 281 | 4 | 837 | 3.66 | 120 |  |
| harness-eval-v1-python-L-04 | heldout | 944 | 1 | 4 | 450 | 2.47 | 120 |  |
| harness-eval-v1-web-S-01 | tune | 10 | 0 | 16 | 1668 | 5.19 | 120 |  |
| harness-eval-v1-web-S-02 | tune | 9 | 0 | 34 | 3004 | 5.47 | 120 |  |
| harness-eval-v1-web-S-03 | heldout | 14 | 0 | 2 | 594 | 5.17 | 120 |  |
| harness-eval-v1-web-S-04 | heldout | 9 | 0 | 34 | 2898 | 5.28 | 120 |  |
| harness-eval-v1-web-M-01 | tune | 15 | 0 | 21 | 1828 | 6.78 | 120 |  |
| harness-eval-v1-web-M-02 | tune | 26 | 0 | 1 | 551 | 5.53 | 120 |  |
| harness-eval-v1-web-M-03 | heldout | 15 | 0 | 25 | 2074 | 8.45 | 120 |  |
| harness-eval-v1-web-M-04 | heldout | 25 | 0 | 11 | 1521 | 5.42 | 120 |  |
| harness-eval-v1-web-L-01 | tune | 36 | 0 | 21 | 1545 | 6.41 | 120 |  |
| harness-eval-v1-web-L-02 | heldout | 36 | 0 | 19 | 1703 | 6.59 | 120 |  |
| harness-eval-v1-kernel-S-01 | tune | 57 | 0 | 2 | 73 | 1.7 | 120 | 3 of 4 |
| harness-eval-v1-kernel-S-02 | tune | 46 | 0 | 2 | 32 | 1.7 | 120 | 4 of 4 |
| harness-eval-v1-kernel-S-03 | heldout | 42 | 0 | 2 | 62 | 1.7 | 120 | 4 of 4 |
| harness-eval-v1-kernel-S-04 | heldout | 50 | 0 | 2 | 70 | 1.72 | 120 | 4 of 4 |
| harness-eval-v1-kernel-M-01 | tune | 53 | 0 | 3 | 95 | 1.95 | 120 | 3 of 4 |
| harness-eval-v1-kernel-M-02 | tune | 55 | 0 | 2 | 230 | 2.55 | 120 | 3 of 4 |
| harness-eval-v1-kernel-M-03 | heldout | 52 | 0 | 3 | 83 | 1.83 | 120 | 4 of 4 |
| harness-eval-v1-kernel-M-04 | heldout | 47 | 0 | 3 | 135 | 1.95 | 120 | 3 of 4 |
| harness-eval-v1-kernel-L-01 | tune | 32 | 0 | 5 | 93 | 2.45 | 120 | 4 of 4 |
| harness-eval-v1-kernel-L-02 | heldout | 134 | 0 | 5 | 356 | 2.77 | 120 | 4 of 4 |

Gold test runs: median 3.7 s, max 56.7 s.
