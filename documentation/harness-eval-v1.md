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

## Baseline (step 5)

rlm-harness v1 step 5: the `default` profile at the commit below, frozen as the user profile `v1-baseline`, run once per task (D-170) on both splits. Numbers and ids only.

- dates (UTC): tune 2026-11-03, held-out 2026-11-04; `harness_commit` `f5e23f2fa0032433964699232f23f607db374a98`
- set: tasks_sha256 `2e3e600f56cb15531359540d318df6935d92a95309aea4728c0bd5fd7a3949fc`, grades_sha256 `347f53ba43b2f1e1409d9c543b58a0632b7ec001615af12c1414eac987ecd82c`
- model: root and sub `openrouter/deepseek/deepseek-v4.1-flash-1017`, call format `auto`, root thinking `high`, sub thinking `default` (the profile's: `subcall_thinking`); GPU backend `hip`
- profile hashes: `v1-baseline` = built-in `default` at that commit `83f0257bcf41091b34bc796f3a2f8771dc043c6653ef46888ef324aa887f83ac`; `training-v4` `b5ffa4b69d4c260b53e740a24919f37c7c9254032b6e9e90edb016da1ad8fa59` (not run, D-249)

### `v1-baseline`, tune split

| | n | passed | pass rate | mean f2p share | mean turns | mean sub-calls | mean compactions | cost | cache hit | mean wall s |
|---|---|---|---|---|---|---|---|---|---|---|
| total | 23 | 18 | 78.3% | 0.941 | 34.8 | 4.7 | 0.00 | $1.8721 | 91.5% | 342 |
| python | 8 | 8 | 100.0% | 1.000 | 35.5 | 0.4 | 0.00 | $0.2157 | 83.6% | 168 |
| web | 5 | 5 | 100.0% | 1.000 | 35.6 | 4.8 | 0.00 | $0.2298 | 93.6% | 344 |
| kernel | 4 | 3 | 75.0% | 0.750 | 68.2 | 8.2 | 0.00 | $1.4030 | 96.0% | 1137 |
| research | 4 | 1 | 25.0% | - | 8.8 | 12.2 | 0.00 | $0.0120 | 73.6% | 26 |
| search | 2 | 1 | 50.0% | - | 15.5 | 0.0 | 0.00 | $0.0116 | 90.9% | 73 |
| R | 6 | 2 | 33.3% | - | 11.0 | 8.2 | 0.00 | $0.0236 | 85.0% | 41 |
| S | 7 | 7 | 100.0% | 1.000 | 30.4 | 3.4 | 0.00 | $0.3372 | 86.2% | 286 |
| M | 7 | 6 | 85.7% | 0.857 | 58.0 | 2.9 | 0.00 | $1.3481 | 94.9% | 665 |
| L | 3 | 3 | 100.0% | 1.000 | 38.7 | 5.3 | 0.00 | $0.1633 | 90.0% | 317 |

- infra errors: 0; reruns: none; capped: false; `test_tamper` 3; `leak_suspect` 12
- mean score 0.815; cost per run $0.0814

### `v1-baseline`, held-out split

| | n | passed | pass rate | mean f2p share | mean turns | mean sub-calls | mean compactions | cost | cache hit | mean wall s |
|---|---|---|---|---|---|---|---|---|---|---|
| total | 24 | 17 | 70.8% | 0.941 | 24.1 | 2.6 | 0.00 | $0.7156 | 85.9% | 263 |
| python | 8 | 7 | 87.5% | 0.993 | 25.0 | 0.5 | 0.00 | $0.1213 | 76.9% | 101 |
| web | 5 | 5 | 100.0% | 1.000 | 36.4 | 3.2 | 0.00 | $0.2739 | 86.8% | 408 |
| kernel | 5 | 4 | 80.0% | 0.800 | 26.6 | 6.6 | 0.00 | $0.2771 | 92.2% | 656 |
| research | 4 | 1 | 25.0% | - | 6.8 | 2.2 | 0.00 | $0.0338 | 36.6% | 15 |
| search | 2 | 0 | 0.0% | - | 18.5 | 0.5 | 0.00 | $0.0096 | 89.7% | 62 |
| R | 6 | 1 | 16.7% | - | 10.7 | 1.7 | 0.00 | $0.0434 | 68.8% | 31 |
| S | 7 | 7 | 100.0% | 1.000 | 27.7 | 2.6 | 0.00 | $0.2640 | 87.8% | 342 |
| M | 7 | 6 | 85.7% | 0.992 | 30.0 | 4.9 | 0.00 | $0.2923 | 89.1% | 339 |
| L | 4 | 3 | 75.0% | 0.750 | 27.8 | 0.2 | 0.00 | $0.1159 | 81.0% | 340 |

- infra errors: 1 (other 1); reruns: none; capped: false; `test_tamper` 4; `leak_suspect` 9
- mean score 0.750; cost per run $0.0298

### Notes

- The tune split has 23 of its 24 tasks: the PC lost power during `harness-eval-v1-kernel-L-01` (at turn 63), so it has no grade line, and it is counted as not run (Rishi, 2026-11-03). Its partial spend is in neither output.
- The held-out split's `harness-eval-v1-kernel-L-02` is its one infra line (`other`: the engine exited at turn 31 when the PC was paused). D-187 would rerun it; Rishi kept it as not run (2026-11-04), so the held-out rows above count it as a failure, and 17 of the 23 tasks that ran passed (73.9%). Its spend, about $0.11, is in the ledger's line but not in its grade line.
- `training-v4` was not run (D-249), so there is no compare row. One replicate on 24 tasks can't show a significant difference between two profiles (D-170); later steps compare against the held-out split above.
- `leak_suspect` is noisy on this set: no base snapshot holds a hidden test file or an f2p name, and no run's messages hold the set path or `hidden/`. The matches are the model's own test names in REPL output (generic pytest names, Playwright titles that repeat a request's wording), plus `harness-eval-v1-web-S-01`, whose task text names two file-path f2p titles.
- `test_tamper`: every kernel task that passed (7 of 7) changed a protected file (`kernels.py`, `gpu.py` or `gpu_compat.h`); grading reverted them first, and they passed anyway.
- No run compacted, in either split.
- Did not pass (one run each): `harness-eval-v1-research-R-02`, `harness-eval-v1-research-R-01`, `harness-eval-v1-research-R-03`, `harness-eval-v1-search-R-02`, `harness-eval-v1-kernel-M-02`, `harness-eval-v1-research-R-06`, `harness-eval-v1-research-R-05`, `harness-eval-v1-research-R-07`, `harness-eval-v1-search-R-04`, `harness-eval-v1-search-R-03`, `harness-eval-v1-python-M-04`.
- Hit the turn limit: `harness-eval-v1-kernel-M-02` (150).
