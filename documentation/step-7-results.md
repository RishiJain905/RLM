# Step 7 results: data

> **Step 7 of [production-build-plan.md](production-build-plan.md)** (holes H7 and H8, the licence bar, research §3 and §8.1). Built and tested 2026-10-11 on Rishi's PC, on Windows and inside WSL2 with Docker.
>
> **Decisions applied (Rishi, 2026-10-11):**
> - **7a:** the research mix of 35% cyber, 30% OOLONG-style aggregation, 20% code generators and 15% repo localization.
> - **7b:** permissive licences only in training, with everything else eval-only.
> - **7c:** drop cyber, and split its 35% between better coding and one more thing we already have. The production mix is now **45% aggregation, 40% code generators and 15% repo localization** (§6).
>
> **Status: complete, committed and pushed to `claude/project-thread-9nnh3n`**, on top of step 6's `claude/project-thread-2apfvb`.
>
> **No W&B credit was used.** Nothing here calls a model. The `.env` file was never opened. Nothing was installed. The data was downloaded with the Hugging Face CLI to `D:\RLM-data`, outside the repo, as Rishi asked. That is the one exception to "everything inside the repo folder".
>
> Anything marked *(mine)* is my choice, not the spec's. Anything marked *(inferred)* hasn't been checked against the live system.

## Summary

**Step 7 is built, and every done-when check passes.** Four new training families fill the whole mix as decision 7c set it, and the aggregation family gained two sources (LEDGAR contracts and arXiv titles) for its bigger share. The production selection set, the held-out test, the capability check's 100 prompts and a dataset manifest are in. The regression suite is now **241 tests** (214 before); all pass in WSL with Docker, except the step 4 tokenizer check, which skips there as before.

| Done-when check | Result | Evidence |
|---|---|---|
| The top levels have a median context above 130K characters (H8) | **Pass** | Level 5 medians: aggregation 381K, call graph 376K, needle 371K, repo localization 404K characters (`scripts/data_check.py`, 20 tasks per family and level, §4). |
| No generator, template or source overlaps training (H7) | **Pass** | Held-out question templates share no 6-word sequence with training templates. Held-out behaviours, line formats, call styles, sources and LCA items are disjoint from training (§5). |
| Every training source passes the licence bar (7b) | **Pass** | Training reads only CLINC150 (CC-BY-3.0), MASSIVE (CC-BY-4.0), LEDGAR (CC-BY-4.0), arXiv titles (CC0) and Long Code Arena (Apache-2.0, repos MIT or Apache-2.0). `tasks.check_mix()` refuses an eval-only family or source in any training mix, and the production config runs it (§3). |
| A dataset manifest pins every version | **Pass** | `manifests/dataset-manifest.json` pins each source's revision, licence and role, and the sha256 of all 111 data files and 4 derived files. Its hash goes into every checkpoint's `config.json` (§7). |

| Domain | 7a | After 7c | Families |
|---|---|---|---|
| Cyber | 35% | dropped | none |
| Aggregation | 30% | **45%** | `aggregation` |
| Code generators | 20% | **40%** | `callgraph` 20%, `needle` 20% |
| Repo localization | 15% | 15% | `repo_loc` |

**Leftover file removed:** the unused AIT-LDS file from the cyber work (7.1 GB) was moved to the Recycle Bin at Rishi's request on 2026-10-11, and its folder removed.

---

## 1. What was built

| Path | What it is |
|---|---|
| `rlm_rl/data.py` | Where the data lives (`D:\RLM-data`, `/mnt/d/RLM-data` from WSL, or `$RLM_DATA_DIR`), every source's origin, revision, licence and role, the licence bar, and the text pools (CLINC150, MASSIVE, LEDGAR, arXiv titles, HuffPost) |
| `rlm_rl/tasks/aggregation.py` | The OOLONG-style aggregation family, training and held-out (§2.1) |
| `rlm_rl/tasks/synthcode.py` | Generated Python packages: the call-graph and needle-function families, training and held-out (§2.2) |
| `rlm_rl/tasks/behaviours.py` | The needle family's 57 behaviours in 20 groups of look-alikes, each with a reference implementation |
| `rlm_rl/tasks/repoloc.py` | Long Code Arena bug localization: training, selection and test families (§2.3) |
| `rlm_rl/tasks/oolong.py` | A fixed sample of OOLONG-synth's test split, for the held-out test |
| `rlm_rl/tasks/__init__.py` | Gains the new families, the domain mix (`DOMAIN_MIX`, decisions 7a and 7c), `production_mix()`, `check_mix()` and `private_seed_list()` |
| `rlm_rl/capability_set.py` | Builds the capability check's 100 prompts (§2.5) |
| `rlm_rl/evals.py` | Gains `selection-v1` (64 tasks) and `heldout-test-v1` (170 tasks) (§2.4) |
| `rlm_rl/config.py` | Production now uses the production mix, `selection-v1` and the capability prompts, and pins the dataset manifest's hash. `tasks_from="tickets"` keeps step 6's stand-in for tests and dry runs. |
| `rlm_rl/capability.py`, `rlm_rl/training.py` | A capability path can be `data:<path>`, relative to the data root |
| `scripts/build_data.py` | Unpacks the LCA repositories' git data, indexes the usable LCA items, samples the arXiv titles, and builds the capability prompts |
| `scripts/build_manifest.py` | Writes `manifests/dataset-manifest.json` |
| `scripts/data_check.py` | The done-when checks above, written to `manifests/data-check.json` |
| `scripts/train.py` | Gains `--ticket-tasks` |
| `tests/test_data_families.py` | 26 tests (§8) |
| `tests/test_training.py`, `tests/dryrun.py` | A production dry run on the production families. The dry run's oracle can now answer lists and names, and the step 6 loop tests use the ticket stand-in explicitly. |
| `README.md` | The Layout section lists the new files |

### On the D drive (outside git)

| Folder | Holds | Size |
|---|---|---|
| `hf/` | The Hugging Face downloads, at pinned revisions: CLINC150, MASSIVE, LEDGAR (LexGLUE), arXiv metadata, LCA bug localization (Python split and the 48 repository archives), HuffPost headlines, OOLONG-synth's test split, IFEval, MMLU-Pro and CRUXEval | ~22 GB |
| `github/clinc__oos-eval/` | CLINC150's intent-to-domain map | 4 KB |
| `derived/lca-git/` | The git data unpacked from each LCA repository archive | 11 GB |
| `derived/lca/` | The index of usable LCA items | 6 MB |
| `derived/aggregation/` | The fixed sample of arXiv titles, 6,000 per archive | 2 MB |
| `derived/capability/` | `capability-v1.jsonl`, the 100 capability prompts | 55 KB |
| `logs/` | Build and check logs | small |

The earlier thread downloaded everything under `hf/` and `github/` except LEDGAR and arXiv, which I added for the bigger aggregation share.

## 2. The task families

Every family follows the rules from research §3.1 and spec §3:
- **Generated on the host** from a random 64-bit seed that never enters the sandbox (H3). Only the context text does.
- **A verifiable answer:** an integer, a name or a list, scored by the step 5 reward. No multiple choice and no judge.
- **A blind guess** in `task.grading`, so near-miss credit pays only for beating it (H1).
- **Five levels**, from the 20-step tasks' size (~35K characters) to ~400K, which ~20 sub-calls of ~20K characters cover (spec §6).
- **Renamed entities per seed**, except in LCA, whose code is real.

### 2.1 Aggregation (45% after 7c)

OOLONG-synth fails the licence bar, so this generator rebuilds its method on permissive data.

- **The context** is a log of short texts, each with a date and a user id but no label. Each task draws one source, so the model practises on four registers:

  | Source | Licence | What a line holds | Labels | Draw weight |
  |---|---|---|---|---|
  | CLINC150 | CC-BY-3.0 | A voice-assistant request (~38 characters) | 10 domains, such as banking, travel, small talk | 30% |
  | MASSIVE | CC-BY-4.0 | A voice-assistant request (~32 characters) | 16 scenarios, such as alarm, email, weather | 20% |
  | LEDGAR (via LexGLUE) | CC-BY-4.0 | A contract provision from an SEC filing (~525 characters) | 16 clause types, such as governing law, severability, indemnification | 25% |
  | arXiv titles | CC0 | A paper title (~67 characters) | 12 archives, such as mathematics, astrophysics, computer science | 25% |

  **LEDGAR and arXiv were added when 7c raised aggregation to 45%**, so the bigger share isn't spent on one kind of text. Rishi asked for more data if the bigger split needed it. Both were downloaded with the HF CLI to `D:\RLM-data`. The arXiv titles are a fixed sample of 6,000 per archive, drawn once by `scripts/build_data.py aggregation-pools`. arXiv's label sets can't reach 8 compatible labels, so it serves levels 1-3, and CLINC150 and LEDGAR fill levels 4-5.
- **The question** gives the topic list with a one-line description of each, then asks for an aggregate. There are 7 kinds of question, worded for the source ("messages", "provisions", "papers"):

  | Kind | Answer | Needs labels | Share |
  |---|---|---|---|
  | How many lines are about a topic | Integer | Yes | 22% |
  | How many lines from one user are about a topic | Integer | Yes | 12% |
  | Which topics have more than N lines | List of topics | Yes | 13% |
  | Which users have at least N lines about a topic | List of user ids | Yes | 8% |
  | How many lines about a topic between two dates | Integer | Yes | 15% |
  | How many lines one user has between two dates | Integer | No | 18% |
  | Which user has the most lines | User id | No | 12% |

  The last two need no labels, so code alone answers them. That keeps the whole mix near research §3.1's ~20% code-solvable tasks (§4.3).
- **Per seed**, the source, the topics, their mix, the users and their ids, the date range, the line format (one of three) and the wording all change.
- **Levels** add lines, topics (4 at level 1, 8 at level 5) and users (6-10 at level 1, 26-40 at level 5). A level's line count follows from its size and the source's typical line length.
- **Topics too close to grade fairly** are never asked about together:
  - banking and credit cards, alarm and calendar
  - in LEDGAR, amendments and entire agreements
  - in arXiv, high-energy physics and nuclear physics
  
  MASSIVE's `general` and `play` scenarios are left out *(mine)*. A text that appears under two labels in its source is dropped, because its label can't be read off.
- **Repeats.** A text is never repeated until its topic's pool runs out. That only happens with small MASSIVE scenarios at levels 2 and 3, and then for under 7% of lines.

The held-out variant, `aggregation_heldout`, uses HuffPost headlines (a source never trained on), its own two line formats and its own question wording.

### 2.2 Code generators (40% after 7c)

Each task is a generated Python package of 15 to 60 modules, rendered file after file. Function and module names come from neutral word lists, fresh per seed, so a name says nothing about what the function does. Every file parses as valid Python.

**`callgraph`**: "Which functions can end up calling `X`, directly or through other functions?" The answer is a list of qualified names (`module.function`), scored by set-F1. The gold set comes from the generator's own call graph, so it is exact even where calls are hard to see.
- At level 1, and in a quarter of tasks above it, every call is a plain call, so Python's `ast` finds the answer. A test checks that it does.
- Otherwise some calls are written so that `ast` and name matching miss them: an alias (`step = mod.f; step(x)`), a dispatch table (`_ROUTES["key"](x)`), or a `getattr` built from string pieces (level 4 and up). Comments also mention functions that are never called (level 2 and up). A test checks that plain `ast` gets the wrong answer on most of these tasks.
- Levels add size, the share of hidden calls (0% to 50%), call depth, and the number of callers (2-6 at level 1, 6-18 at level 5).
- The blind guess is every function in the package *(mine)*, which scores a low F1.

**`needle`**, RepoQA's method: "Somewhere in this package is a function that adds up the squares of the even numbers in a list. What is it?" The answer is one qualified name, by exact match.
- The 46 training behaviours come in 15 groups of look-alikes, for example the sum of squares of the even items, of the odd items, of the cubes of the even items, and the count of the even items. The other members of the target's group sit elsewhere in the package as decoys.
- Each behaviour has a reference implementation. A test runs every generated target against it and checks that no decoy matches.
- Behaviour functions call nothing, so running one is a fair way to check it.
- Levels add size, decoys (1 to 3 from the target's group) and other behaviour functions (6 to 32).

**Held-out variants:** `callgraph_heldout` uses its own question wording and two call styles training never sees (`functools.partial` and lambda wrappers). `needle_heldout` asks about 11 behaviours in 5 groups that training never describes.

### 2.3 Repo localization (15%)

Long Code Arena bug localization, Python. Each task is a real GitHub issue and its repository's Python files at the commit the issue was filed against. The answer is the set of files the fix changed, scored by set-F1.

- **What's usable.** LCA ships repository archives only for its 48 test repositories. 40 of them also have issues in the Python train split, so 698 train issues have their code available. Of those, **559 are usable**: 112 changed files too big for any level, and 27 changed more than 5 files.
- **Splits:**

  | Family | Use | Items | Repositories |
  |---|---|---|---|
  | `repo_loc` | Training | 531 | 37 |
  | `repo_loc_select` | Selection set only | 28 | 3 (keras, modin, sanic) |
  | `repo_loc_test` | Held-out test only | 40 of LCA's 50 test items | 40 |

- **Fitting a level.** A repository is usually bigger than a level (the median is ~0.9M characters). So a context is the changed files plus other files from the same repository, up to the level's size. Each level up draws more of that filler from the changed files' own folders, which look most like the answer (20% at level 1, 70% at level 5).
- **Balance.** jax has 135 items and dvc 123. A repository's chance of being drawn is its item count capped at 60 *(mine)*, so those two don't dominate.
- **The blind guess** is every context file whose name the issue mentions. At level 1 that is exactly right for 40% of tasks, falling to 5% at level 5, because the issue names the file. Those tasks count as code-solvable (§4.3).
- **Git history never reaches the sandbox.** The unpacked git data stays on the host, and the context holds files at the issue's base commit only.

### 2.4 Eval tiers (H7)

**Selection set, `selection-v1`:** 64 fixed tasks from private seeds, run at every eval to pick the checkpoint.

| Family | Levels | Tasks |
|---|---|---|
| `aggregation` (held-out seeds) | 1-5 | 20 |
| `callgraph` (held-out seeds) | 1-5 | 10 |
| `needle` (held-out seeds) | 1-5 | 10 |
| `repo_loc_select` (repositories training never sees) | 1-4 | 12 |
| `tickets_all`, `tickets_customer` (the held-out generator family) | 1, 3, 5 | 12 |

Its seeds, and the held-out test's, were drawn once when `scripts/data_check.py` first built the sets, and live in `private/seeds/` (git-ignored). The ticket tasks are the held-out generator family, because production never trains on them. When the cyber family exists, a `selection-v2` gives it about 22 of the 64 tasks.

**Held-out test, `heldout-test-v1`:** 170 tasks, touched once in step 13.

| Part | Tasks |
|---|---|
| `aggregation_heldout`: HuffPost source, its own line formats and wording | 40 |
| `callgraph_heldout`: its own wording and call styles | 20 |
| `needle_heldout`: behaviours training never describes | 20 |
| `repo_loc_test`: LCA's Python test split, each usable item once | 40 |
| OOLONG-synth's test split: 10 per length bucket from 16K to 256K tokens, over its 8 source sets | 50 |

**Public benchmarks** (LCA Java/Kotlin, OOLONG-real, LongBench-v2 code, BrowseComp-Plus, LocBench, LongCodeBench, MRCR) are not downloaded. The cyber benchmarks (CyberSOCEval, ExCyTIn-Bench) went with cyber. They are touched once in step 13, so fetching them belongs there.

### 2.5 The capability check

`capability-v1.jsonl` holds 100 prompts in step 6's format, each graded by machine checks:

| Source | Prompts | Check |
|---|---|---|
| IFEval (Apache-2.0) | 30 | Only prompts whose every instruction maps to a check: lowercase, capitals only, no commas, JSON, a `<<title>>`, quotation, end phrase, postscript, required or forbidden words, word counts |
| MMLU-Pro (MIT) | 40 | The option letter, stratified over its 14 subjects |
| CRUXEval (MIT) | 20 | The function's return value, for integer, boolean and short string outputs |
| Chat, written here | 10 | Short instructions with exact, numeric, JSON, word-count or keyword checks |

## 3. The licence bar (7b)

| Source | Licence | Role |
|---|---|---|
| CLINC150 and its domain map | CC-BY-3.0 | **Training** (aggregation) |
| MASSIVE | CC-BY-4.0 upstream (the MTEB mirror's card says Apache-2.0) | **Training** (aggregation) |
| LEDGAR, via LexGLUE | CC-BY-4.0 | **Training** (aggregation) |
| arXiv metadata (`gfissore/arxiv-abstracts-2021`), titles only | CC0-1.0 | **Training** (aggregation) |
| Long Code Arena bug localization | Apache-2.0; every repository in it is MIT or Apache-2.0 | **Training** (repo localization) |
| HuffPost News Category | CC-BY-4.0 | Eval only, held out by design |
| OOLONG-synth | None stated, and built from sets that include non-commercial ones | Eval only (fails 7b) |
| IFEval, MMLU-Pro, CRUXEval | Apache-2.0, MIT, MIT | Eval only (capability check) |

The call-graph and needle families read no dataset at all.

**Not used:**
- **OOLONG-synth's validation split** fails 7b. Our aggregation generator replaces it.
- **Loghub** fails 7b (research use only).
- **SWE-smith** isn't needed: LCA covers the 15% repo share.
- **`mteb/amazon_massive_intent`** was downloaded by the earlier thread but isn't used, because the generator needs the scenario labels.
- **The cyber sources** (OTRF Security-Datasets, splunk attack_data, EVTX-ATTACK-SAMPLES, AIT-LDS v2) went with decision 7c (§6).

**How it's enforced.** `rlm_rl/data.py` records each source's licence and role, and each generator lists the sources it reads. `tasks.check_mix()` refuses any training mix with an eval-only family or a family that reads an eval-only source. The production config calls it, and a test covers it.

## 4. H8 and H1: sizes and blind guesses

### 4.1 Context sizes

Median characters over 20 tasks per family and level (`manifests/data-check.json`):

| Family | Level 1 | Level 2 | Level 3 | Level 4 | Level 5 |
|---|---|---|---|---|---|
| `aggregation` | 36K | 80K | 149K | 253K | **381K** |
| `callgraph` | 37K | 79K | 151K | 247K | **376K** |
| `needle` | 36K | 80K | 147K | 237K | **371K** |
| `repo_loc` | 39K | 84K | 160K | 272K | **404K** |
| `aggregation_heldout` | 36K | 82K | 152K | 256K | 388K |
| `callgraph_heldout` | 37K | 84K | 157K | 271K | 403K |
| `needle_heldout` | 34K | 79K | 142K | 250K | 376K |
| `repo_loc_select` | 37K | 84K | 157K | 281K | 406K |

Making one level-5 task takes about 1 second for the call graph (its call-graph search), 0.2 s for the needle family and under 0.1 s for the others, on the host.

Every training family's top level is far above 130K characters, and each level is coverable in about 20 sub-calls of ~20K characters.

### 4.2 Blind guesses (H1)

At every family and level, the task's own blind guess earns exactly its exact-hit rate and nothing more, as the step 5 reward guarantees. Exact-hit rates:

| Family | Level 1 | Level 2 | Level 3 | Level 4 | Level 5 |
|---|---|---|---|---|---|
| `aggregation` | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| `callgraph` | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| `needle` | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| `repo_loc` | **0.40** | 0.20 | 0.15 | 0.05 | 0.05 |
| `repo_loc_select` | 0.20 | 0.25 | 0.25 | 0.20 | 0.20 |
| Held-out code and aggregation | 0.00-0.10 at every level | | | | |

The odd held-out aggregation hit is an integer guess that happened to land on the count. Needle tasks have no blind guess, because a name can't be guessed from metadata, so they score 0.

Repo localization is the outlier: at level 1, 40% of issues name exactly the files their fix changed. A policy that learns "submit the files the issue mentions" gets those right without reading code. That is a real skill, LCA's own benchmark keeps such issues, and it falls to 5% by level 5. These tasks count toward the code-solvable share.

### 4.3 Code-solvable share

Research §3.1 asks for about 20% of training tasks to be solvable by code alone. Drawn by the production mix:

| Level 1 | Level 2 | Level 3 | Level 4 | Level 5 |
|---|---|---|---|---|
| 36% | 24% | 28% | 25% | 20% |

Level 1 is higher because every level-1 call graph is plain and LCA issues at level 1 often name their files. A task counts as code-solvable when it's an aggregation question that needs no labels, a plain call graph, or an LCA issue that names exactly its changed files.

## 5. H7: what the checks found

| Check | Result |
|---|---|
| Held-out vs training question templates, shared 6-word sequences | Aggregation 0 of 105, code 0 of 28, repo localization 0 of 21 |
| Held-out vs training needle descriptions, shared 6-word sequences | 0 of 104 |
| Needle behaviours, line formats and call styles in both | None |
| HuffPost headlines that also appear in any training source (CLINC150, MASSIVE, LEDGAR, arXiv titles) | 0 of 58,954 headlines; no shared 8-word sequence |
| OOLONG-synth test messages that also appear in our training pools | 20 of 78,006, all short stock phrases such as "hello", "thanks!" and "i like it" |
| LCA test items in training | 0 (one test issue shares its base commit with a training issue; the issues and fixes differ) |
| Selection repositories in training | None |

**One overlap to know about.** 32 of the 40 LCA test items come from repositories that training also uses, with different issues and base commits. Only 8 are from repositories training never sees. LCA only ships code for its test repositories, so holding all of them out would leave training almost nothing. The held-out test reports the 8 separately (`grading["repo"]` names each task's repository). The selection set uses three repositories that training never sees.

## 6. Decision 7c: cyber dropped

**Decided by Rishi, 2026-10-11:** skip cyber as a whole, and split its 35% between better coding and one more thing we already have.

**The split, as I read it** *(mine)*: code generators +20 points, aggregation +15.
- **Code to 40%:** call graph 20% and needle 20%. That is "better coding": half again more code practice than any other single domain.
- **Aggregation to 45%:** it's the canonical RLM skill (OOLONG), its generator makes fresh tasks without limit, and OOLONG's own test split measures it in the held-out test. Rishi asked for more data if the bigger share needed it, so it gained LEDGAR contract provisions and arXiv titles (§2.1).
- **Repo localization stays at 15%:** it draws on a fixed pool of 531 issues, so a bigger share would repeat issues sooner.

Moving the split, for example code 40%, repo 25%, aggregation 35%, is a one-line change to `tasks.DOMAIN_MIX`.

**What changed in the code:**
- `tasks.DOMAIN_MIX` is now aggregation 45%, code 40%, repo 15%, and `config.production()` takes it directly. The machinery for an unbuilt domain (`missing_domains()`, `--leave-out-missing`) is gone.
- The four cyber sources registered earlier the same day (OTRF Security-Datasets, splunk attack_data, EVTX-ATTACK-SAMPLES, AIT-LDS v2) are out of `rlm_rl/data.py`. They're listed under `NOT_USED` with the reason. `scripts/fetch_cyber_data.sh` is deleted.
- The selection set and held-out test had no cyber tasks, so their plans are unchanged. Their aggregation tasks now come from generator v2, with the new sources.

**What happened before the decision.** Rishi first cut cyber from step 7. Later he asked for its data to be downloaded without anyone opening it, leaving Atomic Red Team out. The session's automatic permission check refused every cyber download except finishing the AIT-LDS file the earlier thread had started. I didn't work around the refusal. Rishi then dropped cyber. No cyber generator was written, and no cyber file was opened.

## 7. The dataset manifest

`manifests/dataset-manifest.json` (sha256 `357589b52c1e…`) records:
- decisions 7a, 7b and 7c, the domains and their families, and the production mix
- every family's generator version, role (train or eval), sources and levels
- the selection set and held-out test plans, and the capability file
- every source's origin, pinned revision, licence and role, with the size and sha256 of each of its 111 files (~22 GB)
- the 4 derived files with their hashes, and what was left out and why

`config.manifest()` puts the manifest file's hash, the generator versions and the mix into every checkpoint's `config.json` (spec §8). Rebuild it with `scripts/build_manifest.py` after any data change. It takes about two minutes, most of it hashing.

## 8. How it was tested

| File | Tests | What they cover |
|---|---|---|
| `tests/test_data_families.py` | 26 | **Code families:** every template matches its reference and decoys differ; every family and level gives valid Python and a scorable task; plain call graphs are what `ast` finds; hidden calls defeat plain `ast`; needle targets do what the question says and decoys don't; held-out wording and call styles never appear in training; sizes. **Aggregation:** every family and level within its size range; recounting each answer from the rendered text alone (labelling each line by its source pool) gives the gold answer, over all 7 kinds, every source and both variants; every training source gets drawn, and no label name is shared between sources; no label appears in the context; training never reads held-out sources. **Repo localization:** every family and level; selection repositories and test items never reach training; the test split's own wording. **Mix and licences:** the 7c mix; the production config; the licence bar; the selection set's 64 tasks; eval sets rebuilt identically from private seeds; the capability prompts. Seeds never appear in a task id, and every family is deterministic per seed. |
| `tests/test_training.py` | +1 | A production dry run on the production families: 2 steps, the 64-task selection set at checkpoint 0, and the mix and manifest in `config.json` |

- **In WSL with Docker:** 241 tests, 240 pass and 1 skips (the tokenizer check, as before), in about 10 minutes. No sandbox container was left behind.
- **On Windows:** 205 tests run, 189 pass and 16 skip because they need Docker or a Unix socket.
- **Existing tests, two changes:**
  - The step 6 loop tests ask for the ticket stand-in explicitly (`tasks_from="tickets"`), because production now defaults to the production families.
  - Step 5's blind-guesser test runs over every registered family, so it now meets lists and names. It counts an exact hit through the reward instead of `==`, and makes 12 tasks per level for the new families (300 for the ticket ones). The first WSL run caught this.

## 9. Defaults I picked

| Fork | Default |
|---|---|
| What fills the aggregation share | Our OOLONG-style generator, since OOLONG-synth fails 7b (the earlier thread's finding, accepted by Rishi). At 45% it draws on CLINC150 30%, MASSIVE 20%, LEDGAR 25% and arXiv titles 25% of the time. |
| Aggregation labels | Coarse labels (domains, scenarios, clause types, archives) rather than fine ones, so each label has hundreds to thousands of distinct texts |
| Which labels never meet | Close pairs in every source, such as banking and credit cards, amendments and entire agreements, high-energy and nuclear physics (`aggregation.py` lists them). MASSIVE's `general` and `play` are left out. |
| Code-solvable share | 30% of aggregation questions and 25% of call-graph tasks above level 1, which brings the mix near ~20% |
| The blind guess for lists | The cheap list: every topic, every user, or every function in the context. For LCA, every file the issue names. |
| LCA contexts | The changed files plus filler from the same repository up to the level's size, with more same-folder filler at higher levels |
| LCA balance | A repository's draw weight is its item count, capped at 60 |
| LCA held-out repositories | keras, modin and sanic go to the selection set only |
| The selection set's held-out family | The ticket tasks |
| The held-out test's OOLONG sample | 10 per length bucket from 16K to 256K tokens; date and month answers left out because their format isn't stated |
| The capability check | 30 IFEval, 40 MMLU-Pro, 20 CRUXEval and 10 chat prompts |
| Where the data lives | `D:\RLM-data`, or `$RLM_DATA_DIR`. The capability path in the config is `data:derived/capability/capability-v1.jsonl`, so it works from Windows and WSL. |
| Where 7c's 35% went | Code generators +20 (call graph and needle 20% each), aggregation +15 (§6) |
| More aggregation data | LEDGAR (legal) and arXiv titles (science): permissive, distinct registers, and clean coarse labels. GoEmotions was passed over because its labels are too subjective to count exactly, and DBpedia and MultiEURLEX because they're share-alike. |

## 10. What changes the spec or plan

For the planning thread, which owns both files.

1. **Plan step 7:**
   - The sources table: OOLONG-synth validation is replaced by our aggregation generator; SWE-smith isn't used; the cyber row is dropped (decision 7c, §6).
   - The eval tiers: `selection-v1` and `heldout-test-v1` as in §2.4. Public benchmarks move to step 13.
   - Decision 7c and the new mix (§6), and the two new aggregation sources (§2.1).
2. **Spec §6:** the families and their levels are in §2. The task mix is `tasks.production_mix()`.
3. **Spec §8:** the dataset manifest is `manifests/dataset-manifest.json`, and `config.json` holds its hash.
4. **Spec §10, code layout:** add `rlm_rl/data.py`, `rlm_rl/capability_set.py`, the four new modules in `rlm_rl/tasks/`, and `scripts/build_data.py`, `build_manifest.py` and `data_check.py`.
5. **Research §3.3 and §3.4:** LCA has code for only 40 of its train repositories, so repo localization has 531 training items in 37 repositories.
6. **Plan step 8, the red-team:** see §11.
7. **Research §3.4, plan steps 12 and 13, and the goal line:** cyber is gone. The mix is 45/40/15, and the cyber benchmarks leave step 13's list. The goal says "blue-team cyber and code tasks"; it now reads long-context aggregation and code.

## 11. Known limits, for the red-team (step 8)

| Item | Why it matters | What to check |
|---|---|---|
| **`copied_answer` on repo localization** | When an issue names the gold file, a policy might type the path straight from the issue without printing it, and the detector would zero an honest answer *(inferred)* | Run honest-style episodes on such tasks and count false alarms |
| **Pretraining memory on LCA** | The code and issues are public, and the base model may have seen the fixes *(inferred)* | The step 11 baselines show how far the base model gets without reading |
| **Label noise in aggregation** | The gold labels are the datasets' own. Sub-calls will misread some lines, which caps accuracy at high levels; arXiv titles are the noisiest source. That's OOLONG's nature, and the controller keeps the pass rate in band. | The second seed doesn't test this; production's level curves will |
| **Synthetic code shares its filler** | Held-out code tasks reuse the training filler bodies. They differ in question wording, call styles and behaviours, not in filler. | The selection and held-out results per family |
| **More code-solvable tasks at level 1** | 36% at level 1, then 20-28% above it. Every family starts at level 1, so the first steps lean a little toward code-only wins. | Per-family pass rates in the first 10 steps of production |

## 12. Commands

```bash
python scripts/build_data.py all                 # unpack LCA git data, index LCA items, sample arXiv titles, build the capability prompts
python scripts/build_manifest.py                 # pin every data file in manifests/dataset-manifest.json
python scripts/data_check.py                     # H7, H8, H1, licences, mix and eval sets -> manifests/data-check.json
.venv-linux/bin/python scripts/sandbox_ctl.py check                                              # the full suite, inside Ubuntu
.venv-linux/bin/python scripts/train.py start --kind production --run prod-dry --lr 2e-5 --dry-run   # no W&B, no credit
```
