# Step 7 results: data

> **Step 7 of [production-build-plan.md](production-build-plan.md)** (holes H7 and H8, the licence bar, research §3 and §8.1). Built and tested 2026-10-11 on Rishi's PC, on Windows and inside WSL2 with Docker.
>
> **Decisions applied (Rishi, 2026-10-11):** 7a, the research mix of 35% cyber, 30% OOLONG-style aggregation, 20% code generators and 15% repo localization; 7b, permissive licences only in training, with everything else eval-only. **Scope cut (Rishi, 2026-10-11):** nothing was built for the cyber family in this step. Its 35% share is still open (§6).
>
> **Status: committed and pushed to `claude/project-thread-9nnh3n`**, on top of step 6's `claude/project-thread-2apfvb`. The cyber family is still open (§6).
>
> **No W&B credit was used.** Nothing here calls a model. The `.env` file was never opened. Nothing was installed. The data was downloaded with the Hugging Face CLI to `D:\RLM-data`, outside the repo, as Rishi asked. That is the one exception to "everything inside the repo folder". Later the same day Rishi asked for the cyber sources too (§6.1). Their licences were checked and they are registered and pinned, but only one of their downloads went through.
>
> Anything marked *(mine)* is my choice, not the spec's. Anything marked *(inferred)* hasn't been checked against the live system.

## Summary

**Step 7 is built for every domain except cyber, and every done-when check passes on what was built.** Three new task families fill 65 of the 100 points of the 7a mix. The production selection set, the held-out test, the capability check's 100 prompts and a dataset manifest are in. The regression suite is now **240 tests** (214 before); all pass in WSL with Docker, except the step 4 tokenizer check, which skips there as before.

| Done-when check | Result | Evidence |
|---|---|---|
| The top levels have a median context above 130K characters (H8) | **Pass** | Level 5 medians: aggregation 386K, call graph 376K, needle 371K, repo localization 404K characters (`scripts/data_check.py`, 20 tasks per family and level, §4). |
| No generator, template or source overlaps training (H7) | **Pass** | Held-out question templates share no 6-word sequence with training templates. Held-out behaviours, line formats, call styles, sources and LCA items are disjoint from training (§5). |
| Every training source passes the licence bar (7b) | **Pass** | Training reads only CLINC150 (CC-BY-3.0), MASSIVE (CC-BY-4.0) and Long Code Arena (Apache-2.0, repos MIT or Apache-2.0). `tasks.check_mix()` refuses an eval-only family or source in any training mix, and the production config runs it (§3). |
| A dataset manifest pins every version | **Pass** | `manifests/dataset-manifest.json` pins each source's revision, licence and role, and the sha256 of all 105 data files and 3 derived files. Its hash goes into every checkpoint's `config.json` (§7). |

**The cyber share is not built.** `config.production()` refuses to start a production run until it is, unless you pass `--leave-out-missing`. That trains on the built domains alone, spread over 100% (§6.2, decision 7c).

**Two things need you:**
1. **The rest of the cyber downloads.** My permission settings blocked every cyber download except finishing the AIT-LDS file the earlier thread had started. `bash scripts/fetch_cyber_data.sh` fetches the rest at the pinned versions. Run it yourself, or allow it, and I'll rebuild the manifest (§6.1).
2. **Decision 7c, what fills the cyber share.** The data alone doesn't fill it: the cyber task generator still has to be built, and building it means reading those logs. Nothing before production (steps 8 to 11) needs it (§6.2).

---

## 1. What was built

| Path | What it is |
|---|---|
| `rlm_rl/data.py` | Where the data lives (`D:\RLM-data`, `/mnt/d/RLM-data` from WSL, or `$RLM_DATA_DIR`), every source's origin, revision, licence and role, the licence bar, and the utterance pools |
| `rlm_rl/tasks/aggregation.py` | The OOLONG-style aggregation family, training and held-out (§2.1) |
| `rlm_rl/tasks/synthcode.py` | Generated Python packages: the call-graph and needle-function families, training and held-out (§2.2) |
| `rlm_rl/tasks/behaviours.py` | The needle family's 57 behaviours in 20 groups of look-alikes, each with a reference implementation |
| `rlm_rl/tasks/repoloc.py` | Long Code Arena bug localization: training, selection and test families (§2.3) |
| `rlm_rl/tasks/oolong.py` | A fixed sample of OOLONG-synth's test split, for the held-out test |
| `rlm_rl/tasks/__init__.py` | Gains the new families, decision 7a's mix (`DOMAIN_MIX`), `production_mix()`, `check_mix()` and `private_seed_list()` |
| `rlm_rl/capability_set.py` | Builds the capability check's 100 prompts (§2.5) |
| `rlm_rl/evals.py` | Gains `selection-v1` (64 tasks) and `heldout-test-v1` (170 tasks) (§2.4) |
| `rlm_rl/config.py` | Production now uses the 7a mix, `selection-v1` and the capability prompts, and pins the dataset manifest's hash. `tasks_from="tickets"` keeps step 6's stand-in for tests and dry runs. |
| `rlm_rl/capability.py`, `rlm_rl/training.py` | A capability path can be `data:<path>`, relative to the data root |
| `scripts/build_data.py` | Unpacks the LCA repositories' git data, indexes the usable LCA items, and builds the capability prompts |
| `scripts/build_manifest.py` | Writes `manifests/dataset-manifest.json` |
| `scripts/data_check.py` | The done-when checks above, written to `manifests/data-check.json` |
| `scripts/fetch_cyber_data.sh` | Fetches the cyber sources onto the data root at their pinned versions (§6.1) |
| `scripts/train.py` | Gains `--leave-out-missing` and `--ticket-tasks` |
| `tests/test_data_families.py` | 25 tests (§8) |
| `tests/test_training.py`, `tests/dryrun.py` | A production dry run on the 7a families. The dry run's oracle can now answer lists and names, and the step 6 loop tests use the ticket stand-in explicitly. |
| `README.md` | The Layout section lists the new files |

### On the D drive (outside git)

| Folder | Holds | Size |
|---|---|---|
| `hf/` | The Hugging Face downloads, at pinned revisions: CLINC150, MASSIVE, LCA bug localization (Python split and the 48 repository archives), HuffPost headlines, OOLONG-synth's test split, IFEval, MMLU-Pro and CRUXEval | ~21 GB |
| `github/clinc__oos-eval/` | CLINC150's intent-to-domain map | 4 KB |
| `derived/lca-git/` | The git data unpacked from each LCA repository archive | 11 GB |
| `derived/lca/` | The index of usable LCA items | 6 MB |
| `derived/capability/` | `capability-v1.jsonl`, the 100 capability prompts | 55 KB |
| `zenodo/ait-lds-v2/` | AIT-LDS v2: `russellmitchell.zip` complete, its md5 matching Zenodo's. The other 7 zips aren't fetched yet (§6.1). | 7.1 GB |
| `logs/` | Build and check logs | small |

The earlier thread downloaded everything under `hf/` and `github/`. I reused it and downloaded nothing new.

## 2. The task families

Every family follows the rules from research §3.1 and spec §3:
- **Generated on the host** from a random 64-bit seed that never enters the sandbox (H3). Only the context text does.
- **A verifiable answer:** an integer, a name or a list, scored by the step 5 reward. No multiple choice and no judge.
- **A blind guess** in `task.grading`, so near-miss credit pays only for beating it (H1).
- **Five levels**, from the 20-step tasks' size (~35K characters) to ~400K, which ~20 sub-calls of ~20K characters cover (spec §6).
- **Renamed entities per seed**, except in LCA, whose code is real.

### 2.1 Aggregation (30% of 7a)

OOLONG-synth fails the licence bar, so this generator rebuilds its method on permissive data.

- **The context** is a log of short messages, each with a date and a user id but no label. The messages are CLINC150 utterances grouped by domain (10 labels such as banking, travel and small talk) or MASSIVE utterances grouped by scenario (16 labels such as alarm, email and weather).
- **The question** gives the topic list with a one-line description of each, then asks for an aggregate. There are 7 kinds of question:

  | Kind | Answer | Needs labels | Share |
  |---|---|---|---|
  | How many messages are about a topic | Integer | Yes | 22% |
  | How many messages from one user are about a topic | Integer | Yes | 12% |
  | Which topics have more than N messages | List of topics | Yes | 13% |
  | Which users sent at least N messages about a topic | List of user ids | Yes | 8% |
  | How many messages about a topic between two dates | Integer | Yes | 15% |
  | How many messages one user sent between two dates | Integer | No | 18% |
  | Which user sent the most messages | User id | No | 12% |

  The last two need no labels, so code alone answers them. That keeps the whole mix near research §3.1's ~20% code-solvable tasks (§4.3).
- **Per seed**, the topics, their mix, the users and their ids, the date range, the line format (one of three) and the wording all change.
- **Levels** add messages, topics (4 at level 1, 8 at level 5) and users (6-10 at level 1, 26-40 at level 5).
- **Topics too close to grade fairly** are never asked about together, for example banking and credit cards, or alarm and calendar. MASSIVE's `general` and `play` scenarios are left out *(mine)*. A message that appears under two labels in its source is dropped, because its label can't be read off.
- **A message is never repeated** until its topic's pool runs out. That didn't happen at levels 4 and 5 in testing. It happened for up to 7% of lines at levels 2 and 3 in a few tasks, when a small MASSIVE scenario was drawn.

The held-out variant, `aggregation_heldout`, uses HuffPost headlines (a source never trained on), its own two line formats and its own question wording.

### 2.2 Code generators (20% of 7a)

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

### 2.3 Repo localization (15% of 7a)

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

**Public benchmarks** (CyberSOCEval, ExCyTIn-Bench, LCA Java/Kotlin, OOLONG-real, LongBench-v2 code, BrowseComp-Plus, LocBench, LongCodeBench, MRCR) are not downloaded. They are touched once in step 13, so fetching them belongs there.

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

**How it's enforced.** `rlm_rl/data.py` records each source's licence and role, and each generator lists the sources it reads. `tasks.check_mix()` refuses any training mix with an eval-only family or a family that reads an eval-only source. The production config calls it, and a test covers it.

## 4. H8 and H1: sizes and blind guesses

### 4.1 Context sizes

Median characters over 20 tasks per family and level (`manifests/data-check.json`):

| Family | Level 1 | Level 2 | Level 3 | Level 4 | Level 5 |
|---|---|---|---|---|---|
| `aggregation` | 35K | 76K | 149K | 252K | **386K** |
| `callgraph` | 37K | 79K | 151K | 247K | **376K** |
| `needle` | 36K | 80K | 147K | 237K | **371K** |
| `repo_loc` | 39K | 84K | 160K | 272K | **404K** |
| `aggregation_heldout` | 36K | 80K | 148K | 259K | 380K |
| `callgraph_heldout` | 37K | 84K | 157K | 271K | 403K |
| `needle_heldout` | 34K | 79K | 142K | 250K | 376K |
| `repo_loc_select` | 37K | 84K | 157K | 281K | 406K |

Making one level-5 task takes about 1 second for the call graph (its call-graph search), 0.2 s for the needle family and under 0.1 s for the others, on the host.

Every training family's top level is far above 130K characters, and each level is coverable in about 20 sub-calls of ~20K characters.

### 4.2 Blind guesses (H1)

At every family and level, the task's own blind guess earns exactly its exact-hit rate and nothing more, as the step 5 reward guarantees. Exact-hit rates:

| Family | Level 1 | Level 2 | Level 3 | Level 4 | Level 5 |
|---|---|---|---|---|---|
| `aggregation` | 0.00 | 0.00 | 0.00 | 0.00 | 0.05 |
| `callgraph` | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| `needle` | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| `repo_loc` | **0.40** | 0.20 | 0.15 | 0.05 | 0.05 |
| `repo_loc_select` | 0.20 | 0.25 | 0.25 | 0.20 | 0.20 |
| Held-out code and aggregation | 0.00-0.10 at every level | | | | |

The aggregation hits are integer guesses that happened to land on the count. Needle tasks have no blind guess, because a name can't be guessed from metadata, so they score 0.

Repo localization is the outlier: at level 1, 40% of issues name exactly the files their fix changed. A policy that learns "submit the files the issue mentions" gets those right without reading code. That is a real skill, LCA's own benchmark keeps such issues, and it falls to 5% by level 5. These tasks count toward the code-solvable share.

### 4.3 Code-solvable share

Research §3.1 asks for about 20% of training tasks to be solvable by code alone. Drawn by the production mix (cyber left out):

| Level 1 | Level 2 | Level 3 | Level 4 | Level 5 |
|---|---|---|---|---|
| 34% | 21% | 20% | 22% | 23% |

Level 1 is higher because every level-1 call graph is plain and LCA issues at level 1 often name their files. A task counts as code-solvable when it's an aggregation question that needs no labels, a plain call graph, or an LCA issue that names exactly its changed files.

## 5. H7: what the checks found

| Check | Result |
|---|---|
| Held-out vs training question templates, shared 6-word sequences | Aggregation 0 of 105, code 0 of 28, repo localization 0 of 21 |
| Held-out vs training needle descriptions, shared 6-word sequences | 0 of 104 |
| Needle behaviours, line formats and call styles in both | None |
| HuffPost headlines that also appear in CLINC150 or MASSIVE | 0 of 58,954 headlines; no shared 8-word sequence |
| OOLONG-synth test messages that also appear in our training pools | 20 of 78,006, all short stock phrases such as "hello", "thanks!" and "i like it" |
| LCA test items in training | 0 (one test issue shares its base commit with a training issue; the issues and fixes differ) |
| Selection repositories in training | None |

**One overlap to know about.** 32 of the 40 LCA test items come from repositories that training also uses, with different issues and base commits. Only 8 are from repositories training never sees. LCA only ships code for its test repositories, so holding all of them out would leave training almost nothing. The held-out test reports the 8 separately (`grading["repo"]` names each task's repository). The selection set uses three repositories that training never sees.

## 6. The cyber share (35% of 7a): sources registered, family open

Rishi cut the cyber family from step 7 on 2026-10-11. Later that day he asked for its data to be downloaded and wired up, with Atomic Red Team left out and no cyber file opened. What stands:
- `tasks.DOMAIN_MIX` keeps 7a's 35% for cyber, with no family behind it. `tasks.missing_domains()` returns `["cyber"]`, and the dataset manifest records it.
- `config.production()` refuses to build a production config while the share is empty, unless told to leave it out.
- No cyber generator was written, and no cyber file was opened.

### 6.1 The cyber sources

The four sources are registered in `rlm_rl/data.py` (domain `cyber`) and pinned in the manifest. The licences come from GitHub's licence detection and the Zenodo record.

| Source | Where | Licence | Role (7b) | Pinned at | Size | Downloaded |
|---|---|---|---|---|---|---|
| OTRF Security-Datasets | GitHub | MIT | Training allowed | commit `d9d40ef` | ~0.8 GB | No, blocked |
| splunk attack_data | GitHub, Git LFS | Apache-2.0 | Training allowed | commit `4389fa7` | Unknown (LFS) | No, blocked |
| EVTX-ATTACK-SAMPLES | GitHub | GPL-3.0 | Eval only | commit `4ceed2f` | ~6 MB | No, blocked |
| AIT-LDS v2 | Zenodo record 5789064 | CC-BY-NC-SA-4.0 | Eval only (held-out log sources) | v2.0, md5 per zip | 131 GB in 8 zips | 1 of 8: `russellmitchell.zip` (7.1 GB), md5 matches |

The research listed OTRF's licence as unresolved. GitHub reads its licence file as MIT, so it passes 7b. Loghub still fails 7b, so it isn't registered.

**Why only one download went through.** The session's automatic permission check refused the three `git clone`s and the script that fetches the other AIT-LDS zips. It allowed only the resume of the AIT-LDS file the earlier thread had started. I didn't try to get around the refusal. `scripts/fetch_cyber_data.sh` does the rest:
- It clones the three repos at their pinned commits, pulling splunk's Git LFS content.
- It downloads the 7 remaining zips 3 at a time, resuming partial files, and checks each against Zenodo's md5.
- It skips anything already finished, so it's safe to rerun.
- It opens and runs nothing it downloads.

Zenodo served ~0.6 MB/s per file here, so the zips take many hours even 3 at a time. Windows Defender may quarantine some attack samples, as it did with Atomic Red Team; the manifest counts a quarantined file as missing. After the script finishes, `python scripts/build_manifest.py` records each repo's commit and file count and each zip's md5 check.

### 6.2 Decision 7c: what fills the cyber share

Nothing before production needs this: the red-team (step 8), the dashboard (step 9), the second seed (step 10, ticket tasks) and the baselines (step 11) all run without it. It has to be settled before step 12.

- **Train production without cyber: recommended (mine).** The built domains are spread over 100%: aggregation 46%, call graph 15%, needle 15%, repo localization 23%. It's one flag (`--leave-out-missing`). Production still trains the RLM skills the research cares most about (decomposing long contexts, code for exact work, sub-calls for reading), but the LoRA isn't trained on blue-team logs.
- **Build the cyber family first, in its own step.** That keeps 7a as decided. The earlier attempts hit repeated safeguards refusals, so it would need a different approach that you're comfortable with.
- **Re-split 7a another way,** for example more code. Say which.

## 7. The dataset manifest

`manifests/dataset-manifest.json` (sha256 `bdc9a79de6f0…`) records:
- decisions 7a and 7b, the domains and their families, the unbuilt cyber share, and the mix without it
- every family's generator version, role (train or eval), sources and levels
- the selection set and held-out test plans, and the capability file
- every source's origin, pinned revision, licence and role, with the size and sha256 of each of its 105 files (~21 GB)
- the cyber sources by commit (the repos, with their file and missing-file counts once cloned) or by Zenodo's md5 (the AIT-LDS zips), and which of them are downloaded
- the 3 derived files with their hashes, and what was left out and why

`config.manifest()` puts the manifest file's hash, the generator versions, the mix and the missing domains into every checkpoint's `config.json` (spec §8). Rebuild it with `scripts/build_manifest.py` after any data change. It takes about two minutes, most of it hashing.

## 8. How it was tested

| File | Tests | What they cover |
|---|---|---|
| `tests/test_data_families.py` | 25 | **Code families:** every template matches its reference and decoys differ; every family and level gives valid Python and a scorable task; plain call graphs are what `ast` finds; hidden calls defeat plain `ast`; needle targets do what the question says and decoys don't; held-out wording and call styles never appear in training; sizes. **Aggregation:** every family and level within its size range; recounting each answer from the rendered text alone (labelling each message by its source pool) gives the gold answer, over all 7 kinds and both variants; no label appears in the context; training never reads held-out sources. **Repo localization:** every family and level; selection repositories and test items never reach training; the test split's own wording. **Mix and licences:** the 7a mix and the cyber refusal; the production config; the licence bar; the selection set's 64 tasks; eval sets rebuilt identically from private seeds; the capability prompts. Seeds never appear in a task id, and every family is deterministic per seed. |
| `tests/test_training.py` | +1 | A production dry run on the 7a families with cyber left out: 2 steps, the 64-task selection set at checkpoint 0, and the manifest in `config.json` |

- **In WSL with Docker:** 240 tests, 239 pass and 1 skips (the tokenizer check, as before), in about 10 minutes. No sandbox container was left behind.
- **On Windows:** 204 tests run, 188 pass and 16 skip because they need Docker or a Unix socket.
- **Existing tests, two changes:**
  - The step 6 loop tests ask for the ticket stand-in explicitly (`tasks_from="tickets"`), because production now defaults to the 7a mix.
  - Step 5's blind-guesser test runs over every registered family, so it now meets lists and names. It counts an exact hit through the reward instead of `==`, and makes 12 tasks per level for the new families (300 for the ticket ones). The first WSL run caught this.

## 9. Defaults I picked

| Fork | Default |
|---|---|
| What fills the 30% aggregation share | Our OOLONG-style generator on CLINC150 and MASSIVE, since OOLONG-synth fails 7b (the earlier thread's finding, accepted by Rishi) |
| Aggregation labels | Domains and scenarios rather than fine intents, so each label has hundreds of distinct messages |
| Which labels never meet | Banking and credit cards, small talk and meta, alarm and date-time, calendar and alarm, music and audio, transport and travel, news and Q&A, plus three HuffPost pairs. MASSIVE's `general` and `play` are left out. |
| Code-solvable share | 30% of aggregation questions and 25% of call-graph tasks above level 1, which brings the mix near ~20% |
| The blind guess for lists | The cheap list: every topic, every user, or every function in the context. For LCA, every file the issue names. |
| LCA contexts | The changed files plus filler from the same repository up to the level's size, with more same-folder filler at higher levels |
| LCA balance | A repository's draw weight is its item count, capped at 60 |
| LCA held-out repositories | keras, modin and sanic go to the selection set only |
| The selection set's held-out family | The ticket tasks |
| The held-out test's OOLONG sample | 10 per length bucket from 16K to 256K tokens; date and month answers left out because their format isn't stated |
| The capability check | 30 IFEval, 40 MMLU-Pro, 20 CRUXEval and 10 chat prompts |
| Where the data lives | `D:\RLM-data`, or `$RLM_DATA_DIR`. The capability path in the config is `data:derived/capability/capability-v1.jsonl`, so it works from Windows and WSL. |
| Production without cyber | Refused unless `--leave-out-missing` is passed |

## 10. What changes the spec or plan

For the planning thread, which owns both files.

1. **Plan step 7:**
   - The sources table: OOLONG-synth validation is replaced by our aggregation generator; SWE-smith isn't used; the cyber row is deferred (§6).
   - The eval tiers: `selection-v1` and `heldout-test-v1` as in §2.4. Public benchmarks move to step 13.
   - Decision 7c (§6.2), and the cyber sources in §6.1.
2. **Spec §6:** the families and their levels are in §2. The task mix is `tasks.production_mix()`.
3. **Spec §8:** the dataset manifest is `manifests/dataset-manifest.json`, and `config.json` holds its hash and the missing domains.
4. **Spec §10, code layout:** add `rlm_rl/data.py`, `rlm_rl/capability_set.py`, the four new modules in `rlm_rl/tasks/`, and `scripts/build_data.py`, `build_manifest.py`, `data_check.py` and `fetch_cyber_data.sh`.
5. **Research §3.3 and §3.4:** LCA has code for only 40 of its train repositories, so repo localization has 531 training items in 37 repositories.
6. **Plan step 8, the red-team:** see §11.
7. **Plan step 12:** production needs `--leave-out-missing` until 7c is settled.

## 11. Known limits, for the red-team (step 8)

| Item | Why it matters | What to check |
|---|---|---|
| **`copied_answer` on repo localization** | When an issue names the gold file, a policy might type the path straight from the issue without printing it, and the detector would zero an honest answer *(inferred)* | Run honest-style episodes on such tasks and count false alarms |
| **Pretraining memory on LCA** | The code and issues are public, and the base model may have seen the fixes *(inferred)* | The step 11 baselines show how far the base model gets without reading |
| **Label noise in aggregation** | The gold labels are the datasets' own. Sub-calls will misread some messages, which caps accuracy at high levels. That's OOLONG's nature, and the controller keeps the pass rate in band. | The second seed doesn't test this; production's level curves will |
| **Synthetic code shares its filler** | Held-out code tasks reuse the training filler bodies. They differ in question wording, call styles and behaviours, not in filler. | The selection and held-out results per family |
| **More code-solvable tasks at level 1** | 34% at level 1, then 20-23% above it. Every family starts at level 1, so the first steps lean a little toward code-only wins. | Per-family pass rates in the first 10 steps of production |
| **The cyber share** | 35% of 7a has no family, and 3 of its 4 sources aren't downloaded (§6) | `scripts/fetch_cyber_data.sh`, then decision 7c |

## 12. Commands

```bash
python scripts/build_data.py all                 # unpack LCA git data, index LCA items, build the capability prompts
python scripts/build_manifest.py                 # pin every data file in manifests/dataset-manifest.json
python scripts/data_check.py                     # H7, H8, H1, licences, mix and eval sets -> manifests/data-check.json
.venv-linux/bin/python scripts/sandbox_ctl.py check                                              # the full suite, inside Ubuntu
.venv-linux/bin/python scripts/train.py start --kind production --run prod-dry --lr 2e-5 --dry-run --leave-out-missing   # no W&B, no credit
```
