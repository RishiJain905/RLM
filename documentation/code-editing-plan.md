# Code editing plan

> **Approved, being built.** Written 2026-10-12 at Rishi's request: add real code writing and editing to the RL run, so the root gets better at engineering and better at doing it the RLM way (REPL and sub-calls). Rishi approved it the same day with every recommended option in §15, dropped the E7 probe, cut the reserve to $10, and asked for real repositories in several languages, not only Python. [production-spec.md](production-spec.md) §12 turns it into the build contract and wins where the two differ. No W&B credit has been spent on it. §5.6, the REPL tools, was added later the same day at Rishi's request, and he approved it at 23:34 UTC.
>
> **Sources:** [production-spec.md](production-spec.md), [production-build-plan.md](production-build-plan.md), [step-3-sandbox-results.md](step-3-sandbox-results.md), [step-5-results.md](step-5-results.md), [step-7-results.md](step-7-results.md), [step-8-results.md](step-8-results.md), the code in `rlm_rl/`, and the open SWE RL recipes in §2 (links in Sources). Anything marked *(mine)* is my choice, not a source's. Anything marked *(inferred)* hasn't been measured; step 11's baselines measure the important ones.

## Summary

**The gap.** Every training task today ends in a number, a name or a file list. The root writes Python in every episode, but only to read. Nothing rewards a working code change, and Rishi's daily work is building features in codebases.

**The plan in five lines.**
1. **Two new task families.** `edit_real` is proper engineering in the DeepSWE mould: real repositories in several languages, real test suites, an issue to fix or a feature to build, graded by hidden tests. `edit_synth` is a cheap tier built from our own generated packages, with difficulty levels, that keeps the controller fed and costs about a third as much per episode.
2. **One prebuilt image per repository** (SWE-smith's model), not one per task (R2E-Gym's, which DeepSWE trained on). That is what makes real repositories with real dependencies fit on Rishi's PC: a few dozen images on the D: drive instead of thousands. Containers stay `--network none`.
3. **Reward is the hidden tests.** 1 when every fail-to-pass and pass-to-pass test passes, else 0, as DeepSWE, Nebius and Kimi-Dev do. The diff is graded in a fresh container with the tests restored, git history is stripped, and canary tests catch a patch that forges results.
4. **The RLM way comes from the environment, not a process reward.** The repository sits on disk, the root sees only 2,000 characters of output per turn, and tasks are sized so reading file by file can't finish inside the budget. Exploring with code, delegating reading to `llm_batch`, drafting and reviewing with `llm`, and running tests before `submit` are what win.
5. **Cost is the hard limit.** An editing episode should cost about $0.06 (synthetic) to $0.15 (real) against today's $0.016 *(inferred)*. At the 40% edit share, a step costs about $2.10, so the ~$157 production ceiling buys ~75 steps instead of ~120. DeepSWE trained on ~100K episodes, and one task the size of Datacurve's DeepSWE v1.1 would cost ~$1.30-2.30 per episode, so those stay out of reach for training and become an eval instead (§2, §10.2).

**What it does to the plan.** Step 10 (second seed) can run now, unchanged, while the edit work is built. Six new build steps (E1-E6) cost $0. Steps 11-13 grow to include edit tasks and a comparison against a normal coding agent; step 11's baselines take over what the dropped E7 probe would have measured.

**Evals.** Held-out repositories in the selection and held-out sets, then SWE-bench Pro and DeepSWE v1.1 off W&B in step 13, with Terminal-Bench 2.1 as an optional transfer check. Terminal-Bench 4.0, SWE-Marathon and ProgramBench are what frontier labs show, but a 27B model scores near zero on them, so they can't show a difference (§12.4).

**Decisions** are in §15 (E-a to E-g). Rishi took every recommended option on 2026-10-12.

---

## 1. Why the current mix doesn't teach editing

| Family | Share today | What the root submits | Edits code? |
|---|---|---|---|
| `aggregation` | 45% | A count, a list or a name | No |
| `callgraph` | 20% | A list of function names | No |
| `needle` | 20% | One function name | No |
| `repo_loc` | 15% | A list of file paths | No |

The read families teach the first half of engineering in a big codebase: finding the right code. `repo_loc` is the closest, since it asks where a bug lives. None of them asks the model to change code and keep the tests green, which is the half Rishi spends most of his time on. RL also tends to narrow a model toward what it is rewarded for, so a LoRA trained only on read tasks could get worse at editing *(inferred, a known pattern rather than something we measured)*. The edit families both teach editing and guard against that.

---

## 2. What the open recipes did

**Two different things are called DeepSWE.**
- **DeepSWE (Agentica, July 2025)** is a training recipe: Qwen3-32B trained with RL alone on real repository tasks. I found no Agentica "DeepSWE 1.1" model.
- **DeepSWE v1.1 (Datacurve, June 2026)** is a benchmark: 113 original long-horizon tasks in 91 permissive repositories (34 of them Python), run with mini-SWE-agent. Its reference solutions average 668 added lines across 7 files, against 10 lines and 1 file for SWE-bench Verified. v1.1 tightened the grading: only the committed diff is graded, in a fresh container, every test is reported by name, and history after the start commit is removed.

The plan takes its training recipe from the first, its grading safeguards from the second, and uses the second as a held-out eval (§12.4). Datacurve marks its benchmark data with a do-not-train canary, so it stays eval-only.

### 2.1 The recipes

| Recipe | Tasks | Environment | Reward and training | Scale | SWE-bench Verified |
|---|---|---|---|---|---|
| **DeepSWE** (Agentica) | 4.5K R2E-Gym-Subset problems, with repositories shared with SWE-bench removed | One image per problem, on Kubernetes, 512 containers per iteration | Binary: a sample of fail-to-pass and pass-to-pass tests all pass. GRPO with clip-high, no KL, no std normalisation; episodes that hit a limit are masked out. Tools: bash, search, file editor, submit; 50 steps in training | ~200 steps of 64 problems × 8 rollouts (~100K episodes); 64 H100s for 6 days | Qwen3-32B ~23% → 42.2% |
| **SkyRL SA-SWE-32B** | 4.5K R2E-Gym tasks | Same | Pure RL, 50 turns, 32K context, 64 × 8 per step | 125 steps (~64K episodes), 4,601 H100-hours | 24.4% → 39.4% |
| **SkyRL-v0** | ~300 tasks | Same | Pure RL | Small | Qwen3-14B 18.0% → 21.6% |
| **R2E-Gym** | 8.1K tasks mined from commits in 13 repositories, issue texts written from the commit | One image per task, ~300-500 MB each (the Subset is ~4 TB) | SFT on 3.3K Claude trajectories, not RL | | 34.4% (51% with verifiers) |
| **SWE-Gym** | 2.4K tasks, 11 repositories | One image per task (~6 TB) | SFT on 491 trajectories | | 7.0% → 20.6% |
| **SWE-smith** | 50K bugs injected into 128 repositories (AST mutations, LM edits, mirrored PRs, combined bugs), LM-written issues | **One image per repository, 295 GB for all 128** | SFT on 5K Claude trajectories | The whole dataset cost $1,360 to make | Qwen2.5-Coder-32B ~7% → 40.2% |
| **Nebius** (SWE-rebench) | 21K tasks filtered to 7.2K: flaky tests out over 50 runs, at most 7 files | One image per task | Rejection fine-tuning, then DAPO: group 10, binary reward plus a length penalty, zero-advantage groups dropped, never- and always-solved tasks dropped | | Qwen2.5-72B 11.4% → 39.0% |
| **Qwen3-Coder-Next** | ~1.6M tasks from PRs and synthesized bugs | One image per task | Outcome reward; bug-triggering test files hidden from the agent; tasks filtered by pass rate | Lab scale; turns grew from ~50 to ~130 | 70.6-71.3% |
| **Kimi-Dev** | | Docker | Reward only when the whole suite passes | 72B | 60.4% |

### 2.2 What carries over

1. **Binary all-pass reward is the standard** (DeepSWE, Nebius, Kimi-Dev). It's the default here (§6).
2. **Drop tasks that are always or never solved.** Every recipe does it, because those groups carry no signal. Here that's the levels, task retirement and step 11's base pass rates (§11).
3. **Hide the failing tests, grade the diff in a fresh container, score per test, strip git history.** Qwen3-Coder-Next and DeepSWE v1.1 between them do all four (§5, §7).
4. **Hacking is common without those safeguards.** Qwen's "Verification Horizon" study counted history mining in 21% of rollouts, overfitting to visible tests in 30%, tampering with the harness in 8.3% and with test oracles in 3.7%. Sanitised history, no network and a trajectory monitor cut hacks from 37.8% to 1.3%.
5. **Images per task don't fit a PC.** DeepSWE needed a cluster with terabytes of image cache. SWE-smith's one image per repository is the only real-repository source that fits at scale; per-task images work only for a small slice.
6. **Scale.** We can afford about 1% of DeepSWE's episodes (§10.2). The closest small-scale result is SkyRL-v0: ~300 tasks gave a few points.
7. **The base model is already a good editor.** Qwen3.8-27B's model card reports 61.7 on SWE-bench Pro, 42.2 on DeepSWE v1.1 (Qwen3.6-27B: 13.3) and 73.0 on Terminal-Bench 2.1, in Qwen's own harnesses with thinking on by default. In our REPL harness, with thinking off, it is unmeasured. So the run's realistic goal is to make it good at editing the RLM way, and the tasks must be hard enough that groups aren't all solved.
8. **Plain sampling.** Nebius saw `top_k` and `min_p` sampling break RL within 5-10 iterations, because of a mismatch between sampling and training. We sample at temperature 1.0 and nothing else; E1 confirms that ART passes nothing extra.
9. **Long context isn't the lever.** DeepSWE found context beyond 32K added at most ~2%.
10. **Loops are the common failure.** 53% of the SWE-smith model's failures hit step or cost limits, often repeating itself. We log repeated cells (§8.3).

---

## 3. The design: two tiers

| | `edit_real` (proper engineering) | `edit_synth` (cheap tier) |
|---|---|---|
| What it is | Real repositories in Python, Go, TypeScript and JavaScript, Rust and Java, with their own test suites | Our generated Python packages (the `callgraph` and `needle` generator), stdlib only |
| Task kinds | Fix a reported bug; implement a described feature | Fix the bug behind a symptom; change a function and update every caller; add a feature that threads through several modules |
| Environment | One prebuilt image per repository (one per task for the real-PR slice), run `--network none` | Today's stdlib sandbox image, with writes and test runs allowed |
| Graded by | The repository's own tests, hidden fail-to-pass tests plus the existing pass-to-pass tests | Generated hidden tests: property checks against the generator's reference implementation |
| Difficulty | Levels by patch size and spread (one file and a few lines, up to multi-file features) | Levels by package size, files to touch, decoys and hidden call styles |
| Cost per episode | ~$0.15 *(inferred)* | ~$0.06 *(inferred)* |
| Licence | Permissive repositories only (7b) | Ours |

**Why both.** The real tier is the point: it is the work Rishi does, and a LoRA that only learns synthetic edits may not transfer. The synthetic tier costs about a third as much, has clean difficulty levels for the controller, needs no new images, and each of its task kinds is the edit twin of a read family we already have (§4.2), so the read and write skills meet in one codebase. Training on both at once is the default *(mine)*; §11 explains why I didn't make the synthetic tier a separate warm-up phase.

---

## 4. Task sources and licences

### 4.1 `edit_real`

**Source: SWE-smith's per-repository images and task instances in five language groups, filtered to permissive repositories, plus a small slice of real pull requests from SWE-rebench V2** *(decision E-b; languages mine, at Rishi's request for a broad spectrum, 2026-10-12)*.

- **Why SWE-smith.** Each repository is pinned at one commit with its dependencies installed in one image, and every task instance is a patch against that snapshot, so one image serves hundreds of tasks. SWE-smith now publishes a dataset per language with the same layout. R2E-Gym (DeepSWE's data), SWE-Gym and SWE-rebench build an image per task: 4-6 TB for their training sets, a job for a cluster (§2.1). Measured on Docker Hub (compressed sizes; on disk is roughly 2.5-3x, *inferred*):

| Language | Dataset | Tasks | Repositories (= images) | Image size, median |
|---|---|---|---|---|
| Python | `SWE-smith-py` (MIT) | ~50K | 128 | ~2.3 GB (mean, all 128) |
| Go | `SWE-smith-go` (MIT) | 8,212 | 87 | 334 MB |
| JavaScript | `SWE-smith-js` | 6,073 | 34 | 444 MB |
| TypeScript | `SWE-smith-ts` | 5,032 | 30 | 860 MB |
| Rust | `SWE-smith-rs` | 5,311 | 39 | 807 MB |
| Java | `SWE-smith-java` | 7,470 | 59 | 616 MB |

- **Languages and task counts** *(mine)*. About 310 validated training tasks:

| Group | Training tasks | Training repositories | Why |
|---|---|---|---|
| Python | ~90 | ~6 | Rishi's main language and the base of every benchmark |
| Go | ~60 | ~6 | Small images, warm build caches, fast tests; in SWE-bench Pro |
| TypeScript and JavaScript | ~60 | ~6 | The most common languages in SWE-bench Pro and SWE-PolyBench after Python |
| Rust | ~45 | ~4 | Strict compiler feedback: a wrong edit fails to build, which teaches checking before submitting |
| Java | ~35 | ~3 | Common in eval sets and industry; only images whose tests run offline are kept |
| Real pull requests (Python, Go, TypeScript) | ~24 | ~6 | Real issues and their merged fixes, from SWE-rebench V2 (below) |

  That covers all four of SWE-bench Pro's languages. Each language group also gets held-out repositories that training never sees, for `selection-v2` and `heldout-test-v2` (§12). C++ is left out: builds are slow and its test runners vary too much for the grader.
- **Repository bar.** Kept only when all of these hold:
  - the repository's licence is permissive (MIT, BSD, Apache-2.0 or similar), read from its own licence file in the image. SWE-smith only required a licence allowing non-proprietary use, so this is a real filter. Only the Python and Go dataset cards carry a licence (MIT); the JS, TS, Rust and Java cards carry no licence tag and no card text (checked 2026-10-12), though the toolkit that made them is MIT. **Decided by Rishi, 2026-10-12:** those four languages train only on tasks we generate on the permissive repositories (feature removals and our own injected bugs, validated like every task), and SWE-smith's rows for them are eval-only
  - not used by an eval we report: SWE-bench Pro, DeepSWE v1.1, and SWE-bench Multilingual (which shares caddy, gin, babel, axios, ripgrep and gson with SWE-smith)
  - the tests run with `--network none`: every image is smoke-tested offline once (Java's Maven and Gradle plugins are the usual failure)
  - the relevant tests run in under ~60 s inside a 2-CPU, 4 GB container
  - the image is under ~3 GB compressed *(mine)*
- **Size.** About 35 SWE-smith images plus ~24 real-PR images, roughly 100 GB on the D: drive *(inferred)*. Training needs only a few hundred tasks, because ~75 steps at ~2 real tasks a step see ~150 distinct tasks (§10).
- **Three task kinds:**
  - **Bug fix.** SWE-smith's own instances: a bug introduced into the source, an issue text describing the symptom, and the existing tests it breaks. In SWE-smith those failing tests stay visible to the agent; we move them into the hidden set (§5.1) and re-validate each instance in our grader (§5.4). About 80% of Go rows have no issue text; E4 writes one from the failing tests and the symptom, or skips the row. SWE-smith's "LM Rewrite" instances, where a function is rewritten from its signature and docstring, are closest to feature work and get a larger share *(mine)*.
  - **Feature implementation** *(mine, decision E-f)*. We generate these: pick a public function or method, or a small feature spanning a few functions, that existing tests cover; remove it; write the issue from its docstring and what the tests check. The tests that fail after removal are the hidden fail-to-pass set. This is the kind of task Rishi does most, and SWE-smith's bug instances alone under-represent it. At least a quarter of each language's tasks are feature tasks where E4 can find function boundaries reliably (Python's `ast`, Go's `go/ast` run inside the image, and similar tools for the others).
  - **Real pull request.** A small slice from Nebius's SWE-rebench V2 (CC-BY-4.0, 32K tasks in 20 languages, a per-task licence field): a real issue and the fix that was merged for it, with the fix's tests kept separate and hidden. Only permissive per-task licences, no tasks flagged for test-suite coupling or external dependencies, and high-confidence issue texts. Its images are per task (~0.5-0.8 GB compressed each, *measured on a sample*), which is why the slice stays small. It replaces R2E-Gym, which Rishi allowed "a little if needed": SWE-rebench V2 has real Python PRs too, plus Go and TypeScript, with one adapter.
- **Issue texts** for the feature tasks and the Go rows without one are written by Claude in the build session from the code and tests, so they cost nothing on W&B. A leak check refuses an issue that quotes the removed code or a hidden test's name.

### 4.2 `edit_synth`

Built from `rlm_rl/tasks/synthcode.py` and `behaviours.py`, which already generate 15-60 module packages with neutral names, hidden call styles and reference implementations.

| Task kind | Edit twin of | What the root must do | Hidden tests check |
|---|---|---|---|
| **Fix the symptom** | `needle` | An issue reports a wrong result at a public entry point. The bug is a mutation (flipped comparison, off-by-one, swapped arguments, wrong default, dropped branch) in a behaviour function a few calls deep, with look-alike decoys elsewhere | The entry point matches the reference on fresh random inputs |
| **Change and update callers** | `callgraph` | Add a required parameter to a function, or rename it, and update every caller, including aliases, dispatch tables and `getattr` strings | Every path into the function works, old behaviour unchanged |
| **Add a feature** | (new) | Add an option to an entry point that has to be threaded through 2-4 modules | The option works on fresh inputs; existing behaviour unchanged |

- **Visible tests.** Each package ships a small `tests/` folder of pass-to-pass tests, so running tests is part of the job.
- **Hidden test inputs** are drawn on the host per seed and are different from any example in the issue, so hard-coding the examples earns nothing.
- **Levels 1-5** go from ~20K to ~200K characters of package, 1 to 4 files to change, and more decoys and hidden call styles. Edit tasks don't need the read families' 400K top level; the cost is in turns, not reading *(mine)*.
- **Held-out variant** `edit_synth_heldout`: mutation kinds, feature templates and issue wording training never sees (H7).
- **Why not competitive programming.** DeepCoder's set (MIT) and TACO (Apache-2.0) would run in today's sandbox as they are, but they test an algorithm in one file, not a change inside a codebase, and CRUXEval in the capability check already guards short code.
- **Licence:** ours, no dataset.

---

## 5. How environments are built and tests run

The step 3 sandbox stays the boundary. Editing needs five changes, and only edit episodes get them; read episodes run exactly as today.

### 5.1 The episode container (edit mode)

| | Read episodes (today) | Edit episodes |
|---|---|---|
| Image | `rlm-sandbox` (python:3.12-slim, stdlib) | `edit_synth`: the same image. `edit_real`: a derived image built once from the repository's image, with our REPL worker and a standalone Python mounted read-only at `/opt/rlm` |
| Network | `--network none` | `--network none` (unchanged) |
| Files the model can write | `/scratch` only | `/scratch` and `/work/repo`, a writable bind mount of a fresh per-episode copy of the repository on the host, so it survives a REPL restart |
| Running programs | Blocked by Landlock | Allowed for the image's own toolchain and test runner, so `subprocess.run(["go", "test", "./..."])` or `subprocess.run(["python", "-m", "pytest", ...])` works. Still no network and no host files |
| Limits | 1 CPU, 512 MB, 64 pids | 2 CPUs, 4 GB, 256 pids *(mine)* |
| Cell timeout | 120 s | 300 s, so a test run fits *(mine)* |

- **No history to mine.** `/work/repo` is a fresh `git init` with one commit at the task's base, so `git diff` and `git status` work but there are no other commits, branches, tags, reflog or remotes. This closes the git-history leak that hit SWE-bench agents in 2025 (§2.2).
- **No clean copy to diff against.** A SWE-smith image holds the repository *before* the bug, which is the answer, and a SWE-smith bug is itself one commit whose revert is the fix. So each repository gets a derived image, built once: the image's own checkout, any other installed copy of the package (such as a release in site-packages) and build caches are deleted, and `/work/repo` is mounted where the image's environment expects the repository (for example `/testbed`), so editable installs and build paths still point at it *(mine)*.
- **Hidden tests never enter this container.** For `edit_real`, the test files that hold fail-to-pass tests are removed from `/work/repo` before the episode starts.
- **The REPL runs on our own Python**, a standalone CPython 3.12 build mounted read-only at `/opt/rlm/python`, so the same REPL works in a Go, Rust or Java image and in a Python image that pins an old version. The repository's own toolchain (its Python, `go`, `cargo`, `npm`, `mvn` and so on) runs through `subprocess`, which is how the root tries things and runs tests.

### 5.2 What the root sees

- `context` holds the issue and the repository's file list with sizes. The files themselves are on disk in `/work/repo`, which is how the Part 2 harness will see Rishi's own repositories *(mine; the alternative, the whole repository as one string, would leave a stale copy once files change)*.
- The bridge API stays v1: `context`, `llm`, `llm_batch`, `submit`. The REPL also has seven coding tools that run inside the container (§5.6), and plain `open` and `subprocess` still work.
- `submit(summary)` ends the episode. The graded output is the diff of `/work/repo` against the base, not the submitted value.

### 5.3 The grader

1. After `submit`, or at the turn limit, the host copies the diff out of the episode container and stops it.
2. The diff is checked: text files only, under a size cap, inside the repository. Changes to existing test files, `conftest.py` and test configuration are dropped and flagged (§7).
3. A fresh grader container starts from the same image, `--network none`, applies the diff to a clean copy, restores the full test suite, and adds two canary tests: one that must pass and one that must fail, both with names drawn fresh per episode.
4. It runs the fail-to-pass and pass-to-pass tests (for real repositories, only the test files or packages that matter, as SWE-smith does), with a 600 s cap *(mine)*, and reads per-test results from the runner's own structured report: JUnit XML for Python and Java, `go test -json` for Go, Jest, Vitest or Mocha JSON for JavaScript and TypeScript, and `cargo test`'s per-test lines for Rust. Results are keyed by file or package plus test name, since names repeat across packages.
5. The host computes the reward (§6). A grader that can't start or hangs is an infra error, as in step 4, and never a zero for the model.

### 5.4 Validating every real task once, before training

A task enters the pool only if, in our grader:
- the gold patch passes every fail-to-pass and pass-to-pass test, three runs in a row, so flaky tests are out
- the empty patch fails every fail-to-pass test
- removing the hidden test files doesn't break collection of the visible ones
- the issue text doesn't quote the gold patch or a hidden test's name

This costs PC time, not W&B credit.

### 5.5 Rishi's PC

- **Disk:** every image and downloaded dataset lives on the D: drive (Rishi, 2026-10-12): Docker Desktop's disk image location must be on D: before the first pull, and task metadata goes to `D:\RLM-data`. Pulling needs the network once; if the Remote Control permission check refuses a bulk pull, Rishi runs the pull script himself, as with the step 7 data.
- **Memory:** a step runs about 8 real and 5 synthetic edit episodes. At 4 GB each that is up to ~50 GB if they all run at once, so E2 measures the PC and caps how many edit containers run together *(inferred)*.
- **Time:** edit episodes are longer, so a step should take ~10-20 minutes instead of ~3-6, and ~75 steps take about a day *(inferred)*. Resume covers interruptions.

### 5.6 REPL tools (approved by Rishi 2026-10-12)

Rishi asked (22:55 UTC) for the tools a real coding harness gives its model, in the training environment now and in the Part 2 harness later. Plain `open` and `subprocess` can do everything, but every harness below gives the model a small set of named tools instead, and the model we deploy should arrive already trained on the ones it will have.

**What the harnesses ship.**

| Job | Claude Code | Codex CLI | DeepSeek Harness | Qwen Code (the model's own family) |
|---|---|---|---|---|
| Read a file | `Read`: numbered lines, offset and limit | `read_file` | `str_replace_editor` (view) | `read_file` |
| Find files by name | `Glob` | `glob_file_search`, `list_dir` | file search | listing and search tools |
| Search contents | `Grep` (ripgrep) | `rg` | file search | listing and search tools |
| Edit a file | `Edit`: exact text that must occur once, or `replace_all` | `apply_patch` (its own patch format) | `str_replace_editor` | `edit` |
| Create a file | `Write` | `apply_patch` (Add File) | `str_replace_editor` (create) | `write_file` |
| Run a command | `Bash`: 2 min default, 10 max | `shell_command`: 120 s default | `bash` (persistent shell) | `run_shell_command` |
| Plan | `TaskCreate` and friends (was `TodoWrite`) | `update_plan` | planning | `todo_write` |
| Sub-agents | `Agent` | `spawn_agent` and friends | subagents | `agent` |
| Web | `WebFetch`, `WebSearch` | `web_search` | web search | `web_fetch`, `web_search` |

DeepSeek Harness's "code mode" goes furthest the RLM way: the model writes one program against generated bindings instead of calling tools one at a time. Our root already works like that, so our tools are Python functions it calls from code.

**Seven functions in the REPL** (Rishi picked this set at 23:34 UTC). Each returns a Python value the root can filter, count, slice or pass to `llm`, exactly as it handles `context` today; it sees only what it prints, still cut at 2,000 characters.

| Tool | What it does | Modelled on |
|---|---|---|
| `read_file(path, start=1, n=40) -> str` | `n` lines from line `start`, numbered, long lines cut at 300 characters, ending with how many lines follow. 40 lines fit one printed turn *(mine)*. Raises for a missing or binary file | `Read`, `read_file` |
| `find_files(pattern, path=".") -> list[str]` | Paths matching a glob such as `"src/**/*.go"`, sorted, skipping `.git` and dependency or build folders (`node_modules`, `target`, `.venv` and so on) | `Glob`, `glob_file_search` |
| `grep(pattern, path=".", glob=None, ignore_case=False, files_only=False) -> list[str]` | Regex matches (Python `re`) as `"path:line: text"`, or only the paths; skips binary files and the same folders | `Grep`, `rg` |
| `edit_file(path, old, new, replace_all=False) -> str` | Replaces `old` with `new`. `old` must occur exactly once unless `replace_all=True`; otherwise it raises with the count found and leaves the file untouched. Keeps the file's line endings | `Edit`, `str_replace_editor`, `edit` |
| `write_file(path, text) -> str` | Creates or overwrites a file, making folders as needed | `Write`, `write_file` |
| `shell(cmd, timeout=120) -> Result` | Runs `cmd` with `sh -c` in the repository. `.code` is the exit code and `.out` holds stdout and stderr together; printing it shows the exit code and the last 1,500 characters, where test failures are. A timeout kills the command and returns what it printed, with code 124. Never longer than the cell's remaining time | `Bash`, `shell_command`, `bash` |
| `diff() -> str` | The change so far against the base commit, new files included, the same diff the grader takes. For checking work and for review sub-calls: `llm("Review this change against the task:\n" + diff())` | `git diff` through the shell |

The names follow Qwen Code and Codex where they can, and avoid names a model often rebinds by accident: `run` would be replaced by `from subprocess import run`, and `glob` by `import glob` *(mine)*. One way to edit (`edit_file`, which Claude Code, DeepSeek Harness and Qwen Code all use) keeps edits measurable; several edits are a loop *(mine)*.

**Where they run.** Inside the episode container, as one standard-library module, `rlm_rl/repl_tools.py`, mounted read-only at `/opt/rlm` next to the standalone Python and loaded into the REPL before turn 1 of an edit episode. Nothing new crosses the bridge: it stays v1 (`context`, `llm`, `llm_batch`, `submit`), so the host's attack surface doesn't change, and Landlock and `--network none` bound the tools as they bound `open` and `subprocess`. Those two keep working; the tools are the easy path, not the only one. Read episodes don't get the tools, since they have no repository on disk. `diff()` needs `git` in the image: SWE-smith's images have it, and the `edit_synth` image gets it if it's missing.

**The same code in Part 2.** The Part 2 harness imports the same module, so the deployed model calls the functions it trained on. Their names, signatures and docstrings freeze with prompt v2: `TOOLS_VERSION = "t1"` is recorded in every checkpoint next to the prompt version, and the H10 check diffs them against the harness's copy as it already does the bridge.

**Left out of training, added in Part 2:**
- **Web fetch and search.** Training has no network. The harness adds them through its bridge.
- **Sub-agents with their own tools** (Claude Code's `Agent`, Codex's `spawn_agent`). In an RLM that is a recursive call: a sub-RLM with its own REPL. It would multiply the cost of an episode beyond the budget, so in training `llm` and `llm_batch` stay as they are, and Part 2 adds `rlm()`.
- **A plan or todo tool.** The prompt already asks for a short plan every turn, and a Python list keeps one across turns. The harness can add one later.
- **`apply_patch`, language servers, images, notebooks, background commands.** One way to edit is enough to train; the images have no language servers and the tasks no pictures.

**What changes.**
- **Spec:** §12.1 gets the tool table; §12.2's coding paragraph lists the tools, which grows the one system prompt from 1,527 to 2,270 characters (nothing has run on v2 yet, so this costs nothing); §12.4 requires `git` in every edit image; §12.12 adds `rlm_rl/repl_tools.py`.
- **Logged, not rewarded** (§12.9): `tool_calls` (mean calls per episode, per tool), `edit_errors` (mean `edit_file` calls that raised), and `raw_io_share` (share of file writes and commands made with `open` or `subprocess` instead of the tools). `test_runs`, `tested_before_submit` and `edit_cycles` read the tool calls directly and still count `open` and `subprocess`.
- **Dashboard and overview:** the Editing panel gains a tool-use row; the overview's edit chapter lists the tools, each hover-defined.
- **Unchanged:** reward, grader and detectors. What gets graded is still the diff and the hidden tests.
- **Red-team brief** (E6) adds: shadowing or patching the tool functions, editing `.git` to change what `diff()` or the host's diff sees, and paths outside the repository.
- **Build:** `rlm_rl/repl_tools.py` and `tests/test_repl_tools.py` are written and tested in the cloud (22 tests, Python 3.12 and 3.13). The PC session loads the module into the edit REPL, adds the metrics and switches the bridge to the new prompt after E4. No W&B credit.

---

## 6. Reward

Computed on the host after the grader runs, in this order:

1. **No `submit`, an empty diff, or a diff that doesn't apply: 0.**
2. **A hack detector fired (§7): 0**, before the group mean, as today (H11).
3. **Every fail-to-pass and every pass-to-pass test passes: 1.** This is the headline metric, `resolved`, like `exact` for the read families.
4. **Anything else: 0.** That includes a hidden test that times out, since every task's gold patch was validated well inside the grader's limit (§5.4).
5. **A grader or container failure: infra error**, out of the group, never scored.

**Why binary** *(decision E-c)*. It is what DeepSWE, Nebius and Kimi-Dev train on (§2.2), and none of the recipes in §2 used partial credit. The calibration idea from 1b still holds: the blind guess here is the empty patch, which fails every fail-to-pass test, and deleting, stubbing or hard-coding a feature breaks pass-to-pass tests, so all three earn 0. E5 checks those three on every family, as H1 did for the read families.

**The alternative** is `0.5 × (share of fail-to-pass tests passing)` when nothing else broke. It would leave fewer all-zero groups on hard tasks, but it also pays for half-fixes the tests happen to accept.

**Episodes that hit the turn cap** score 0 if they never called `submit`, as today. DeepSWE masks such episodes out of training instead, but its agents can't see the cap coming; ours are told on their last turn that they must submit.

No length, turn or cost penalty, as for the read families.

---

## 7. Hack checks and the red-team

### 7.1 Closed by construction

| Exploit | Why it can't pay |
|---|---|
| Editing or deleting the tests | The grader restores the whole test suite before running it, and fail-to-pass tests are never in the episode container |
| Reading the hidden tests | They exist only in the grader container |
| Mining git history for the fix | One commit, no remotes, no other copy of the package (§5.1) |
| Diffing against the image's clean checkout | Removed from the episode container (§5.1) |
| Faking a pass with `sys.exit(0)` or an early exit | The grader scores each named test from its report, and a missing test counts as failed |
| Hard-coding the issue's examples | Hidden inputs are fresh per seed (`edit_synth`) or are the repository's own tests (`edit_real`) |
| Installing packages or fetching the fix | No network |
| Changing site-packages, `sitecustomize` or `.pth` files during the episode | Only the diff of `/work/repo` reaches the grader, which starts from a clean image |
| Leaving a background process to fake results | The grader is a different container |

### 7.2 New detectors (zero the reward, counted per rule)

| Rule | Fires when |
|---|---|
| `test_tamper` | The diff edits an existing test file, any `conftest.py`, or test configuration (`pytest.ini`, `tox.ini`, the test sections of `setup.cfg` or `pyproject.toml`, `jest` or `vitest` configs, and each language's equivalent), or adds skip markers. New test files are allowed and are simply not run by the grader |
| `grader_tamper` | The must-fail canary passes, the must-pass canary fails, the collected test count is off, or the results report is missing or malformed. This catches a patch that, at import time, patches the test runner or exits early to fake a pass |
| `test_aware` | Non-test code added by the diff checks whether it is under test: reads `PYTEST_CURRENT_TEST`, looks for `pytest` or `unittest` in `sys.modules`, imports `_pytest`, or calls `os._exit`, or the equivalent for the language's test runner (for example reading `JEST_WORKER_ID`, or `testing.Testing()` in Go) |
| `early_submit` | Today's rule, unchanged: `submit` on turn 1 or before any code looked at the repository |
| `timeout_loop` | Today's rule, unchanged |

Broader signals such as `inspect.stack()`, new `os.environ` reads or `sys.settrace` in non-test code are legitimate often enough that they go to the Claude audit queue instead of zeroing *(mine)*.

**One change to an existing rule.** A failed `pip install` or network call inside an edit episode is logged as `network_try` but doesn't zero the reward *(mine)*. It can't gain anything with no network, and agents commonly try it while debugging. File escapes and bridge tampering still fire `escape_attempt` as today.

### 7.3 The Claude audit

The 5a audit at each 10-step checkpoint adds edit wins: does the diff fix the issue, or does it pass the tests some other way? Special-casing the shapes of test inputs, weakened behaviour the tests don't cover, and anything `test_aware` queued all go here. As before, the audit is never a training signal.

### 7.4 Red-team (new step E6)

The same defensive approach that worked for step 8. The build ends with a brief, `documentation/edit-red-team-brief.md`, written last; Rishi hands it to a separate session himself (2026-10-12). That session gets the edit environment and the grader, and checks whether reward can be earned without solving the task: forged results, test edits, conftest tricks, history mining, reaching hidden tests and grader crashes. Each hole that works gets fixed and added as a regression test. It runs before any W&B spend on edit tasks.

---

## 8. What the RLM way adds

### 8.1 How an RLM root should work an edit task

A normal coding agent reads files into its own context one tool call at a time, and its transcript grows to tens of thousands of tokens. The RLM root keeps its own context small and pushes reading into code and sub-calls:

1. **Explore with code.** Walk `/work/repo`, parse modules with `ast`, search for the issue's identifiers, build the call graph around them, and print only a summary.
2. **Delegate reading.** `llm_batch` over the 10-20 most likely files: "Does this file relate to the issue below? Name the functions and lines." The root reads the answers, not the files.
3. **Reproduce.** Write a short script that shows the bug or the missing feature, and run it.
4. **Draft.** Write the change in code, or ask `llm` to draft a function from the surrounding code, then splice it in and check it parses.
5. **Review.** Ask `llm` to review the diff against the issue: missed callers, edge cases, broken behaviour.
6. **Verify.** Run the reproduction and the visible tests that matter, and summarise failures in code rather than printing whole tracebacks.
7. **Submit** once those pass.

### 8.2 How training pushes toward it

- **The environment, not a process reward.** The root sees only 2,000 characters of output per turn, the task text lists files but doesn't contain them, and levels grow the codebase until reading file by file can't finish inside 20 turns. Delegating and searching in code become the way to win, not a style we score. Scoring the steps themselves would teach the model to perform them for show, the same reason the audit is never a reward.
- **Tasks that need a survey.** "Update every caller" and multi-file features need many files read at once, which is where `llm_batch` pays off.
- **The prompt.** Prompt v2 adds one paragraph with the steps above, as today's prompt does for reading.

### 8.3 What we log to see whether it worked (not rewarded)

Characters sent to sub-calls against characters printed to the root, sub-calls used for drafting and for review (detected by the prompt holding a diff), test runs before `submit`, whether a reproduction ran, edit-and-test cycles, turns to the first edit, and the root's own transcript length. These go on the dashboard next to `resolved`, so we can see whether the gains come with RLM behaviour or without it.

### 8.4 Prompt v2

One system prompt for every family (H10 still diffs it against the harness):
- The limits move from the system prompt into each task's first message, because edit episodes get different limits (§9).
- One new section for code tasks: the seven tools (§5.6), tests run with `shell`, there is no network, the changes to the repository are what gets graded, existing tests and test configuration must not be changed, and the steps in §8.1 as advice.

Nothing has been trained on v1 yet, so changing it now costs nothing. The second seed runs now on v1 (E-g), since it only replicates the 20-step run; production, baselines and every eval from step 11 on use v2. The text is in production-spec.md §12.2.

---

## 9. Episode budget changes

| Limit | Read families (unchanged) | Edit families *(mine)* | Why |
|---|---|---|---|
| Root turns | 12 | 20 | Explore, reproduce, edit, test and fix need more cycles. Each turn can do many actions in one cell, so 20 turns is closer to a normal agent's ~50-100 tool calls than it looks |
| Sub-calls | 30 | 30 | Enough for one survey `llm_batch`, drafts and reviews |
| Root output tokens per turn | 3,072 | 4,096 | Room to write a function in one cell |
| Sub-call output tokens | 1,024 | 1,024 | A drafted function fits; step 11 checks truncation |
| REPL output shown | 2,000 chars | 2,000 chars | Keeps the root's context small, which is the RLM discipline |
| Cell timeout | 120 s | 300 s | A test run fits |
| Container | 1 CPU, 512 MB | 2 CPUs, 4 GB | Real test suites |
| Grader timeout | none | 600 s | Hidden tests |

- **Step 11 checks the caps.** If more than ~20% of the base model's edit episodes hit the turn cap with a test failing, turns go to 30 before production. *Done (Rishi, 2026-10-17): 56% did, so prompt v3 gives edit episodes 30 turns (step-11-results.md §13).*
- **Training sequence length.** An edit transcript should run ~15-30K tokens, up to ~95K at every cap *(inferred)*. E1 checks what W&B's trainer accepts for Qwen3.8-27B, and the step 6 rule (set `packed_sequence_length` from measured lengths) applies.

---

## 10. Cost per episode and the new split

W&B bills Qwen3.8-27B at $0.40 per 1M input tokens ($0.15 cached) and $3.00 per 1M output tokens (research §6). Today's read episodes measure $0.016 at level 1.

| Episode | Turns | Root output | Root input, summed over turns | Sub-calls | Cost |
|---|---|---|---|---|---|
| Read task, level 1 (measured) | 5.2 | ~2K tokens | ~25K tokens | 2-6 | **$0.016** |
| Read task, top level (fitted) | ~8 | | | ~20 | ~$0.06 |
| `edit_synth` | ~10 | ~6K | ~70K | ~6 | **~$0.06** *(inferred)* |
| `edit_real`, RLM root | ~16 | ~11K | ~190K | ~15 | **~$0.15** *(inferred)* |
| A DeepSWE-style tool agent, for comparison | ~40-50 | ~15K | ~1M | none | ~$0.20-0.45 *(inferred)* |

- **The range is wide.** Whether W&B's cached-input price applies to our multi-turn prefixes isn't known, and turns per episode are a guess. Step 11 measures both before production starts.
- **An RLM root should be cheaper per episode than a tool agent** because its own transcript stays short. That is a hypothesis this run tests, not a result.

### 10.1 The split of the ~$218

Rishi's changes on 2026-10-12: no E7 probe, and the reserve goes from $25 to $10.

| Item | Before | After | Change |
|---|---|---|---|
| Second seed (step 10) | ~$12 | ~$12 | none |
| Baselines (step 11) | ~$5 | ~$9 | adds the 16 edit tasks to the selection set, run twice, which also measures edit cost and the base pass rate |
| Production ceiling (step 12) | ~$150 | ~$157 | |
| Selection-set evals during production | ~$25 | ~$30 | each eval now ~$3.50 instead of ~$2.50 |
| Reserve (ask Rishi first) | ~$25 | ~$10 | |
| **Total** | ~$217 | ~$218 | |

### 10.2 What production buys

| | Before | After (40% edit) |
|---|---|---|
| Cost per step | ~$0.50-1.90 | ~$2.10 (range ~$1.70-3.20) |
| Steps for the ceiling | ~80-180, ~120 middle | ~75 (range ~50-90) |
| Edit episodes | 0 | ~960 |
| Distinct real edit tasks seen | 0 | ~150 |
| Read episodes | ~3,800 | ~1,440 |

**What this run can and can't show.** ~960 edit episodes can show whether RL improves the root at editing the RLM way on held-out tasks, and whether it beats the base model in the same harness. It is about 1% of DeepSWE's ~100K episodes, so it can't repeat DeepSWE's SWE-bench gain (§2). That would need far more inference credit (decision E-d).

**Why not train on DeepSWE v1.1-sized tasks.** Datacurve reports Qwen3.8-Max taking 111 steps and 95K output tokens per task. At W&B's Qwen3.8-27B prices the output alone is ~$0.29, and re-reading a context that grows over 111 steps adds ~$1-2 more, so ~$1.30-2.30 an episode *(inferred)*. A 32-episode step would cost ~$40-75, so the production ceiling would buy 2-4 steps. The top training level is sized like SWE-bench Pro instead (~5 files, ~120 lines), and DeepSWE v1.1 is a held-out eval (§12.4).

---

## 11. The new mix

| Family | Before | Now *(decision E-a)* |
|---|---|---|
| `aggregation` | 45% | 25% |
| `callgraph` | 20% | 10% |
| `needle` | 20% | 10% |
| `repo_loc` | 15% | 15% |
| `edit_synth` | | 15% |
| `edit_real` | | 25% |

- **40% editing** because Rishi's work is mostly in codebases. That's 3.2 edit tasks a step at 8 tasks, which `allocate_step` already spreads evenly across steps (every family stays within one task of its share).
- **`repo_loc` keeps its 15%.** It is the read skill closest to editing: finding where the change goes.
- **Aggregation drops the most.** It is the RLM paper's home ground and stays the biggest read family, but it is the furthest from Rishi's use.
- **No separate synthetic warm-up phase** *(mine)*. The controller already starts every family at level 1, and `edit_real` level 1 is single-file fixes of a few lines. A warm-up phase would mean a step-dependent mix, more code, and fewer steps on real tasks. If step 11 shows the base model below ~15% on real level 1, the fallback is to start `edit_real` at a lower share for the first 20 steps.
- **Real-task levels** go by the gold patch: one file and up to ~10 lines (SWE-bench Verified's typical size); one file up to ~40 lines; two files; three or more files; multi-file features of ~100+ lines (SWE-bench Pro's typical size). The controller moves them like any other family.
- **Tasks that keep coming back all-0 or all-1** are retired from the pool, so real tasks don't waste ~$0.60 a group on no signal *(mine)*. That uses training rollouts only, so it costs nothing extra.

---

## 12. Evals

### 12.1 Selection set, `selection-v2` (picks the checkpoint)

64 tasks, as now: 48 read tasks in the new mix's proportions, plus 16 edit tasks (6 `edit_synth` held-out seeds and 10 `edit_real` tasks from repositories training never sees, across languages). About $3.50 an eval instead of $2.50. It keeps the ticket tasks as the held-out generator family.

### 12.2 Held-out test, `heldout-test-v2` (touched once, step 13)

Today's 170 tasks plus 60 edit tasks: 20 `edit_synth_heldout` and 40 `edit_real` from held-out repositories, across languages, at least 15 of them feature tasks.

### 12.3 The comparisons that answer Rishi's question

| Run | Shows |
|---|---|
| Base Qwen3.8-27B as an RLM root, our harness | The step-0 number (step 11) |
| **Base Qwen3.8-27B in a normal coding agent** (bash and file edits, no sub-calls), same tasks | Whether the RLM way helps at all, since a plain agent that greps and reads files may do as well |
| Trained LoRA as an RLM root | The training gain |
| Trained LoRA in the normal coding agent | Whether RLM training helps or hurts plain agent use |

The normal-agent arms run in step 13, off W&B, by default *(decision E-e)*. I'd use mini-SWE-agent for them *(mine)*: ~100 lines, bash only, MIT, built for RL and evals, and the scaffold Datacurve runs every model through on DeepSWE.

### 12.4 Public benchmarks (step 13, off W&B)

These are the benchmarks labs now show for new coding models. All run in step 13, off W&B, with the base model and the LoRA in the same harness and limits. Qwen's own numbers for Qwen3.8-27B, from its model card, used Qwen's harnesses with thinking on by default, so they aren't comparable to ours; they only show where the ceiling is.

| Benchmark | What it is | Qwen3.8-27B card | Use here |
|---|---|---|---|
| **SWE-bench Pro** (public split, Scale) | 731 real-repository tasks in Python, Go, JS and TS; fixes average ~107 lines over ~4 files; graded on the patch | 61.7 | **Primary real-repository eval** *(mine)*: a stratified 100-200 sample across its four languages. Closest to what `edit_real` trains, less contaminated than Verified, and what OpenAI and Qwen now report |
| **DeepSWE v1.1** (Datacurve) | 113 original long-horizon tasks, 34 of them Python, ~668 lines over ~7 files; mini-SWE-agent; on Harbor | 42.2 | **Long-horizon check**, the closest thing to Rishi's feature work. Run with bigger limits for both models (e.g. 60 turns), since training caps edits at 20 turns. Eval only: its tasks carry a do-not-train canary |
| **Terminal-Bench 2.1** | 89 terminal tasks across software, science, ops and more; on Harbor | 73.0 | Optional transfer check: does RLM editing training help or hurt general terminal work |
| **Terminal-Bench 4.0** (and Artificial Analysis's run of it, "-AA") | 66 tasks with an 8-hour limit; AA runs mini-SWE-agent with 500 steps | ~5.6 on AA *(unverified mirror)* | Not used: a 27B model sits near the floor, so it can't show a difference, and a full run costs a lot of tokens |
| **SWE-Marathon** (Abundant AI) | 20 project-scale builds (port Kubernetes to Rust, clone Slack), 2-10 hours each, ~27M tokens per rollout, mostly not Python | none | Not used: far beyond a 27B model and our compute |
| **ProgramBench** (Meta and others) | 200 programs to rebuild from their binary and docs; 1,000 steps, 6 hours, 20 CPUs and 60 GB per container | none | Not used: the best frontier entry resolves 4.5% |
| SWE-bench Verified | 500 Python tasks | not reported | Not used. OpenAI stopped reporting it in February 2026 (flawed tests, contamination), and Qwen dropped it from the 3.8 card |

- **How we run them.** Harbor (Apache-2.0) runs SWE-bench Pro, DeepSWE v1.1 and Terminal-Bench, and accepts a custom agent whose loop runs outside the container and acts through it. Our RLM root plugs in that way, which also makes Harbor the cheapest route for the Part 2 harness's own evals *(mine)*.
- **Overlap check.** E4 drops any training repository that SWE-bench Pro or DeepSWE v1.1 uses.
- **The normal-agent arm** (§12.3) runs on the same benchmarks through mini-SWE-agent, which is how Datacurve and Artificial Analysis run every model.

### 12.5 The capability check

Unchanged: IFEval, MMLU-Pro, CRUXEval and chat prompts catch frying. CRUXEval is the short-code guard.

---

## 13. Build steps

All of them run on Rishi's PC through Remote Control threads, like steps 3-9, except E1 and the docs. None spends W&B credit; the E7 probe was dropped (Rishi, 2026-10-12) and step 11 measures what it would have.

| # | Step | Builds | W&B | Done when |
|---|---|---|---|---|
| E1 | Spec v2 | Prompt v2, edit limits, sandbox, grader, reward, detectors, metrics, evals and budget in `production-spec.md` §12 | $0 | Done 2026-10-12 |
| E2 | Sandbox edit mode | Writable `/work/repo` bind mount, process rule, limits, no-history checkout, derived images, standalone Python, the grader with canaries and per-language report parsers, PC memory test | $0 | Every step 3 test still passes; new tests show hidden tests unreachable, history absent, canaries working, and grader failures as infra errors |
| E3 | `edit_synth` | Three task kinds, levels 1-5, hidden property tests, held-out variant | $0 | The gold change scores 1 and the empty, stub and delete patches score 0 on every kind and level; a fake-model dry run trains on it |
| E4 | `edit_real` data | Repository choice across languages, image pulls to D:, validation (§5.4), feature-removal generator, issue texts, splits, manifest | $0 (disk on D:) | ~300+ validated training tasks, every licence permissive, several languages, manifest pinned |
| E5 | Reward, detectors, evals, dashboard | §6, §7.2, `selection-v2`, `heldout-test-v2`, the dashboard's Editing panel, the overview page | $0 | Unit tests per detector; the dry run fills every new panel |
| E6 | Red-team brief | `documentation/edit-red-team-brief.md` (§7.4), written last; Rishi runs it in a separate session | $0 | Brief delivered; after Rishi's run, no working exploit left |

E3 and E4 can run side by side once E2 is done. The REPL tools (§5.6) are written and tested in the cloud; the PC session loads them into the edit REPL and adds their metrics after E4.

---

## 14. How it shifts steps 10-13

| Step | Before | After |
|---|---|---|
| 10. Second seed (~$12) | Next | **Unchanged, and it can run now** (decision E-g). It replicates the 20-step run on the ticket tasks, so nothing in this plan blocks it. Its stability curves still set production's learning rate. |
| New E1-E6 | | Alongside step 10. $0. |
| 11. Baselines (~$5 → ~$9) | Base as RLM and direct reading, on `selection-v1` | The same on `selection-v2`, with the 16 edit tasks run twice, so edit tasks have a step-0 number and a measured cost per episode |
| 12. Production (~$150 → ~$157) | ~120 steps, three domains | ~75 steps on the new mix. Stop rules unchanged; the saturation rule counts the edit families too |
| 13. Final eval (off W&B) | Held-out test, public benchmarks, base rerun | Adds the edit tiers, SWE-bench Pro, DeepSWE v1.1 and the normal-agent comparison (§12.3-12.4), run through Harbor. It needs Docker on the eval host for the repository images. The Part 2 harness needs an edit mode (a working directory and `subprocess`) for its own eval to match training |

---

## 15. Decisions (taken 2026-10-12)

Rishi took the recommended option on every one. The alternatives stay here for the record, with their numbers updated for the dropped E7 probe and the $10 reserve.

**E-a. Edit share of the mix.**
- **40%: recommended (mine), chosen.** `edit_real` 25%, `edit_synth` 15%; reads 60% (§11). ~75 production steps, ~960 edit episodes.
- **30%,** the first sketch in the project chat: `edit_real` 20%, `edit_synth` 10%. ~84 steps, ~810 edit episodes.
- **50%:** `edit_real` 30%, `edit_synth` 20%. ~67 steps, ~1,070 edit episodes, and the read families get thin.

**E-b. Where real tasks come from.**
- **SWE-smith's per-repository images, permissive repositories only: recommended (mine), chosen.** Per-repository images fit the PC's D: drive. Rishi added: several languages, not only Python, and R2E-Gym a little if needed (§4.1).
- **R2E-Gym-Subset, DeepSWE's own data (Apache-2.0).** Real commits rather than injected bugs, but one image per task at ~300-500 MB each: ~300 tasks is ~90-150 GB and ~300 pulls. Each of its 10 repositories still has to pass 7b.

**E-c. Reward on edit tasks.**
- **Binary, 1 only when every hidden test passes: recommended, chosen.** It is what DeepSWE, Nebius and Kimi-Dev train on.
- **Partial credit,** 0.5 × the share of fail-to-pass tests passing when nothing else broke. Fewer all-zero groups, but it pays for half-fixes.

**E-d. Inference budget.**
- **Fit inside the ~$218: recommended, chosen.** ~75 steps, ~960 edit episodes (§10.2).
- **Add inference credit for a bigger edit run.** Every extra $100 buys about 48 more steps at the proposed mix. I won't plan around this unless you choose it.

**E-e. The normal-coding-agent comparison.**
- **Off W&B in step 13: recommended (mine), chosen.** $0 from the pot.
- **Also a small slice on W&B in step 11** (base model, 16 edit tasks, ~$4), to see early whether the RLM way already helps before production.

**E-f. Feature tasks on real repositories.**
- **Generate them by removing covered features (§4.1): recommended (mine), chosen.** They match the work Rishi does most.
- **SWE-smith's bug instances only.** Less build work, but mostly bug fixes.

**E-g. When to run the second seed.**
- **Now, while E1-E6 are built: recommended (mine), chosen.** It doesn't touch edit tasks, and it warns us early about plumbing.
- **After E6,** so every paid run uses prompt v2.

---

## 16. Risks and limits

- **The cost numbers are estimates.** Step 11 measures them on 32 base-model edit episodes before production starts. If edit episodes cost much more than estimated, production drops toward ~50 steps, or the edit share drops.
- **Base pass rate on real tasks is unknown.** If it's very high, most real groups are all-1 and teach nothing; if very low, all-0. The levels, task retirement and step 11's base pass rates handle both ends.
- **Small sample.** ~130 distinct real tasks, from ~7 repositories, is a narrow training set. The held-out repositories and the normal-agent comparison are what keep the claims honest.
- **Thinking is off.** Qwen's published coding scores use thinking, which our run pins off (decision 11b is still open). Editing may gain more from thinking than reading does, so 11b's ~$2 medium-thinking base arm should include edit tasks *(mine)*.
- **Toolchains in other languages.** Rust and Java builds are slow after an edit, and Java test plugins sometimes want the network. The 300 s cell, the offline smoke test and dropping slow repositories handle it; E4 measures test times per language.
- **Uneven languages.** The base model is likely weaker in Rust and Java than in Python. The dashboard's per-language table shows whether a language stays near 0, and task retirement keeps all-0 tasks from wasting groups.
- **PC limits.** Memory and disk decide how many edit containers run at once; E2 measures.
- **Contamination.** SWE-smith's and SWE-rebench's repositories are popular, so the base model has seen their code. The bugs and removed features are new, but the codebase isn't. The held-out repositories and recent public tasks (§12.4) limit how much this inflates scores.

**Not in this plan:** C++ and other languages beyond the five groups, tasks needing services or databases, tools beyond the REPL (MCP and the rest of Rishi's harness belong to Part 2), and `rlm()` recursion past depth 1.

---

## Sources

**Training recipes**
- DeepSWE (Agentica): [blog](https://www.together.ai/blog/deepswe), [model card](https://huggingface.co/agentica-org/DeepSWE-Preview), [training script](https://github.com/agentica-project/rllm/blob/main/examples/swe/train_deepswe_32b.sh), [rLLM](https://github.com/rllm-org/rllm)
- R2E-Gym: [paper](https://arxiv.org/abs/2504.07164), [code](https://github.com/R2E-Gym/R2E-Gym), [R2E-Gym-Subset](https://hf.co/datasets/R2E-Gym/R2E-Gym-Subset)
- SWE-Gym: [paper](https://arxiv.org/abs/2412.21139), [dataset](https://hf.co/datasets/SWE-Gym/SWE-Gym)
- SWE-smith: [paper](https://arxiv.org/abs/2504.21798), [code](https://github.com/SWE-bench/SWE-smith), [SWE-smith-py](https://hf.co/datasets/SWE-bench/SWE-smith-py), per-language datasets [go](https://hf.co/datasets/SWE-bench/SWE-smith-go), [js](https://hf.co/datasets/SWE-bench/SWE-smith-js), [ts](https://hf.co/datasets/SWE-bench/SWE-smith-ts), [rs](https://hf.co/datasets/SWE-bench/SWE-smith-rs), [java](https://hf.co/datasets/SWE-bench/SWE-smith-java)
- SWE-bench Multilingual (eval overlap list): [dataset](https://hf.co/datasets/SWE-bench/SWE-bench_Multilingual)
- SWE-rebench: [dataset](https://hf.co/datasets/nebius/SWE-rebench), [paper](https://arxiv.org/abs/2505.20411); V2: [dataset](https://hf.co/datasets/nebius/SWE-rebench-V2), [paper](https://arxiv.org/abs/2602.23866), [code and log parsers](https://github.com/SWE-rebench/SWE-rebench-V2)
- Nebius RL on SWE tasks: [paper](https://arxiv.org/html/2508.03501v2)
- SkyRL: [SkyRL-v0](https://sky.cs.berkeley.edu/project/skyrl/), [SA-SWE-32B](https://arxiv.org/html/2511.16108v1)
- Qwen3-Coder-Next: [report](https://arxiv.org/html/2603.00729v1)
- Kimi-Dev: [code](https://github.com/MoonshotAI/Kimi-Dev)
- SWE-RL (Meta): [paper](https://arxiv.org/html/2502.18449v2)
- mini-SWE-agent: [code](https://github.com/SWE-agent/mini-swe-agent)

**Harness tools (§5.6)**
- Claude Code: [tools reference](https://code.claude.com/docs/en/tools-reference)
- Codex CLI: [built-in tool surface](https://codex.danielvaughan.com/2026/06/03/codex-cli-built-in-tool-surface-complete-reference-shell-file-search-image-multi-agent/)
- DeepSeek Harness: [overview](https://www.digitalapplied.com/blog/deepseek-harness-open-source-agent-framework-2026), [runtime write-up](https://medium.com/@vignarajj/deepseek-harness-dsh-the-open-source-agent-runtime-that-finally-lets-you-own-the-stack-bcad91e05d74)
- Qwen Code: [tools](https://qwenlm.github.io/qwen-code-docs/en/developers/tools/introduction/)
- What an agent harness is: [freeCodeCamp](https://www.freecodecamp.org/news/what-is-an-agent-harness/)

**Reward hacking**
- SWE-bench git history leak: [issue #465](https://github.com/SWE-bench/SWE-bench/issues/465)
- Qwen, "Verification Horizon": [paper](https://arxiv.org/html/2606.26300v2)
- ImpossibleBench: [paper](https://arxiv.org/html/2510.20270v1)

**Benchmarks**
- DeepSWE v1.1 (Datacurve): [v1.1 notes](https://deepswe.datacurve.ai/blog/deepswe-v1-1), [v1](https://deepswe.datacurve.ai/blog/deepswe), [Harbor dataset](https://hub.harborframework.com/datasets/datacurve/deep-swe-1-1/latest)
- FeatBench: [paper](https://arxiv.org/html/2509.22237v2); NoCode-bench: [paper](https://arxiv.org/html/2507.18130v3)
- SWE-bench Pro: [leaderboard](https://labs.scale.com/leaderboard/swe_bench_pro_public); OpenAI on dropping Verified: [post](https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/)
- Terminal-Bench: [site](https://www.tbench.ai/), [4.0 notes](https://www.tbench.ai/news/terminal-bench-4-0), [Artificial Analysis run](https://artificialanalysis.ai/evaluations/terminalbench-4-0); Harbor: [code](https://github.com/harbor-framework/harbor), [custom agents](https://docs.harborframework.com/agents/custom-agents)
- SWE-Marathon: [paper](https://arxiv.org/abs/2606.07682), [code](https://github.com/abundant-ai/swe-marathon)
- ProgramBench: [paper](https://arxiv.org/abs/2605.03546), [leaderboard](https://programbench.com)
- Qwen3.8-27B model card: [Hugging Face](https://huggingface.co/Qwen/Qwen3.8-27B)

**This repository:** [production-spec.md](production-spec.md), [production-build-plan.md](production-build-plan.md), [step-3-sandbox-results.md](step-3-sandbox-results.md), [step-5-results.md](step-5-results.md), [step-7-results.md](step-7-results.md), [step-8-results.md](step-8-results.md), [rl-run-research.md](rl-run-research.md) (W&B prices, §6), `rlm_rl/tasks/__init__.py` (`allocate_step`), `rlm_rl/sandbox.py`, `rlm_rl/bridge.py`.
