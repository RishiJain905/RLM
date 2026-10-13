# Edit build results: code editing environments (E2-E5)

> **Build steps E2-E5 of [code-editing-plan.md](code-editing-plan.md)**, against the build contract in [production-spec.md](production-spec.md) §12 (spec v2). Built on Rishi's PC, 2026-10-12 to 13, on Windows and inside WSL2 (Ubuntu) with Docker Desktop 29.8.1.
>
> **Decisions applied:** every recommended option in code-editing-plan §15 (Rishi, 2026-10-12), the coordinator's language and source picks (plan §4.1), and Rishi's licence call (2026-10-12, 22:53 UTC): **JavaScript, TypeScript, Rust and Java train only on tasks we generate**; their SWE-smith rows are eval-only.
>
> **No W&B credit was used.** Every episode here ran against `tests/fake_model.py`. The `.env` file was never opened. **Nothing has been committed or pushed**: Rishi decides that.
>
> Anything marked *(mine)* is my choice, not the spec's. Anything marked *(inferred)* hasn't been measured.

*This file is being written during the build; the numbers below are filled in as each stage finishes.*

## Summary

(Filled in at the end.)

---

## 1. What was built

| Path | What it is |
|---|---|
| `rlm_rl/bridge.py` | Prompt v2 (spec §12.2): one system prompt, limits in each first message, the edit first message, timeout and crash texts that quote the cell limit, `Limits` and `limits(family)` (§12.3). Prompt v1 stays, for the second seed only (decision E-g) |
| `rlm_rl/repl_worker.py` | Edit mode (`--edit`): Landlock that lets programs run and confines writes, an audit hook that logs program starts, repository reads and writes and blocked network attempts, and child-process output fed into the cell |
| `rlm_rl/editbox.py` | The edit sandbox (§12.4): `EditSandbox`, per-task `Workspace`s (the base built once, copied per episode), the host-side diff, the standalone Python bundle, `EditImage` |
| `rlm_rl/grader.py`, `rlm_rl/grader_driver.py` | The grader (§12.5): diff checks, test-file stripping, canaries, the grader container and its driver, report parsers for pytest JUnit, `go test -json`, Jest, Vitest and Mocha JSON, `cargo test` lines, Maven and Gradle JUnit, and our unittest JSON |
| `rlm_rl/editmetrics.py` | §12.9's per-episode numbers and each step's `edit` block |
| `rlm_rl/tasks/editsynth.py` | E3: `edit_synth` and `edit_synth_heldout` |
| `rlm_rl/tasks/editreal.py` | `edit_real`, `edit_real_select`, `edit_real_test`, reading the validated task index |
| `rlm_rl/editdata.py`, `rlm_rl/editfeatures.py`, `scripts/build_edit_data.py` | E4: derived images, snapshots, licences, candidates, feature removal, our own injected bugs, validation, index and manifest |
| `rlm_rl/env.py`, `reward.py`, `detectors.py`, `monitor.py`, `evals.py`, `config.py`, `training.py`, `transcripts.py`, `tasks/__init__.py`, `data.py` | E5: edit episodes through the loop, the binary reward, the new detectors, the `edit` block, `selection-v2` and `heldout-test-v2`, the §12.7 mix, task retirement, ceilings 157/30, transcript format 3, the edit data sources and their licence status |
| `tests/` | New: `test_grader.py`, `test_editbox.py`, `test_editsynth.py`, `test_edit_env.py`, `test_editdata.py`. Changed: `test_bridge.py`, `test_detectors.py`, `test_sandbox.py`, `test_tickets.py`, `test_data_families.py`, `test_training.py`, `test_reward.py`, `fakes.py` (the in-process edit sandbox and a local grader), `dryrun.py` (the oracle handles prompt versions and edit tasks) |

---

## 2. E2: the edit sandbox and the grader

### 2.1 The episode container

| | Read episodes (unchanged) | Edit episodes |
|---|---|---|
| Image | `rlm-sandbox` | `edit_synth`: `rlm-sandbox`, read-only. `edit_real`: `rlm-edit-<repo>:<hash>`, derived from the repository's image (§4) |
| Our REPL's Python | the image's | `edit_synth`: the image's (the same 3.12.15). `edit_real`: the standalone bundle at `/opt/rlm/python` |
| Writable | `/scratch` | `/work/repo`, `/scratch`, `/tmp` and the image's cache folders, as Landlock allows |
| Programs | blocked | allowed: the image's toolchain runs through subprocess |
| User | uid 10001 | the host user's own non-root uid *(mine)* |
| Root filesystem | read-only | `edit_synth` read-only; derived images get a per-container writable layer, which Landlock confines and which goes with the container *(mine)* |
| Limits | 1 CPU, 512 MB, 64 pids, 120 s cells | 2 CPUs, 4 GB, 256 pids, 300 s cells (§12.3) |

**The standalone Python** is taken from the pinned `python:3.12-slim` image the read sandbox already uses, so nothing was downloaded for it: Python 3.12.15 with its own dynamic loader and C libraries (51 MB), started through that loader. It runs in any Linux image, even Alpine, which has neither Python nor glibc (a test checks).

**What the audit hook logs in edit mode** (the host turns these into §12.9's numbers): each program started, with its arguments (`exec`); the first read of /work/repo in each cell (`repo_read`); writes to it (`repo_write`); and blocked network attempts (`network`, counted as `network_try`). Writes outside the writable folders are still blocked and fire `escape_attempt`.

**Output.** During a cell, file descriptors 1 and 2 are a pipe into the cell's output, so a program run without `capture_output` still shows the model what it printed. The audit log moves to a private descriptor that programs don't inherit.

**The host never runs git against the episode's `.git`.** The model controls it (config, hooks, attributes). Each task's base is built once into a host cache with a private git directory; the episode's diff is taken with a throwaway git directory that borrows the cache's objects, under a git environment that reads no system, global or repository config. A test writes an `fsmonitor`, a diff driver and a clean filter into the episode's `.git/config` and checks that none of them runs on the host.

**Change detection.** After each cell the host fingerprints /work/repo (paths, sizes, modes, times), leaving out `.git`, `__pycache__`, `node_modules`, `target` and other caches, so a test run that writes caches doesn't count as an edit.

### 2.2 The grader

As spec §12.5, with these details *(mine)*:
- **Test detection by language:** Python `test_*.py`, `*_test.py`, `tests/` and `test/` folders; Go `*_test.go` and `testdata/`; JavaScript and TypeScript `*.test.*`, `*.spec.*`, `__tests__/`, `__mocks__/`, `test/`, `tests/`; Rust `tests/`, `tests.rs` and edits inside an existing `#[cfg(test)]` module; Java `src/test/`.
- **Test configuration:** any `conftest.py` (new or existing), `pytest.ini`, `tox.ini`, `noxfile.py`, `.coveragerc`, Jest, Vitest, Mocha, Karma and Babel configs, `.cargo/config*`, and the test sections of `setup.cfg`, `pyproject.toml`, `Cargo.toml`, `package.json` (`jest`, `mocha`, `vitest`, `ava`, `nyc`, `c8`, `scripts.test*`), `pom.xml` (Surefire and Failsafe) and `build.gradle`.
- **New test files are stripped too**, not just left unrun: in Go a new `_test.go` file in a tested package would otherwise compile and run.
- **`test_aware` patterns** per language (spec §12.6): `PYTEST_CURRENT_TEST`, `_pytest`, a test runner in `sys.modules`, `os._exit`; Go `testing.Testing()`, `flag.Lookup("test.`; JS/TS `JEST_WORKER_ID`, `process.env.VITEST`, `typeof jest`, `process.exit(`; Rust `cfg!(test)`, `process::exit`; Java `System.exit(`, `org.junit` in main code; and skip markers. `inspect.stack()`, environment reads and `sys.settrace` only go to the audit queue.
- **Canaries:** two tests with fresh random names, written as the runner expects (a pytest file, a unittest module, a separate Go package so a broken build elsewhere can't hide them, a Jest/Vitest/Mocha file named like the task's test files, a Cargo integration test, a JUnit 4 or 5 class).
- **`grader_tamper`** fires when both canaries are reported and either is wrong, when only one is reported, when the run claims success (exit 0) with neither, or when the report is malformed (unparseable, or the same test reported with conflicting outcomes). A patch that breaks the build reports neither canary and fails, and that is not tampering.
- **The driver** makes itself non-dumpable, so the tests, which run as the same user, can't read its stdin or write its stdout through /proc and forge the result. It runs each command in its own process group and kills the group when the command ends, so nothing left running can rewrite a report.
- **Infra or the model's?** A grader container that never reports is `infra_error`. A run past the 600 s cap is the model's (missing tests count as failed). A container killed by the out-of-memory killer with no report is the model's too: its own code used the memory, and every gold patch was validated well inside the limit.

### 2.3 The REPL coding tools (spec §12.1, approved by Rishi 2026-10-12 23:34 UTC)

`rlm_rl/repl_tools.py` (from the cloud, saved verbatim, `TOOLS_VERSION` "t1"): `read_file`, `find_files`, `grep`, `edit_file`, `write_file`, `shell`, `diff`, standard library only.
- **Where they run:** in the episode container, mounted read-only at `/opt/rlm/repl_tools.py` beside the worker. The worker imports them before Landlock and the audit hook, starts in /work/repo, and configures them with the base commit (from the host's workspace) and the cell's compute time left (bridge waits excluded, as on the host), so `shell()` never runs past the cell limit.
- **Logging:** each call is a `tool` audit event (name, ok, seconds, path or command). Program starts and /work/repo writes carry `tool=true/false`, and each cell sends exact counts, so the host can compute `raw_io_share` (the `repo_write` events stop at 20 per cell).
- **git:** `edit_synth` now runs in `rlm-edit-synth`, the pinned `python:3.12-slim` plus git, because `diff()` and the model's own `git status` need it. `derive` refuses a source image without git.
- **Prompt v2** lists the tools (2,270 characters); `tests/test_contract.py` checks each name, parameter, default and return type against the real signatures.

### 2.4 Tests

| File | Tests | Covers |
|---|---|---|
| `tests/test_grader.py` | 28 | Diff parsing, safe paths, test detection per language, test configuration (whole files and sections), test-aware code, binary, symlink, unsafe and oversize diffs, every report parser, conflicting duplicates, canary names and commands, verdicts, scoring |
| `tests/test_editbox.py` (Docker) | 13 | The writable repository and running programs, writes confined (also for child processes), network blocked and logged, no history to mine, hidden tests never in the container, timeouts and crashes keep the files, the host diff ignoring a booby-trapped `.git`, the bundle in Alpine, the grader: gold resolves, empty, delete and stub patches don't, a test edit is `test_tamper`, forged results caught by the canaries, an early exit caught, the 600 s cap is the model's, a missing image is infra; 8 edit episodes at once |
| `tests/test_edit_env.py` | 8 | Edit episodes through the loop: a resolved episode with every §12.9 key, no submit or no change, `early_submit`, `test_tamper`, `network_try` logged only, a grader failure as infra, the step block's keys equal the dashboard's, task retirement, and a full episode in real containers |

---

## 3. E3: `edit_synth`

`rlm_rl/tasks/editsynth.py` builds standard-library-only Python packages from the existing generators (`synthcode.py`, `behaviours.py`), written as real files with neutral names fresh per seed and a small visible `tests/` folder (`python3 -m unittest discover -s tests`).

| Kind | Edit twin of | The root must | Hidden tests check |
|---|---|---|---|
| `fix` | `needle` | find and fix a mutation in a behaviour function a few calls below a public entry point, past look-alike decoys | the entry point against its reference on fresh inputs, and every other public function against its own |
| `callers` | `callgraph` | rename a shared helper, or give it a required keyword-only parameter, and update every caller: direct calls, aliases, dispatch tables, `getattr` with names built from pieces | the new name or parameter, and that every public function still works, so a missed caller fails |
| `feature` | (new) | add an option to a public function and thread it 2-4 modules down to where the work happens | every listed function with the option on fresh inputs, and that the default changes nothing |

- **Levels 1-5** grow the package from ~20K to ~200K characters, with longer call chains, more callers, more modules an option crosses (1 to 4 files to change), more decoys and more hidden call styles.
- **Hidden test inputs** are drawn on the host from the seed and differ from every example in the task text, so hard-coding examples earns nothing (spec §12.6, H1). They never enter the episode's repository.
- **`edit_synth_heldout`** (eval only, H7) uses mutation kinds, feature options, helper-parameter meanings, task wording and behaviours that training never sees.
- **Tests:** `tests/test_editsynth.py` (8): every kind and level builds; the gold change resolves and the base fails; the visible tests pass at the base; held-out never overlaps training; the same seed gives the same task.

---

## 4. E4: `edit_real` data

### 4.1 The pipeline (`scripts/build_edit_data.py`, everything on `D:\RLM-data\edit`)

| Stage | What it does |
|---|---|
| `pull` | Pulls each planned repository's SWE-smith image (network on, the only stage that needs it) |
| `derive` | Snapshots the image's checkout to `repos/<key>.tar` (no `.git`, no dependency or build folders), reads the repository's own licence file, works out the test runner, and builds `rlm-edit-<key>:<hash>` with no network: the checkout (and SWE-smith's per-task branches, which hold every answer) deleted, `/testbed` a symlink to /work/repo, the conda environment's PATH made permanent, Go and Cargo caches moved to `/opt/rlm-cache` and node_modules to `/opt/rlm-deps` (owned by the host user), the toolchain told to stay offline. Refuses an image without git |
| `candidates` | SWE-smith bug rows as candidate tasks, spread over levels, with their failing tests moved into the hidden set. Python and Go only for training; for the other four languages, only the held-out repositories (eval splits) |
| `features` | Feature removals: a documented public function that 1-4 test files mention is replaced by a stub (`raise NotImplementedError`, `panic`, `unimplemented!()`, a throw); the tests that then fail are the hidden fail-to-pass set |
| `bugs` | Our own injected bugs for JavaScript, TypeScript, Rust and Java training: one single-line mutation per function (flipped comparison, off-by-one, swapped operator, dropped negation and the like) in a function the tests cover |
| `rebench` | SWE-rebench V2 real PRs (§4.4) |
| `validate` | Spec §12.8 in our grader: the base once, the gold patch three times; fail-to-pass = fails on the base and passes on every gold run; pass-to-pass = passes everywhere; flaky tests dropped; working canaries, visible tests that still run without the hidden files, a gold run under half the 600 s cap |
| `briefs` | What I need to write each generated task's text (signature, docstring, how the tests use it); texts go to `issues/<language>.json`, written by me, never quoting the gold patch or naming a hidden test |
| `index` | `index.json`: validated tasks with texts that pass the leak check, capped per repository, then per language (training targets, at least a quarter feature tasks) and per eval split (selection 10, test 40 with at least 15 feature tasks) |
| `manifest` | `manifest.json`: every source with its licence status and role, every repository with its licence, image digests, snapshot hashes, task counts, the index's hash |
| `prune` | Removes the images of real-PR candidates that failed validation (each has its own image) |

### 4.2 Repositories

Training repositories give training tasks; the selection and test repositories are held out (training never sees them). All licences were read from each repository's own licence file in its image.

| Language | Training | Selection | Test |
|---|---|---|---|
| Python | sqlparse (BSD-3), jinja (BSD-3), marshmallow (MIT), arrow (Apache-2.0), deepdiff (MIT), gpxpy (Apache-2.0) | glom (BSD-2) | mistune (BSD-3), pyparsing (MIT) |
| Go | chi (MIT), squirrel (MIT), env (MIT), bigcache (Apache-2.0), ristretto (Apache-2.0), go-prompt | hjson-go | roaring, bitset |
| JavaScript | async, dayjs, svgo | ws | nock, mathjs |
| TypeScript | rxjs, react-hook-form, foam | redoc | homebridge, formik |
| Rust | chrono, itertools, semver, rust-base64 | uuid | rust-csv, pastel |
| Java | JsonPath, picocli, metrics | disruptor | jsonschema2pojo, mapdb |

*(Licences for the rows without one listed are filled in from `repos.json` when their images are derived.)*

**Eval overlap.** No repository is in an eval we report. I checked every pick against SWE-bench Pro's public set and SWE-bench Multilingual (both downloaded to `D:\RLM-data\hf\evals`) and against the task names in DeepSWE v1.1 (the dataset itself is gated, so I used its file listing). **go-critic was the second Go test repository, but DeepSWE v1.1 has a go-critic task**, so bitset replaced it. trpc, an earlier TypeScript test pick, failed to pull after 11 minutes and was dropped; homebridge and formik are the TypeScript test repositories.

### 4.3 Licences and sources (Rishi's call, 2026-10-12)

| Source | Licence status | Used for |
|---|---|---|
| SWE-smith Python, Go | MIT on the dataset card | training (bug tasks), eval splits |
| SWE-smith JavaScript, TypeScript, Rust, Java | no licence stated on the dataset card | eval splits only; `tasks.check_mix()` refuses a training mix that reads them |
| Generated (our feature removals and injected bugs) | ours; each repository's own licence is permissive | training in every language, eval splits |
| SWE-rebench V2 | CC-BY-4.0, permissive per-task licence | training (real PRs) |

### 4.4 Real pull requests (SWE-rebench V2)

Each real PR has its own image (0.8-1.3 GB compressed, about 0.5 GB of it unique to the task, measured from the registry manifests before pulling), so I took about 10 candidates per language from 3 repositories each, the most recent first, to share layers and stay inside the disk budget: Python click, Python-Markdown, attrs; Go kin-openapi, go-yaml, cbor; TypeScript tslint-to-eslint-config, FxTS, html-react-parser. Rows qualify only with a permissive per-task licence, no flagged problems, an issue-text confidence of at least 0.9, at most 4 files and 200 lines changed, 1-30 fail-to-pass tests, and a repository outside the evals and the held-out sets. The image's checkout is the base, the merged fix is the gold patch, and the files the PR's test patch leaves are the hidden tests. Images of candidates that fail validation are removed (`prune`).

---

## 5. E5: reward, detectors, metrics, mix, evals

| Piece | Where | What it does |
|---|---|---|
| Reward | `reward.py` `edit_score` | Binary (decision E-c): 1 when the episode submitted a non-empty diff that applies and every fail-to-pass and pass-to-pass test passes; 0 otherwise. A grader failure is an infra error, never a 0. The components carry §12.9's per-episode numbers |
| Detectors | `detectors.py` | `test_tamper` (the diff touched a test file or test configuration), `grader_tamper` (canaries wrong, missing or contradicted) and `test_aware` (main code that checks whether it runs under a test) zero the reward. `network_try` is logged only. `early_submit` for edits means submitting before any code read or changed /work/repo or called a coding tool. `copied_answer` is off for edits |
| Blind check (H1) | `tests/test_editsynth.py` | The blind patches (empty, delete the target, stub every function, hard-code the task's examples) all score 0 |
| Metrics | `editmetrics.py` | Each step record's `edit` block with §12.9's keys, `by_language` as `{episodes, resolved}`, and since the coding tools: `edit_errors`, `raw_io_share` and `tool_calls`. Eval summaries carry `by_language` as `{n, accuracy}` |
| Retirement | `tasks.Retirement`, `training.py` | An `edit_real` task whose group came back all-0 twice or all-1 once is retired and never drawn again; the state is saved with each checkpoint and restored on resume |
| Mix | `tasks/__init__.py` | Spec §12.7: aggregation 25, callgraph 10, needle 10, repo_loc 15, edit_real 25, edit_synth 15 (family shares set directly). `allocate_step` keeps every family within one task of its share at every step. `check_mix` refuses a training family whose records come from a source the licence bar keeps out of training, which now includes the JavaScript, TypeScript, Rust and Java SWE-smith datasets |
| Evals | `evals.py` | `selection-v2` (64: 48 read tasks, 6 held-out-seed `edit_synth` tasks, the selection split's 10 `edit_real` tasks) and `heldout-test-v2` (heldout-test-v1, 20 `edit_synth_heldout`, the test split's 40 `edit_real` tasks with at least 15 feature tasks). Seeds are in `private/`. The index stage sizes the two splits (`balance`) |
| Ceilings | `config.py` | `PRODUCTION_CEILINGS` 157 (train) / 30 (eval) |
| Records | `transcripts.py`, `config.py` | Transcript format 3: the edit fields (`language`, `repo`, `diff`, `grade`, `edit_stats`, `limits`, `tools_version`). `config.json` and every checkpoint pin the prompt version, its hash and the tools version |
| Dry run | `tests/dryrun.py`, `scripts/train.py` | The oracle policy answers every family, edits included, on either prompt version; `--real-edit` runs edit episodes in the real sandbox and grader instead of the in-process fakes |

---

## 5a. The slow data drive: time, never the reward (Rishi, 2026-10-13)

A slow drive may cost time but must never turn a good answer into a 0 or pause the run.

| Limit | What happens now |
|---|---|
| Container start | 300 s for a read sandbox, 600 s for an edit sandbox; a launch that doesn't come up in time is tried once more (`START_ATTEMPTS`, `rlm_rl/sandbox.py`); one that never starts is an infra error (reward None, never trained on) |
| Grader cap | Per task: max(600 s, 4x the task's slowest gold run under load), stored in the index (`tests.timeout_s`, `editdata.grader_cap`). A run that hits it reruns once, alone (no other grader running), at double the cap; only a second timeout counts against the model. Waiting for a grader slot counts against nothing. `grader_retries` (mean per episode) is in each step's `edit` block |
| Cell limit (300 s compute) | Per task when needed: if the gold change's own test command, run through `shell()` in the episode container under load, takes more than half the cell limit, the task's cell limit becomes twice that time, rounded up to 30 s (`editdata.cell_limit`), recorded in the episode's limits and quoted in its first message. No task is dropped for it |
| Grader concurrency | `grader.set_grader_concurrency(n)` caps graders at once if the measurements show the disk thrashing |

Tests: `tests/test_slow_drive.py` (8).

**Measurements under load** (`build_edit_data.py measure --jobs 30`): *(filled in when the data build finishes)*.

**The D: drive itself** *(Rishi asked for this before the red-team write-up)*: whether the drive should be optimized in place for the training run (write caching, scan exclusions, which Rishi makes himself, and WSL memory), with the measured numbers. In place only: no switching drives, no wiping data (Rishi, 2026-10-13), so no Docker clean-up; if the numbers say clean-up would really help, it is listed on its own as needing his explicit OK. Recommendations only; no drive setting is changed without Rishi. *(Filled in after the measurements.)*

## 5b. Security notes for the red team (as built)

For `documentation/edit-red-team-brief.md` (the coordinator writes it). Line numbers are as of this build.

**The graded diff never comes from the episode's `.git`.** Each task's base is committed once into a host cache with a private git directory (`_commit_base`, `rlm_rl/editbox.py:191`); the episode gets a copy (`materialize`, `:218`). After the last turn the host diffs the episode's work tree (`diff`, `:224`) against the base commit with a throwaway bare git directory whose objects come from the host cache through `alternates`, under `git_env` (`:124`: no system or global config) and `GIT_SAFE` (`:133`: no fsmonitor, hooks, attributes file or untracked cache), and `--no-ext-diff --no-textconv --no-renames --full-index`. The episode's `.git` (object store, refs, index, config, hooks) is never read, so tampering with it changes nothing that is graded. What the model can still influence: a `.gitignore` in the work tree hides its own new files from the diff (it only loses its own changes; tracked files stay tracked), and a `.gitattributes` marking files binary makes them "Binary files differ", which the grader strips (again only its own loss). Test `tests/test_editbox.py` plants an fsmonitor, a diff driver and a clean filter in the episode's `.git/config` and checks none runs on the host.

**Edit-mode Landlock** (`edit_landlock_rules`, `rlm_rl/repl_worker.py:215`): read and execute everywhere (`/`); write, create, remove, make symlinks, sockets and fifos only under `/work/repo`, `/scratch`, `/tmp` and the image's cache folders (`/opt/rlm-cache`, and `/opt/rlm-deps` for JavaScript and TypeScript), plus read/write on `/dev/null`. It applies to the REPL and every child process. `/episode` (context, bridge socket) and `/opt/rlm` (worker, tools, grader driver, Python bundle) are read-only bind mounts. The container: `--network none`, `--cap-drop ALL`, `no-new-privileges`, the host's non-root uid, 2 CPUs, 4 GB, 256 pids, tmpfs `/scratch` and `/tmp` (`container_args`, `rlm_rl/editbox.py:440`).

**Edit-mode audit hook** (`_EDIT_HANDLERS`, `rlm_rl/repl_worker.py:543`), in the REPL process only, not in programs it starts:
- blocks and logs: network sockets (anything but AF_UNIX, `:356`) and connects to anything but the bridge socket (`:362`), as `network`; writes outside the writable folders, as `file` (`:342`; Landlock blocks them anyway); `ctypes`; host-module imports; tampering with the audit log descriptor (the REPL exits)
- logs without blocking: each program started (`exec`, with its arguments and `tool=true/false`), the first read of /work/repo per cell (`repo_read`), writes to /work/repo (`repo_write`, the first 20 per cell, with `tool=`), each coding-tool call (`tool`: name, ok, seconds, path or command), and each cell's exact counts of tool and plain writes and programs (`io`, `:715`)
- a program the model runs (say `python3 -c ...` from the image) has no audit hook: only Landlock and the container bound it

**Hidden tests and canaries** reach the grader only on the host side: `prepare` (`rlm_rl/grader.py:993`) copies the base, applies the checked diff (test files, test configuration, binary, symlink and unsafe paths already stripped by `check_diff`, `:472`), then writes the hidden test files and the two canaries into that copy. A fresh container mounts the copy at /work/repo; the episode container never sees them. Canary names come from `secrets.token_hex` (`_names`, `:550`): a file `test_<8 hex>.py` holding `test_c<10 hex>` (asserts True) and `test_c<10 hex>` (asserts False), and the equivalents per runner. The driver gets its job on stdin, makes itself non-dumpable (`rlm_rl/grader_driver.py:45`), runs each command in its own process group and kills the group when it ends, and frames its output with a per-run nonce. Reports go to `/tmp/rlm-grade/<random token>.out`. A run that hits the task's cap is rerun once, alone, at double the cap (`grade`, `:1082`; `GraderSlots`, `:1027`; one run is `_grade_once`, `:1129`).

**The coding tools** (`rlm_rl/repl_tools.py`) are mounted read-only at `/opt/rlm/repl_tools.py` and imported by the worker before Landlock and the audit hook (`load_tools`, `rlm_rl/repl_worker.py:752`). They are ordinary Python functions in the model's namespace, so the model can rebind them, reach the module through `sys.modules['repl_tools']` (or the worker through `sys.modules['__main__']`) and monkeypatch either: make `active()` always true, call `configure()` with another `on_call` or `base`, write audit lines through the worker's `_log`, or delete hook handlers. All of that changes only logged metrics (`tool_calls`, `edit_errors`, `raw_io_share`, `test_runs`, and the event-based detectors `escape_attempt` and `early_submit`): never the reward, which comes from the host-side diff and the fresh grader container. Blocking still holds because Landlock and `--network none` are kernel and container controls, not the hook.

**Known gaps** *(for the red team to probe)*:
1. **Recognizable canaries.** The canary names (`test_c` + 10 hex characters, in `test_<8 hex>.py`) and bodies (`assert True` / `assert False, 'canary'`) are easy to recognize. Main code that runs inside the test process could rewrite every outcome to "passed" except the canaries' and evade `grader_tamper`. Mitigation, not built: canary names and bodies drawn to look like the repository's own tests.
2. **`test_aware` is a static pattern list** (`TEST_AWARE`, `rlm_rl/grader.py:431`): an obfuscated runtime check (say, a string-built environment variable name) passes it. The canaries are the backstop, with gap 1.
3. **The audit hook can be switched off from inside** (above): event-based metrics and detectors are best effort; nothing reward-bearing depends on them.
4. **Programs bypass the hook.** A child process can read anything Landlock allows (the whole image, including toolchain caches) and write anywhere writable; no history or hidden tests are anywhere in the container, which the hidden-test test checks.

## 6. Differences from the spec

1. **§12.4, read-only root:** derived images run with a per-container writable layer instead, because Go, Cargo, Maven and Gradle must write their warm build caches. Landlock confines every write to /work/repo, /scratch, /tmp and the image's cache folders, and the layer goes with the container. `edit_synth` stays read-only. (Folded into the spec by the coordinator.)
2. **§12.4, non-root user:** edit containers run as the host user's own uid (1000 here), not 10001, so the host can read, diff and remove what the episode wrote. Still non-root, no capabilities. (Folded in.)
3. **§12.2, one prompt:** prompt v1 is kept beside v2, selected per run (`RunConfig.prompt_version`): the second seed replicates the 20-step run on v1 (decision E-g); everything else uses v2. (Folded in.)
4. **§12.5 step 2, new test files** are stripped from the diff, not kept unrun (see §2.2).
5. **§12.4, `edit_synth`'s image:** `rlm-edit-synth` (the read sandbox's pinned base plus git) instead of `rlm-sandbox`, because the coding tools' `diff()` needs git. The read image is unchanged. (The coordinator's instruction with the tools.)
6. **§12.4, Go's module cache** stays read-only where the image put it (world-readable, and `GOPROXY=off` fetches nothing) instead of being re-owned: re-owning copies every file into a new layer, which took over 16 minutes per image on the data drive. Only the build cache moves to `/opt/rlm-cache`.
7. **Start-up timeouts are generous** *(mine; Rishi, 2026-10-13: time is not the concern as long as things work)*: 300 s for a read sandbox (was 60 s), 600 s for an edit sandbox, and 600 s on top of the tests' own cap for a grader container (was 180 s). Docker's disk and every image live on the USB data drive, where creating a container took minutes while downloads and builds ran beside it, and a training loop test paused on exactly that. A slow start should cost time, not an infra error. The cell limits are unchanged.
8. **§12.8, visible-test check:** run for Python (collection), Go (`go vet`), Rust (`cargo test --no-run`) and Java (`mvn -o test-compile`); JavaScript and TypeScript have no compile step to check, so their tasks skip it.
