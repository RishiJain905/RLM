# Edit build results: code editing environments (E2-E5)

> **Build steps E2-E5 of [code-editing-plan.md](code-editing-plan.md)**, against the build contract in [production-spec.md](production-spec.md) §12 (spec v2). Built on Rishi's PC, 2026-10-12 to 13, on Windows and inside WSL2 (Ubuntu) with Docker Desktop 29.8.1.
>
> **Decisions applied:** every recommended option in code-editing-plan §15 (Rishi, 2026-10-12), the coordinator's language and source picks (plan §4.1), and Rishi's licence call (2026-10-12, 22:53 UTC): **JavaScript, TypeScript, Rust and Java train only on tasks we generate**; their SWE-smith rows are eval-only.
>
> **No W&B credit was used.** Every episode here ran against `tests/fake_model.py`. The `.env` file was never opened. Everything is committed and pushed to `claude/project-thread-9nnh3n` (Rishi's go-ahead, 2026-10-13); §7 has this round's final state.
>
> Anything marked *(mine)* is my choice, not the spec's. Anything marked *(inferred)* hasn't been measured.

*Written during the build and completed on 2026-10-13, when the first round of edit_real data was frozen (§7).*

## Summary

Step 9.5 built the code-editing environments: the edit sandbox and grader with the REPL coding tools (E2), the synthetic `edit_synth` family (E3), the `edit_real` data pipeline over SWE-smith, our own generated tasks and SWE-rebench real PRs (E4), and the reward, detectors, metrics, mix and evals for edit episodes (E5). The first round of `edit_real` data was frozen on 2026-10-13 so the step finished that day (Rishi): Python, Go and JavaScript.

- **The index:** 157 tasks: training 107 (Python 86, Go 16 including 2 real PRs, JavaScript 5), selection 10, test 40 (8 of them feature tasks). Every task passed validation in the grader (the base fails its tests, the gold change passes them three times running, the visible tests still build without the hidden ones) and was timed under load (§7).
- **Checks:** a real-sandbox production dry run passed (evals, one trained step, a clean stop, no infra), and the full suite stands at 399 passed, 1 skipped, after fixing a container leak the Docker tests found (§7).
- **Bugs validation caught on the way:** canary results counted as task tests (§5), dropped Go tests from a too-short check timeout and from hidden helper files, Rust test modules removed from under their `mod` lines, Java and TypeScript repositories whose tests never ran (wrong runner, missing dependencies, read-only caches), SWE-smith TypeScript patches the parser couldn't read (§4.2, §6).
- **Known gaps (§7):** Go training under its 60 target; 8 feature tasks in the test split, not 15; Rust, TypeScript and Java left for a later round (their data and fixes are on D:); JsonPath parked; the training pool smaller than production's draws.
- **The drive (§5a):** D: as it is is correct for production, since a slow drive costs time and never reward; expect about 30-35 minutes per training step on average. Recommendations are in place only.

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
| `validate` | Spec §12.8 in our grader: the base once, the gold patch three times; fail-to-pass = fails on the base and passes on every gold run; pass-to-pass = passes everywhere; flaky tests dropped; working canaries, visible tests that still run without the hidden files, a gold run under half the 600 s cap. Eval candidates are validated only until the language holds its even share of the selection and test splits plus a spare (test 8 with 3 features, selection 3 with 1: `eval_need`); the rest stay unvalidated, listed in `validated/<language>.spare.json`, and `--all-evals` validates them (Rishi, 2026-10-13: skip the spare evals). Training, selection and test tasks keep every check |
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

**How each repository's tests run** (`REPO_TESTS` in `scripts/build_edit_data.py`, where the derive stage's guess from the files was wrong; `retest` applies it to existing candidates):
- *picocli*: a pom.xml for the core module only and no Maven in the image; every module builds with `./gradlew`, whose cache holds the dependencies.
- *dropwizard metrics*: `./mvnw` (no system Maven), its Maven under `MAVEN_USER_HOME=/opt/rlm-cache/m2`. A Maven module's tests run from the root with `-pl <module> -am`, so the modules it depends on build in the same reactor rather than being looked up in the local repository.
- *JsonPath*: the image's Gradle cache has none of the test dependencies, so offline the tests can't compile. The image is derived from a warmed copy (`warm_image`: the repository's own `testClasses` once, network on); episodes and graders stay offline.
- *rxjs*: **not used** (`UNUSABLE`). Its specs import the built package (`exports` point at `dist/`, which the image never built), and mapping the import to `src/` with tsconfig-paths sends mocha down the ESM loader, which can't load `.ts`. *foam* and *formik*: the derive stage found no runner it supports (formik is a monorepo, and its SWE-smith rows name their failing tests opaquely, `jest_summary_64`), so they have no tasks; TypeScript's eval tasks come from homebridge (test) and redoc (selection).
- SWE-smith's TypeScript patches have no `diff --git` lines and name their failing tests by file (`suite::<file>`): `editdata.with_git_headers` adds the lines and the file is the scope.
- *react-hook-form*: jest with its own config (`--config ./scripts/jest/jest.config.js`, which sets the SWC transform); without it Babel can't parse the TypeScript tests. *homebridge*: `--forceExit`, as its own script. *nock*: mocha (the derive stage guessed jest). *svgo*: Yarn's pnpm linker makes no `node_modules/.bin`, so jest runs as `node node_modules/jest/bin/jest.js`, with `NODE_OPTIONS=--experimental-vm-modules` for its ES modules.
- A wrapper (`./gradlew`, `./mvnw`) is named from the repository root wherever the tests run from; a task whose module has no build file for its runner is dropped before any container.

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

**Canary results leaked into validation (found and fixed 2026-10-13).** The grader kept the two canaries' results beside the real tests'. Canary names are fresh every run, so validation counted the gold run's must-pass canary as a fail-to-pass test: 12 Python, 61 Go, 8 Rust and 5 TypeScript results carried one. Where a task's real tests reported nothing, the canary was its only fail-to-pass test, and the task passed validation with nothing behind it (1 Go, 6 Rust, 5 TypeScript); in training, any task holding a stale canary name could never be solved. The grader now drops canary results once they have vouched for the run (`grader.is_canary_key`), and `decanary` removed the canary keys from the results already written (appended corrections; a task left with no fail-to-pass test fails). Python's 173 passing tasks were unaffected.

## 5a. The slow data drive: time, never the reward (Rishi, 2026-10-13)

A slow drive may cost time but must never turn a good answer into a 0 or pause the run.

| Limit | What happens now |
|---|---|
| Container start | 300 s for a read sandbox, 600 s for an edit sandbox; a launch that doesn't come up in time is tried once more (`START_ATTEMPTS`, `rlm_rl/sandbox.py`); one that never starts is an infra error (reward None, never trained on) |
| Grader cap | Per task: max(600 s, 4x the task's slowest gold run under load), stored in the index (`tests.timeout_s`, `editdata.grader_cap`). A run that hits it reruns once, alone (no other grader running), at double the cap; only a second timeout counts against the model. Waiting for a grader slot counts against nothing. `grader_retries` (mean per episode) is in each step's `edit` block |
| Cell limit (300 s compute) | Per task when needed: if the gold change's own test command, run through `shell()` in the episode container under load, takes more than half the cell limit, the task's cell limit becomes twice that time, rounded up to 30 s (`editdata.cell_limit`), recorded in the episode's limits and quoted in its first message. No task is dropped for it |
| Grader concurrency | `grader.set_grader_concurrency(n)` caps graders at once if the measurements show the disk thrashing |
| Grader containers across processes | `grader.global_slot`: at most `RLM_GRADER_SLOTS` grader containers at once across every process (lock files under /tmp/rlm-grader-slots). The data build's validation chains set 5: run side by side, each with its own cap, they had started more containers than the drive could bring up before the grader's deadlines (Rishi, 2026-10-13: keep pacing on D:). Waiting for a slot counts against nothing. Unset in training |
| Visible-test check (validation) | 1800 s, and once more at 3600 s if it runs out of time (`editdata.VISIBLE_TIMEOUT_S`); a second timeout is an infra error, which the next `validate` run tries again, never a dropped task. It was a fixed 300 s, and compiling every test target of a Go module or a Rust crate under load ran past it, so good tasks were dropped as "visible tests break". `revisible` gives the tasks an older check dropped the current check again, alone: their base and gold runs passed, so nothing else is repeated |

Tests: `tests/test_slow_drive.py` (17).

**Measurements under load** (`build_edit_data.py measure --per-repo 2 --jobs 15`, 2026-10-13 evening: 44 sample tasks, 2 per repository, 15 at once, the drive otherwise quiet):

| | Python (18) | Go (18) | JavaScript (8) |
|---|---|---|---|
| Episode container up, repository copied in (median / max) | 174 / 326 s | 147 / 272 s | 189 / 292 s |
| The gold change's test command, run in the episode (median / max) | 145 / 490 s | 865 / 1379 s | 240 / 1770 s |
| Grader container, start to report (median / max) | 258 / 477 s | 1184 / 2051 s | 422 / 677 s |
| Solo reruns (a grader run that hit its cap) | 0 | 12 | 0 |
| Tasks in the index with a raised cell limit (largest) | 61 of 105 (990 s) | 35 of 35 (2760 s) | 14 of 17 (3570 s) |
| Grader cap in the index (median / largest) | 600 / 1907 s | 2215 / 8206 s | 600 / 2709 s |

`measure`'s own start figure subtracts a copy timed separately, so a few rows read negative: container up plus copy is the honest number. One grader waited up to 1.9 hours for a slot behind Go's solo reruns, which run with the drive to themselves. No infra errors. The real-sandbox dry run (`dry-edit-final3`, fake model): 32 episodes in its step, 0 infra, 0 grader reruns; the step took 273 s (rollouts 235 s) with an instant model.

*Per production step on D: as it is* (8 tasks x 4 rollouts; the mix gives edit_real 25%, about 2 tasks or 8 episodes, edit_synth 15%, read families 60%): a step ends with its slowest episode. An edit_real episode is about container up (~3 min) + the model's turns (~5-10 min with W&B inference, *inferred*) + its own test runs (say two: Python ~5 min, JavaScript ~8, Go ~30) + grading (Python ~4, JavaScript ~7, Go ~20). So about 20-25 minutes for a step whose edit_real tasks are Python or JavaScript, and about an hour when one is Go (15% of the training pool, so about 28% of steps). **Average about 30-35 minutes per step** *(inferred from 15 tasks at once; production runs up to 32 episodes at once, though read episodes barely touch the drive)*, against a median 290 s for the 20-step read-only run. About 75 steps would take about 40 hours, plus selection evals every 10 steps (64 tasks, 10 of them edit_real: about an hour each).

**The D: drive itself** (Rishi asked for this before the red-team write-up; read-only review, 2026-10-13, no setting changed):

*Current state:*
- Seagate One Touch HDD, 4 TB, on **USB 3 with UAS** (the UASPStor driver, under an AMD USB 3.10 controller's USB 3.0 root hub): the link is not the bottleneck.
- **Write caching on the device is already enabled** (UserWriteCacheSetting = 1); "turn off Windows write-cache buffer flushing" is not set, the safe setting for a USB drive.
- Under the build's load: 100% busy, about 90 operations a second, 216 ms each at a queue depth of 5, 7.8 MB/s: a hard disk's random-IO ceiling. The internal WD_BLACK SN850X NVMe sat 99.9% idle.
- The Docker Desktop data disk (`D:\Docker-Image\DockerDesktopWSL\DockerDesktopWSL\disk\docker_data.vhdx`, 152 GB) and the Ubuntu distro (`D:\WSL\Ubuntu`, per `.wslconfig`) are both on D:, so images, containers and WSL's repository copies and grader folders all share it.
- `.wslconfig`: memory=24GB of the machine's 31 GB, processors=14, swap=8GB; Docker Desktop runs in the same WSL2 VM and shares them. Another distro (EXL3-Runtime) was running in that VM.
- Free space: D: 196 GB, C: 122 GB, F: 73 GB.
- Defender real-time protection is on. Its exclusions can't be read without administrator rights, so whether the .vhdx files and `D:\RLM-data` are excluded is unknown.

*Recommendations, in place, each one Rishi's to make (none is needed for correct training):*
1. **Defender exclusions** (as administrator: Windows Security, Virus & threat protection, Manage settings, Exclusions, Add an exclusion, Folder): `D:\Docker-Image`, `D:\WSL` and `D:\RLM-data`. Real-time scanning adds work to files Docker and WSL write constantly, on a drive with no IO to spare. Expected: fewer stalls on container starts and layer writes, perhaps 10-30% on IO-heavy steps *(inferred)*.
2. **Keep the drive and the VM to the run:** stop other WSL distros during a run (`wsl --terminate EXL3-Runtime`), other projects' jobs on D: and any image builds. The numbers above came from a quiet drive; every competing read or write stretches every container start.
3. **Moderate parallelism** (configuration, not a drive setting): the grader's cross-process limit (`RLM_GRADER_SLOTS`, 5 in validation) can cap graders in training too. On this drive more containers at once did not mean more throughput; past about 8-10 it meant timeouts.
4. **WSL memory:** leave 24 GB on a 31 GB machine. More page cache would make repeated container starts from the same images faster, but Windows needs the rest; revisit only with more RAM.
- Already right, no change: the USB 3 / UAS link, device write caching on.
- Not recommended: turning off write-cache buffer flushing (faster flushes, but a power cut or an unplug could corrupt the 152 GB Docker disk).
- Clean-up, only with Rishi's explicit OK and of little use for speed: BuildKit's build cache and superseded derived images (older TypeScript and JsonPath derives, the JsonPath warm image) would free space, not IO. D: has 196 GB free, so it isn't needed now.

*Verdict:* **D: as it is is good enough for production.** Every limit the drive can stretch (container start, grader cap, cell limit) was set from measurements under load, a timeout is retried before it counts, and an infra error is never a zero: the slow drive costs time, never reward. The cost is about 30-35 minutes per training step on average, against about 5 for read-only steps.

## 5b. Security notes for the red team (as built)

For [step-9.5-red-team.md](step-9.5-red-team.md), the red-team brief built from these notes. Line numbers are as of this build.

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
4. **§12.5 step 2, new test files** are dropped from the diff, not kept unrun (see §2.2): Go, Rust and Java compile every test file in scope, so a kept file would still build and a Go `init()` would run. `tests/test_grader.py` checks it per language. (Folded into the spec by the coordinator, e2c240f.)
5. **§12.4, `edit_synth`'s image:** `rlm-edit-synth` (the read sandbox's pinned base plus git) instead of `rlm-sandbox`, because the coding tools' `diff()` needs git. The read image is unchanged. (The coordinator's instruction with the tools.)
6. **§12.4, Go's module cache** stays read-only where the image put it (world-readable, and `GOPROXY=off` fetches nothing) instead of being re-owned: re-owning copies every file into a new layer, which took over 16 minutes per image on the data drive. Only the build cache moves to `/opt/rlm-cache`.
7. **Start-up timeouts are generous** *(mine; Rishi, 2026-10-13: time is not the concern as long as things work)*: 300 s for a read sandbox (was 60 s), 600 s for an edit sandbox, and 1800 s on top of the tests' own cap for a grader container (was 180 s; 600 s was still too little for Rust builds under load). Docker's disk and every image live on the USB data drive, where creating a container took minutes while downloads and builds ran beside it, and a training loop test paused on exactly that. A slow start should cost time, not an infra error. The cell limits are unchanged.
8. **§12.8, visible-test check:** run for Python (collection), Go (`go test -run '^$' -exec /bin/true ./...`: every package's tests compile, none run, not even a `TestMain`), Rust statically (`rust_hidden_named`: a removed file breaks the build only when a `mod` line, a `#[path]` or a Cargo.toml `path` names it, since each integration test is its own crate and a test module under src/ stays as an empty file; compiling every test target with `cargo test --no-run` ran past an hour on the data drive) and Java (`test-compile` / `testClasses` with the repository's own build tool); JavaScript and TypeScript have no compile step to check, so their tasks skip it. (Go first used `go vet`, which also fails on lint the repository already had, so it dropped good tasks; the records it dropped were validated again.)
9. **§12.8, the three gold runs** share one grader container where the runner writes its report where it's told (pytest, unittest, Go, Cargo, Jest, Vitest, Mocha), each run with its own report and the task's cap; Maven and Gradle keep a container per run. On the USB drive each container start cost minutes, and validation started five per task.
10. **Go tasks hide test functions, not whole test files** *(mine)*: in Go, hiding a whole `_test.go` file also removed helpers the package's other test files use, so the visible tests stopped compiling (all 26 Go drops in the first 53 validations). A Go task's repository now keeps each hidden test file without the hidden test functions (helpers, types and other tests kept, unused imports dropped), and the grader writes the full file back (`editfeatures.strip_go_tests`, the record's `replace`).
11. **§12.4, TypeScript's node_modules stay where the image installed them** *(mine)*: moving them out of the checkout copied every file into a new layer, and TypeScript's images are 2.8-6.5 GB (rxjs ran four hours and hadn't finished). A TypeScript image derived now keeps its node_modules folders in place, deletes `.git` and every other file in the checkout, and links `/opt/rlm-deps` to that folder, so an episode's `/work/repo/node_modules -> /opt/rlm-deps/node_modules` links are unchanged. The node_modules stay owned by root, read-only to the sandbox user (tests only read them), and the image's repository path stays a folder instead of a link to /work/repo. The JavaScript images were derived before this and keep the moved, host-owned copies.
12. **Rust test modules in their own files stay as empty modules** *(mine)*: a generated Rust task whose hidden tests sit in `src/**/tests.rs` (or `src/**/tests/mod.rs`) can't simply remove the file: a `mod tests;` line in the code names it, and the crate's tests stop compiling (27 generated tasks were heading for "visible tests break"). The file stays in the repository as an empty module and the grader writes the full file back (`hide_plan`, the record's `replace`, as for Go).

## 7. This round: Python, Go and JavaScript (Rishi, 2026-10-13 2:12 pm ET: "Python and Go today")

Validation stopped at 2:25 pm ET on 2026-10-13 (Rishi: finish step 9.5 now) so the build could finish the same day. Training uses every Python, Go and JavaScript task that passed with real tests by then; the index is built with `index --languages python go javascript`.

| Language | Passed validation (train / test / select) | In the index's training split |
|---|---|---|
| Python | 173 (149 / 19 / 5) | 86 (55 bug, 31 feature: the per-repository caps and the 90 target) |
| Go | 38 (17 + 2 real PRs / 15 / 4) | 16 (4 bug, 10 feature, 2 real PRs) |
| JavaScript | 18 (5 / 9 / 4) | 5 (feature) |

**The index** (`D:\RLM-data\edit\index.json`, 157 tasks): training 107 (Python 86, Go 16 including 2 real PRs, JavaScript 5), selection 10 (Python 3, Go 4, JavaScript 3), test 40 (Python 16, Go 15, JavaScript 9; 8 of them feature tasks). `measure` timed 40 of 44 sample tasks under 15-at-once load (median test command: Python 143 s, JavaScript 263 s, Go 861 s); `index` set each task's grader cap and, where needed, its cell limit from those timings (unmeasured tasks take their repository's slowest).

The eval splits are filled from all three languages: selection 10 of 13 passing, test 40 of 43 passing (about 8 feature tasks). The first index left them short (6 and 24) because a per-repository cap meant for training bug tasks also applied to the eval splits and counted feature tasks; it now applies to training only.

**Known gaps in this round:**
- **Go training is under its 60-task target** (the slow drive left most Go candidates unvalidated; our own injected Go bugs are in the pool for the next round).
- **The test split has fewer than 15 feature tasks** (spec §12.10 asks for at least 15 of its 40): `balance` takes the feature tasks Python, Go and JavaScript have.
- **Rust, TypeScript and Java are not in this round.** Their candidates, texts, images and partial results stay on `D:\RLM-data\edit` for a later round. TypeScript's runner problems are fixed (react-hook-form passes with real tests); Rust works but its builds are slow on the drive; Java's images are derived with their own build wrappers.
- **JsonPath is parked**: its Gradle tests still report nothing on the warmed image, cause not yet found.
- **rxjs, foam and formik have no tasks** (see §4.2).
- **The edit_real training pool (107 tasks) is smaller than production's draws.** About 75 steps x 2 edit_real tasks per step is about 150 draws, and retirement (a task whose group came back all-0 twice or all-1 once) takes tasks out of the pool for good. Levels of the 107: level 1: 46, level 2: 48, level 3: 13, none at 4 or 5 (above level 3 a draw takes the nearest level). If every task retires, `editreal.make_task` raises `ValueError("no edit_real task left to draw")`, nothing catches it, and the step stops. Step 12 decides: more tasks from the later round (Rust, TypeScript, Java, the Go top-up), or a fallback such as bringing back the oldest retired tasks or moving edit_real's share to edit_synth. No fallback was built (Rishi: finish 9.5 only).

**Checks run at the end of this round (2026-10-13 evening):**
- `measure` timed 40 of 44 sample tasks (2 per repository) 15 at a time in episode containers, the gold change written through the REPL and graded in a fresh grader container; `index` took each task's grader cap and cell limit from those timings.
- **Real-sandbox dry run** (`train.py start --dry-run --kind production --lr 2e-5 --real-edit --stop-after 1`, run `dry-edit-final3`): the selection eval at checkpoint 0 (64 tasks, its 10 edit_real tasks in the real edit sandbox), the capability eval (100%), one trained step, and a clean operator stop; no errors and no infra. The first attempt showed the dry run's oracle answering every edit_real task with the read policy: it keyed its gold changes on the task's first message with the default 300 s cell limit, while `index` now raises slow tasks' limits and the environment quotes the raised one. Fixed in `tests/dryrun.py`.
- **Full test suite:** 400 tests, 395 passed, 1 skipped. The 5 failures were Docker tests run beside `measure` and the dry runs on the saturated drive: a sandbox that didn't start among 32 concurrent ones, the Docker training-loop test pausing on that infra, and teardowns that found leftover containers. The leftovers were a real leak (`rlm_rl/sandbox.py`): a launch that timed out removed its container before the slow daemon had finished creating it, so it stayed "Created" for good (66 had piled up). The container is now removed again once the docker client has exited. Rerun of those Docker test classes alone after the fix (`EditReplTest`, `DockerTest`, `DockerLoopTest`): **14 tests, all passed** (632 s). With them, the suite stands at 399 passed and 1 skipped.

*Follow-up, not built:* a launch the daemon takes longer to create than the second removal pass can still leave a "Created" container; a periodic sweep of the run's own never-started containers (by its `rlm.run` label) would close that.
