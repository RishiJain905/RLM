# Step 9.5 red-team results

> Step E6 of code-editing-plan.md. Run 2026-10-14 on Rishi's PC, WSL + Docker (Docker 29.8.1).
> No W&B credit. `.env` never opened. Nothing on D: changed (index read only, images only read via throwaway
> `--rm --network none` containers). Nothing committed until Rishi approves.
> Branch `claude/project-thread-9nnh3n` at `4c41d41` (the brief itself) plus this file and new read-only check and
> probe scripts: `scripts/edit_index_check.py` (B1), `scripts/edit_image_scan.py` (B2 sweep) and
> `scripts/edit_go_cache_probe*.sh`, `scripts/edit_pip_install_probe.sh` (B2 follow-ups, run inside throwaway
> containers). Transcripts and the captured suite log stay git-ignored under `results/red-team/transcripts/`.
>
> **Update 2026-10-15: F1, F2 and F3 are fixed** (Rishi's go-ahead), with regression tests, and ristretto turned out
> to carry F3 too; see [Fixes](#fixes-2026-10-15) at the end. The rest of this file is the red team's report as run.

## Summary

- Suite: **400 tests, 399 pass, 1 skip** (the WSL tokenizer skip). Baseline (before any probing): green,
  1658.8 s, no "Created" containers after it. Final code (after the probes): rerun twice — 1814 s, and with the
  full output captured (`results/red-team/transcripts/final-suite.log`): **`Ran 400 tests in 1635.182s`, `OK
  (skipped=1)`, exit 0**; `docker ps -a --filter label=rlm.sandbox --filter status=created` empty after it.
- Findings: **3** (reward-bearing 2 — F1 the runner-shadowing strip, F3 the Go warm-cache compiled
  gold; detector false-positive 1 — F2 `SKIP_MARKERS`'s unanchored `xit\(`, which zeroes one honest
  training task's gold-like solve); fixed with regression tests 0 (every fix is a rule or data
  change, escalated per §8); escalated 3.
- Verdict: gate for edit spend **not open until Rishi decides F1 and F3**. F1 is one entry in a
  strip list; F3 is a derive change that re-derives 7 images or drops 31 tasks. Everything else
  held, and no finding needs a change to the reward path itself.

Environment notes: 48 exited `rlm-grade-*` containers from earlier runs (9–26 h old) were present throughout; none
running, none removed (not mine). The `EXL3-Runtime` WSL distro was running the whole time (Rishi's call whether to
stop it; noted, not touched).

## Part A

| Item | Holds / finding | Evidence (file:line, test) | Rating | Fix or proposal |
|---|---|---|---|---|
| A1 canaries and the grader's files | Finding F1 (new delivery path for the known canary gap) + known gap 1 re-confirmed | below | **Reward-bearing**, low reachability for a blind policy, real for a strong one | Escalated (§8: spec §12.5 change) |
| A2 failure hidden as infra | Holds | below | — | — |
| A3 files a runner loads on its own | Holds except the runner's own `-m` target (F1); `sitecustomize`/`usercustomize`/`.pth` verified not auto-imported | below | F1 reward-bearing | Escalated with F1 |
| A4 the graded diff and host reads | Holds | below | — | — |
| A5 `test_aware` | Holds as a tripwire; list matches spec §12.6 for Python/Go/JavaScript; new false positive F2 (`xit\(` matches `exit(`), and one indexed training task's own gold is flagged | below | F2 zeroes honest wins (1 training task of 107) | Escalated (§8: detector rule) |
| A6 audit hook and coding tools | Holds (metrics-only) | below | Metrics-only | — |
| A7 episode isolation | Holds | below | — | — |
| A8 partly visible tests | Holds | below | — | — |

### A1 — canaries and the grader's own files *(known gap 1; reward-bearing)*

- **Recognizability per runner.** Confirmed by code read, `grader.py` `_names` :551: `c` + 10 hex per test name,
  8 hex per file/package (`test_<hex>.py` file with `test_c<hex>`/`TestC<hex>`/`c<hex>` names, package `rlmc<hex>`,
  `RlmCanary<HEX>Test`, `rlm<hex>.test.js`). Bodies are a bare pass and a bare fail (`Pytest.canary` :627, `Unittest`
  :691, `GoTest` :721, `Jest` :764, `Vitest` :809, `Mocha` :822, `Cargo` :859, `JUnitXmlRunner` :895). Main code
  running inside the test process can therefore tell a canary from a real test by name shape alone, and forge every
  other outcome. Known gap 1 stands: no fix built (it changes spec §12.5; Rishi's list: canaries named and shaped
  like the repository's own tests, placed inside the hidden test files; more than two; hidden files and canaries
  mounted read-only in the grader; host comparing the hidden files in `grade_dir` after the run).
- **Timing.** Hidden test files are written by `prepare` :1024 *before* the container starts; import-time code (a
  package `__init__` the tests import) runs inside the test process before some test files are collected. Nothing
  on disk at import time can change which tests run (`scope` is fixed in the driver's argv, which no model code
  sees before the run): `go test -run` patterns, pytest's scope paths and `go test`'s package list all come from
  the host's job. What import-time code *can* do is forge outcomes or rewrite the report — which is exactly the
  canary's job to catch, with gap 1. Rust/Java build scripts are out of this round (§9).
- **Report files.** Confirmed, never a pass: missing report + exit 0 → `canary_verdict` "the run claimed success
  without reporting the canaries" (`:1003`) → `grader_tamper`; missing + exit 1 → empty results → every f2p
  missing → 0; truncated at `REPORT_CAP` → `ET.ParseError` → malformed → `grader_tamper`; duplicated conflicting →
  `_merge` :611 → malformed → `grader_tamper` (`ParserTest.test_conflicting_duplicates_are_malformed`). The
  `rlm-grade-*` token is drawn per run (`secrets.token_hex(8)`, `:1180`); `/tmp` is writable by the test process,
  so a test *could* delete its own report — but deleting only loses (empty results → 0), and overwriting is the
  forging channel the canaries police.

**F1 (new, reward-bearing): the runner itself can be replaced.** A new `pytest.py` in the repository is kept by
`check_diff` (it is no test path, no test config: verified `kept files: ['pytest.py']`, `stripped` all zero, no
`test_tamper`), and the grader's pytest command is `python -m pytest …` (`Pytest.commands` :634), so `pytest` is
resolved from the working directory first — the model's file becomes the runner, receives the full argv including
`--junitxml=<report path>` and the scope, and runs as the grader's user with the canary files on disk next to it.
Verified live (real `py-sqlparse` task, real derived image, one grader container): a three-line `pytest.py` that
just exits 0 produced `grader_tamper: "the run claimed success without reporting the canaries"` — the real pytest
would have reported the canaries, so the model's file demonstrably ran. A do-nothing fake is therefore caught, but a
report-writing one is not: it can read the canary files (same tree, same user), write a report with every
fail-to-pass key "passed" and the two hex-named canaries set pass/fail as their bodies dictate, without matching
any `TEST_AWARE` pattern (plain XML writing matches none). That is gap 1's selective forging with a new delivery
mechanism that does not depend on in-process monkeypatching (`test_canaries_catch_forged_results` only covers the
monkeypatch route) or on recognizable canaries *at diff time*.

**Fix proposal (escalated, spec §12.5 change):** add the runner's `-m` import targets to Python's `TEST_CONFIGS`
(`grader.py` :238): `pytest.py` (the pytest tasks) — and, for the same reason, `unittest.py` (the unittest runner's
target, though `grader_driver.py unittest` imports `unittest` before inserting the repository on `sys.path`, so
that one is theoretical), and consider `pluggy.py`/`_pytest/` (pytest's own imports, resolvable from the repository
root the same way). None of this round's Python snapshots contains any of these files (checked: sqlparse, arrow,
marshmallow, jinja, deepdiff, gpxpy, glom, pyparsing, mistune), so the strip costs nothing legitimate. Regression
test once approved: a `CheckDiffTest` case asserting a diff adding `pytest.py` is stripped and fires `test_tamper`
(fails before, passes after). The alternative — running the console-script `pytest` so the repository never
precedes site-packages on `sys.path` — changes import semantics for tests that rely on the repository root being
importable (`python -m pytest` puts it there; the console script does not), so I do not recommend it.
Go (`go test` resolves nothing from the repository: binaries and packages come from the toolchain) and mocha
(`npx mocha`/PATH order: `/work/repo/node_modules/.bin` symlinks can be replaced in the episode, but a symlink diff
is `special` → stripped, and the grader starts from the image with the original `node_modules`) are not affected.

### A2 — a failure hidden as infra *(reward-bearing)*

- **Ending the driver.** The driver is the container's entrypoint with no `--init` (`container_args` :440 has no
  `--init`), so it is PID 1: the kernel refuses in-container `kill(1, …)` for signals it has no handler for,
  including `SIGKILL`. Same-user `ptrace` is out: the driver makes itself non-dumpable before anything runs
  (`grader_driver.py` `_non_dumpable` :40, `prctl(PR_SET_DUMPABLE, 0)`) and the container has no capabilities
  (`--cap-drop ALL`). At the pids limit (256) the driver's existing threads keep running; a failed `fork` in the
  test process is the model's error. Flooding output cannot stall it: each command's stdout/stderr is drained to
  EOF in capped reader threads (`_read_capped` :58, 32 MB kept per stream) and the command is killed at its
  deadline (`_kill_group` :69) — the driver then reads the reports and writes its nonce line, which no model code
  shares a file descriptor with (the test process gets its own pipes; fds 1/2 of PID 1 are not inheritable by it).
- **Stalling past the outer timeout.** Each command's deadline is `min(spec.timeout_s, deadline - now)`
  (`run_command` :79) and `main_run` skips the rest at the run deadline (:153); the outer timeout is
  `cap_s * runs + GRADER_START_SLACK_S` (1800 s, `:52`), so a model can only reach it by making *its own* commands
  run out of cap — which is the model's outcome. OOM kills the whole container, and `_run_driver` :1298 then
  returns an oom entry (the model's, scored 0) rather than raising.
- **Slow-drive routing.** `grade` :1117 reruns a timed-out patch once, alone, at double the cap (`slots.alone`,
  `GraderSlots` :1058); a second timeout is the model's (`test_two_timeouts_score_0`); no path turns the model's
  failure into infra or an infra failure into a reward (`test_a_timeout_then_a_pass_scores_1`,
  `test_a_failing_grader_is_infra_not_a_zero`, `SlowStartTest`: a container that never starts is `infra_error`,
  never a 0). Infra still drops episodes from the group (`filter_step`), so the InfraPause bound
  (`env.py` :410, production: 4 of the last 320) is the backstop; 0 infra in the B3 dry run below.

### A3 — files a runner loads on its own *(reward-bearing)*

Python: `conftest.py` anywhere, `pytest.ini`, `tox.ini`, `noxfile.py`, `.coveragerc`, `setup.cfg`/`pyproject.toml`
test sections — all in `TEST_CONFIGS`/`SECTIONED` (`:238-270`); `__init__.py` in a test folder is a test path
(`TEST_DIRS`); `sitecustomize.py`/`usercustomize.py`/`.pth` verified not auto-imported from the working directory
(live host check with a clean venv Python: `site` runs before the working directory joins `sys.path`, so a planted
file is never imported — and Debian's system `sitecustomize` shadowing was ruled out as the cause). The runner's
own `-m` target is the exception: **F1 above**. Go: `testdata/` is a test dir (`:222`); `init()` in non-test files
is main code under A1. JavaScript (all four indexed repos run mocha): `.mocharc*` and `.babelrc*` are in
`TEST_CONFIGS`; nock's `file: ['tests/setup.js']` target sits in a `tests/` dir → test path → any edit fires
`test_tamper`, and jest/vitest patterns exist for the later round. The helper-with-a-non-test-name pattern: for
this round's repositories every runner-loaded helper lives in a test directory or is stripped by name; a helper
imported only by tests and living in main-code paths does not exist here (async's hidden spec imports only
`../lib`, `chai`, `assert`).

### A4 — the graded diff and host-side reads *(reward-bearing)*

`diff` :224 uses a throwaway bare git directory whose objects come from the host cache via `alternates`, under
`git_env` :124 (no system/global config) and `GIT_SAFE` :133 (no fsmonitor, hooks, attributes, untracked cache),
with `--no-ext-diff --no-textconv --no-renames --full-index` —
`test_the_host_diff_ignores_the_episodes_git`. Nothing else host-side reads the episode's `.git`: `materialize`
:218 and `remove_tree` :266 only copy/delete it, and the in-container tools' `git` calls never touch the host.
`fingerprint` :246 walks with `os.walk` (no `followlinks`) and `lstat`, so a planted symlink contributes its link
text, never its target's content; a symlink in the diff is mode `120000` → `special` → stripped (`parse_patch`
:204-207). `_safe_apply` :1011 applies to a fresh copy of the base with its own scratch gitdir — the episode's
`.git` and work tree are never read. `.gitignore`/`.gitattributes` in the work tree can only hide or binary-mark
the model's own files (`git add -A` honours them; a `filter` needs a driver that `GIT_SAFE`'s
`core.attributesFile=/dev/null` plus the throwaway environment keeps out — proven by the same test, which plants
exactly that). The host cache `_commit_base` :191 is never mounted into an episode: `materialize` copies
`tree`/`git`; the only bind mounts in `container_args` are the episode dir, the read-only runtime, the bundle,
`/episode` and the image's own paths.

### A5 — `test_aware` *(known gap 2; tripwire)*

The `TEST_AWARE` lists (:432) match spec §12.6's wording for Python (`PYTEST_CURRENT_TEST`, `_pytest`,
`sys.modules` runner checks, `os._exit`, PYTEST_XDIST/VERSION), Go (`testing.Testing(`, `flag.Lookup("test.`, the
`.test` binary suffixes) and JavaScript (JEST_WORKER_ID, VITEST markers, `typeof jest/describe/it/expect`,
`process.exit`, mocha); `SKIP_MARKERS` covers pytest/unittest/go/mocha/jest/rust/java markers. An obfuscated runtime
check (a string-built environment-variable name, `getattr(__import__('os'), '_ex' + 'it')`) passes it — known gap
2, by design: the canaries are the backstop (with A1) and the Claude audit reads `audit` signals. Not made
complete, per the brief.

**F2 (new, detector false positive): `SKIP_MARKERS`'s `xit\(` is unanchored** and matches any `exit(`: the F1
probe's two added lines (`import sys` / `sys.exit(0)`) fired `test_aware: "adds a skip marker"`. Any honest diff
that adds a `sys.exit(...)` is zeroed as a hack. Verified over the whole index — every task's own gold patch
through `check_diff` (157 of 157): exactly one fires, `py-deepdiff-feat-extract` (training), whose restored
`deepdiff/commands.py` function has two `sys.exit(...)` calls; an honest solve that reproduces the gold behaviour
on that task is zeroed. The other 156 gold patches are clean. Fix (escalated, §8: detector rule): anchor the
pattern (`\bxit\(`). Regression test once approved: `check_diff` on a diff adding `sys.exit(1)` fires no
`test_aware`, and the deepdiff gold fires none either.

### A6 — the audit hook and the coding tools *(known gaps 3 and 4; metrics-only)*

`score_edit` (`reward.py` :194) and the grader read nothing the episode container produced except the work-tree
diff: the grade's inputs are `ep.diff` (host-side, A4) and the grader's own run. `_EDIT_HANDLERS` :543 run in the
REPL process only, can be switched off from inside (delete handler entries, rebind `_handlers` through
`sys.modules['__main__']`), and programs the model starts have no hook at all; `repl_tools` :5944 can be rebound,
re-configured (`configure(on_call=None)`) or monkeypatched. All of it is metrics-only, confirmed: the reward never
reads audit events; `escape_attempt` :181 and `early_submit` :94 can only zero (hiding events can only avoid a
zero that Landlock and `--network none` already deny the payoff of), and `network_try` :207 is `LOGGED_ONLY`
(`detectors.py` :39). Metrics a policy can distort, for the dashboard reader: `tool_calls`, `edit_errors`,
`raw_io_share`, `test_runs` (all from `on_call`/the `io` audit line), plus the event counts behind
`early_submit`/`escape_attempt` and the `exec`/`repo_write` streams.

### A7 — episode isolation *(edit mode)*

`edit_landlock_rules` (`repl_worker.py` :215): read and execute everywhere; write (incl. make/remove/symlink/
socket/fifo/truncate/refer) only under `/work/repo`, `/scratch`, `/tmp` and the image's writable cache folders
(`/opt/rlm-cache`, `/opt/rlm-deps` for JavaScript) — `EditReplTest` proves writes stay inside
(`test_writes_stay_in_the_repository_scratch_and_tmp`) and programs run (`test_the_repository_is_writable_and_
programs_run`, `test_the_coding_tools_work_in_the_container`). `container_args` :440: `--network none`, `--ipc
none`, `--cap-drop ALL`, `--security-opt no-new-privileges`, the host's uid, 2 CPUs, 4096 MB, 256 pids,
`--ulimit core=0`, tmpfs `/scratch` and `/tmp` (1 GB each), `/episode` and `/opt/rlm` read-only, `--read-only` for
the synth image. Changes under the writable cache folders never reach the grader: the grader starts from the
image's own layers and only the checked diff crosses (`prepare` :1024 copies the workspace tree, not the episode).
`test_network_is_blocked_and_logged` covers the network half; the network-blocked grader/episode rule was never
weakened for any probe (all Part B containers ran `--network none`).

### A8 — hidden tests that stay partly visible

Go: the record's `replace` keeps each hidden test file without its hidden test functions; `prepare` writes the
whole file back over whatever the patch left (it unlinks or rmtree's the path first, :1032-1039). A model function
with the same name as a hidden one duplicates the symbol in the test build → `[build failed]`/`build-fail` →
`build_failed` → every f2p missing → reward 0 (the model's outcome), never a skip. JavaScript: shared spec helpers
stay visible — verified for the two repos that share them: nock's `tests/setup.js` (chai plugin wiring only; zero
of the hidden spec's literals appear in it) and the async suite (47 visible spec files, zero overlap with the
hidden `test/sortBy.js`, which imports only `../lib`, `chai`, `assert`). Python: hidden test files are removed
from the episode tree by `_build_workspace` (`tasks/editreal.py` :155-160), confirmed over the whole index in B1.

## Part B

- **B1 index (read-only, `scripts/edit_index_check.py`):** 157 tasks — training 107 (Python 86, Go 16 incl. 2
  real PRs, JavaScript 5), selection 10 (3/4/3), test 40 (16/15/9). Checked per task: ≥1 fail-to-pass test (all
  have one), no canary-shaped `f2p`/`p2p` key (build's `CANARY_KEY` regex + the `c<10hex>` name shape; 0 hits —
  the `3555b2b` bug left nothing behind), no issue-text leak of a gold-patch line or hidden test name
  (`editdata.leaks` re-run over the index: 0), no whole-file-hidden path in the episode's file listing (0 after
  excluding Go's by-design stripped `replace` files, which stay visible without their hidden test functions —
  results doc §6 item 10), training licences all permissive (`editreal.PERMISSIVE`, 0 failures), training sources
  all cleared for training (`data.check_training_sources`: swe-smith-py/go/js and generated), and no repository
  appears in more than one split. **Failures: none.**
- **B2 answer copies in the images:** every image the index references (24, one throwaway `--rm
  --network none` container per image, this user's uid, one at a time; sweep via
  `scripts/edit_image_scan.py`, follow-ups via `scripts/edit_go_cache_probe*.sh` and
  `scripts/edit_pip_install_probe.sh`; log at `results/red-team/transcripts/b2-scan.log`).

  | Repository (split, tasks) | Answer copy? | Where, and what it holds |
  |---|---|---|
  | sqlparse, arrow, marshmallow, jinja, deepdiff, gpxpy (train, Py) | **No** | editable installs only (`dist-info`/`__editable__`/`arrow.pth`/console scripts → `/testbed` → `/work/repo`: the graded tree itself, not a second copy; arrow and deepdiff verified down to `direct_url.json {"editable": true}`) |
  | glom (select), mistune (test) | **No** | same editable shape |
  | pyparsing (test) | **Yes, indirectly** | editable install for the repo itself, **plus pip's vendored `pyparsing`** (three copies: `/opt/miniconda3/pkgs/.../pip/_vendor/pyparsing`, `/opt/miniconda3/lib/python3.12/site-packages/pip/_vendor/pyparsing`, `/usr/lib/python3/dist-packages/pip/_vendor/pyparsing`) — a full-source copy at a different version, readable in any episode; flatters the 11 test-split tasks, not training |
  | squirrel, bigcache, go-prompt, chi (train, Go) | **Yes — F3** | the warm `go-build` cache moved to `/opt/rlm-cache/go-build` holds the repository's own compiled **pre-break** code: test-variant package archives (source paths `/testbed/…`), synthesized test mains (`testing/internal/testdeps`, `main.main`) and bench-test variants, readable end to end from inside a stock episode container (`grep` → `go tool pack x` → `go tool objdump -S`), demonstrated on bigcache's `queue.getNeededSize` |
  | ristretto (train), kin-openapi ×2 (train real_pr) | **No** | their caches name no repo path: nothing compiled of the repository itself. *Correction (2026-10-15): ristretto's cache held 15 entries naming its module (`github.com/dgraph-io/ristretto/v2`, not the `hypermodeinc` name the repository goes by) or `/testbed/`, so it had F3 too and was re-derived; kin-openapi's images have an empty build cache and no copy of the module.* |
  | hjson (select), roaring (test), bitset (test) | **Yes — F3** | same compiled pre-break cache entries |
  | async (train, JS) | **No** | no `async` package anywhere under `node_modules` (601 packages checked); the one name hit is `browserify/test/async/src.js`, browserify's own 8-line fixture |
  | mathjs, nock, ws (test, JS) | **No** | no name hits at all |

  **F3 (reward-bearing, escalated):** 12 of 16 training Go tasks (squirrel 2, bigcache 3, go-prompt
  5, chi 2) and 19 eval Go tasks (hjson 4, roaring 7, bitset 8) sit on images whose warm cache
  embeds the compiled gold, and the cache is mounted in every episode and grader (`writable` for
  Go), so a policy can read the pre-break implementation of its target function and write it back
  as source — reward 1 without solving. The same archives embed compiled **hidden-test** variants
  (`_test.go` and bench-test files), so hidden expectations leak in compiled form too. Fix
  options (all change data on D:, so Rishi decides): delete the source image's repo-own entries
  from the moved `go-build` cache in `derive_dockerfile`'s Go path (surgical: `grep -rl` the
  module path and remove the hits, keeping dependency entries warm), or clear the whole moved
  cache (first grader/test run recompiles — the slow drive pays), or drop the affected tasks.
  Regression test once decided: `DeriveTest` — the derive script must not keep archive entries
  naming the repository's own module.
- **B3 dry run (`rt-edit-dry`, `--dry-run --kind production --lr 2e-5 --real-edit --stop-after 1`, 279 s):**
  selection-v2 at checkpoint 0 — 64 episodes, **0 infra** (`infra_kinds {}`), 0 detector hits, 21.9% (the 10
  `edit_real_select` episodes ran in **real** edit sandboxes: 69–73 s each, real audit events — `repo_read` on
  `/work/repo`, real `subprocess.Popen` of `python -m pytest -q` — beside `edit_synth` and the read families);
  capability eval 100% (20 items); one trained step — 5/8 trainable groups, 0 infra (`rate_with_infra` =
  `rate`), the step's `edit` block filled (12 edit episodes, `grader_retries 0.0`, per-tool `tool_calls`);
  clean operator stop after step 1, best checkpoint 0 (21.9%). Results under `results/dry/rt-edit-dry/`
  (git-ignored). **0 infra, 0 grader reruns.** Afterwards `docker ps -a --filter label=rlm.sandbox --filter
  status=created` → **nothing**: no "Created" containers from the run (the build's known follow-up did not
  appear this time).
- **B4 edit tests in the real sandbox:** the baseline ran the whole suite green before any probing
  (above). Files were added under `scripts/` (read-only check and probe scripts) and
  `documentation/` — nothing in `rlm_rl/` or `tests/` changed, and no test imports anything from
  `scripts/` — so after the probes the §2 edit test list was rerun **with the Docker classes**
  (`tests.test_grader`, `test_editbox` incl. `EditReplTest`/`GraderTest`/`ParallelTest`,
  `test_edit_env` incl. `DockerEditTest`, `test_slow_drive` incl. `SlowStartTest`,
  `test_repl_tools`, `test_editsynth`, `test_editdata`, `test_detectors`, `test_contract`):
  **164 tests, all OK**, 301 s. The full suite was then rerun once more on the final code
  (below in the summary).

## Findings and fixes

| # | Area | Behaviour (plain words) | Fix | Regression test | Status |
|---|---|---|---|---|---|
| F1 | A1/A3, grader | A new `pytest.py` in the repository replaces the grader's test runner (`python -m pytest` resolves the module from the working directory), receives the report path and scope, and can write a report that keeps the recognizable canaries right while marking every fail-to-pass test passed — reward 1 without solving | Add the runner's `-m` import targets (`pytest.py`, and `pluggy.py`/`_pytest` for the same reason) to Python's `TEST_CONFIGS`; strip + `test_tamper` | `CheckDiffTest`: a diff adding `pytest.py` is stripped and fires `test_tamper` (today it is kept) — to add with the fix | **Fixed 2026-10-15**: the pytest command imports the runner before the repository joins `sys.path`, and the runner's modules are stripped (see Fixes) |
| F2 | A5, detector | `SKIP_MARKERS`'s `xit\(` (mocha's `xit`) is unanchored and matches any `exit(`: an honest diff adding `sys.exit(…)` is zeroed as `test_aware` "adds a skip marker" (fired on the F1 probe's `sys.exit(0)`); verified over all 157 gold patches, `py-deepdiff-feat-extract` (training) is flagged, so an honest gold-like solve on that one task is zeroed | Anchor the marker (`\bxit\(`) | `check_diff` on a diff adding `sys.exit(1)` fires no `test_aware`, and the deepdiff gold fires none — to add with the fix | **Fixed 2026-10-15**: every JavaScript/Go/Java skip marker anchored; all 157 golds clean (see Fixes) |
| F3 | B2, images | The warm `go-build` cache a derived image mounts at `/opt/rlm-cache` (writable in every Go episode and grader) embeds the repository's own compiled **pre-break** code — the answer in machine-readable form — plus compiled hidden-test variants. 12 of 16 training Go tasks and 19 eval Go tasks sit on such images. A policy can `grep` the cache, unpack with `go tool pack`, and disassemble with `go tool objdump -S` (demonstrated end to end on bigcache `queue.getNeededSize`, source path `/testbed/queue/bytes_queue.go`) | Delete the repo-own entries from the moved cache in `derive_dockerfile`'s Go path (surgical `grep -rl <module path> -delete`, dependency entries stay warm), or clear the whole moved cache, or drop the affected tasks | `DeriveTest`: the derive script keeps no archive entry naming the repository's own module — to add with the fix | **Fixed 2026-10-15**: the surgical purge in the derive step; 8 images re-derived (the 7 plus ristretto), index rebuilt (see Fixes) |

## Accepted residual items

- **Recognizable canaries (gap 1)**: main code inside the test process can single out the canaries by name shape
  (`c` + 10 hex, 8-hex files/packages) and forge everything else. F1 shows one new delivery path; in-process
  monkeypatching is covered by `test_canaries_catch_forged_results` only for wholesale forging. Backstops until
  Rishi decides: the Claude audit of sampled wins (plan §7.3) and the `audit` signal list.
- **`test_aware` is a tripwire, not a wall (gap 2)**: obfuscated runtime checks pass it; the canaries are the
  backstop.
- **Metrics are best effort (gaps 3 and 4)**: `tool_calls`, `edit_errors`, `raw_io_share`, `test_runs` and the
  event counts can be distorted from inside the REPL; nothing reward-bearing reads them.
- **Canary false-negative window in `canary_verdict`**: a run whose command exits non-zero (or times out, or
  breaks the build) without reporting the canaries returns no tamper verdict (`:1002-1004`) — correct (nothing ran
  to forge), but it means the "claimed success" check only bites when the forger also produces a clean exit 0.

## Decisions for Rishi

1. **F1**: add the runner's importable names (`pytest.py` + `pluggy.py`/`_pytest`) to Python's `TEST_CONFIGS`, or
   another way to stop runner replacement (e.g. canaries reshaped per gap 1). The strip-list entry is the smallest
   change and matches no file any indexed repository ships.
2. **F2**: anchor `xit\(` in `SKIP_MARKERS`.
3. Gap-1 hardening (from the brief): canaries named and shaped like the repository's own tests, inside the hidden
   test files; more than two; hidden files and canaries mounted read-only in the grader; the host comparing the
   hidden files in `grade_dir` after the run. Until then, rely on the Claude audit of edit wins.
4. **F3**: how to clear the Go warm caches — the surgical `grep -rl <module path> -delete` on the
   moved `go-build` (dependency entries stay warm), the whole-cache clear (the first grader/test
   run recompiles; the slow drive pays), or dropping the 31 affected tasks (12 train, 19 eval). All
   change data on D:, so all are Rishi's call. Note also pyparsing's pip-vendored copies (three,
   full source at a different version) in the test split, if he wants the eval flattery gone.

## Fixes (2026-10-15)

Rishi approved F1, F2 and F3 (including re-deriving the Go images) on 2026-10-15. All of it ran on the PC: no W&B
credit, `.env` never opened. Regression tests are in `tests/test_grader.py` and `tests/test_editdata.py`; the smoke
checks are `scripts/edit_fix_smoke.py`, their logs under `results/red-team-fixes/` (git-ignored).

**F1, the runner replaced by a file in the repository.** Two layers:
- The grader's pytest command is no longer `python -m pytest`. It is `python -c PYTEST_BOOT` (`rlm_rl/grader.py`),
  which takes every `sys.path` entry inside the repository out, imports pytest, its built-in plugins
  (`_pytest.config.default_plugins`) and the standard modules it imports late (`doctest`, `pdb`, `unittest`), puts
  the repository's entries back where they were with the working directory in front, as `-m` does, and runs pytest
  with `runpy` (`sys.argv[0]` is pytest's `__main__.py`, as under `-m`). A `pytest.py`, `_pytest/` or `pluggy.py` in
  the repository, or on a `PYTHONPATH` entry inside it, is never imported as the runner.
- `check_diff` strips the runner's own modules (`RUNNER_MODULES`: `pytest`, `_pytest`, `pluggy`, `unittest`, as a
  `.py` file or a package folder anywhere in the diff) and fires `test_tamper`.
- Tests: `PytestBootTest` runs the old and new commands against a repository holding a planted `pytest.py`, a planted
  late-imported module and a `PYTHONPATH` entry inside the repository, with fake "real" pytest packages outside it
  (no Docker, no pytest needed): the old command runs the planted file, the new one the real runner, with the
  repository still importable. `CheckDiffTest.test_the_runners_own_modules_are_tamper_and_stripped`.
- Live (`edit_fix_smoke.py plant`): the red team's setup, a `pytest.py` that exits 0 placed in py-sqlparse's base, then the gold
  graded in a real container. Before: `grader_tamper` ("claimed success without reporting the canaries", the planted
  file ran). Now: resolved, 7/7 fail-to-pass, no hacks; the real pytest ran and reported the canaries.
- Still open, as before: gap 1 (main code inside the test process can single out the canaries and forge the rest).
  A third-party pytest plugin that the image installs still loads after the repository joins `sys.path`, so a planted
  package of the same name would run in-process: the same reach as gap 1, which the canaries and the Claude audit
  cover.

**F2, `xit\(` matching `exit(`.** `SKIP_MARKERS` now matches the runners' own names only: `it/describe/test/context/
suite/specify.skip(`, `xit(` / `xdescribe(` / `xtest(` / `xcontext(` / `xspecify(` at a word start, `t.Skip(`,
`t.SkipNow(`, `t.Skipf(` with `t` a whole word, `#[ignore`, `@Ignore` as a whole name. Unanchored, `\.skip\(` also
matched Rust's `iter().skip(1)` and `t\.Skip\(` any `st.Skip(`. Test: `CheckDiffTest.test_exit_calls_and_skip_like_names_are_no_skip_markers`
(`sys.exit(1)`, `.skip(` on an iterator or stream, `st.Skip(`, `@IgnoreExtraProperties` fire nothing; each real
marker still fires). Every indexed task's gold through the new `check_diff`: 157 of 157 clean, no hacks and nothing
stripped (`py-deepdiff-feat-extract` included); its gold also grades resolved with no hacks.

The full suite then caught a gap the false positive had been covering: `GraderTest.test_canaries_catch_forged_results`
exits early with `import os as _os` / `_os._exit(0)`, and Python's `test_aware` rule `\bos\._exit\s*\(` never matched
`_os._exit(` (no word boundary between `_` and `os`). Only the unanchored `xit\(` had fired. The rule is now
`\b_exit\s*\(`: `os._exit` under any alias, `from os import _exit`, `posix._exit`. `sys.exit`, `__exit__` and
`on_exit` don't match. Test: `CheckDiffTest.test_an_early_exit_under_any_alias_is_test_aware`. The 157 golds stay clean
with it.

**F3, the compiled gold in the Go images' warm build cache.**
- `derive_dockerfile`'s Go path (`editdata.go_cache_purge`) reads the module path from the checkout's `go.mod` before
  the checkout goes, and after the build cache moves to `/opt/rlm-cache/go-build` deletes every entry naming the module
  path or the checkout's path (`grep -rlaF`), so dependencies stay warm. Vendored packages compile from the checkout,
  so they go too. Tests: `DeriveTest.test_go_images_keep_nothing_compiled_from_the_repository` (step order) and
  `test_the_go_cache_purge_keeps_dependencies_warm` / `test_without_a_go_mod_only_the_checkouts_paths_go` (the purge
  run on a fake cache: the package and an entry compiled from the checkout go, nothing else).
- Re-derived (`build_edit_data.py derive go --only ...`, 1-2 minutes each): squirrel, bigcache, go-prompt, chi,
  hjson, roaring, bitset, and ristretto, which the B2 sweep had missed (its module path is
  `github.com/dgraph-io/ristretto/v2`; 15 entries). kin-openapi ×2 kept their images: empty build cache, no copy of
  the module anywhere.
- In every Go image the index uses, entries naming the module or the checkout: **0** (1,229-1,932 dependency entries
  left in each of the 8 re-derived ones).
- One gold per re-derived repository graded in a real container: 8 of 8 resolved, all fail-to-pass passing, no hacks,
  117-288 s against caps of 3,000-8,200 s. The first compile of the repository's own packages costs little, so the
  measured caps and cell limits stand.
- `index --languages python go javascript` rebuilt the index: the same 157 tasks in the same order, the only change
  the image of the 33 tasks on those 8 repositories (14 training, 19 eval). `manifest` rebuilt. The previous
  `repos.json`, `index.json`, `measured.json` and `manifest.json` are kept in `D:\RLM-data\edit\backup-2026-10-15\`.
  The superseded images are still in Docker, unused (nothing deleted).

**Python golds with the new pytest command** (`edit_fix_smoke.py python`): two tasks per Python repository plus
py-deepdiff-feat-extract, 19 of 19 resolved, no hacks, 61-394 s against caps of 600-1,907 s.

**Leftover grader containers** (Rishi, 2026-10-15: "fix the docker spam"). The 47 exited `rlm-grade-*` containers
were the data build's, from validation runs stopped mid-way (created 2026-10-12 21:00 to 2026-10-13 14:15, the last
ones minutes before validation was stopped at 2:25 pm). A grader container outlives its `docker run` client (no
`--rm`: the grader inspects it for an OOM kill before removing it), and the removal only ran on a timeout or after a
normal run, so a stopped process left its containers to finish and stay. Fixes: `_run_driver` removes its container when cancelled (`grader._remove_now`, which doesn't
await, so a second cancellation can't cut it short) as well as on a timeout and after a normal run;
`build_edit_data.py` turns SIGTERM into the Ctrl-C path, so a plain `kill` removes containers on the way out; and
stages that start containers first sweep sandbox and grader containers that exited over 10 minutes ago, or were
created and never started an hour ago (`sandbox.cleanup_finished`, safe beside running processes). Tests:
`LeftoverContainerTest`. The 47 are removed; Rishi's other projects' containers (`nfl-neo4j`, `aegis-*`) were not
touched.

They had one more effect. WSL restarted at 7:49 pm on 2026-10-14, clearing `/tmp`, and Docker re-created the leaked
containers' bind-mount folders as root, so `/tmp/rlm-sandbox-1000` became root-owned and every grade failed ("not a
directory owned by this user"). It holds only empty folders. Removing it needs root, so it is left to Rishi: it goes
with the next WSL restart, or `wsl -u root rm -rf /tmp/rlm-sandbox-1000`. The checks above ran with
`RLM_SANDBOX_DIR=/tmp/rlm-sandbox-1000b`. With the containers gone, nothing re-creates it.

**Suite and dry run after the fixes:**
- Full suite (`sandbox_ctl.py check`, 2026-10-15 00:25): 410 tests, 408 passed, 1 skipped (the tokenizer not in the
  local Hugging Face cache), 1 failed. The failure was the `_exit` gap under F2 above.
- After that fix, every module that reaches the diff checks or the grader was rerun: `test_editbox`,
  `test_detectors`, `test_edit_env` and `test_grader` (94 tests, OK, 219 s), then `test_slow_drive` and
  `test_training` (34 tests, OK, 2,036 s).
- No "Created" or exited sandbox or grader containers were left after any of them.
- Dry run (`train.py start --dry-run --kind production --lr 2e-5 --real-edit --stop-after 1`, run `rt-fixes-dry`;
  fake model and backend, no W&B):
  - selection-v2 at checkpoint 0: 21.9% over 64 episodes, its 10 edit_real tasks in real edit sandboxes on the new
    images;
  - capability eval 100%;
  - one trained step, 32 episodes, 8 of them edit_real;
  - a clean operator stop.
  0 infra errors, 0 grader retries, no detector hits, no containers left.
