# Edit environment: red-team brief

> **Step E6 of [code-editing-plan.md](code-editing-plan.md)** (§7.4). It is the gate before any W&B credit is spent on edit tasks (the baselines in step 11 and production in step 12).
>
> **What this is.** A defensive QA plan for *our own* code-editing environment: the edit sandbox, the host-side diff, the grader and the edit detectors that step 9.5 built. The question is the one step 8 asked of the read families: **can reward be earned without solving the task?** It turns the security notes the build left ([edit-build-results.md](edit-build-results.md) §5b) and the points gathered during the build into an ordered checklist. Part A is a read-only code review. Part B is a small set of live checks on Rishi's PC.
>
> **What this is not.** It is not an exploit manual. It contains no attack code, no payloads and no escape recipes, and the results file should contain none either. A finding is written as "this behaviour is possible, here is the fix, here is the regression test". A regression test uses the smallest literal case that shows the behaviour, built on the existing test fixtures.
>
> **Cost.** Everything here is **$0**: local, against the fake model (`tests/fake_model.py`) and the real sandbox and grader in Docker. Nothing calls W&B.
>
> **Status when you start:** step 9.5 is finished. Branch `claude/project-thread-9nnh3n` at `65d04ec` or later; the suite is 400 tests, 399 pass and 1 skip. The task index is `D:\RLM-data\edit\index.json` (`/mnt/d/RLM-data/edit/index.json` from WSL): 157 tasks in Python, Go and JavaScript (training 107, selection 10, test 40). Rust, TypeScript and Java wait for a later round; §9 lists what to check when they arrive.
>
> **Where this file lives.** On the PC branch `claude/project-thread-9nnh3n` as `documentation/step-9.5-red-team.md` (committed at Rishi's request, 2026-10-14). It was first written on the cloud branch as `edit-red-team-brief.md`.

Line numbers below are as of `65d04ec`; function names are the stable reference.

---

## 0. DO NOT (read first)

None of the work below needs any of these broken.

- **No W&B credit.** No `scripts/train.py ... --live`, no real model calls. If a check seems to need one, stop and ask Rishi.
- **Never open, read, print, log or echo the W&B key.** It lives in the git-ignored `.env`. Nothing here needs `.env`: don't `cat` it, load it or print `os.environ`.
- **No commit and no push without asking Rishi.** Write the results file and any new tests, show Rishi, commit when he says so.
- **Files:** everything you write goes inside the repo folder (`F:\Personal\RLM\RLM`), docs in `documentation\`. Data stays on `D:\RLM-data`. Never write large files into the repo.
- **The D: drive is Rishi's.** Never delete, move or rewrite anything on D:, never switch drives, never move Docker's disk image. No `docker system prune`, `docker image prune`, `docker builder prune` or `docker rmi`. Re-deriving an image or changing `index.json` is a data change: ask Rishi first.
- **Containers you did not start are not yours.** Never stop, remove or exec into an `rlm-grade-*` or `rlm-ep-*` container you didn't start: a run or the data build may own it. Before any live section, check that nothing else is running (§2). `scripts/sandbox_ctl.py clean` removes *every* sandbox container, so run it only when nothing else is running.
- **Few containers at a time.** D: is a slow USB hard disk. Start at most 4 containers of your own at once (the test suite starts its own; let it). A container that takes minutes to start is the drive, not a finding.
- **No network from episodes or graders.** Both run `--network none`. Don't add a network, a port, a mount or a flag to make a check easier. Containers you start for Part B also run `--network none`.
- **Never skip, disable, `xfail` or quarantine a test.** A red test is a finding. If a test is wrong, fix it and say why.
- **Don't weaken a layer to see what it blocks.** The existing tests prove each layer; re-run them.
- **Probe this repo's environment on this PC only.** No other hosts or services.
- **If a safety check stops you on any section,** don't reword around it. Do the rest, mark that section "not done", and tell Rishi which part was stopped.

---

## 1. What this proves, and how

An edit episode earns reward 1 only when its diff of `/work/repo`, taken on the host, applies to a clean copy of the base, and every fail-to-pass and pass-to-pass test passes in a fresh grader container that also holds the hidden tests and two canaries (`reward.py` `score_edit`, `grader.py` `score` :1051). So the reward-bearing surface is narrow: **the diff, the grader's run, and what the episode could read**. Everything the in-episode audit hook records only feeds metrics and the two detectors that can *zero* a reward (`escape_attempt`, `early_submit`); none can raise one.

| Area | What we confirm | Main evidence |
|---|---|---|
| A. Grading inputs | The graded diff comes only from the work tree, read safely on the host; nothing the model controls in `.git` or the file system changes it | `editbox.py` `diff` :224, `fingerprint` :246; `test_editbox.py` |
| B. Diff checks | Test files, test configuration and files a runner loads on its own are stripped or flagged | `grader.py` `check_diff` :473; `test_grader.py` |
| C. The grader's run | Code under test can't make a failing patch pass, or hide a failure as infra | `prepare` :1024, canaries, `_run_driver` :1274; `GraderTest`, `test_slow_drive.py` |
| D. What the episode can read | No hidden test, gold patch or other copy of the answer anywhere in the container | `EditReplTest`; the index and image scans in Part B |
| E. Isolation | Landlock, the container flags and the audit hook hold in edit mode | `edit_landlock_rules` :215, `container_args` :440; `EditReplTest` |
| F. Detectors and metrics | The edit detectors fire when they should; nothing reward-bearing depends on state the model can change | `detectors.py`; `test_detectors.py`, `test_edit_env.py` |

The loop for every item is step 8's:

1. **Measure** (read the code, run the named test or check).
2. **Compare** with the pass line.
3. On a pass, **record** the evidence (file:line or test name) and move on.
4. On a fail, **rate it**: reward-bearing (could turn a 0 into a 1, or a 0 into a dropped episode) or metrics-only; and how reachable it is for a policy over ~950 edit episodes. Then **harden** (a diff rule, a grader check, a routing fix, a derive-time removal), **add a regression test** that fails before and passes after, re-run the suite, and record both. A fix that changes a spec rule (§8) goes to Rishi first.

---

## 2. Before you start

Run everything inside WSL (Ubuntu) from the repo root, with Docker Desktop running, using `.venv-linux/bin/python`.

**Nothing else running.** These should print nothing (or only containers you can tie to a crashed run of your own):

```bash
docker ps -a --filter label=rlm.sandbox --format '{{.Names}} {{.Status}}'
pgrep -af 'train.py|build_edit_data.py|sandbox_ctl.py'
```

If a training run, the data build or another test run is going, stop and ask Rishi. Other WSL distros and other jobs on D: slow every container start; whether to stop them is Rishi's call, so note what you see.

**Baseline: the whole suite green before you look for anything.** It is long on this drive; run it alone.

```bash
.venv-linux/bin/python scripts/sandbox_ctl.py check
```

Expected: 400 tests, 399 pass, 1 skip (the tokenizer check, which skips in WSL). Record the counts and the time. Anything else red means fix the environment first; never proceed by skipping it. Then confirm nothing was left behind:

```bash
docker ps -a --filter label=rlm.sandbox --filter status=created --format '{{.Names}}'   # expect nothing
```

The edit tests alone, for faster reruns while you work:

```bash
.venv-linux/bin/python -m unittest -v tests.test_grader tests.test_editbox tests.test_edit_env tests.test_slow_drive \
  tests.test_repl_tools tests.test_editsynth tests.test_editdata tests.test_detectors tests.test_contract
```

---

## 3. Order of operations

1. §2 baseline.
2. **Part A** (§4), the code review, highest stakes first: A1-A4 are reward-bearing.
3. **Part B** (§5), the live checks.
4. Fixes and regression tests for what A and B found (§1 loop), then the suite again.
5. §7 results file, and tell Rishi.

---

## 4. Part A: code review (read-only)

Each item: what to confirm, where, and what a fix would look like. Record "holds" with evidence, or a finding rated per §1.

### A1. Code under test and the grader's own files *(reward-bearing; known gap 1)*

The grader's repository is a writable bind mount (`container_args` :440 with `repo_dir=grade_dir`), and the hidden tests, the two canaries and the report files (`/tmp/rlm-grade/<token>.out`) all sit where code under test, which runs as the same user inside the test process, can reach them. The protection is the canaries: a run that forges outcomes wholesale gets one canary wrong (`canary_verdict` :985).

Confirm:
- **How recognizable the canaries are, per runner.** Names come from `_names` :551: `c` plus 10 hex characters (`test_c…` in pytest, `TestC…` in Go), in a file or package named from 8 more (`test_<hex>.py`, package `rlmc<hex>`, `rlm<hex>.test.js`). The bodies are a bare pass and a bare fail (`Pytest.canary` :627 and each runner's `canary`). Could main code tell a canary from a real test by name, file or body? If so, selective forging (every outcome passed except the canaries') would evade `grader_tamper`.
- **When hidden test files are read relative to when code under test first runs.** Import-time code, a module the test files import, and in later languages build scripts, run before some test files are loaded or compiled. Can anything they do on disk change which tests run or what they assert?
- **Report files.** The runner writes its report from inside the test process. Confirm the parser treats a missing, partial or duplicated report as a failure or as `grader_tamper` (`_merge` :611, `ParserTest.test_conflicting_duplicates_are_malformed`), never as a pass.

Fix options to weigh (each is a grader change, so Rishi decides, §8): canaries named and shaped like the repository's own tests and placed inside the hidden test files, more than two of them; hidden test files and canaries mounted read-only in the grader; the host comparing the hidden files in `grade_dir` after the run with what `prepare` wrote. Rate how reachable selective forging is for an RL policy, and say whether you'd build the fix before production or rely on the Claude audit of sampled wins (plan §7.3) until then.

### A2. A failure hidden as infra *(reward-bearing)*

An infra error drops the episode from its group (reward None), so a patch that could make its own grading come back as infra would turn a 0 into "not counted". `_run_driver` :1274 raises `InfraError` when the driver never prints its nonce line, unless Docker reports the container out of memory (then it's the model's); the outer timeout is also infra.

Confirm:
- Can code under test end the grader driver, close or flood its output, or exhaust the container's processes so the driver can't report? The driver is the container's entrypoint with no `--init`, so it should be PID 1, which the kernel shields from most in-container signals; check that, and check what happens at the pids limit and when output is huge.
- Can it stall the driver past the outer timeout (`cap_s * runs + GRADER_START_SLACK_S`) without the driver's own per-command deadline firing first?
- The slow-drive rules: a timed-out run reruns once alone at twice the cap and a second timeout is the model's (`grade` :1117, `test_slow_drive.py`). Confirm no path turns the model's failure into infra or an infra failure into a 0 or a 1.

Fix pattern: once the test commands have started, a grader that ends without a report is the model's outcome (reward 0); only a container that never starts, or a host-side failure, is infra. Add a `test_slow_drive.py` or `GraderTest` case. Meanwhile the infra pause bounds it (`InfraPause`, `env.py` :410: production pauses when more than 1% of the last 10 steps' episodes were infra, 4 of 320); record the infra rate from the dry runs.

### A3. Files a runner loads on its own *(reward-bearing)*

`check_diff` :473 strips test files (`is_test_path` :249), test configuration (`is_test_config` :259, `touches_test_config` :351) and new test files, and drops binary, symlink, unsafe-path and oversize diffs (`safe_path` :210, `MAX_DIFF_LINES` :46). For this round's languages, list every file a runner or interpreter loads without being named as a test, and confirm each is stripped or is ordinary main code:
- Python: `conftest.py` anywhere (covered), `sitecustomize.py` and `usercustomize.py`, `.pth` files, `pytest` plugin entry points in `setup.cfg` or `pyproject.toml`, `__init__.py` in a test folder.
- Go: `testdata/` (covered); `init()` in a package's non-test files is main code and runs in the test binary, so it falls under A1.
- JavaScript: Jest, Vitest and Mocha configs, `.babelrc` and `babel.config.*`, the `package.json` test keys (covered per §2.2 of the results doc), `setupFiles` targets, `.npmrc`.
- The pattern gap from the build: a helper with a non-test name that only test files import. Is a change to it kept? (It's main code by path; decide whether that matters.)

Fix pattern: add the path to the language's test-configuration list, with a `CheckDiffTest` case.

### A4. The graded diff and host-side reads *(reward-bearing)*

The host never runs git against the episode's `.git`: `diff` :224 uses a throwaway bare git directory that borrows the host cache's objects, `git_env` :124 and `GIT_SAFE` :133, and `--no-ext-diff --no-textconv --no-renames --full-index` (`test_the_host_diff_ignores_the_episodes_git`). Confirm:
- Nothing else on the host reads the episode's `.git`, and nothing host-side follows a symlink the model planted in `/work/repo`: `fingerprint` :246 walks with `os.walk` and `lstat`; check the diff, `_safe_apply` :1011, and episode clean-up do the same.
- A model `.gitignore` or `.gitattributes` in the work tree can only hide or binary-mark its own changes (its loss, §5b of the results doc), never a test file's restoration or another file's content.
- The host cache (`_commit_base` :191) is never mounted into an episode (`materialize` :218 copies it).

### A5. `test_aware` *(known gap 2)*

`TEST_AWARE` :432 is a static pattern list over the diff's added lines, per language; `AUDIT_SIGNALS` only go to the audit queue. An indirect runtime check passes it; the canaries (A1) are the backstop. Confirm the list matches spec §12.6 for Python, Go and JavaScript, and that skip markers are caught (`SKIP_MARKERS`). Don't try to make the list complete: record the gap, list what the Claude audit should look for in edit wins, and leave any new rule to Rishi (§8).

### A6. The audit hook and the coding tools *(known gaps 3 and 4; metrics-only expected)*

The edit-mode hook (`_EDIT_HANDLERS` :543) runs in the REPL process only and can be switched off from inside; programs the model starts have no hook (§5b of the results doc). The coding tools (`repl_tools.py`, loaded by `load_tools` :752) can be rebound or monkeypatched. Confirm the claim that all of this is metrics-only:
- `score_edit` and the grader read nothing the episode container produced except the work tree.
- Event-based detectors (`early_submit` :94, `escape_attempt` :181) can only zero a reward, so hiding events can only avoid a zero that Landlock and `--network none` make moot.
- `network_try` (:207) is logged only.

Record which metrics a policy could distort (`tool_calls`, `edit_errors`, `raw_io_share`, `test_runs`) so whoever reads the dashboard knows they're best effort.

### A7. Episode isolation in edit mode

`edit_landlock_rules` :215: reads and runs everywhere, writes only under `/work/repo`, `/scratch`, `/tmp` and the image's cache folders (`/opt/rlm-cache`; `/opt/rlm-deps` for JavaScript). `container_args` :440: `--network none`, `--ipc none`, `--cap-drop ALL`, `no-new-privileges`, the host's non-root uid, memory, CPU and pids limits, tmpfs `/scratch` and `/tmp`, `/episode` and `/opt/rlm` read-only. Confirm against `EditReplTest`, and confirm that changes under the writable cache folders never reach the grader (it starts from the image, and only the `/work/repo` diff crosses).

### A8. Hidden tests that stay partly visible

- **Go:** an episode keeps each hidden test file without its hidden test functions; the grader writes the whole file back (`prepare` :1024, the record's `replace`). Confirm a model function with the same name only breaks the build (reward 0).
- **JavaScript:** shared spec helpers are no longer hidden. Confirm the visible helpers carry no hidden test's expected values.
- **Python:** confirm hidden test files are absent from the episode tree (`_build_workspace` in `tasks/editreal.py` :130, the record's `remove`).

---

## 5. Part B: live checks ($0, a few containers at a time)

### B1. The index, with no containers

A short read-only script over `index.json` (keep it in the repo if it's worth rerunning; it reads the data, never writes it). For every task, confirm:
- at least one fail-to-pass test (`tests.f2p`), and no `f2p` or `p2p` key that looks like a canary (`c` plus 10 hex characters, in either case, as in A1); the canary bug fixed in `3555b2b` made such tasks unsolvable or left them with no real test;
- the issue text names no hidden test and quotes no gold-patch line (`editdata.leaks` :500 does this at build time; re-run it over the index);
- no hidden test path is in the episode's file listing;
- selection and test repositories never appear in the training split;
- every training record's licence is permissive and its source is allowed for training (`tasks.check_mix`).

Record the counts per split and language, and any task that fails. Don't edit the index: list the task ids for Rishi.

### B2. No other copy of the answer in the images *(reward-bearing)*

`test_no_history_to_mine` and `test_hidden_tests_never_enter_the_container` check git history and hidden tests. They don't check for **another copy of the repository's own code** that a program in the container could read: an installed release of the package in site-packages, a copy in `node_modules` pulled in as some tool's dependency (likely for widely used packages such as `async` or `dayjs`), the same module in the Go module cache, or built output (`dist/`, `.pyc`). A clean copy holds the unbroken function for a bug task and the full body for a feature task.

For each training repository's derived image (`rlm-edit-<repo>:<hash>`; 6 Python, 6 Go, 3 JavaScript) and the images of the 2 Go real PRs in the training split, start one throwaway container from it (`--rm --network none --user <your uid>`, a name of your own, at most 4 at once) and search the file system, outside `/work/repo`, for files with the repository's package or module name, and compare a few against the snapshot. Record per repository: copy found or not, where, and whether it holds a gold version for any indexed task. The selection and test repositories matter too (a copy there flatters both the base and the trained model); check them after the training ones.

Fix pattern: remove the copy in `derive` (as it already removes the checkout and SWE-smith's task branches) with a `DeriveTest` case, or drop the affected tasks. Both change data on D:, so Rishi decides; list the repositories and task counts.

### B3. A dry run with real edit episodes

```bash
.venv-linux/bin/python scripts/train.py start --run rt-edit-dry --dry-run --kind production --lr 2e-5 --real-edit --stop-after 1
```

Expected, as in the build's `dry-edit-final3`: the selection eval at checkpoint 0, the capability eval, one trained step and a clean stop; 0 infra, 0 grader reruns, the step's `edit` block filled. Results go under `results/dry/` (git-ignored). Then:

```bash
docker ps -a --filter label=rlm.sandbox --filter status=created --format '{{.Names}}'   # expect nothing
```

A "Created" container left behind is the known follow-up from the build (a launch the daemon takes longer to create than the second removal pass); record it with the run id, and propose the sweep by the `rlm.run` label rather than removing other runs' containers.

### B4. The edit tests in the real sandbox

If the baseline ran them green, record that. If you change anything in A or B, rerun the edit test list from §2 and the classes that use Docker (`EditReplTest`, `GraderTest`, `ParallelTest`, `DockerEditTest`, `SlowStartTest`) before the full suite.

---

## 6. Done when

1. The full suite is green (399 pass, 1 skip) on the final code.
2. Every Part A item reads "holds" with evidence, or is a finding rated per §1.
3. Every reward-bearing finding has a fix and a regression test that fails before and passes after, or is escalated to Rishi because the fix changes a spec rule.
4. B1 and B2 are clean, or their task and repository lists are with Rishi. B3 had 0 infra and left no "Created" containers.
5. No credit spent, `.env` never opened, nothing on D: changed, nothing committed until Rishi says so.

In the plan's words: no working way to earn reward without solving the task is left, each closed path backed by a regression test.

---

## 7. Results template

Write `documentation/step-9.5-red-team-results.md`. Keep any transcripts or grader output under `results/` (git-ignored). No attack code in the file.

```markdown
# Step 9.5 red-team results

> Step E6 of code-editing-plan.md. Run <date> on Rishi's PC, WSL + Docker.
> No W&B credit. .env never opened. Nothing on D: changed. Nothing committed until Rishi approves.

## Summary
- Suite: <pass>/<total>, <skip> skipped, <time>. Baseline green: yes/no.
- Findings: <n> (reward-bearing <n>, metrics-only <n>); fixed with regression tests <n>; escalated <n>.
- Verdict: gate for edit spend open / not open, and why.

## Part A
| Item | Holds / finding | Evidence (file:line, test) | Rating | Fix or proposal |
(A1 canaries and the grader's files, A2 infra, A3 auto-loaded files, A4 host diff, A5 test_aware,
 A6 audit hook and tools, A7 isolation, A8 partly visible tests)

## Part B
- B1 index: tasks checked per split and language; failures: <ids or none>.
- B2 answer copies: per repository, found / not found, where, tasks affected.
- B3 dry run: run id, infra <n>, grader reruns <n>, Created containers left <n>.
- B4 Docker edit tests: <counts>.

## Findings and fixes
| # | Area | Behaviour (plain words) | Fix | Regression test | Status |

## Accepted residual items
- <e.g. test_aware is a tripwire; metrics the model can distort>

## Decisions for Rishi
- <canary redesign? read-only hidden tests? infra routing? derive changes or dropped tasks?>
```

---

## 8. Escalate to Rishi (don't decide alone)

- Any change to how canaries are named, shaped or placed, or any new grader check (spec §12.5-§12.6).
- Any new or changed detector rule, including `test_aware` patterns.
- Changing what counts as infra versus the model's outcome.
- Re-deriving images, removing tasks or anything else that changes data on D: or `index.json`.
- Anything that would need a live model call or credit.

---

## 9. When Rust, TypeScript and Java join (later round)

Not in this round's index; check these when their tasks are added:
- **Java:** JsonPath's image was derived from a copy warmed once *with network* (`warm_image`). Confirm episodes and graders of every Java image still run `--network none`, and that the Gradle and Maven caches hold no hidden tests, no answers and no built copy of the project itself (for example a snapshot jar of the repository). Maven and Gradle build sections can run plugins: confirm `touches_test_config` covers what runs around the tests, not only Surefire and Failsafe.
- **Rust:** hidden `src/**/tests.rs` modules stay as empty modules and the grader writes the real ones back; confirm a model edit to the empty module is overwritten, and that a deleted file with an `append` block fails (reward 0) rather than skipping the hidden tests. Confirm a `build.rs` change is handled (A1, A3).
- **TypeScript:** the same spec-helper check as JavaScript (A8), and react-hook-form's own Jest config stays test configuration.
- **All three:** the `test_aware` patterns (A5), the report parsers (`cargo` lines, Maven and Gradle JUnit; `ParserTest`) on real tasks, B1 and B2 for their repositories, and every shipped task with at least one real fail-to-pass test.
