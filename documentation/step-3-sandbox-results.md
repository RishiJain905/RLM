# Step 3 results: sandbox and bridge

> **Step 3 of [production-build-plan.md](production-build-plan.md)** (holes H2, H3, H4 and H10). Built and tested 2026-10-11 on Rishi's PC, inside WSL2 (Ubuntu) with Docker Desktop 4.93 (engine 29.8.1). Decisions applied: 3a a fresh container per episode, 3b WSL2 with a Unix socket and `--network none`, 3c the standard library only.
>
> **No W&B credit was used.** The tests serve sub-calls from a fake model on the host, so nothing called W&B. The `.env` file was never opened. Nothing has been committed or pushed. The one download was the `python:3.12-slim` base image, approved by Rishi.
>
> Anything marked *(mine)* is my choice, not the spec's. Anything marked *(inferred)* hasn't been checked against the live system.

## Summary

Every done-when check in the plan passes except the H10 prompt diff, which is built but has nothing to diff yet. The regression suite is 52 tests, and all of them pass in WSL2 in about 35 seconds (`python3 scripts/sandbox_ctl.py check`).

| Done-when check | Result | Evidence |
|---|---|---|
| Opening `.env`, `.art/`, the repo, home and `/proc` fails from inside | **Pass** | `FilesTest.test_host_files_and_proc_are_unreadable` (27 paths), `LayersTest` |
| Writing outside scratch fails | **Pass** | `FilesTest.test_writes_only_in_scratch` (12 paths, plus mkdir, remove, rename) |
| `pip install`, `curl` and DNS fail | **Pass** | `NetworkTest.test_no_network` (14 attempts, including IPv6, UDP, HTTPS and the Docker socket) |
| Importing any project module fails | **Pass** | `ImportsAndSecretsTest.test_project_modules_are_not_importable` (15 modules, two spellings each) |
| Bridge calls work | **Pass** | `NetworkTest.test_bridge_is_the_way_out`, `ReplTest`, `ScaleTest` |
| No seed is visible inside | **Pass** | `ImportsAndSecretsTest.test_no_keys_or_seed_inside` |
| The environment and the harness import the same bridge module, and a test diffs their system prompts | **Built, nothing to diff yet** | `tests/test_contract.py`. `env.py` arrives in step 5 and the harness in Part 2. Until then the test pins the prompt to the spec text and fails if any other module defines its own. |
| No environment variables or keys inside (from the brief) | **Pass** | The REPL sees only `PATH`, `LANG`, `HOME` and `TMPDIR`. Canary keys set on the host never appear. |
| One container per episode, cleaned up (3a) | **Pass** | Every test checks that no container or episode directory is left. 32 episodes run at once in 3-7 s. |
| Failures surface as infra errors, not wrong answers (H5) | **Pass** | `FailureTest`, 9 cases |

**Nothing is needed from Rishi to use it.** Docker Desktop has to be running, with the WSL integration for Ubuntu left on.

---

## 1. What was built

| Path | What it is |
|---|---|
| `rlm_rl/bridge.py` | Bridge API v1 (`context`, `llm`, `llm_batch`, `submit`), the pinned prompt text and `PROMPT_HASH`, the episode limits, the wire protocol, and `EpisodeBridge`, the host end of one episode's socket |
| `rlm_rl/repl_worker.py` | Runs inside the container: a PID 1 supervisor plus the REPL. Standard library only, and it imports nothing from the project |
| `rlm_rl/sandbox.py` | `Sandbox`: one container per episode, cell timeouts, crash and infra handling, the image build and stale-container cleanup |
| `rlm_rl/tasks/__init__.py` | Host-only task generation helpers: `new_seed()` (random 64-bit) and private eval seed lists |
| `scripts/sandbox_ctl.py` | `build` the image, `check` (run the tests), `clean` up after a crash |
| `tests/` | `test_bridge.py`, `test_contract.py`, `test_sandbox.py`, `test_tasks.py`: the regression suite for every later change |
| `.gitignore` | Adds `private/`, where eval seed lists go |
| `README.md` | The Layout section now lists the production code and the archive |

How the environment uses it (step 5):

```python
async with Sandbox(task.context, complete=client.subcall, run_id=run_id) as box:
    result = await box.run_cell(code)   # CellResult, or raises InfraError
```

`complete` is the async function that serves one sub-call. Step 4's `client.py` provides it, with the shared limiter and retries. It raises `InfraError` when retries run out and `SubcallError` for a request the API rejects.

---

## 2. How the sandbox works

### 2.1 The container (3a, 3b, 3c)

Each episode gets a fresh container from `rlm-sandbox:<hash>`. The image is `python:3.12-slim` pinned by digest (`sha256:a6e34c59…`, Python 3.12.15), with pip removed, a `sandbox` user (uid 10001) and the worker at `/opt/rlm`. The image tag is a hash of the Dockerfile and the worker, so a change to either builds a new image. The build runs with `--network none`.

| Flag | Why |
|---|---|
| `--network none`, `--ipc none` | Loopback only, no `/dev/shm` |
| `--read-only`, `--tmpfs /scratch` (64 MB, `noexec`) | One writable directory |
| `--user 10001`, `--cap-drop ALL`, `--security-opt no-new-privileges` | Not root, no capabilities |
| `--cpus 1`, `--memory 512m` (no swap), `--pids-limit 64` | Resource limits |
| `--mount` of the episode directory at `/episode`, read-only | Holds `context.txt` and `bridge.sock`, nothing else |
| `--hostname sandbox`, `--log-driver none`, `env -i` entrypoint | Nothing about the host or the task leaks in, and the REPL's environment is empty apart from `PATH`, `LANG`, `HOME` and `TMPDIR` |

The episode directory lives under `/tmp/rlm-sandbox-<uid>/` in Ubuntu, which is mode 0700. Each episode's directory name is random, and the container name is random too, so neither carries a seed.

### 2.2 Three layers

The container is the boundary. Two more layers sit inside it, and `LayersTest` checks each one on its own.

| Layer | What it stops | Checked without the other layers |
|---|---|---|
| **Container** | Network (no route, no DNS), host files (none are mounted), writes outside `/scratch` and `/episode` (read-only root), keys (empty environment) | TCP: `Network is unreachable`. DNS: `Temporary failure in name resolution`. Writes: `Read-only file system`. Repo path: `No such file`. |
| **Landlock** (kernel, ABI 3) | Reads outside `/usr`, `/etc`, `/opt/rlm`, `/episode` and `/scratch`, so `/proc`, `/sys` and `/dev` are unreadable. Writes outside `/scratch`. Running any program at all. | `/proc/self/environ` and `/proc`: `Permission denied`. Running `/bin/sh` or `python3`: `Permission denied`. |
| **Audit hook** (Python) | Blocks and logs network, file, process, signal and ctypes attempts, and logs imports of host modules | Every attempt in the probes became a logged event and a readable `PermissionError` |

The REPL applies Landlock to itself before any model code runs. The Windows WSL kernel (6.6.87.2) has Landlock compiled in and enabled, and Docker's seccomp profile allows it. The sandbox refuses to start without it (`require_landlock=True`), so a future kernel without Landlock turns every episode into an infra error rather than running with less protection.

The audit hook writes each blocked attempt to stderr as `@@rlm-audit {json}`. `Sandbox.events` collects them for the `escape_attempt` detector in step 5 (§5 below lists the event kinds). If model code redirects or closes stderr to hide attempts, the next attempt kills the REPL with exit code 86, which is logged as a `tamper` event.

### 2.3 The bridge

- Each episode has its own Unix socket, served by `EpisodeBridge` on the host. It accepts `{"op": "llm" | "llm_batch", "prompts": [...]}` and nothing else, as spec §1.4 says. `submit` comes back with the cell result.
- The 30-call budget, the 24,000-character prompt cut and the request size limit are all enforced on the host, because the REPL is untrusted.
- Malformed or unknown requests get an error and are recorded in `bridge.rejected`. The REPL never sends one, so a rejection means model code wrote to the socket directly.
- Time spent waiting on sub-calls is measured, and it doesn't count toward the 120 s cell timeout (spec §4). A test with a 2 s timeout and a 3 s sub-call passes.
- Lone surrogates in prompts or printed output are replaced on the host, so model code can't produce text that later fails to encode and turns into an infra error.

### 2.4 Failures: the model's outcome or infra (H5)

PID 1 in the container is a small supervisor. It ignores every signal it can, forks the REPL and waits. When the REPL dies, the supervisor reports how and exits. Model code can't signal PID 1, so every REPL death is the model's doing, whatever exit code it fakes. If the container stops without that report, something outside stopped it.

| What happens | Outcome | Test |
|---|---|---|
| Cell runs over 120 s of compute | Container restarted, model told, variables lost | `test_timeout_restarts_and_tells_the_model` |
| REPL exits, segfaults, runs out of memory, or kills itself (`os._exit(137)`, `raise_signal(SIGKILL)`) | Model's outcome: restarted and told | `test_model_crashes_are_not_infra`, `test_self_inflicted_kills_are_the_models` |
| A thread the model left running kills the REPL between cells | Model's outcome; the next cell isn't run and the model is told | same |
| `sys.exit()` or `input()` in a cell | A normal error, no restart | `test_stdin_and_exit_do_not_hang_or_crash` |
| A sub-call fails after retries | `InfraError`; the REPL is never shown a fake answer | `test_subcall_infra_error_discards_the_episode` |
| A bug in the host's sub-call code | `InfraError` | `test_host_bug_in_a_subcall_is_infra` |
| The container can't start | `InfraError` | `test_container_that_cannot_start_is_infra` |
| The container is killed from outside, idle or mid-cell | `InfraError` | `test_killed_from_outside_is_infra` |
| A cell waits more than 30 minutes on sub-calls *(mine)* | `InfraError` | (a safety cap, not tested) |

What the model is told after a timeout or crash is pinned in `bridge.py` and included in the prompt hash (see §5, item 2).

### 2.5 Host-only task generation (H3)

A `Sandbox` takes the context text and nothing else, so generators, seeds, gold answers and graders can't reach it. `rlm_rl/tasks.new_seed()` gives a random 64-bit seed per task. Eval seed lists go in `private/seeds/`, which is git-ignored. The seed test builds a task from a fresh seed, then searches everything the REPL can see (environment, argv, hostname, working directory, `/episode`, globals) for the seed in decimal and hex, and finds nothing.

---

## 3. How it was tested

Run from the repo root inside Ubuntu:

```bash
python3 scripts/sandbox_ctl.py check
```

| File | Tests | Needs Docker |
|---|---|---|
| `tests/test_bridge.py` | 19: the prompt text against the spec, the observation format, the hash, and the socket server (budget, order, truncation, rejected requests, infra errors, blocked time) | No |
| `tests/test_contract.py` | 7: the worker matches the bridge (constants, API names, truncation, answer types), there's one system prompt, and the consumer diff | No |
| `tests/test_tasks.py` | 3: seeds, private seed lists, `private/` is git-ignored | No |
| `tests/test_sandbox.py` | 23: network, files, imports, keys and seed, standard-library use, the bridge in the REPL, failures, the three layers, and 32 parallel episodes | Yes |

- **Probes never print what they read.** An attempt that works is recorded as `ALLOWED` and fails the test.
- **Normal work leaves no trace.** One test runs `re`, `json`, `collections`, `statistics`, `ast`, `decimal`, `zoneinfo`, `sqlite3`, `uuid`, `tempfile`, compression modules and a thread pool calling `llm`. It checks that none of it produces an audit event, so the `escape_attempt` detector won't fire on honest code.
- **The first container run caught one bug.** `importlib.import_module("rlm_rl")` raised no audit event, so host-import attempts written that way weren't logged. An import watcher on `sys.meta_path` now logs every spelling, and the test tries both.
- **Speed.** A container starts in about 0.6-0.7 s when 8 start together. A full step's 32 episodes take 3-7 s in total (start median 1.9-3.5 s). That's small next to the 150-390 s steps of the 20-step run.
- **Flakiness.** The full suite passed on four consecutive runs.

---

## 4. Files a later step will touch

- **Step 4** writes `rlm_rl/client.py` and passes its sub-call function to `Sandbox(complete=...)`. The cell-timeout half of H5 (queue time excluded) is already done here.
- **Step 5** writes `rlm_rl/env.py` on top of `Sandbox.run_cell`, `bridge.initial_messages` and `bridge.observation`, and adds `rlm_rl.env` to the consumer diff (it's already listed). The detectors read `Sandbox.events`, `bridge.rejected` and `Sandbox.timeouts`.
- **Step 8** (red-team) starts from `tests/test_sandbox.py` and the known limits in §6.

---

## 5. What changes the spec or plan

For the planning thread, which owns both files.

1. **Plan step 3 says the socket accepts `llm`, `llm_batch` and `submit`.** Spec §1.4 says `submit` never crosses the bridge, and that's what was built. The plan's wording should drop `submit`.
2. **Two new model-visible texts belong in spec §2.3.** They're in `bridge.py` and in `PROMPT_HASH`:
   - Timeout: `The cell timed out after 120 seconds of compute. The REPL was restarted and all variables were lost; `context` and your sub-call budget are unchanged.`
   - Crash: `The REPL crashed ({reason}). It was restarted and all variables were lost; `context` and your sub-call budget are unchanged.` When the REPL died between cells, ` This cell was not run.` follows. `{reason}` is `exit code N`, `killed by SIGNAL`, `out of memory: the 512 MB limit was reached`, or `the sandbox log was tampered with`.
3. **`submit` details for spec §1.2** *(mine)*:
   - A tuple or set becomes a list, so a set of hosts isn't turned into a string. Sets are sorted.
   - `True` and `False` become strings, since spec §1.2 doesn't list booleans.
   - An answer over 100,000 characters as JSON raises an error in the REPL and isn't submitted.
   - `llm_batch("one string")` raises a `TypeError` instead of sending one sub-call per character.
4. **Spec §1.2's "kernel crash not caused by model code" now has a concrete rule.** A REPL death is always the model's. A container that stops without the supervisor's report is infra, unless Docker says it ran out of memory.
5. **Spec §2.1: the published spec page has the stdlib line, and the `.md` copy doesn't yet.** The code and `tests/test_bridge.py` pin the published text, with `Only the Python standard library is available.` as its own paragraph before "You have at most 12 turns". The `.md` copy's §1.3 also still says a cap of 8; the coordinator flagged both as known.
6. **The environment runs inside WSL2 from here on (3b).** Steps 4-6 need a Linux virtual environment with ART 0.5.20 and the rest of `requirements.txt`, because `.venv` is a Windows one. Ubuntu's Python is 3.12.3 and `python3 -m venv` works there. The sandbox runs Python 3.12.15 and the Windows `.venv` 3.12.0, all with the same standard library, which is what 3c asks the harness to match.
7. **Spec §10's code layout gains** `tests/`, `scripts/sandbox_ctl.py` and the seed helpers in `rlm_rl/tasks/__init__.py`. Eval seed lists go in the git-ignored `private/`.
8. **Plan 3a's start-up estimate** of "a second or two" is about right: 0.6-3.5 s depending on how many start at once.
9. **Event kinds for the `escape_attempt` detector (step 5).** My suggestion *(mine)*: `network`, `file`, `process`, `ctypes`, `signal`, `tamper` and `host_import` fire it, and `nonstdlib_import` (for example `import numpy`, which simply fails) doesn't, since that's a mistake rather than an escape. Any `bridge.rejected` entry should fire it too.
10. **Spec §1.2's request limits** gain an 8 MB cap on a single bridge request *(mine)*.

---

## 6. Known limits, for the red-team (step 8)

- **The audit hook is a tripwire, not a wall.** `_posixsubprocess.fork_exec` forks without an audit event, so that attempt isn't logged. Landlock still stops the exec (the child exits with code 255 and runs nothing), and the pids limit caps forking.
- **`signal.raise_signal` has no audit event.** A REPL that kills itself with it is still attributed to the model, because the supervisor reports the death.
- **Model code can write to the protocol pipe** and fake a result or a death report. Every in-band message is treated as the model's outcome, so this can only hurt its own episode, never mark it as infra.
- **`/etc` is readable at the Landlock layer**, because the dynamic loader and libc need it. The audit hook blocks Python-level reads of it. It holds nothing from the host beyond Docker's `hosts`, `hostname` and `resolv.conf`.
- **The worker source at `/opt/rlm/repl_worker.py` is readable.** It holds no secrets, and tracebacks show its lines anyway.
- **Landlock ABI 3 doesn't cover connecting to Unix sockets.** The only socket in the container is the bridge, which is meant to be reachable.
- **Docker Desktop must be running** with WSL integration on for Ubuntu. If it stops mid-run, every episode becomes an infra error and the >1% rule pauses the run.

---

## 7. Commands

```bash
python3 scripts/sandbox_ctl.py build    # build the sandbox image (needs the pinned base image)
python3 scripts/sandbox_ctl.py check    # run all 52 tests
python3 scripts/sandbox_ctl.py clean    # remove containers and episode directories left by a crash
```
