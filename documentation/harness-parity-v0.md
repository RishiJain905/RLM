# Harness parity v0

Written by `scripts/harness_parity.py --write-summary` on 2026-11-01. It replays recorded v4 episodes through the rlm-harness engine (step 8 of its v0 build) and compares every root request with the messages the trained model saw. The full report, with the diffs, is in `results/harness-parity/2026-11-01/` (git-ignored: it holds task text).

- date: 2026-11-01
- harness commit: 807eca561ca0bae0b8fa054e43b65250d650f69a
- RLM commit: 7a10b01ca4ddae26c31c6895682ff62b4c01a01a
- engine: 0.1.1, python 3.12.0
- OS: Windows-11-10.0.26300-SP0
- git: git version 2.53.0.windows.2
- shell: Git Bash (F:\Personal\Git\bin\bash.exe)
- permission mode: accept-edits

- episodes: 5 (read 3, edit 2)
- requests compared: 72
- byte-for-byte requests: 53 of 72
- accepted differences: python-version: 1
- unexplained differences: 12
- failures: none
- replaced: 0a2b4a7e0eca5904: toolchain, 14084d8bb91854b7: toolchain, 3b92aeef03f8dc12: toolchain, 4ebb4f42a9d95314: toolchain, 6b8b3e6c00e975cd: toolchain, 2c4d4ffb2d661e87: toolchain
- skipped: flags: 19, hacks: 7, infra_error: 46, limits: 158, pace_unknown: 6, prompt_version: 832, seed_not_recorded: 285, timeouts: 9
- digest: 7b75823190da
- verdict: FAIL

53 of 72 root requests matched byte for byte; 12 differences are unexplained and the failures above stand, so this run failed. A `--strict` run exits 1 whenever that count is short of the total.

## Episodes

| episode | run | family | mode | profile | turns | sub-calls | requests matched | verdict |
|---|---|---|---|---|---|---|---|---|
| 32ef101070fcdaa8 | prod-v2 | aggregation | read | training-v4 | 10 | 54 | 10 of 10 | PASS |
| 398c0696523fddd8 | prod-v2 | research_select | read | training-v4 | 4 | 0 | 4 of 4 | PASS |
| 2cfec3a9ceaef5ca | prod-v2 | aggregation | read | training-v4 | 13 | 80 | 13 of 13 | PASS |
| 0a2b4a7e0eca5904 | prod-v2 | edit_spec | repo | training-v4 | 23 | 0 | 5 of 23 | replaced(toolchain) |
| 14084d8bb91854b7 | prod-v2 | edit_spec | repo | training-v4 | 18 | 0 | 1 of 18 | replaced(toolchain) |
| 3b92aeef03f8dc12 | prod-v2 | edit_spec | repo | training-v4 | 28 | 0 | 10 of 28 | replaced(toolchain) |
| 4ebb4f42a9d95314 | prod-v2 | edit_spec | repo | training-v4 | 23 | 0 | 7 of 23 | replaced(toolchain) |
| 6b8b3e6c00e975cd | prod-v2 | edit_spec | repo | training-v4 | 21 | 0 | 1 of 21 | replaced(toolchain) |
| f37a7b19e6c7d399 | prod-v2 | edit_spec | repo | training-v4 | 29 | 0 | 10 of 29 | FAIL |
| 2c4d4ffb2d661e87 | prod-v2 | edit_bugs | repo | training-v4 | 24 | 0 | 1 of 24 | replaced(toolchain) |
| 91e79ef444700cc5 | prod-v2 | edit_bugs | repo | training-v4 | 16 | 0 | 16 of 16 | PASS |

## Difference classes

| class | differences | accepted | reason |
|---|---|---|---|
| path-form | 1 | yes | Yes, if a cell prints os.getcwd() or an absolute path. Tools return relative paths, and /work/repo/... still works through the alias (3a). Accepted only in text the cell's own code printed (it computes a path), never in a tool's return value. |
| python-version | 1 | yes | No. Docker had 3.12.15 and uv's 3.12 is another patch release. Patch releases keep the language the cells use, and the string shows only if a cell prints `sys.version` (its build string with it). |
| timing | 2 | yes | No. The same cell prints different times on two runs in training too, so no behaviour can depend on the value. Times are masked, and pytest's `=` padding, whose width follows the time's length; test counts stay. |
| traceback-path | 2 | yes | Only inside the last lines of a traceback of an error raised in a harness function. Training showed /opt/rlm/repl_worker.py and /opt/rlm/repl_tools.py. The file is not part of the task and the name doesn't change what was raised. |
| unexplained | 57 | no | not accepted |

## Engine fixes

- the worker names its own files in a traceback as training did, /opt/rlm/repl_worker.py and /opt/rlm/repl_tools.py, never the install folder (D-109); line numbers stay its own (rlm-harness d0e7a1c; engine/tests/test_worker.py::test_traceback_names_training_files)
- shell() maps /work/repo in the command to the workspace, as the tools do (D-150); training's sandbox had that folder, so `cd /work/repo && ...` failed natively (rlm-harness 807eca5; engine/tests/test_tools.py::test_shell_maps_the_repo_alias)
- on Windows /tmp is one folder (<scratch>/tmp) for the file tools and the shell, as training's one /tmp was (D-150) (rlm-harness 807eca5; engine/tests/test_tools.py::test_tmp_is_one_folder_for_the_tools_and_the_shell)
- on Windows the shell's \r\n reaches the model as \n, as on training's Linux (D-150) (rlm-harness 807eca5; engine/tests/test_tools.py::test_shell_lines_end_as_in_training)
- the model's commands no longer inherit the engine's PYTHONUNBUFFERED, which put a child Python's traceback after its buffered prints (D-150) (rlm-harness 807eca5; engine/tests/test_worker.py::test_env_strips_keys_and_temp_is_scratch)
