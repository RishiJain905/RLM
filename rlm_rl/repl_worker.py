"""Persistent REPL subprocess for one RLM episode.

Started by rlm_env.ReplWorker with `python repl_worker.py <context_file> <bridge_url>`.
Reads one JSON request per stdin line ({"code": ...}) and writes one JSON reply per
stdout line. Sub-model calls go back to the host over HTTP, so this process never
holds API keys.
"""

import contextlib
import io
import json
import sys
import traceback
import urllib.request

_real_stdout = sys.stdout
_bridge_url = sys.argv[2]
_state = {"submitted": False, "answer": None}


def _bridge(prompts: list[str]) -> list[str]:
    body = json.dumps({"prompts": [str(p) for p in prompts]}).encode()
    request = urllib.request.Request(_bridge_url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=600) as response:
        reply = json.loads(response.read())
    if "error" in reply:
        raise RuntimeError(reply["error"])
    return reply["outputs"]


def llm_query(prompt: str) -> str:
    """Ask the sub-model one question. It can read about 20,000 characters per call."""
    return _bridge([prompt])[0]


def llm_batch(prompts: list[str]) -> list[str]:
    """Ask the sub-model several questions in parallel."""
    return _bridge(list(prompts))


def submit(answer) -> None:
    """Record the final answer and end the episode."""
    _state["submitted"] = True
    _state["answer"] = answer


with open(sys.argv[1], encoding="utf-8") as f:
    _namespace = {
        "context": f.read(),
        "llm_query": llm_query,
        "llm_batch": llm_batch,
        "submit": submit,
        "__name__": "__repl__",
    }

for line in sys.stdin:
    code = json.loads(line)["code"]
    buffer = io.StringIO()
    error = None
    try:
        with contextlib.redirect_stdout(buffer):
            exec(compile(code, "<repl>", "exec"), _namespace)
    except BaseException:
        lines = traceback.format_exc().splitlines()
        error = "\n".join(lines[-6:])
    _real_stdout.write(
        json.dumps(
            {
                "stdout": buffer.getvalue(),
                "error": error,
                "submitted": _state["submitted"],
                "answer": None if _state["answer"] is None else str(_state["answer"]),
            }
        )
        + "\n"
    )
    _real_stdout.flush()
