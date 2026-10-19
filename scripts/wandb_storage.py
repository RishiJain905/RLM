"""See and free W&B artifact storage (2026-10-19: the account held 68.6 GB against Serverless RL's ~5 GB included).

Run inside WSL from the repo root. The W&B key is read from .env and never printed.

    .venv-linux/bin/python scripts/wandb_storage.py list                       # every project, read-only
    .venv-linux/bin/python scripts/wandb_storage.py list rlm-production        # one project's collections
    .venv-linux/bin/python scripts/wandb_storage.py delete rlm-production prod-v1-b1 prod-v1-b2 prod-v1-b3

`delete` removes every version of the named models' LoRA and train-state collections (`<model>` and
`<model>-train-state`) in one project. It prints exactly what it will delete and asks you to type the
project name before deleting anything. Deleted artifacts can't be recovered. prod-v1 is refused: prod-v2
serves its base model from prod-v1:step0 (--base-inference).
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENTITY = "models-ontario-tech-university"
PROTECTED = {("rlm-production", "prod-v1"), ("rlm-production", "prod-v2")}


def api():
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")  # provides WANDB_API_KEY; never printed
    import wandb

    return wandb.Api(api_key=os.environ["WANDB_API_KEY"], timeout=60)


def collections(a, project: str):
    for t in a.artifact_types(f"{ENTITY}/{project}"):
        for col in a.artifact_collections(f"{ENTITY}/{project}", t.name):
            versions = list(col.artifacts())
            yield t.name, col.name, versions, sum(v.size or 0 for v in versions)


def cmd_list(projects: list[str]) -> None:
    a = api()
    names = projects or [p.name for p in a.projects(ENTITY)]
    grand = 0
    for project in names:
        total = 0
        for tname, cname, versions, size in collections(a, project):
            total += size
            if projects and size > 1e7:
                print(f"  {project} [{tname}] {cname}: {len(versions)} versions, {size / 1e9:.2f} GB")
        grand += total
        print(f"{project}: {total / 1e9:.2f} GB")
    print(f"total: {grand / 1e9:.2f} GB")


def cmd_delete(project: str, models: list[str]) -> None:
    refused = [m for m in models if (project, m) in PROTECTED]
    if refused:
        sys.exit(f"refusing {refused}: prod-v2 trains on its own model and serves its base from prod-v1:step0")
    a = api()
    wanted = {name for m in models for name in (m, f"{m}-train-state")}
    plan = [(t, c, v, s) for t, c, v, s in collections(a, project) if c in wanted]
    if not plan:
        sys.exit("nothing matched")
    for tname, cname, versions, size in plan:
        print(f"  {project} [{tname}] {cname}: {len(versions)} versions, {size / 1e9:.2f} GB")
    print(f"{sum(p[3] for p in plan) / 1e9:.2f} GB in {sum(len(p[2]) for p in plan)} versions. This can't be undone.")
    if input(f"Type {project} to delete these: ").strip() != project:
        sys.exit("not deleted")
    for tname, cname, versions, _ in plan:
        for v in versions:
            v.delete(delete_aliases=True)
        print(f"deleted {cname} ({len(versions)} versions)")


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in ("list", "delete"):
        sys.exit(__doc__)
    if sys.argv[1] == "list":
        cmd_list(sys.argv[2:])
    elif len(sys.argv) < 4:
        sys.exit("delete needs a project and at least one model name")
    else:
        cmd_delete(sys.argv[2], sys.argv[3:])


if __name__ == "__main__":
    main()
