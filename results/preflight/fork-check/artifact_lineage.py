"""$0, read-only: who logged each LoRA version of the second seed's branches, when, with what metadata.

Uses the wandb login already in ~/.netrc; does not read .env and prints no key.
"""
import json
import wandb

api = wandb.Api(timeout=60)
ent = api.default_entity
for coll in ("second-seed-v1", "second-seed-v1-b1"):
    try:
        vs = list(api.artifacts("lora", f"{ent}/rlm-production/{coll}"))
    except Exception as exc:
        print(coll, "not readable:", type(exc).__name__, str(exc)[:200])
        continue
    print(coll, len(vs), "versions")
    for a in sorted(vs, key=lambda a: a.created_at):
        try:
            run = a.logged_by()
            by = f"{run.name} ({run.id}, job_type={run.job_type})" if run else None
        except Exception as exc:
            by = f"? {type(exc).__name__}"
        files = [f.name for f in a.files()]
        print(f"  {a.version:>4} {a.aliases} {round(a.size / 1e6, 1)}MB created={a.created_at} by={by}")
        print(f"       files={files}")
        print(f"       metadata={json.dumps(a.metadata, default=str)[:600]}")
