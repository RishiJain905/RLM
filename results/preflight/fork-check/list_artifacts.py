"""$0, read-only: list the LoRA artifact versions of the second seed's branches on W&B. No key printed."""
from pathlib import Path
from dotenv import load_dotenv
ROOT = Path(__file__).resolve().parents[3]
load_dotenv(ROOT / ".env")
import wandb
api = wandb.Api(timeout=60)
ent = api.default_entity
for coll in ("second-seed-v1", "second-seed-v1-b1"):
    try:
        vs = list(api.artifacts("lora", f"{ent}/rlm-production/{coll}"))
    except Exception as exc:
        print(coll, "not readable:", type(exc).__name__, str(exc)[:200]); continue
    print(coll, len(vs), "versions")
    for a in vs:
        print("  ", a.version, a.aliases, a.metadata.get("step") if a.metadata else None, round(a.size / 1e6, 1), "MB", a.created_at)
