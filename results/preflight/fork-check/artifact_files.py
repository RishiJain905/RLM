"""$0, read-only: list the files inside the LoRA artifacts of the second seed and its fork branch."""
from pathlib import Path
from dotenv import load_dotenv
ROOT = Path(__file__).resolve().parents[3]
load_dotenv(ROOT / ".env")
import wandb
api = wandb.Api(timeout=60)
ent = api.default_entity
for ref in ("second-seed-v1:step0", "second-seed-v1:step15", "second-seed-v1-b1:step0", "second-seed-v1-b1:step15",
            "second-seed-v1-b1:step1"):
    a = api.artifact(f"{ent}/rlm-production/{ref}")
    files = [(f.name, round(f.size / 1e6, 1)) for f in a.files()]
    print(ref, a.version, round(a.size / 1e6, 1), "MB:", files, "| logged by run:", getattr(a.logged_by(), "name", None))
