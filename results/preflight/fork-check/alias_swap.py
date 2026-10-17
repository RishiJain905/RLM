"""Put one checkpoint's LoRA under another model's step alias (W&B artifacts only; no training, $0).

    alias_swap.py <source model:stepN> <dest model> <dest alias>

Tests whether W&B's training service loads the newest step's weights from the artifact alias: log
second-seed-v1:step15 into second-seed-v1-b1 as `step1`, then train one step on b1 and compare.
The WANDB_API_KEY comes from .env through load_dotenv and is never printed.
"""
import sys
import tempfile
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[3]
load_dotenv(ROOT / ".env")
import wandb  # noqa: E402

src_ref, dest, alias = sys.argv[1:4]
api = wandb.Api(timeout=120)
ent = api.default_entity
before = {a.version: a.aliases for a in api.artifacts("lora", f"{ent}/rlm-production/{dest}")}
print("before:", before)
with tempfile.TemporaryDirectory() as tmp:
    api.artifact(f"{ent}/rlm-production/{src_ref}").download(root=tmp)
    run = wandb.init(project="rlm-production", entity=ent, job_type="checkpoint-alias-swap",
                     name=f"alias-{src_ref.replace(':', '-')}-to-{dest}-{alias}", settings=wandb.Settings(silent=True))
    art = wandb.Artifact(name=dest, type="lora", metadata={"copied_from": src_ref})
    art.add_dir(tmp)
    run.log_artifact(art, aliases=[alias, "latest"])
    run.finish()
after = {a.version: a.aliases for a in wandb.Api(timeout=120).artifacts("lora", f"{ent}/rlm-production/{dest}")}
print("after:", after)
