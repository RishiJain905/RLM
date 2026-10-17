"""$0, read-only: download three LoRA checkpoints and compare their weights.

Did the 10d fork train from checkpoint 15's weights, or from a fresh LoRA? LoRA's B matrices start
at zero, so their size after 1 step from scratch is far smaller than after 15 or 16 steps.
"""
import sys
from pathlib import Path
from dotenv import load_dotenv
ROOT = Path(__file__).resolve().parents[3]
load_dotenv(ROOT / ".env")
import numpy as np
import wandb
import json as _json


def load_file(path):
    """Read a .safetensors file with numpy alone (no safetensors package here): header length, JSON header, data."""
    raw = Path(path).read_bytes()
    n = int.from_bytes(raw[:8], "little")
    header = _json.loads(raw[8:8 + n])
    base, out = 8 + n, {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        a, b = info["data_offsets"]
        buf = raw[base + a:base + b]
        dt = info["dtype"]
        if dt == "BF16":
            arr = (np.frombuffer(buf, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)
        else:
            arr = np.frombuffer(buf, dtype={"F32": np.float32, "F16": np.float16, "F64": np.float64}[dt])
        out[name] = arr.reshape(info["shape"])
    return out

api = wandb.Api(timeout=120)
ent = api.default_entity
OUT = Path(__file__).resolve().parent / "loras"
names = {"orig15": "second-seed-v1:step15", "orig20": "second-seed-v1:step20", "orig5": "second-seed-v1:step5",
         "fork15": "second-seed-v1-b1:step15", "fork1": "second-seed-v1-b1:step1"}
if len(sys.argv) > 1:  # e.g. orig15=second-seed-v1:step15 new=second-seed-v1:step16 fork1=second-seed-v1-b1:step1
    names = dict(a.split("=", 1) for a in sys.argv[1:])
w = {}
for key, ref in names.items():
    d = OUT / key
    if not any(d.glob("**/*.safetensors")):
        api.artifact(f"{ent}/rlm-production/{ref}").download(root=str(d))
    files = sorted(d.glob("**/*.safetensors"))
    t = {}
    for f in files:
        t.update(load_file(str(f)))
    w[key] = t
    print(key, ref, len(t), "tensors", [f.name for f in files])

keys = sorted(set.intersection(*(set(t) for t in w.values())))
print("shared tensors", len(keys))
def norm(t, part):
    ks = [k for k in keys if part in k]
    return float(np.sqrt(sum(float(np.sum(w[t][k].astype(np.float64) ** 2)) for k in ks)))
def dist(a, b, part):
    ks = [k for k in keys if part in k]
    return float(np.sqrt(sum(float(np.sum((w[a][k].astype(np.float64) - w[b][k].astype(np.float64)) ** 2)) for k in ks)))
for part in ("lora_B", "lora_A"):
    print(part, "norms:", {t: round(norm(t, part), 4) for t in w})
    pairs = [(a, b) for i, a in enumerate(w) for b in list(w)[i + 1:]]
    for a, b in pairs:
        print(f"  {part} distance {a} - {b}: {dist(a, b, part):.5f}")
