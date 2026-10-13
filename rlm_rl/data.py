"""Where the downloaded datasets live, what each one may be used for, and the loaders the task generators read (plan step 7).

The data itself sits outside the repo, on Rishi's D drive (`D:\\RLM-data`, `/mnt/d/RLM-data` from WSL), and
never enters git. `RLM_DATA_DIR` overrides the location. Everything under `hf/` was fetched with the Hugging
Face CLI at the pinned revisions below. Everything under `derived/` is built from it by
`scripts/build_data.py` and pinned by hash in `manifests/dataset-manifest.json`.

The licence bar (decision 7b, Rishi 2026-10-11): only permissive licences (Apache-2.0, MIT, CC-BY and similar)
in training. Anything else, and every benchmark, is eval-only. `SOURCES` records each source's licence and
role, and `check_training_sources()` refuses a training family that reads a source not cleared for training.

Everything here runs on the host (H3). Nothing in this module, and no path under the data root, is mounted
into the sandbox: only a task's rendered `context` is.
"""

from __future__ import annotations

import functools
import gzip
import json
import os
from pathlib import Path

DATA_ENV = "RLM_DATA_DIR"
DEFAULT_ROOTS = ("D:/RLM-data", "/mnt/d/RLM-data")

TRAIN, EVAL = "train", "eval"

# One entry per source. `files` are relative to the data root. `role` is what the licence bar allows:
# TRAIN sources may also be used in evals; EVAL sources never reach a training family.
SOURCES = {
    "clinc150": {
        "what": "CLINC150 intent utterances (the `plus` config), 150 intents in 10 domains",
        "origin": "hf:clinc/clinc_oos", "revision": "155b9c710419136e17307b80d0a13e68cd46b4ec",
        "licence": "CC-BY-3.0", "permissive": True, "role": TRAIN,
        "files": ["hf/clinc__clinc_oos/plus/train-00000-of-00001.parquet",
                  "hf/clinc__clinc_oos/plus/validation-00000-of-00001.parquet",
                  "hf/clinc__clinc_oos/plus/test-00000-of-00001.parquet",
                  "hf/clinc__clinc_oos/README.md"],
    },
    "clinc150-domains": {
        "what": "CLINC150's intent-to-domain map (domains.json from the dataset's own GitHub repo)",
        "origin": "github:clinc/oos-eval/data/domains.json", "revision": None,
        "licence": "CC-BY-3.0", "permissive": True, "role": TRAIN,
        "files": ["github/clinc__oos-eval/domains.json"],
    },
    "massive": {
        "what": "Amazon MASSIVE, English utterances with their scenario labels (MTEB's mirror)",
        "origin": "hf:mteb/amazon_massive_scenario", "revision": "58871793b91addb7c5f7afff26ccf08737fb6697",
        "licence": "CC-BY-4.0 (upstream MASSIVE; the mirror's card says Apache-2.0)", "permissive": True,
        "role": TRAIN,
        "files": ["hf/mteb__amazon_massive_scenario/train/en.json.gz",
                  "hf/mteb__amazon_massive_scenario/validation/en.json.gz",
                  "hf/mteb__amazon_massive_scenario/test/en.json.gz",
                  "hf/mteb__amazon_massive_scenario/README.md"],
    },
    "lca-bug-localization": {
        "what": "Long Code Arena bug localization, Python: issues, the files their fix changed, and repo archives",
        "origin": "hf:JetBrains-Research/lca-bug-localization", "revision": "f2bb26a1c3320ccc1d7b02d2630077ff327a80f6",
        "licence": "Apache-2.0 (dataset); every repo in it is MIT or Apache-2.0", "permissive": True, "role": TRAIN,
        "files": ["hf/JetBrains-Research__lca-bug-localization/py/train-00000-of-00001.parquet",
                  "hf/JetBrains-Research__lca-bug-localization/py/test-00000-of-00001.parquet",
                  "hf/JetBrains-Research__lca-bug-localization/README.md"],
        "dirs": ["hf/JetBrains-Research__lca-bug-localization/repos/py"],
    },
    "huffpost": {
        "what": "HuffPost News Category headlines (held-out aggregation source)",
        "origin": "hf:heegyu/news-category-dataset", "revision": "304a05a55bc6abc0446d8fae0d0771716b6a271a",
        "licence": "CC-BY-4.0", "permissive": True, "role": EVAL,  # permissive, but held out by design (H7)
        "files": ["hf/heegyu__news-category-dataset/data.json"],
    },
    "oolong-synth": {
        "what": "OOLONG-synth test split: 5,200 aggregation questions over 8 source sets, 1K-4M tokens",
        "origin": "hf:oolongbench/oolong-synth", "revision": "f0d59eaf0febf130664cfceb710436c8e3216b2b",
        "licence": "none stated; built from sets that include non-commercial ones", "permissive": False,
        "role": EVAL,
        "dirs": ["hf/oolongbench__oolong-synth/data"],
    },
    "ifeval": {
        "what": "IFEval prompts (capability check)", "origin": "hf:google/IFEval",
        "revision": "966cd89545d6b6acfd7638bc708b98261ca58e84", "licence": "Apache-2.0", "permissive": True,
        "role": EVAL, "files": ["hf/google__IFEval/ifeval_input_data.jsonl"],
    },
    "mmlu-pro": {
        "what": "MMLU-Pro test questions (capability check)", "origin": "hf:TIGER-Lab/MMLU-Pro",
        "revision": "b189ec765aa7ed75c8acfea42df31fdae71f97be", "licence": "MIT", "permissive": True,
        "role": EVAL, "files": ["hf/TIGER-Lab__MMLU-Pro/data/test-00000-of-00001.parquet"],
    },
    "cruxeval": {
        "what": "CRUXEval output prediction (capability check's short coding set)", "origin": "hf:cruxeval-org/cruxeval",
        "revision": "b96af0450242eb4da433032b90998f25588a5d0f", "licence": "MIT", "permissive": True,
        "role": EVAL, "files": ["hf/cruxeval-org__cruxeval/test.jsonl"],
    },
    # Added 2026-10-11 when decision 7c raised aggregation to 45%: other registers beside voice commands.
    "ledgar": {
        "what": "LEDGAR contract provisions from SEC filings, labelled by clause type (LexGLUE's `ledgar` config)",
        "origin": "hf:coastalcph/lex_glue", "revision": "c23fdff1a6bf74e0e1a71cb86f1e781d37da888c",
        "licence": "CC-BY-4.0", "permissive": True, "role": TRAIN,
        "files": ["hf/coastalcph__lex_glue/ledgar/train-00000-of-00001.parquet",
                  "hf/coastalcph__lex_glue/ledgar/validation-00000-of-00001.parquet",
                  "hf/coastalcph__lex_glue/ledgar/test-00000-of-00001.parquet",
                  "hf/coastalcph__lex_glue/README.md"],
    },
    "arxiv-titles": {
        "what": "arXiv paper titles to the end of 2021, labelled by their primary archive",
        "origin": "hf:gfissore/arxiv-abstracts-2021", "revision": "e4c5fbd4dec8e46a5dc869216fe1c94cc585757a",
        "licence": "CC0-1.0", "permissive": True, "role": TRAIN,
        "files": ["hf/gfissore__arxiv-abstracts-2021/arxiv-abstracts.jsonl.gz",
                  "hf/gfissore__arxiv-abstracts-2021/README.md"],
    },
    # The edit families' task sources (spec §12.7, plan step E4). Each repository's own licence file is checked too
    # (tasks/editreal.check_licences). Only the Python and Go SWE-smith cards state a licence; Rishi chose on
    # 2026-10-12 to train JavaScript, TypeScript, Rust and Java only on tasks we generate, so those four datasets'
    # rows are eval-only.
    **{f"swe-smith-{lang}": {
        "what": f"SWE-smith {name} task instances (injected bugs with issue texts) and their per-repository images",
        "origin": f"hf:SWE-bench/SWE-smith-{lang}", "revision": None, "licence": "MIT", "permissive": True,
        "role": TRAIN, "files": [f"hf/SWE-bench__SWE-smith-{lang}/README.md"],
        "dirs": [f"hf/SWE-bench__SWE-smith-{lang}/data"]} for lang, name in (("py", "Python"),
                                                                                              ("go", "Go"))},
    **{f"swe-smith-{lang}": {
        "what": f"SWE-smith {name} task instances: eval splits only (the dataset card states no licence)",
        "origin": f"hf:SWE-bench/SWE-smith-{lang}", "revision": None, "licence": "none stated on the dataset card",
        "permissive": False, "role": EVAL, "files": [f"hf/SWE-bench__SWE-smith-{lang}/README.md"],
        "dirs": [f"hf/SWE-bench__SWE-smith-{lang}/data"]}
       for lang, name in (("js", "JavaScript"), ("ts", "TypeScript"), ("rs", "Rust"), ("java", "Java"))},
    "swe-rebench-v2": {
        "what": "SWE-rebench V2 real issues and merged pull requests, per-task images; permissive per-task licences only",
        "origin": "hf:nebius/SWE-rebench-V2", "revision": None, "licence": "CC-BY-4.0", "permissive": True,
        "role": TRAIN, "files": ["hf/nebius__SWE-rebench-V2/README.md"], "dirs": ["hf/nebius__SWE-rebench-V2/data"],
    },
    "generated": {
        "what": "edit_real tasks we generate (feature removals, our own injected bugs) in permissive repositories",
        "origin": "scripts/build_edit_data.py", "revision": None, "licence": "ours; each repository's own licence "
        "is permissive", "permissive": True, "role": TRAIN, "files": [],
    },
}

# Sources the plan names that are not used, and why (decision 7b and the research's avoid list).
NOT_USED = {
    "oolong-synth validation": "fails 7b: no licence, and built from non-commercial sets; replaced by our own "
                               "aggregation generator on CLINC150 and MASSIVE",
    "loghub": "fails 7b: research use only",
    "mteb/amazon_massive_intent": "downloaded, unused: the scenario labels are the ones the generator needs",
    "cyber sources (OTRF Security-Datasets, splunk attack_data, EVTX-ATTACK-SAMPLES, AIT-LDS v2)":
        "the cyber domain was dropped (decision 7c, Rishi 2026-10-11); its 35% went to code and aggregation",
}

# ----------------------------------------------------------------------------- locations


def root() -> Path:
    """The data root: $RLM_DATA_DIR, else the first default that exists. Raises if none does."""
    env = os.environ.get(DATA_ENV)
    candidates = [env] if env else list(DEFAULT_ROOTS)
    for candidate in candidates:
        path = Path(candidate)
        if path.is_dir():
            return path
    raise FileNotFoundError(f"no data root: set {DATA_ENV} or create one of {candidates}")


def available() -> bool:
    try:
        root()
        return True
    except FileNotFoundError:
        return False


def path(*parts: str) -> Path:
    return root().joinpath(*parts)


def derived(*parts: str) -> Path:
    return root().joinpath("derived", *parts)


def check_training_sources(family: str, sources: list[str]) -> None:
    """The licence bar (7b): a training family may only read sources cleared for training."""
    for name in sources:
        info = SOURCES.get(name)
        if info is None:
            raise KeyError(f"{family}: unknown source {name!r}")
        if info["role"] != TRAIN or not info["permissive"]:
            raise ValueError(f"{family}: source {name!r} is eval-only ({info['licence']}), so it can't train (7b)")


# ----------------------------------------------------------------------------- utterance pools


def _parquet_rows(rel: str, columns: list[str]) -> list[dict]:
    import pyarrow.parquet as pq

    return pq.read_table(path(rel), columns=columns).to_pylist()


@functools.lru_cache(maxsize=None)
def clinc_domain_pools(splits: tuple[str, ...] = ("train", "validation")) -> dict[str, tuple[str, ...]]:
    """CLINC150 utterances grouped by domain (10 domains of 15 intents). Out-of-scope utterances are dropped."""
    import pyarrow.parquet as pq

    domains = json.loads(path("github/clinc__oos-eval/domains.json").read_text(encoding="utf-8"))
    intent_domain = {intent: domain for domain, intents in domains.items() for intent in intents}
    meta = pq.read_schema(path(SOURCES["clinc150"]["files"][0])).metadata or {}
    names = _clinc_label_names(meta)
    pools: dict[str, set] = {d: set() for d in domains}
    for split in splits:
        for row in _parquet_rows(f"hf/clinc__clinc_oos/plus/{split}-00000-of-00001.parquet", ["text", "intent"]):
            domain = intent_domain.get(names[row["intent"]])
            if domain:
                pools[domain].add(" ".join(row["text"].split()))
    return _unambiguous(pools)


def _unambiguous(pools: dict[str, set]) -> dict[str, tuple[str, ...]]:
    """Each label's texts, minus any text that also appears under another label: its label couldn't be read off."""
    seen: dict[str, int] = {}
    for texts in pools.values():
        for text in texts:
            seen[text] = seen.get(text, 0) + 1
    return {label: tuple(sorted(t for t in texts if seen[t] == 1)) for label, texts in sorted(pools.items())}


def _clinc_label_names(meta: dict) -> list[str]:
    """The intent names, in label order, from the parquet's Hugging Face features metadata."""
    info = json.loads(meta[b"huggingface"])
    feature = info["info"]["features"]["intent"]
    return list(feature["names"])


@functools.lru_cache(maxsize=None)
def massive_scenario_pools(splits: tuple[str, ...] = ("train", "validation")) -> dict[str, tuple[str, ...]]:
    """MASSIVE English utterances grouped by scenario (18 scenarios)."""
    pools: dict[str, set] = {}
    for split in splits:
        with gzip.open(path(f"hf/mteb__amazon_massive_scenario/{split}/en.json.gz"), "rt", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    pools.setdefault(row["label_text"], set()).add(" ".join(row["text"].split()))
    return _unambiguous(pools)


@functools.lru_cache(maxsize=None)
def huffpost_pools(min_len: int = 25) -> dict[str, tuple[str, ...]]:
    """HuffPost headlines grouped by category. Held-out source: eval only."""
    pools: dict[str, set] = {}
    with open(path("hf/heegyu__news-category-dataset/data.json"), encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            headline = " ".join(row.get("headline", "").split())
            if len(headline) >= min_len:
                pools.setdefault(row["category"], set()).add(headline)
    return _unambiguous(pools)


LEDGAR_MAX_CHARS = 1_500  # longer provisions are left out, so a context keeps a few hundred lines


@functools.lru_cache(maxsize=None)
def ledgar_pools(splits: tuple[str, ...] = ("train", "validation")) -> dict[str, tuple[str, ...]]:
    """LEDGAR contract provisions grouped by clause type (100 types). Mis-decoded quotes become apostrophes."""
    import pyarrow.parquet as pq

    first = path("hf/coastalcph__lex_glue/ledgar/train-00000-of-00001.parquet")
    names = json.loads(pq.read_schema(first).metadata[b"huggingface"])["info"]["features"]["label"]["names"]
    pools: dict[str, set] = {}
    for split in splits:
        for row in _parquet_rows(f"hf/coastalcph__lex_glue/ledgar/{split}-00000-of-00001.parquet", ["text", "label"]):
            text = " ".join(row["text"].replace("�", "'").split())
            if 40 <= len(text) <= LEDGAR_MAX_CHARS:
                pools.setdefault(names[row["label"]], set()).add(text)
    return _unambiguous(pools)


# arXiv primary archives, grouped into the labels the aggregation family asks about. Others are left out.
ARXIV_GROUPS = {"math": "mathematics", "astro-ph": "astrophysics", "cond-mat": "condensed matter physics",
                "cs": "computer science", "hep-ph": "high-energy physics", "hep-th": "high-energy physics",
                "hep-ex": "high-energy physics", "hep-lat": "high-energy physics", "quant-ph": "quantum physics",
                "gr-qc": "general relativity", "nucl-th": "nuclear physics", "nucl-ex": "nuclear physics",
                "q-bio": "quantitative biology", "stat": "statistics", "q-fin": "quantitative finance",
                "nlin": "nonlinear sciences"}
ARXIV_PER_LABEL = 6_000
ARXIV_POOL = "aggregation/arxiv-titles-v1.json.gz"


def build_arxiv_pool() -> Path:
    """A fixed sample of up to ARXIV_PER_LABEL titles per label, written to derived/ (scripts/build_data.py)."""
    import random

    groups: dict[str, set] = {}
    with gzip.open(path("hf/gfissore__arxiv-abstracts-2021/arxiv-abstracts.jsonl.gz"), "rt", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            cats = row["categories"][0] if isinstance(row["categories"], list) else row["categories"]
            first = cats.strip("[]'\" ").split()[0].strip("'\",") if cats else ""
            label = ARXIV_GROUPS.get(first.split(".")[0])
            title = " ".join((row.get("title") or "").split())
            if label and 20 <= len(title) <= 200:
                groups.setdefault(label, set()).add(title)
    pools = _unambiguous(groups)
    rng = random.Random(0)
    sample = {label: sorted(rng.sample(texts, min(ARXIV_PER_LABEL, len(texts)))) for label, texts in pools.items()}
    out = derived(ARXIV_POOL)
    out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out, "wt", encoding="utf-8") as f:
        json.dump(sample, f)
    return out


@functools.lru_cache(maxsize=None)
def arxiv_title_pools() -> dict[str, tuple[str, ...]]:
    """arXiv titles grouped by archive, from the sample `build_arxiv_pool()` wrote."""
    with gzip.open(derived(ARXIV_POOL), "rt", encoding="utf-8") as f:
        return {label: tuple(texts) for label, texts in sorted(json.load(f).items())}
