"""Building the edit_real task data (spec §12.7-12.8, plan step E4). scripts/build_edit_data.py drives it.

Everything it writes lives on the data drive (`data.path("edit", ...)`, D:\\RLM-data\\edit), never in
the repository or git:
- `repos/<key>.tar`: each repository's checkout at its image's commit, without .git, dependency
  folders or build outputs: the snapshot every task's base is built from.
- `candidates/`, `validated/`: task records before and after validation.
- `issues/`: task texts written for the tasks we generate (feature removals, our own bugs).
- `index.json`: the validated tasks edit_real reads (tasks/editreal.py); `manifest.json`: images,
  digests, licences, sources and counts.

Images: each repository's source image (SWE-smith's `swebench/swesmith.x86_64.<owner>_1776_<repo>.<sha8>`,
or SWE-rebench V2's per-task image) is pulled once and a derived image `rlm-edit-<key>:<hash>` is
built from it with no network (`derive`): the image's own checkout goes (for an injected bug it is
the answer, and SWE-smith keeps every task's bug commit as a branch in it), `/testbed` becomes a
symlink to /work/repo, warm build caches move to /opt/rlm-cache (and JavaScript's node_modules to
/opt/rlm-deps) owned by the host user, and the toolchain is told to stay offline.

Validation (spec §12.8), in our grader (grader.grade with keep_all): the base runs once and the gold
patch three times. A test that fails on the base and passes on every gold run is fail-to-pass; one
that passes everywhere is pass-to-pass; anything else is dropped as flaky. A task enters the pool
only with at least one fail-to-pass test, working canaries, visible tests that still run without
the hidden files, a gold run under half the grader's cap, and a task text that quotes neither the
gold patch nor a hidden test's name.
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import io
import json
import math
import re
import secrets
import shutil
import subprocess
import tarfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from . import bridge, data, editbox, grader
from .sandbox import docker_cli_env
from .tasks import editreal

EDIT_DIR = "edit"
REPO_DIR = "/testbed"
DERIVE_VERSION = 2  # 2: JavaScript and TypeScript keep node_modules where the image installed them (no copy)
# Copying a JavaScript project's node_modules (or a Rust target/) into a writable layer took over an hour on the USB data
# drive under load (TypeScript's 5-6 GB images ran for hours); a build cut off at the old 1-hour limit starts again from
# nothing.
DERIVE_TIMEOUT_S = 6 * 3600

# Per language: the runner, a generic command for the visible tests, what the snapshot leaves out, the derived
# image's offline settings and the cache folders the toolchain writes.
LANGS = {
    "python": {"runner": "pytest", "test_hint": "python -m pytest -q", "exclude": [],
               "env": {"PIP_NO_INDEX": "1", "PIP_DISABLE_PIP_VERSION_CHECK": "1", "PYTHONDONTWRITEBYTECODE": "1"},
               "writable": []},
    "go": {"runner": "go", "test_hint": "go test ./...", "exclude": [],
           "env": {"GOCACHE": "/opt/rlm-cache/go-build", "GOFLAGS": "-mod=mod", "GOPROXY": "off", "GOSUMDB": "off",
                   "GOTOOLCHAIN": "local", "GONOSUMCHECK": "1"},
           "writable": ["/opt/rlm-cache"]},
    "javascript": {"runner": "jest", "test_hint": "npx jest", "exclude": ["node_modules"],
                   "env": {"CI": "true", "npm_config_offline": "true"}, "writable": ["/opt/rlm-deps", "/opt/rlm-cache"]},
    "typescript": {"runner": "jest", "test_hint": "npx jest", "exclude": ["node_modules"],
                   "env": {"CI": "true", "npm_config_offline": "true"}, "writable": ["/opt/rlm-deps", "/opt/rlm-cache"]},
    "rust": {"runner": "cargo", "test_hint": "cargo test", "exclude": ["target"],
             "env": {"CARGO_TARGET_DIR": "/opt/rlm-cache/target", "CARGO_NET_OFFLINE": "true", "CARGO_TERM_COLOR": "never"},
             "writable": ["/opt/rlm-cache"]},
    "java": {"runner": "maven", "test_hint": "mvn -o -q test", "exclude": [],
             "env": {"MAVEN_OPTS": "-Dmaven.repo.local=/opt/rlm-cache/m2/repository", "GRADLE_USER_HOME": "/opt/rlm-cache/gradle",
                     "GRADLE_OPTS": "-Dorg.gradle.daemon=false"},
             "writable": ["/opt/rlm-cache"]},
}
SWE_SMITH_DATASETS = {"python": "SWE-bench__SWE-smith-py", "go": "SWE-bench__SWE-smith-go",
                      "javascript": "SWE-bench__SWE-smith-js", "typescript": "SWE-bench__SWE-smith-ts",
                      "rust": "SWE-bench__SWE-smith-rs", "java": "SWE-bench__SWE-smith-java"}
# Repositories that an eval we report also uses (spec §12.7): SWE-bench Multilingual shares these with SWE-smith.
EVAL_OVERLAP = {"caddyserver/caddy", "gin-gonic/gin", "babel/babel", "axios/axios", "BurntSushi/ripgrep", "google/gson"}


def edit_path(*parts: str) -> Path:
    return data.path(EDIT_DIR, *parts)


def repo_key(image: str) -> str:
    """'swebench/swesmith.x86_64.go-chi_1776_chi.23c395f8' -> 'go-chi__chi.23c395f8'."""
    name = image.split("/", 1)[-1].split(":", 1)[0]
    name = name.removeprefix("swesmith.x86_64.")
    return re.sub(r"[^A-Za-z0-9_.\-]", "-", name.replace("_1776_", "__")).lower()


def _docker(*args: str, check: bool = True, input: bytes | None = None, timeout: float = 3600) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], capture_output=True, check=check, input=input, env=docker_cli_env(),
                          timeout=timeout)


# ----------------------------------------------------------------------------- derived images


PROBE = ('echo "@@PATH=$PATH"; if command -v git >/dev/null 2>&1; then echo "@@GIT=1"; fi; '
         'if command -v go >/dev/null 2>&1; then echo "@@GOMODCACHE=$(go env GOMODCACHE)"; '
         'echo "@@GOCACHE=$(go env GOCACHE)"; fi')
# Cache and config folders an image may point at a path the episode can't write (SWE-rebench V2 uses /workspace).
CACHE_ENV = ("XDG_CACHE_HOME", "XDG_CONFIG_HOME", "XDG_DATA_HOME", "PIP_CACHE_DIR", "PYTHONPYCACHEPREFIX",
             "npm_config_cache", "YARN_CACHE_FOLDER", "PNPM_HOME", "UV_CACHE_DIR", "POETRY_CACHE_DIR")
WRITABLE_PREFIXES = ("/scratch", "/tmp", "/work/repo", "/opt/rlm-cache")


def image_env(source: str) -> dict:
    out = _docker("image", "inspect", "--format", "{{json .Config.Env}}", source, check=False).stdout.decode()
    return dict(e.split("=", 1) for e in json.loads(out or "[]") or [] if "=" in e)


def probe_source(source: str) -> dict:
    """What a login shell in the source image sees: its PATH (SWE-smith activates its conda env, and nvm and the like,
    in .bashrc, which a plain container never runs), whether git is installed (the coding tools' diff() needs it),
    and Go's module and build caches. One container."""
    out = _docker("run", "--rm", "--network", "none", "--user", "0", "--entrypoint", "bash", source, "-lc", PROBE,
                  check=False)
    found = {}
    for line in out.stdout.decode(errors="replace").splitlines():
        if line.startswith("@@") and "=" in line:
            key, _, value = line[2:].partition("=")
            found[key] = value.strip()
    return {"path": found.get("PATH") if (found.get("PATH") or "").startswith("/") else None,
            "git": found.get("GIT") == "1", "gomodcache": found.get("GOMODCACHE") or None,
            "gocache": found.get("GOCACHE") or None}


def login_path(source: str, probe: dict | None = None) -> str | None:
    """The PATH episodes get: the login shell's entries first (SWE-smith's conda env and the like), then the image's
    own that the login shell dropped (Debian's /etc/profile resets PATH, which loses /usr/local/go/bin, Cargo's and
    the JDK's bin folders). None when that is just the image's own PATH."""
    path = (probe or probe_source(source))["path"]
    own = image_env(source).get("PATH") or ""
    if not path or path == own:
        return None
    entries = list(dict.fromkeys(p for p in path.split(":") + own.split(":") if p))
    merged = ":".join(entries)
    return None if merged == own else merged


def cache_env(source: str) -> dict:
    """Overrides for cache folders the image points outside what an episode may write: they move under /scratch."""
    env = image_env(source)
    return {k: f"/scratch/.cache/{k.lower()}" for k in CACHE_ENV
            if k in env and env[k].startswith("/") and not env[k].startswith(WRITABLE_PREFIXES)}


def derive_dockerfile(source: str, language: str, uid: int, gid: int, repo_dir: str = REPO_DIR,
                      path: str | None = None, go: dict | None = None, extra_env: dict | None = None,
                      deps: list[str] | None = None) -> str:
    """The derived image (spec §12.4): no checkout, the image's repository path -> /work/repo, caches the sandbox user
    can write, offline. `go` (SWE-rebench V2 images, from probe_source) says where Go's module and build caches are.

    `deps` (JavaScript, TypeScript: the checkout's outermost node_modules folders) stay where the image installed them:
    moving or re-owning them copies every file into a new layer, hours for a 5 GB node_modules on the data drive. The
    checkout around them goes instead (.git and every file outside node_modules; folders stay, empty), the repository
    path stays a folder rather than a link to /work/repo, and /opt/rlm-deps is a link to it, so an episode's
    /work/repo/<folder> -> /opt/rlm-deps/<folder> links reach them. They stay owned by root: read-only to the
    sandbox user, which the tests only read."""
    cfg = LANGS[language]
    if deps:
        steps = [f"rm -rf {repo_dir}/.git /root/setup_env.sh",
                 f"find {repo_dir} -mindepth 1 \\( -type d -name node_modules -prune \\) -o \\( ! -type d -exec rm -f {{}} + \\)",
                 "mkdir -p /work /opt/rlm /scratch /episode /opt/rlm-cache", f"ln -s {repo_dir} /opt/rlm-deps"]
        chown = ["/opt/rlm-cache"]
    else:
        steps = [f"rm -rf {repo_dir} /root/setup_env.sh",
                 "mkdir -p /work /opt/rlm /scratch /episode /opt/rlm-cache /opt/rlm-deps", f"ln -s /work/repo {repo_dir}"]
        chown = ["/opt/rlm-cache", "/opt/rlm-deps"]
    env_go = {}
    if language == "go" and go:
        # The module cache stays where it is, read-only (GOPROXY=off fetches nothing): its folders are opened up
        # (chmod on the folders only, so nothing is copied into a new layer) and GOMODCACHE points at it.
        modcache, gocache = go.get("gomodcache"), go.get("gocache") or "/root/.cache/go-build"
        if modcache:
            parts = [p for p in modcache.split("/") if p]
            steps += ["chmod o+rx " + " ".join("/" + "/".join(parts[:i]) for i in range(1, len(parts) + 1))]
            env_go["GOMODCACHE"] = modcache
        steps += [f'if [ -d "{gocache}" ]; then mv "{gocache}" /opt/rlm-cache/go-build; '
                  "else mkdir -p /opt/rlm-cache/go-build; fi"]
    elif language == "go":
        # Go only reads its module cache (world-readable; GOPROXY=off fetches nothing), so /go keeps its owner: re-owning
        # it copies every file into a new layer, which takes many minutes on the data drive.
        steps += ["if [ -d /root/.cache/go-build ]; then mv /root/.cache/go-build /opt/rlm-cache/go-build; "
                  "else mkdir -p /opt/rlm-cache/go-build; fi"]
    if language == "rust":
        # The image's warm build (dependencies compiled) lives in the checkout's target/: keep it as the cache,
        # before the checkout goes, or every test run rebuilds every dependency.
        steps.insert(0, f"mkdir -p /opt/rlm-cache && if [ -d {repo_dir}/target ]; then mv {repo_dir}/target "
                        "/opt/rlm-cache/target; fi")  # chown -R /opt/rlm-cache below makes it writable
        steps += ["mkdir -p /opt/rlm-cache/target",
                  'CH="${CARGO_HOME:-/usr/local/cargo}"; if [ -d "$CH" ]; then chown -R ' + f"{uid}:{gid}" + ' "$CH"; fi']
    if language == "java":
        steps += ["mkdir -p /opt/rlm-cache/m2 /opt/rlm-cache/gradle",
                  "if [ -d /root/.m2 ]; then cp -a /root/.m2/. /opt/rlm-cache/m2/; fi",
                  "if [ -d /root/.gradle ]; then cp -a /root/.gradle/. /opt/rlm-cache/gradle/; fi"]
    steps += ["rm -rf /root/.cache", f"chown -R {uid}:{gid} " + " ".join(chown)]
    env = {**cfg["env"], **env_go, **(extra_env or {}), "HOME": "/scratch"}
    if path:
        env["PATH"] = path
    if language in ("javascript", "typescript"):
        env["PATH"] = "/work/repo/node_modules/.bin:" + (path or "$PATH")
    labels = {editbox.L_UID: str(uid), editbox.L_LANGUAGE: language, editbox.L_ALIAS: repo_dir,
              editbox.L_WRITABLE: ":".join(cfg["writable"]), "rlm.edit.source": source,
              "rlm.edit.derive_version": str(DERIVE_VERSION)}
    lines = [f"FROM {source}", "USER root", "RUN set -e; " + "; ".join(steps)]
    lines += [f'ENV {k}="{v}"' for k, v in env.items()]
    lines += [f'LABEL {k}="{v}"' for k, v in labels.items()]
    return "\n".join(lines) + "\n"


def node_modules_moves(source: str, repo_dir: str = REPO_DIR) -> list[str]:
    """JavaScript: every node_modules folder in the checkout, relative to it, outermost only."""
    out = _docker("run", "--rm", "--network", "none", "--entrypoint", "/bin/sh", source, "-c",
                  f"cd {repo_dir} && find . -name node_modules -type d -prune", check=False)
    return sorted(p[2:] for p in out.stdout.decode().split() if p.startswith("./"))


def warm_image(source: str, warm: str, repo_dir: str = REPO_DIR, name: str | None = None) -> str:
    """The source image with its build tool's dependency cache filled: `warm` (a shell command, e.g. compiling the
    tests) runs once in the checkout, with the network on. Some source images never resolved their test
    dependencies (JsonPath's Gradle cache has none), and episodes and graders run offline. The derived image is built
    from this one; nothing in an episode or a grader ever has the network."""
    digest = hashlib.sha256(f"{source}\n{repo_dir}\n{warm}".encode()).hexdigest()[:12]
    tag = f"rlm-warm-{name or repo_key(source)}:{digest}"
    if _docker("image", "inspect", tag, check=False).returncode == 0:
        return tag
    dockerfile = f"FROM {source}\nUSER root\nRUN cd {repo_dir} && {warm}\n"
    context = io.BytesIO()
    with tarfile.open(fileobj=context, mode="w") as tar:
        info = tarfile.TarInfo("Dockerfile")
        info.size = len(dockerfile.encode())
        tar.addfile(info, io.BytesIO(dockerfile.encode()))
    out = _docker("build", "--pull=false", "-t", tag, "-", input=context.getvalue(), check=False,
                  timeout=DERIVE_TIMEOUT_S)
    if out.returncode:
        raise RuntimeError(f"warming {source} failed: {out.stderr.decode()[-800:]}")
    return tag


def derive(source: str, language: str, uid: int | None = None, gid: int | None = None, repo_dir: str = REPO_DIR,
           name: str | None = None, rebench: bool = False, warm: str | None = None) -> str:
    """Build (or find) the derived image for one source image. Returns its tag. `rebench`: a SWE-rebench V2 image,
    whose repository is at `repo_dir`, whose caches may sit outside what an episode can write, and whose Go caches are
    found by probing. `warm`: fill the build tool's dependency cache first (warm_image)."""
    if warm:
        name = name or repo_key(source)  # the derived image keeps the repository's name
        source = warm_image(source, warm, repo_dir, name)
    uid, gid = (uid, gid) if uid is not None else editbox.host_user()
    probe = probe_source(source)
    if not probe["git"]:
        raise RuntimeError(f"{source} has no git, which the coding tools need (spec §12.1)")
    moves = node_modules_moves(source, repo_dir) if language in ("javascript", "typescript") else []
    dockerfile = derive_dockerfile(source, language, uid, gid, repo_dir=repo_dir, path=login_path(source, probe),
                                   go=probe if rebench and language == "go" else None,
                                   extra_env=cache_env(source) if rebench else None, deps=moves)
    digest = hashlib.sha256(dockerfile.encode()).hexdigest()[:12]
    tag = f"rlm-edit-{name or repo_key(source)}:{digest}"
    if _docker("image", "inspect", tag, check=False).returncode == 0:
        return tag
    context = io.BytesIO()
    with tarfile.open(fileobj=context, mode="w") as tar:
        info = tarfile.TarInfo("Dockerfile")
        body = dockerfile.encode()
        info.size = len(body)
        tar.addfile(info, io.BytesIO(body))
    out = _docker("build", "--network", "none", "--pull=false", "-t", tag, "-", input=context.getvalue(), check=False,
                  timeout=DERIVE_TIMEOUT_S)
    if out.returncode:
        raise RuntimeError(f"deriving {source} failed: {out.stderr.decode()[-800:]}")
    return tag


# ----------------------------------------------------------------------------- snapshots and licences


def export_snapshot(source: str, language: str, dest: Path, repo_dir: str = REPO_DIR) -> dict:
    """Tar the source image's checkout (no .git, no dependency folders or build outputs) to `dest` on the data drive.
    Returns {"sha256", "bytes", "files", "node_modules"}."""
    excludes = ["--exclude=./.git"] + [f"--exclude={e}" for e in LANGS[language]["exclude"]]
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + f".tmp-{secrets.token_hex(4)}")
    with open(tmp, "wb") as f:
        proc = subprocess.run(["docker", "run", "--rm", "--network", "none", "--entrypoint", "tar", source, "-C",
                               repo_dir, *excludes, "-cf", "-", "."], stdout=f, stderr=subprocess.PIPE,
                              env=docker_cli_env(), timeout=3600)
    if proc.returncode:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"exporting {source}'s checkout failed: {proc.stderr.decode()[-500:]}")
    tmp.replace(dest)
    return snapshot_stats(dest, source, language, repo_dir)


def snapshot_stats(dest: Path, source: str, language: str, repo_dir: str = REPO_DIR) -> dict:
    """{"sha256", "bytes", "files", "node_modules"} of a snapshot already on the data drive."""
    sha = hashlib.sha256()
    with open(dest, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            sha.update(chunk)
    with tarfile.open(dest) as tar:
        n = sum(1 for m in tar.getmembers() if m.isfile())
    moves = node_modules_moves(source, repo_dir) if language in ("javascript", "typescript") else []
    return {"sha256": sha.hexdigest(), "bytes": dest.stat().st_size, "files": n, "node_modules": moves}


LICENCE_FILES = ("LICENSE", "LICENSE.md", "LICENSE.txt", "LICENSE.rst", "LICENCE", "COPYING", "COPYING.md",
                 "LICENSE-MIT", "LICENSE-APACHE", "LICENSE-MIT.md", "LICENSE-APACHE.md", "UNLICENSE", "license",
                 "License.txt", "license.txt", "LICENSE.MIT")


def classify_licence(texts: dict[str, str]) -> str:
    """An SPDX id from the repository's licence files, or "unknown"."""
    joined = "\n".join(texts.values())
    low = joined.lower()
    found = []
    if re.search(r"gnu affero general public license", low):
        found.append("AGPL-3.0")
    elif re.search(r"gnu lesser general public license|gnu library general public license", low):
        found.append("LGPL")
    elif re.search(r"gnu general public license", low):
        found.append("GPL")
    if "mozilla public license" in low:
        found.append("MPL-2.0")
    if "eclipse public license" in low:
        found.append("EPL")
    if "european union public licence" in low:
        found.append("EUPL")
    if "business source license" in low or "server side public license" in low or "elastic license" in low:
        found.append("source-available")
    if found:
        return found[0]
    ids = []
    if "apache license" in low and "version 2.0" in low:
        ids.append("Apache-2.0")
    if re.search(r"permission is hereby granted, free of charge", low):
        ids.append("MIT")
    if re.search(r"redistribution and use in source and binary forms", low):
        ids.append("BSD-3-Clause" if re.search(r"neither the name|names of its\s+contributors", low) else "BSD-2-Clause")
    if re.search(r"permission to use, copy, modify, and/?or distribute this software for any", low):
        ids.append("ISC")
    if "this is free and unencumbered software released into the public domain" in low:
        ids.append("Unlicense")
    if "boost software license" in low:
        ids.append("BSL-1.0")
    if "python software foundation license" in low:
        ids.append("PSF-2.0")
    if not ids:
        return "unknown"
    if set(ids) == {"MIT", "Apache-2.0"}:
        return "MIT OR Apache-2.0"
    return ids[0]


def snapshot_licence(snapshot: Path) -> tuple[str, list[str]]:
    texts = {}
    with tarfile.open(snapshot) as tar:
        for m in tar.getmembers():
            name = m.name.removeprefix("./")
            if "/" not in name and name in LICENCE_FILES and m.isfile():
                texts[name] = tar.extractfile(m).read(200_000).decode("utf-8", "replace")
    return classify_licence(texts), sorted(texts)


def detect_runner(snapshot: Path, language: str) -> dict:
    """How a repository runs its tests, from its files: {"runner", "bin", "test_hint", "junit"} (only what differs from
    the language's default)."""
    files = snapshot_files(snapshot)
    if language in ("javascript", "typescript"):
        pkg = json.loads(snapshot_read(snapshot, ["package.json"]).get("package.json") or "{}")
        deps = {**(pkg.get("dependencies") or {}), **(pkg.get("devDependencies") or {})}
        script = (pkg.get("scripts") or {}).get("test", "")
        for runner in ("vitest", "jest", "mocha"):
            if runner in deps or runner in script:
                return {"runner": runner, "bin": [runner], "test_hint": "npm test" if script else f"npx {runner}"}
        return {"runner": "unsupported"}
    if language == "java":
        tests = [p for p in files if "/src/test/java/" in "/" + p and p.endswith(".java")]
        sample = "\n".join(snapshot_read(snapshot, tests[:20]).values())
        junit = "5" if "org.junit.jupiter" in sample else "4"
        if "pom.xml" in files:
            return {"runner": "maven", "bin": ["mvn"], "test_hint": "mvn -o -q test", "junit": junit}
        if "build.gradle" in files or "build.gradle.kts" in files:
            gradle = ["./gradlew"] if "gradlew" in files else ["gradle"]
            return {"runner": "gradle", "bin": gradle, "test_hint": f"{gradle[0]} test --offline", "junit": junit}
        return {"runner": "unsupported"}
    return {}


def permissive(licence: str) -> bool:
    return licence in editreal.PERMISSIVE or licence == "MIT OR Apache-2.0"


def snapshot_files(snapshot: Path) -> dict[str, int]:
    with tarfile.open(snapshot) as tar:
        return {m.name.removeprefix("./"): m.size for m in tar.getmembers() if m.isfile()}


def snapshot_read(snapshot: Path, paths: Iterable[str]) -> dict[str, str]:
    want = set(paths)
    out = {}
    with tarfile.open(snapshot) as tar:
        for m in tar.getmembers():
            name = m.name.removeprefix("./")
            if name in want and m.isfile():
                out[name] = tar.extractfile(m).read().decode("utf-8", "replace")
    return out


# ----------------------------------------------------------------------------- SWE-smith rows


def swe_smith_rows(language: str, repo_prefix: str | None = None, columns=None) -> list[dict]:
    import pyarrow.parquet as pq

    folder = data.path("hf", SWE_SMITH_DATASETS[language], "data")
    rows = []
    for f in sorted(folder.glob("*.parquet")):
        table = pq.read_table(f, columns=columns)
        for r in table.to_pylist():
            if repo_prefix is None or r["repo"].split("/", 1)[-1].startswith(repo_prefix):
                rows.append(r)
    return rows


def reverse_patch(patch: str, tree: Path) -> str:
    """The gold patch for an injected bug: the bug patch reversed, as git would write it."""
    scratch = tree.parent / f".rev-{secrets.token_hex(4)}"
    shutil.copytree(tree, scratch, symlinks=True)
    try:
        env = editbox.git_env(scratch.parent)
        editbox.git("init", "-q", str(scratch), env=env)
        editbox.git("add", "-A", env=env, cwd=scratch)
        editbox.git("commit", "-q", "-m", "clean", env=env, cwd=scratch)
        out = editbox.git("apply", "--whitespace=nowarn", "-", env=env, cwd=scratch, check=False, input=patch.encode())
        if out.returncode:
            raise ValueError(f"the bug patch doesn't apply: {out.stderr.decode()[-200:]}")
        editbox.git("add", "-A", env=env, cwd=scratch)
        return editbox.git("diff", "--cached", "-R", "--no-color", "--no-renames", "--full-index", env=env,
                           cwd=scratch).stdout.decode()
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def patch_size(patch: str) -> tuple[int, int]:
    files = [fp for fp in grader.parse_patch(patch) if not fp.binary]
    return len(files), sum(fp.added + fp.removed for fp in files)


def go_test_files(names: Iterable[str], files: dict[str, str]) -> dict[str, set[str]]:
    """Which _test.go files define each top-level Go test function."""
    out: dict[str, set[str]] = {}
    tops = {n.split("/")[0] for n in names}
    for path, text in files.items():
        if path.endswith("_test.go"):
            for name in tops:
                if re.search(rf"^func {re.escape(name)}\(", text, re.M):
                    out.setdefault(name, set()).add(path)
    return out


def leaks(issue: str, gold_patch: str, test_names: Iterable[str]) -> list[str]:
    """Why a task text gives away its answer (spec §12.8): a gold patch line or a hidden test's name, quoted."""
    found = []
    for fp in grader.parse_patch(gold_patch):
        for line in fp.added_lines():
            s = line.strip()
            if len(s) >= 25 and s in issue:
                found.append(f"quotes the gold patch: {s[:60]}")
                break
    for name in test_names:
        short = name.split("::")[-1].split(".")[-1].split("/")[0]
        if len(short) >= 8 and re.search(rf"(?<![\w]){re.escape(short)}(?![\w])", issue):
            found.append(f"names a hidden test: {short}")
    return found


# ----------------------------------------------------------------------------- validation


BUILD_FILES = {"maven": ("pom.xml",), "gradle": ("build.gradle", "build.gradle.kts")}  # in the module's folder
GOLD_MAX_S = 900  # a gold run slower than this (under load) makes a task too slow to train on
CAP_FACTOR = 4  # a task's grader cap: max(the spec's 600 s, CAP_FACTOR x its slowest gold run under load)


def grader_cap(gold_s: float) -> int:
    """A task's grader cap (spec §12.5, as amended for the slow data drive)."""
    return max(int(bridge.EDIT_LIMITS.grader_timeout_s), int(math.ceil(CAP_FACTOR * gold_s)))


def cell_limit(test_s: float | None) -> int | None:
    """A task's own cell limit when its gold test command, run through shell() under load, takes more than half the
    standard one; None keeps the standard limit (spec §12.3)."""
    base = int(bridge.EDIT_LIMITS.cell_timeout_s)
    if test_s is None or test_s <= base / 2:
        return None
    return int(math.ceil(2 * test_s / 30.0) * 30)


@dataclass
class Validation:
    ok: bool
    why: str = ""
    f2p: list = field(default_factory=list)
    p2p: list = field(default_factory=list)
    flaky: int = 0
    gold_s: float = 0.0
    base_s: float = 0.0


async def validate(record: dict, image: editbox.EditImage, runs: int = 3, run_id: str = "edit-validate") -> Validation:
    """Spec §12.8 in our grader. `record` is an index record with tests.f2p/p2p still empty."""
    task = editreal.to_task(record, "edit_real")
    ws_dir = await asyncio.to_thread(task.workspace.ensure)
    spec = grader.TestSpec.from_dict({**record["tests"], "f2p": [], "p2p": []})
    build_files = BUILD_FILES.get(spec.runner)
    if build_files and not any((ws_dir / "tree" / spec.cwd / f).exists() for f in build_files):
        # e.g. picocli: Maven builds only the root project, its other modules are Gradle's
        return Validation(False, f"no {' or '.join(build_files)} in {spec.cwd}: {spec.runner} can't run this module")
    base = await grader.grade("", spec, ws_dir=ws_dir, image=image, run_id=run_id, keep_all=True)
    if any(h["rule"] == "grader_tamper" for h in base.hacks):
        return Validation(False, f"canaries failed on the base: {base.hacks}")
    golds = []
    if spec.runner in grader.MULTI_RUN_RUNNERS:  # the gold runs back to back in one container: two starts fewer
        g = await grader.grade(record["gold_patch"], spec, ws_dir=ws_dir, image=image, run_id=run_id, keep_all=True,
                               runs=runs)
        if not g.applied:
            return Validation(False, "the gold patch doesn't apply to the base")
        if any(h["rule"] in ("grader_tamper", "test_tamper") for h in g.hacks):
            return Validation(False, f"gold run flagged: {g.hacks}")
        if g.deadline_hit:
            return Validation(False, "the gold run hit the cap")
        each = g.seconds / max(1, runs)
        golds = [dataclasses.replace(g, results=r, seconds=each) for r in g.run_results]
    for _ in range(runs - len(golds)):
        g = await grader.grade(record["gold_patch"], spec, ws_dir=ws_dir, image=image, run_id=run_id, keep_all=True)
        if not g.applied:
            return Validation(False, "the gold patch doesn't apply to the base")
        if any(h["rule"] in ("grader_tamper", "test_tamper") for h in g.hacks):
            return Validation(False, f"gold run flagged: {g.hacks}")
        if g.deadline_hit:
            return Validation(False, "the gold run hit the cap")
        golds.append(g)
    keys = set(base.results) | {k for g in golds for k in g.results}
    f2p, p2p, flaky = [], [], 0
    for k in sorted(keys):
        outs = [g.results.get(k) for g in golds]
        if len(set(outs)) != 1:
            flaky += 1
            continue
        if outs[0] == grader.PASSED:
            if base.results.get(k) == grader.PASSED:
                p2p.append(k)
            elif base.results.get(k) in (grader.FAILED, grader.ERROR, None):
                f2p.append(k)
    gold_s = max(g.seconds for g in golds)
    if not f2p:
        return Validation(False, "no test fails on the base and passes with the gold patch", flaky=flaky)
    if gold_s > GOLD_MAX_S:  # the task's cap is max(600 s, 4x its slowest gold run), so the gold run is always under
        return Validation(False, f"the gold run took {gold_s:.0f} s", flaky=flaky)  # half; this drops only runaways
    hidden_names = [k for k in f2p]
    bad = leaks(record["issue"], record["gold_patch"], hidden_names)
    if bad:
        return Validation(False, "; ".join(bad))
    return Validation(True, "", f2p, p2p, flaky, gold_s, base.seconds)


# Bumped when the visible-test check changes; a task dropped by an older check is validated again (validate stage).
VISIBLE_CHECK_VERSION = 4  # 2: Go compiles tests instead of `go vet`; 3: and doesn't run TestMain (-exec /bin/true);
# 4: a check that runs out of time is tried again with more, and never counts as broken visible tests; Gradle builds
# compile their tests with Gradle
VISIBLE_TIMEOUT_S = 1800  # compiling every test target (Go packages, a Rust crate) on the slow data drive, under load


async def visible_tests_run(record: dict, image: editbox.EditImage, run_id: str = "edit-validate") -> tuple[bool, str]:
    """Spec §12.8: removing the hidden test files doesn't break the visible tests (they still collect or compile).
    A check that runs out of time runs once more with double the time; a second timeout is an InfraError (the drive's
    problem, not the task's), never a broken-visible-tests verdict."""
    task = editreal.to_task(record, "edit_real")
    ws_dir = await asyncio.to_thread(task.workspace.ensure)
    language = record["language"]
    checks = {"python": ["python", "-m", "pytest", "-q", "--collect-only", "-p", "no:cacheprovider"],
              "go": ["go", "test", "-count=1", "-run", "^$", "-exec", "/bin/true", "./..."], "rust": ["cargo", "test", "--no-run", "--offline"],
              "java": ["mvn", "-o", "-q", "test-compile"]}
    spec = grader.TestSpec(runner=record["tests"]["runner"], scope=[], f2p=[], p2p=[], language=language,
                           cwd=record["tests"].get("cwd", "."), env=record["tests"].get("env", {}),
                           bin=record["tests"].get("bin") or [])
    if spec.runner == "gradle":  # Gradle builds compile their tests with testClasses, not Maven's test-compile
        checks["java"] = grader.Gradle.bin(spec) + ["testClasses", "--offline", "--no-daemon", "--console=plain"]
    argv = record.get("visible_check") or checks.get(language)
    if not argv:
        return True, "no visible check for this language"
    for cap in (VISIBLE_TIMEOUT_S, 2 * VISIBLE_TIMEOUT_S):
        job = {"nonce": secrets.token_hex(16), "commands": [{"argv": argv, "cwd": grader.Runner._cwd(spec),
                                                              "env": dict(LANGS[language]["env"], **spec.env),
                                                              "timeout": cap, "merge": True}],
               "reports": [], "deadline_s": cap, "writable": ["/work/repo", "/scratch", "/tmp", *image.writable]}
        base = editbox.cache_root() / f"visible-{secrets.token_hex(6)}"
        try:
            editbox.materialize(ws_dir, base)
            driver = await grader.run_driver(job, image=image, grade_dir=base, run_id=run_id, docker="docker",
                                             runtime=editbox.runtime_dir(), bundle=editbox.python_bundle_dir(),
                                             limits=editbox.edit_sandbox_limits(),
                                             timeout=cap + grader.GRADER_START_SLACK_S)
        finally:
            editbox.remove_tree(base)
        cmd = (driver.get("commands") or [{}])[0]
        if not (cmd.get("timed_out") or driver.get("deadline_hit")):
            break
    else:
        raise grader.InfraError(f"the visible-test check ran past {cap} s on its second try")
    out = (cmd.get("stdout") or "") + (cmd.get("stderr") or "")
    if language == "python":
        ok = cmd.get("exit") in (0, 5) and "error" not in out.lower().split("collected")[-1][:400].lower()
        ok = cmd.get("exit") in (0, 5) and not re.search(r"\d+ errors? (during collection|in )", out)
    else:
        ok = cmd.get("exit") == 0
    return ok, out[-600:]


def level_for(record: dict) -> int:
    files, lines = patch_size(record["gold_patch"])
    return editreal.level_of(files, lines, record["kind"] == "feature")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp-{secrets.token_hex(3)}")
    tmp.write_text(json.dumps(value, indent=1, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def write_lines(path: Path, lines: list[str]) -> None:
    """Replace a JSONL file in one rename, so a reader never sees half of it."""
    tmp = path.with_name(path.name + f".tmp-{secrets.token_hex(3)}")
    tmp.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default
