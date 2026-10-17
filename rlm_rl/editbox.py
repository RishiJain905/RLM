"""The edit sandbox: one container per edit episode, with the task's repository writable at /work/repo (spec §12.4).

Read episodes are unchanged (sandbox.py). An edit episode differs in five ways:

1. **Image.** `edit_synth` runs in `rlm-edit-synth`: the read sandbox's pinned base plus git. Each `edit_real`
   repository has a derived image,
   `rlm-edit-<repo>:<hash>`, built once by scripts/build_edit_data.py: the image's own checkout
   is deleted (for an injected bug it is the answer), other installed copies and caches go, and
   the path the image's environment expects (e.g. /testbed) becomes a symlink to /work/repo.
2. **Our Python.** The REPL runs on a standalone CPython 3.12 mounted read-only at
   /opt/rlm/python, so it works in a Go, Rust or Java image and in a Python image pinned to an old
   version. The bundle is taken from the pinned `python:3.12-slim` image the read sandbox is built
   from, with its own dynamic loader and C libraries, and started through that loader, so it needs
   nothing from the image it runs in (`build_python_bundle`). The repository's toolchain runs
   through subprocess.
3. **A writable repository.** /work/repo is a bind mount of a per-episode host folder with a random
   name, holding the repository at the task's base: one commit in a fresh `git init`, with no other
   history, branches, tags, reflog or remotes. It survives a REPL restart. Hidden test files are
   never in it.
4. **Processes allowed.** Landlock lets the image's programs run and confines writes to
   /work/repo, /scratch, /tmp and the image's cache folders (repl_worker.py, edit mode). The REPL
   has the seven coding tools of repl_tools.py, mounted with the worker at /opt/rlm (spec §12.1).
5. **Limits** from `bridge.EDIT_LIMITS`: 2 CPUs, 4 GB, 256 pids, 300 s cells.

Everything else from step 3 holds: --network none, no capabilities, a non-root user, an
environment that holds nothing from the host, the bridge socket as the only way out, and the
infra rules. Two deliberate differences *(mine)*: a derived image runs with a per-container
writable layer instead of a read-only root, because Go, Cargo, Maven and Gradle must write their
warm build caches (Landlock still confines every write, and the layer is discarded with the
container); and edit containers run as the host user's own non-root uid, so the host can read,
diff and remove what the episode wrote.

**The diff never trusts the episode's .git.** The model controls /work/repo/.git (config,
hooks, attributes), so the host never runs git against it. Each task's base is built once into a
host cache (`Workspace.ensure`): the files and a private git directory with the base commit. The
episode's diff is taken with a throwaway git directory that borrows the cache's objects, and a git
environment that reads no system, global or repository config (`diff`).
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import hashlib
import json
import os
import secrets
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from . import bridge, sandbox
from .bridge import Complete, InfraError
from .sandbox import CellResult, Sandbox, SandboxLimits, docker_cli_env

REPO_PATH = bridge.EDIT_REPO_PATH
RUNTIME_PATH = "/opt/rlm"
PYTHON_PATH = "/opt/rlm/python"
WORKER = Path(__file__).with_name("repl_worker.py")
DRIVER = Path(__file__).with_name("grader_driver.py")
TOOLS = Path(__file__).with_name("repl_tools.py")  # the coding tools, imported by the worker in edit mode

# edit_synth's image: the read sandbox's pinned base plus git, which the coding tools' diff() and the model's own
# `git status` need. The worker comes from the /opt/rlm mount, so the image doesn't change when the worker does.
SYNTH_REPO = "rlm-edit-synth"
SYNTH_DOCKERFILE = """\
FROM {base}
RUN apt-get update \\
 && apt-get install -y --no-install-recommends git \\
 && rm -rf /var/lib/apt/lists/* \\
 && rm -rf /usr/local/lib/python3.12/site-packages/pip /usr/local/lib/python3.12/site-packages/pip-* \\
           /usr/local/bin/pip /usr/local/bin/pip3 /usr/local/bin/pip3.12 /usr/local/lib/python3.12/ensurepip/_bundled \\
 && mkdir -p /opt/rlm /episode /scratch /work/repo
"""

# The bundle's source: the read sandbox's own pinned base image (Python 3.12.15 on Debian bookworm).
PYTHON_SOURCE_IMAGE = sandbox.BASE_IMAGE
BUNDLE_PYTHON = [f"{PYTHON_PATH}/lib/ld-linux-x86-64.so.2", "--library-path", f"{PYTHON_PATH}/lib",
                 f"{PYTHON_PATH}/bin/python3.12"]
SANDBOX_PYTHON = ["/usr/local/bin/python3"]  # rlm-sandbox's own Python, the same build as the bundle
# rlm-sandbox's image environment is the python image's; this is all the REPL needs from it
SANDBOX_ENV = {"PATH": "/usr/local/bin:/usr/bin:/bin", "LANG": "C.UTF-8", "PYTHONDONTWRITEBYTECODE": "1"}

GIT_DATE = "2000-01-01T00:00:00+00:00"  # the base commit's date, so the same tree always has the same commit
WORKSPACE_FORMAT = 1
CACHE_MAX_GB = 20.0  # workspaces kept on the host before the oldest unused ones go *(mine)*

EDIT_LABEL = "rlm.edit"

# Image labels a derived image carries (scripts/build_edit_data.py writes them).
L_WRITABLE, L_UID, L_ALIAS, L_LANGUAGE = "rlm.edit.writable", "rlm.edit.uid", "rlm.edit.repo_alias", "rlm.edit.language"


def edit_sandbox_limits(lim: bridge.Limits = bridge.EDIT_LIMITS) -> SandboxLimits:
    """Spec §12.3's container limits for the edit sandbox. Startup gets longer: real images are large, and they live on
    the USB data drive, where creating a container took minutes under load (a slow start must not be an infra error).

    The wall-clock cap (for a cell stuck on sub-calls) always sits a full default cap above the cell's own limit. It was
    a fixed 1,800 s, under the raised limits of slow tasks (JavaScript 3,570 s, Go up to 2,760 s), so a cell that ran or
    hung past 30 minutes was flagged infra instead of timing out as the model's own (step 11, 2026-10-17: a
    `node` script the model left hanging)."""
    return SandboxLimits(cpus=lim.cpus, memory_mb=lim.memory_mb, pids=lim.pids, scratch_mb=256,
                         cell_timeout_s=lim.cell_timeout_s, startup_timeout_s=600.0,
                         cell_wall_cap_s=lim.cell_timeout_s + SandboxLimits.cell_wall_cap_s)


TMP_MB = 1024  # /tmp in an edit container: test binaries, jest caches, pytest's tmp_path


def cache_root() -> Path:
    """Where workspaces, the runtime folder and the Python bundle live on the host: WSL's own disk, not the Windows
    drive, so containers start fast. `$RLM_EDIT_CACHE` overrides."""
    return Path(os.environ.get("RLM_EDIT_CACHE") or Path.home() / ".cache" / "rlm-edit")


def host_user() -> tuple[int, int]:
    uid, gid = os.getuid(), os.getgid()
    if uid == 0:
        raise RuntimeError("edit containers run as the host user, which must not be root")
    return uid, gid


# ----------------------------------------------------------------------------- git on the host


def git_env(home: Path | str) -> dict:
    """An environment for host git that reads no system or global config and runs nothing from the work tree."""
    env = {k: os.environ[k] for k in ("PATH", "LANG") if k in os.environ}
    env.update(HOME=str(home), GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL="/dev/null", GIT_TERMINAL_PROMPT="0",
               GIT_AUTHOR_NAME="rlm", GIT_AUTHOR_EMAIL="rlm@localhost", GIT_COMMITTER_NAME="rlm",
               GIT_COMMITTER_EMAIL="rlm@localhost", GIT_AUTHOR_DATE=GIT_DATE, GIT_COMMITTER_DATE=GIT_DATE)
    return env


GIT_SAFE = ["-c", "core.fsmonitor=false", "-c", "core.untrackedCache=false", "-c", "core.hooksPath=/dev/null",
            "-c", "core.quotePath=false", "-c", "core.autocrlf=false", "-c", "core.symlinks=true",
            "-c", "diff.noprefix=false", "-c", "diff.mnemonicPrefix=false", "-c", "diff.relative=false",
            "-c", "core.attributesFile=/dev/null", "-c", "advice.detachedHead=false"]


def git(*args: str, env: dict, cwd: Path | None = None, check: bool = True, input: bytes | None = None,
        timeout: float = 300) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *GIT_SAFE, *args], env=env, cwd=cwd, capture_output=True, check=check,
                          input=input, timeout=timeout)


# ----------------------------------------------------------------------------- workspaces (a task's base)


@dataclass
class Workspace:
    """A task's repository at its base, built once on the host and copied into each episode (spec §12.4).

    `build(folder)` writes the base tree into an empty folder: generated files for edit_synth; for
    edit_real, the repository snapshot with the task's bug applied or feature removed and its hidden
    test files taken out. `key` must change whenever what `build` writes would.
    """

    key: str
    build: Callable[[Path], None] = field(repr=False)

    def path(self, root: Path | None = None) -> Path:
        return (root or cache_root()) / "workspaces" / f"ws-{self.key}"

    def ensure(self, root: Path | None = None) -> Path:
        """Build the workspace if it isn't cached. Safe to call from several processes and threads at once."""
        final = self.path(root)
        if (final / "meta.json").exists():
            os.utime(final / "meta.json")  # marks it used, for the cache's least-recently-used cleanup
            return final
        import fcntl  # Linux only, like the sandbox

        final.parent.mkdir(parents=True, exist_ok=True)
        with open(final.with_suffix(".lock"), "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if (final / "meta.json").exists():
                return final
            tmp = final.with_name(f"{final.name}.tmp-{secrets.token_hex(4)}")
            try:
                tree = tmp / "tree"
                tree.mkdir(parents=True)
                self.build(tree)
                meta = _commit_base(tmp)
                (tmp / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
                if final.exists():
                    shutil.rmtree(final)
                tmp.rename(final)
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
        return final


def _commit_base(tmp: Path) -> dict:
    """Commit tmp/tree as the single base commit, then move its git directory to tmp/git, packed, with no reflog."""
    tree, env = tmp / "tree", git_env(tmp)
    git("-c", "init.defaultBranch=main", "init", "-q", str(tree), env=env)
    git("add", "-A", env=env, cwd=tree)
    git("commit", "-q", "--allow-empty", "--no-verify", "-m", "base", env=env, cwd=tree)
    base = git("rev-parse", "HEAD", env=env, cwd=tree).stdout.decode().strip()
    git("repack", "-a", "-d", "-q", env=env, cwd=tree)
    git("prune-packed", env=env, cwd=tree)
    files = git("ls-files", "-z", env=env, cwd=tree).stdout.split(b"\0")
    (tree / ".git").rename(tmp / "git")
    gitdir = tmp / "git"
    for name in ("logs", "hooks", "description", "COMMIT_EDITMSG", "ORIG_HEAD", "FETCH_HEAD"):
        target = gitdir / name
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()
    (gitdir / "config").write_text("[core]\n\trepositoryformatversion = 0\n\tfilemode = true\n\tbare = false\n"
                                   "\tlogallrefupdates = false\n", encoding="utf-8")
    return {"format": WORKSPACE_FORMAT, "base": base, "files": sum(1 for f in files if f), "built_at": time.time()}


def workspace_meta(ws_dir: Path) -> dict:
    return json.loads((ws_dir / "meta.json").read_text(encoding="utf-8"))


def materialize(ws_dir: Path, dest: Path) -> None:
    """Copy a workspace's base into a new episode folder, with its own .git holding only the base commit."""
    subprocess.run(["cp", "-a", "--", str(ws_dir / "tree"), str(dest)], check=True, capture_output=True)
    subprocess.run(["cp", "-a", "--", str(ws_dir / "git"), str(dest / ".git")], check=True, capture_output=True)


def diff(ws_dir: Path, worktree: Path, scratch: Path) -> str:
    """The episode's change against its base, as a git diff (spec §12.5 step 1). New files are included; files the
    repository's .gitignore ignores are not. `scratch` is a private empty folder for the throwaway git directory."""
    base = workspace_meta(ws_dir)["base"]
    gitdir = scratch / "git"
    env = git_env(scratch)
    git("init", "-q", "--bare", str(gitdir), env=env)
    (gitdir / "objects" / "info" / "alternates").write_text(str((ws_dir / "git" / "objects").resolve()) + "\n")
    env = dict(env, GIT_DIR=str(gitdir), GIT_WORK_TREE=str(worktree), GIT_INDEX_FILE=str(scratch / "index"))
    git("read-tree", base, env=env, cwd=scratch)
    git("add", "-A", "--", ".", env=env, cwd=worktree)
    out = git("diff", "--cached", "--no-color", "--no-ext-diff", "--no-textconv", "--no-renames", "--full-index",
              base, env=env, cwd=worktree)
    return out.stdout.decode("utf-8", "replace")


# Build outputs and caches a test run writes into the repository: not an edit.
CACHE_DIRS = frozenset({"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".hypothesis", ".tox", ".nox",
                        "node_modules", ".gradle", ".cache", "target", ".jest-cache", ".turbo", ".parcel-cache"})
CACHE_SUFFIXES = (".pyc", ".pyo")


def fingerprint(worktree: Path) -> str:
    """A cheap fingerprint of a work tree's files (paths, sizes, modes and modification times), with .git, build
    outputs and caches left out. A cell that changes it edited the repository."""
    h = hashlib.sha1()
    for dirpath, dirnames, filenames in os.walk(worktree):
        dirnames[:] = sorted(d for d in dirnames if d not in CACHE_DIRS
                             and not (dirpath == str(worktree) and d == ".git"))
        for name in sorted(filenames):
            if name.endswith(CACHE_SUFFIXES):
                continue
            path = os.path.join(dirpath, name)
            try:
                st = os.lstat(path)
            except OSError:
                continue
            h.update(f"{os.path.relpath(path, worktree)}\0{st.st_size}\0{st.st_mtime_ns}\0{st.st_mode}\n".encode(
                "utf-8", "replace"))
    return h.hexdigest()


def remove_tree(path: Path) -> None:
    """Remove an episode folder. The model may have made folders it can't be removed from; fix their modes first."""
    if not path.exists():
        return
    try:
        shutil.rmtree(path)
        return
    except OSError:
        pass
    for dirpath, dirnames, _ in os.walk(path):
        for d in dirnames:
            with contextlib.suppress(OSError):
                os.chmod(os.path.join(dirpath, d), 0o700)
    with contextlib.suppress(OSError):
        os.chmod(path, 0o700)
    shutil.rmtree(path, ignore_errors=True)


def prune_cache(root: Path | None = None, max_gb: float = CACHE_MAX_GB) -> int:
    """Drop the least recently used workspaces until the cache fits in `max_gb`. Returns how many went."""
    folder = (root or cache_root()) / "workspaces"
    if not folder.is_dir():
        return 0
    entries = []
    for ws in folder.glob("ws-*"):
        meta = ws / "meta.json"
        if ws.is_dir() and meta.exists():
            size = sum(f.stat().st_size for f in ws.rglob("*") if f.is_file() and not f.is_symlink())
            entries.append((meta.stat().st_mtime, size, ws))
    total, removed = sum(e[1] for e in entries), 0
    for _, size, ws in sorted(entries):
        if total <= max_gb * 1e9:
            break
        shutil.rmtree(ws, ignore_errors=True)
        total -= size
        removed += 1
    return removed


# ----------------------------------------------------------------------------- the runtime mounted at /opt/rlm


def runtime_dir(root: Path | None = None) -> Path:
    """The folder mounted read-only at /opt/rlm in every edit container: the REPL worker, its coding tools and the
    grader driver, and an empty `python/` mount point for the bundle."""
    blob = WORKER.read_bytes() + b"\0" + DRIVER.read_bytes() + b"\0" + TOOLS.read_bytes()
    folder = (root or cache_root()) / f"runtime-{hashlib.sha256(blob).hexdigest()[:12]}"
    if not (folder / "grader_driver.py").exists():
        tmp = folder.with_name(folder.name + f".tmp-{secrets.token_hex(4)}")
        (tmp / "python").mkdir(parents=True)
        shutil.copy2(WORKER, tmp / "repl_worker.py")
        shutil.copy2(TOOLS, tmp / "repl_tools.py")
        shutil.copy2(DRIVER, tmp / "grader_driver.py")
        for f in tmp.iterdir():
            os.chmod(f, 0o755 if f.is_dir() else 0o644)
        try:
            tmp.rename(folder)
        except OSError:  # another process made it first
            shutil.rmtree(tmp, ignore_errors=True)
    return folder


BUNDLE_SCRIPT = r"""
set -e
mkdir -p /out/bin /out/lib
cp -a /usr/local/lib/python3.12 /out/lib/
cp -L /usr/local/bin/python3.12 /out/bin/python3.12
cp -L /usr/local/lib/libpython3.12.so.1.0 /out/lib/
for f in $(ldd /usr/local/bin/python3.12 /usr/local/lib/libpython3.12.so.1.0 /usr/local/lib/python3.12/lib-dynload/*.so \
           | awk '$2 == "=>" && $3 ~ /^\// {print $3}' | sort -u); do cp -L "$f" /out/lib/; done
cp -L /lib64/ld-linux-x86-64.so.2 /out/lib/ld-linux-x86-64.so.2
rm -rf /out/lib/python3.12/site-packages /out/lib/python3.12/ensurepip /out/lib/python3.12/idlelib \
       /out/lib/python3.12/tkinter /out/lib/python3.12/turtledemo /out/lib/python3.12/lib2to3
mkdir -p /out/lib/python3.12/site-packages
/usr/local/bin/python3.12 -m compileall -q -j 0 /out/lib/python3.12 >/dev/null || true
/out/lib/ld-linux-x86-64.so.2 --library-path /out/lib /out/bin/python3.12 -I -c 'import ssl, sqlite3, zlib, lzma, bz2, json, decimal, ctypes; print("bundle ok")'
"""


def python_bundle_id() -> str:
    return hashlib.sha256((PYTHON_SOURCE_IMAGE + BUNDLE_SCRIPT).encode()).hexdigest()[:12]


def python_bundle_dir(root: Path | None = None) -> Path:
    return (root or cache_root()) / f"python-{python_bundle_id()}"


def build_python_bundle(root: Path | None = None, docker: str = "docker", force: bool = False) -> Path:
    """Assemble the standalone Python from the pinned python:3.12-slim image, with no network (spec §12.4)."""
    folder = python_bundle_dir(root)
    if (folder / "bin" / "python3.12").exists() and not force:
        return folder
    uid, gid = host_user()
    tmp = folder.with_name(folder.name + f".tmp-{secrets.token_hex(4)}")
    tmp.mkdir(parents=True)
    try:
        out = subprocess.run([docker, "run", "--rm", "--network", "none", "--pull", "never", "--user", f"{uid}:{gid}",
                              "--mount", f"type=bind,source={tmp},target=/out", "--entrypoint", "/bin/sh",
                              PYTHON_SOURCE_IMAGE, "-c", BUNDLE_SCRIPT], capture_output=True, text=True,
                             env=docker_cli_env(), timeout=600)
        if out.returncode or "bundle ok" not in out.stdout:
            raise RuntimeError(f"building the Python bundle failed: {(out.stderr or out.stdout)[-800:]}")
        if folder.exists():
            shutil.rmtree(folder)
        tmp.rename(folder)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return folder


# ----------------------------------------------------------------------------- images


def synth_dockerfile(base: str = sandbox.BASE_IMAGE) -> str:
    return SYNTH_DOCKERFILE.format(base=base)


def synth_image_tag(base: str = sandbox.BASE_IMAGE) -> str:
    return f"{SYNTH_REPO}:{hashlib.sha256(synth_dockerfile(base).encode()).hexdigest()[:12]}"


def build_synth_image(base: str = sandbox.BASE_IMAGE, docker: str = "docker", force: bool = False) -> str:
    """Build edit_synth's image once. The build fetches git from Debian, so it needs the network; episodes never do."""
    tag = synth_image_tag(base)
    exists = subprocess.run([docker, "image", "inspect", tag], capture_output=True, env=docker_cli_env()).returncode == 0
    if exists and not force:
        return tag
    subprocess.run([docker, "build", "--pull=false", "-t", tag, "-"], input=synth_dockerfile(base).encode(), check=True,
                   capture_output=True, env=docker_cli_env())
    return tag


@dataclass(frozen=True)
class EditImage:
    """How to run an edit episode's (and its grader's) container in one image (spec §12.4)."""

    tag: str
    python: tuple[str, ...]  # how our Python starts inside it
    env: tuple[tuple[str, str], ...] = ()  # set on top of the image's own environment
    writable: tuple[str, ...] = ()  # the toolchain's cache folders, besides /work/repo, /scratch and /tmp
    read_only: bool = True
    bundle: bool = False  # mount the standalone Python at /opt/rlm/python
    env_reset: bool = False  # start from an empty environment (rlm-sandbox, as in read mode)
    language: str = "python"

    @classmethod
    def synth(cls, tag: str | None = None) -> EditImage:
        """edit_synth's image: the read sandbox's base plus git (build_synth_image), read-only, with its own Python."""
        return cls(tag or synth_image_tag(), tuple(SANDBOX_PYTHON), tuple(SANDBOX_ENV.items()), (), True, False,
                   True, "python")

    @classmethod
    def derived(cls, tag: str, docker: str = "docker") -> EditImage:
        """An edit_real repository's derived image, described by the labels build_edit_data.py gave it."""
        out = subprocess.run([docker, "image", "inspect", "--format", "{{json .Config.Labels}}", tag],
                             capture_output=True, text=True, env=docker_cli_env())
        if out.returncode != 0:
            raise InfraError(f"edit image {tag} is missing: build it with scripts/build_edit_data.py")
        labels = json.loads(out.stdout or "{}") or {}
        if labels.get(L_UID) and int(labels[L_UID]) != os.getuid():
            raise RuntimeError(f"{tag} was built for uid {labels[L_UID]}, not this user's {os.getuid()}")
        writable = tuple(p for p in (labels.get(L_WRITABLE) or "").split(":") if p.startswith("/"))
        return cls(tag, tuple(BUNDLE_PYTHON), (), writable, False, True, False, labels.get(L_LANGUAGE) or "python")


_DERIVED: dict[str, EditImage] = {}


def derived_image(tag: str) -> EditImage:
    if tag not in _DERIVED:
        _DERIVED[tag] = EditImage.derived(tag)
    return _DERIVED[tag]


def container_args(docker: str, name: str, image: EditImage, limits: SandboxLimits, run_id: str, *,
                   repo_dir: Path, runtime: Path, bundle: Path | None, episode_dir: Path | None = None,
                   extra_mounts: list[str] = (), kind: str = "episode") -> list[str]:
    """`docker run` flags shared by an edit episode's container and its grader's (everything but the command)."""
    uid, gid = host_user()
    lim = limits
    args = [
        docker, "run", "-i", "--pull", "never", "--name", name,
        "--label", f"{sandbox.LABEL}=1", "--label", f"rlm.run={run_id}", "--label", f"{EDIT_LABEL}={kind}",
        "--network", "none", "--ipc", "none",
        "--tmpfs", f"/scratch:rw,nosuid,nodev,exec,size={lim.scratch_mb}m,uid={uid},gid={gid},mode=0700",
        "--tmpfs", f"/tmp:rw,nosuid,nodev,exec,size={TMP_MB}m,mode=1777",
        "--user", f"{uid}:{gid}", "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
        "--cpus", str(lim.cpus), "--memory", f"{lim.memory_mb}m", "--memory-swap", f"{lim.memory_mb}m",
        "--pids-limit", str(lim.pids), "--ulimit", "core=0", "--hostname", "sandbox", "--log-driver", "none",
        "--workdir", REPO_PATH,
        "--mount", f"type=bind,source={repo_dir},target={REPO_PATH}",
        "--mount", f"type=bind,source={runtime},target={RUNTIME_PATH},readonly",
        "-e", "HOME=/scratch", "-e", "TMPDIR=/tmp",
    ]
    if image.read_only:
        args.append("--read-only")
    if image.bundle:
        if bundle is None:
            raise InfraError("the standalone Python bundle is missing: run scripts/build_edit_data.py runtime")
        args += ["--mount", f"type=bind,source={bundle},target={PYTHON_PATH},readonly"]
    if episode_dir is not None:
        args += ["--mount", f"type=bind,source={episode_dir},target=/episode,readonly"]
    for key, value in image.env:
        args += ["-e", f"{key}={value}"]
    return args + list(extra_mounts)


def python_command(image: EditImage, script: str, *args: str) -> tuple[str, list[str]]:
    """(entrypoint, arguments after the image) to run one of our scripts on our Python in `image`."""
    if image.env_reset:  # rlm-sandbox: an empty environment, as in read mode, plus what edit mode needs
        env = [f"{k}={v}" for k, v in image.env] + ["HOME=/scratch", "TMPDIR=/tmp"]
        return "/usr/bin/env", ["-i", *env, *image.python, "-I", "-B", script, *args]
    return image.python[0], [*image.python[1:], "-I", "-B", script, *args]


# ----------------------------------------------------------------------------- the sandbox


class EditSandbox(Sandbox):
    """One edit episode's REPL: the repository at /work/repo, writable, in the task's image (spec §12.4).

        async with EditSandbox(context, complete, workspace=ws, image=EditImage.synth(), run_id="run-1") as box:
            result = await box.run_cell("print(open('README.md').read()[:200])")
            patch = await box.diff()
    """

    def __init__(self, context: str, complete: Complete, *, workspace: Workspace, image: EditImage,
                 limits: SandboxLimits | None = None, run_id: str = "adhoc", docker: str = "docker",
                 root: Path | None = None, max_subcalls: int = bridge.EDIT_LIMITS.max_subcalls,
                 messages: bridge.CellMessages | None = None, cache: Path | None = None):
        limits = limits or edit_sandbox_limits()
        messages = messages or bridge.cell_messages(limits=dataclasses.replace(
            bridge.EDIT_LIMITS, cell_timeout_s=int(limits.cell_timeout_s)), edit=True)
        super().__init__(context, complete, limits=limits, image=image.tag, run_id=run_id, docker=docker, root=root,
                         max_subcalls=max_subcalls, messages=messages)
        self.edit_image = image
        self.workspace = workspace
        self.cache = cache
        self.repo_dir = self.dir.parent / f"repo-{self.episode_id}"
        self.diff_dir = self.dir.parent / f"diff-{self.episode_id}"
        self.ws_dir: Path | None = None
        self.runtime: Path | None = None
        self.bundle: Path | None = None
        self._fingerprint: str | None = None

    async def start(self) -> None:
        try:
            self.ws_dir = await asyncio.to_thread(self.workspace.ensure, self.cache)
            self.runtime = await asyncio.to_thread(runtime_dir, self.cache)
            if self.edit_image.bundle:
                self.bundle = python_bundle_dir(self.cache)
                if not (self.bundle / "bin" / "python3.12").exists():
                    raise InfraError("the standalone Python bundle is missing: run scripts/build_edit_data.py runtime")
            await asyncio.to_thread(materialize, self.ws_dir, self.repo_dir)
            self._fingerprint = await asyncio.to_thread(fingerprint, self.repo_dir)
        except InfraError:
            await asyncio.to_thread(remove_tree, self.repo_dir)
            raise
        except Exception as exc:  # a workspace that can't be built is the host's problem, never the model's
            await asyncio.to_thread(remove_tree, self.repo_dir)
            raise InfraError(f"edit workspace failed: {exc!r}") from exc
        await super().start()

    async def close(self) -> None:
        if self._closed:
            return
        await super().close()
        await asyncio.to_thread(remove_tree, self.repo_dir)
        await asyncio.to_thread(remove_tree, self.diff_dir)

    def _run_args(self, name: str) -> list[str]:
        args = container_args(self.docker, name, self.edit_image, self.limits, self.run_id, repo_dir=self.repo_dir,
                              runtime=self.runtime, bundle=self.bundle, episode_dir=self.dir)
        writable = ":".join(self.edit_image.writable)
        entry, rest = python_command(self.edit_image, f"{RUNTIME_PATH}/repl_worker.py", "--edit",
                                     *(["--writable", writable] if writable else []),
                                     "--base", workspace_meta(self.ws_dir)["base"],
                                     "--cell-timeout", f"{self.limits.cell_timeout_s:g}")
        return args + ["--entrypoint", entry, self.image, *rest]

    async def run_cell(self, code: str) -> CellResult:
        result = await super().run_cell(code)
        try:
            now = await asyncio.to_thread(fingerprint, self.repo_dir)
        except OSError as exc:
            raise InfraError(f"edit workspace unreadable: {exc!r}") from exc
        result.repo_changed = now != self._fingerprint
        self._fingerprint = now
        return result

    async def diff(self) -> str:
        """The repository's change against its base (spec §12.5 step 1). Raises InfraError if the host can't take it."""
        remove_tree(self.diff_dir)
        self.diff_dir.mkdir(mode=0o700)
        try:
            return await asyncio.to_thread(diff, self.ws_dir, self.repo_dir, self.diff_dir)
        except (OSError, subprocess.SubprocessError) as exc:
            raise InfraError(f"taking the episode's diff failed: {exc!r}") from exc
        finally:
            await asyncio.to_thread(remove_tree, self.diff_dir)
