"""Build, check and clean up the episode sandbox (production plan step 3).

Run inside WSL2 (Ubuntu) from the repo root, with Docker Desktop's WSL integration on and the
Linux virtual environment from the README:

    .venv-linux/bin/python scripts/sandbox_ctl.py build    # build the sandbox images from the pinned base image
    .venv-linux/bin/python scripts/sandbox_ctl.py check    # run the regression tests in tests/
    .venv-linux/bin/python scripts/sandbox_ctl.py clean    # remove containers and episode directories a crashed run left behind
"""

import argparse
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rlm_rl import bridge, editbox, repl_tools, sandbox  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["build", "check", "clean"])
    parser.add_argument("--force", action="store_true", help="rebuild the image even if it exists")
    args = parser.parse_args()

    if args.command == "build":
        tag = sandbox.build_image(force=args.force)
        synth = editbox.build_synth_image(force=args.force)  # edit_synth's image; its build fetches git
        bundle = editbox.build_python_bundle(force=args.force)  # edit_real's standalone Python
        print(f"image {tag}, edit_synth image {synth} (base {sandbox.BASE_IMAGE}), Python bundle {bundle.name}, "
              f"bridge API {bridge.API_VERSION}, prompt {bridge.PROMPT_VERSION} {bridge.PROMPT_HASH[:12]}, "
              f"tools {repl_tools.TOOLS_VERSION}")
        return 0
    if args.command == "clean":
        print(f"removed {sandbox.cleanup_stale()} containers")
        return 0
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"), top_level_dir=str(ROOT))
    return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
