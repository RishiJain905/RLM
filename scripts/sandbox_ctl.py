"""Build, check and clean up the episode sandbox (production plan step 3).

Run inside WSL2 (Ubuntu) from the repo root, with Docker Desktop's WSL integration on and the
Linux virtual environment from the README:

    .venv-linux/bin/python scripts/sandbox_ctl.py build    # build the sandbox image from the pinned base image
    .venv-linux/bin/python scripts/sandbox_ctl.py check    # run the regression tests in tests/
    .venv-linux/bin/python scripts/sandbox_ctl.py clean    # remove containers and episode directories a crashed run left behind
"""

import argparse
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rlm_rl import bridge, sandbox  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["build", "check", "clean"])
    parser.add_argument("--force", action="store_true", help="rebuild the image even if it exists")
    args = parser.parse_args()

    if args.command == "build":
        tag = sandbox.build_image(force=args.force)
        print(f"image {tag} (base {sandbox.BASE_IMAGE}), bridge API {bridge.API_VERSION}, "
              f"prompt {bridge.PROMPT_VERSION} {bridge.PROMPT_HASH[:12]}")
        return 0
    if args.command == "clean":
        print(f"removed {sandbox.cleanup_stale()} containers")
        return 0
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"), top_level_dir=str(ROOT))
    return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
