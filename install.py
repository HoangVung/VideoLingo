"""Compatibility wrapper for the stage-based installer.

Historically users ran ``python install.py`` and the app launched at the end.
Keep that behavior here while moving the real installation logic to
``installer.py`` so setup_env.py and launchers can reuse it safely.
"""

from __future__ import annotations

import sys

from installer import main


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--check" not in args and "--launch" not in args and "--no-launch" not in args:
        args.append("--launch")
    raise SystemExit(main(args))
