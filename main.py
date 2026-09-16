#!/usr/bin/env python3
"""Entry point.

Usage::

    python main.py                       # start the GUI
    python main.py gui                   # start the GUI
    python main.py convert song.mp3      # headless CLI (all subcommands)
    python main.py analyze song.wav
    python main.py transcribe song.mp3 --output song.mid
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow ``python main.py`` (and ``python <abs path>/main.py``) from any cwd.
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else list(argv)
    if not args or (len(args) == 1 and args[0] == "gui"):
        from app.gui.main_window import run_gui

        return run_gui()
    if args[0] == "--smoke-test":
        from app.gui.main_window import run_smoke_test

        return run_smoke_test(args[1:])
    from app.cli import main as cli_main

    return cli_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
