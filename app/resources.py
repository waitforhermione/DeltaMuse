"""Resource path resolution that works from source *and* from a bundled EXE.

Never resolve assets relative to the current working directory: a bundled app
may be started from any folder. Use :func:`resource_path` for read-only bundled
assets and :func:`writable_dir` for anything the app writes.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def is_bundled() -> bool:
    """``True`` when running from a PyInstaller bundle."""
    return getattr(sys, "frozen", False)


def bundle_dir() -> Path:
    """Directory that contains bundled resources.

    - source run: the repository root (the parent of ``app/``)
    - onedir EXE: the directory holding the EXE and its ``_internal`` folder
    - one-file EXE: the temporary extraction dir (``sys._MEIPASS``)
    """
    if is_bundled():
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            return Path(meipass)
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def resource_path(*parts: str) -> Path:
    """Resolve a bundled, read-only resource relative to the bundle."""
    return bundle_dir().joinpath(*parts)


def app_data_dir() -> Path:
    """Per-user writable directory (roaming, created on demand)."""
    appdata = os.environ.get("APPDATA")
    if appdata:
        base = Path(appdata)
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    directory = base / "piano_to_harmonica_score"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


__all__ = ["is_bundled", "bundle_dir", "resource_path", "app_data_dir"]
