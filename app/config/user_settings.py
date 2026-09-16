"""Lightweight user settings (preferences only, never music data).

Stored as a small JSON file in the user's config directory:

- Windows: ``%APPDATA%/piano_to_harmonica_score/settings.json``
- others:  ``$XDG_CONFIG_HOME/piano_to_harmonica_score/settings.json``
           or ``~/.config/piano_to_harmonica_score/settings.json``

Priority is always: **CLI argument > user settings > built-in default**.

A missing file, a corrupt file or an invalid field must never prevent the CLI
from running: loaders fall back to defaults (per field where possible) and
report what happened.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any

from app.core.errors import PianoScoreError

APP_DIR_NAME = "piano_to_harmonica_score"
SETTINGS_FILE_NAME = "settings.json"

#: Allowed keys and their built-in defaults. Only preferences live here; score
#: data (notes, transpose actually applied by an explicit flag, ...) does not.
DEFAULTS: dict[str, Any] = {
    "default_style": "practice",
    "default_section_duration": 6.0,
    "default_range_mode": "octave_fold",
    "default_transpose": "auto",
    "segment_context_before": 2.0,
    "segment_context_after": 2.0,
    # GUI preferences (remembered conveniences, never score data).
    "gui_last_input_dir": "",
    "gui_last_output_dir": "",
    "gui_window_geometry": "",
}

_KNOWN_STYLES = frozenset({"compact", "practice"})
_KNOWN_RANGE_MODES = frozenset({"octave_fold", "drop", "nearest"})


@dataclass(frozen=True)
class UserSettings:
    default_style: str = "practice"
    default_section_duration: float = 6.0
    default_range_mode: str = "octave_fold"
    default_transpose: str | int = "auto"
    segment_context_before: float = 2.0
    segment_context_after: float = 2.0
    gui_last_input_dir: str = ""
    gui_last_output_dir: str = ""
    gui_window_geometry: str = ""


def settings_path() -> Path:
    """Cross-platform location of ``settings.json``."""
    appdata = os.environ.get("APPDATA")
    if appdata:
        base = Path(appdata)
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return base / APP_DIR_NAME / SETTINGS_FILE_NAME


def _validate_field(key: str, value: Any) -> Any:
    """Return the coerced value, or raise ``ValueError`` when invalid."""
    if key in ("default_style",):
        text = str(value).strip().lower()
        if text not in _KNOWN_STYLES:
            raise ValueError(f"expected one of {sorted(_KNOWN_STYLES)}, got {value!r}")
        return text
    if key in ("default_range_mode",):
        text = str(value).strip().lower()
        if text not in _KNOWN_RANGE_MODES:
            raise ValueError(f"expected one of {sorted(_KNOWN_RANGE_MODES)}, got {value!r}")
        return text
    if key in ("default_transpose",):
        if isinstance(value, str):
            text = value.strip().lower()
            if text == "auto":
                return "auto"
            value = int(text)
        number = int(value)
        if not -12 <= number <= 12:
            raise ValueError(f"transpose must be 'auto' or within -12..+12, got {value!r}")
        return number
    if key in ("default_section_duration", "segment_context_before", "segment_context_after"):
        number = float(value)
        if number <= 0 or number != number or number in (float("inf"), float("-inf")):
            raise ValueError(f"{key} must be a positive number, got {value!r}")
        return number
    if key in ("gui_last_input_dir", "gui_last_output_dir"):
        return str(value) if value is None else str(value)
    if key == "gui_window_geometry":
        text = str(value).strip()
        if text and ("x" not in text or not all(part.strip().isdigit() for part in text.split("x", 1))):
            raise ValueError(f"{key} must look like '1200x800', got {value!r}")
        return text
    raise ValueError(f"unknown setting {key!r}")


def parse_setting_value(key: str, raw: str) -> Any:
    """Coerce a raw CLI/config string (``"practice"``, ``"1.5"``, ...) for *key*."""
    if key not in DEFAULTS:
        raise PianoScoreError(
            f"unknown setting {key!r}; available: {', '.join(sorted(DEFAULTS))}"
        )
    try:
        return _validate_field(key, raw)
    except ValueError as exc:
        raise PianoScoreError(f"invalid value for {key}: {exc}") from exc


def load_settings(path: str | Path | None = None) -> tuple[UserSettings, list[str]]:
    """Load the user settings, falling back per field.

    Returns ``(settings, warnings)``; *warnings* carries human-readable notes
    about corrupt JSON or invalid fields (the CLI prints them as ``Warning:``).
    """
    target = Path(path) if path is not None else settings_path()
    notes: list[str] = []
    if not target.is_file():
        return UserSettings(), notes

    try:
        raw = target.read_text(encoding="utf-8")
        data = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        notes.append(f"settings file {target} is unreadable ({exc}); using defaults")
        return UserSettings(), notes

    if not isinstance(data, dict):
        notes.append(f"settings file {target} is not a JSON object; using defaults")
        return UserSettings(), notes

    values: dict[str, Any] = {}
    for key, value in data.items():
        if key not in DEFAULTS:
            notes.append(f"ignored unknown setting {key!r}")
            continue
        try:
            values[key] = _validate_field(key, value)
        except ValueError as exc:
            notes.append(f"ignored invalid setting {key}: {exc}")
    settings = UserSettings(**values)  # type: ignore[arg-type]
    return settings, notes


def save_settings(settings: UserSettings, path: str | Path | None = None) -> Path:
    """Write *settings* as JSON; returns the path."""
    target = Path(path) if path is not None else settings_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {field.name: getattr(settings, field.name) for field in fields(UserSettings)}
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return target


def with_updated_field(settings: UserSettings, key: str, value: Any) -> UserSettings:
    return replace(settings, **{key: value})  # type: ignore[arg-type]


__all__ = [
    "APP_DIR_NAME",
    "SETTINGS_FILE_NAME",
    "DEFAULTS",
    "UserSettings",
    "settings_path",
    "load_settings",
    "save_settings",
    "parse_setting_value",
    "with_updated_field",
]
