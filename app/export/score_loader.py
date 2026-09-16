"""Read a ``score.json`` back into typed objects.

Needed by the layout stage (and later by the renderer) so a score can be
re-laid-out or re-rendered without re-running the whole transcription pipeline.

The reader is tolerant about *optional* fields (``confidence``) and strict about
everything the schema requires.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from app.core.errors import ScoreDocumentError
from app.core.models import MAX_MIDI_PITCH, MIN_MIDI_PITCH, PlayableNote
from app.score.labels import NoteLabelSymbols

SUPPORTED_SCORE_VERSION = 1

_REQUIRED_DOCUMENT_FIELDS = ("version", "profile", "lanes", "duration", "notes")
_REQUIRED_NOTE_FIELDS = ("start", "duration", "source_pitch", "pitch", "lane", "key", "semitone", "octave")


@dataclass(frozen=True)
class ScoreDocument:
    """A parsed ``score.json``."""

    version: int
    profile_name: str
    lanes: tuple[str, ...]
    symbols: NoteLabelSymbols
    duration: float
    transpose: int
    auto_transpose: bool
    range_mode: str
    report: dict[str, Any]
    notes: tuple[PlayableNote, ...]
    #: Where the document came from (set by :func:`load_score_document`).
    source: str | None = None
    #: Original media range when the score was cropped (``None`` = full song).
    source_segment: tuple[float, float] | None = None

    @property
    def note_count(self) -> int:
        return len(self.notes)


def _require(document: dict[str, Any], key: str, path: Path) -> Any:
    if key not in document:
        raise ScoreDocumentError(f"{path} is missing the required field {key!r}")
    return document[key]


def _as_int(value: Any, name: str, path: Path) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ScoreDocumentError(f"{path}: {name} must be an int, got {value!r}")
    return value


def _as_float(value: Any, name: str, path: Path) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ScoreDocumentError(f"{path}: {name} must be a number, got {value!r}")
    return float(value)


def _as_bool(value: Any, name: str, path: Path) -> bool:
    if not isinstance(value, bool):
        raise ScoreDocumentError(f"{path}: {name} must be a bool, got {value!r}")
    return value


def _parse_note(raw: Any, path: Path, index: int) -> PlayableNote:
    if not isinstance(raw, dict):
        raise ScoreDocumentError(f"{path}: note #{index} must be a JSON object")
    missing = [key for key in _REQUIRED_NOTE_FIELDS if key not in raw]
    if missing:
        raise ScoreDocumentError(f"{path}: note #{index} is missing {', '.join(missing)}")

    pitch = _as_int(raw["pitch"], "pitch", path)
    source_pitch = _as_int(raw["source_pitch"], "source_pitch", path)
    lane = _as_int(raw["lane"], "lane", path)
    octave = _as_int(raw["octave"], "octave", path)
    start = _as_float(raw["start"], "start", path)
    duration = _as_float(raw["duration"], "duration", path)
    key = raw["key"]
    semitone = _as_bool(raw["semitone"], "semitone", path)

    for name, value in (("pitch", pitch), ("source_pitch", source_pitch)):
        if not MIN_MIDI_PITCH <= value <= MAX_MIDI_PITCH:
            raise ScoreDocumentError(
                f"{path}: note #{index} has {name} {value} outside 0..127"
            )
    if start < 0:
        raise ScoreDocumentError(f"{path}: note #{index} has a negative start")
    if duration <= 0:
        raise ScoreDocumentError(f"{path}: note #{index} has a non-positive duration")

    confidence = raw.get("confidence")
    if confidence is not None:
        confidence = _as_float(confidence, "confidence", path)
        if not 0.0 <= confidence <= 1.0:
            raise ScoreDocumentError(f"{path}: note #{index} has confidence outside 0..1")

    if not isinstance(key, str) or not key:
        raise ScoreDocumentError(f"{path}: note #{index} has an invalid key {key!r}")

    try:
        return PlayableNote(
            source_pitch=source_pitch,
            pitch=pitch,
            start=start,
            duration=duration,
            lane=lane,
            key_label=key,
            semitone=semitone,
            octave_offset=octave,
            confidence=confidence,
        )
    except (TypeError, ValueError) as exc:
        raise ScoreDocumentError(f"{path}: note #{index} is invalid: {exc}") from exc


def parse_score_document(data: Any, path: Path | None = None) -> ScoreDocument:
    """Validate parsed JSON data and turn it into a :class:`ScoreDocument`."""
    label = str(path) if path is not None else "<score document>"
    if not isinstance(data, dict):
        raise ScoreDocumentError(f"{label} must contain a JSON object")

    missing = [key for key in _REQUIRED_DOCUMENT_FIELDS if key not in data]
    if missing:
        raise ScoreDocumentError(f"{label} is missing required field(s): {', '.join(missing)}")

    version = _as_int(data["version"], "version", label)
    if version != SUPPORTED_SCORE_VERSION:
        raise ScoreDocumentError(
            f"{label} has unsupported score version {version} (expected {SUPPORTED_SCORE_VERSION})"
        )

    lanes_raw = data["lanes"]
    if not isinstance(lanes_raw, list) or not lanes_raw:
        raise ScoreDocumentError(f"{label}: lanes must be a non-empty JSON array")
    if not all(isinstance(lane, str) and lane for lane in lanes_raw):
        raise ScoreDocumentError(f"{label}: every lane must be a non-empty string")
    lanes = tuple(lanes_raw)

    notes_raw = data["notes"]
    if not isinstance(notes_raw, list):
        raise ScoreDocumentError(f"{label}: notes must be a JSON array")

    notes = tuple(_parse_note(raw, Path(label), index) for index, raw in enumerate(notes_raw))

    for note in notes:
        if not 0 <= note.lane < len(lanes):
            raise ScoreDocumentError(
                f"{label}: note lane {note.lane} is outside 0..{len(lanes) - 1}"
            )

    conversion = data.get("conversion", {})
    if not isinstance(conversion, dict):
        raise ScoreDocumentError(f"{label}: conversion must be a JSON object")

    segment_raw = data.get("source_segment")
    segment = _parse_source_segment(segment_raw, label)

    return ScoreDocument(
        version=version,
        profile_name=str(data["profile"]),
        lanes=lanes,
        symbols=NoteLabelSymbols.from_dict(data.get("symbols")),
        duration=_as_float(data["duration"], "duration", label),
        source_segment=segment,
        transpose=_as_int(conversion.get("transpose", 0), "transpose", label),
        auto_transpose=_as_bool(conversion.get("auto_transpose", False), "auto_transpose", label),
        range_mode=str(conversion.get("range_mode", "")),
        report=dict(data.get("report", {})),
        notes=notes,
    )


def _parse_source_segment(raw: Any, label: str) -> tuple[float, float] | None:
    """``{"start": 48.0, "end": 82.0}`` -> tuple; ``null``/absent -> ``None``."""
    if raw is None:
        return None
    if not isinstance(raw, dict) or set(raw) != {"start", "end"}:
        raise ScoreDocumentError(
            f"{label}: source_segment must be an object with exactly 'start' and 'end'"
        )
    start = _as_float(raw["start"], "source_segment.start", label)
    end = _as_float(raw["end"], "source_segment.end", label)
    if end <= start:
        raise ScoreDocumentError(
            f"{label}: source_segment end ({end:g}) must be greater than start ({start:g})"
        )
    return (start, end)


def load_score_document(path: str | Path) -> ScoreDocument:
    """Load and validate a ``score.json`` file."""
    resolved = Path(path)
    if not resolved.is_file():
        raise ScoreDocumentError(f"score file not found: {resolved}")
    try:
        raw = resolved.read_text(encoding="utf-8")
    except OSError as exc:
        raise ScoreDocumentError(f"cannot read score file {resolved}: {exc}") from exc
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ScoreDocumentError(f"{resolved} is not valid JSON: {exc}") from exc

    document = parse_score_document(data, resolved)
    return replace(document, source=str(resolved))


__all__ = [
    "SUPPORTED_SCORE_VERSION",
    "ScoreDocument",
    "parse_score_document",
    "load_score_document",
]
