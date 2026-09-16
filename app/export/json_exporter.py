"""``score.json`` export.

The document is a complete, self-describing description of the score: for every
note it says *when* to play, *which key* to press, whether a semitone modifier is
needed and whether to shift an octave.

Determinism is a hard requirement: exporting the same melody with the same config
twice must produce byte-for-byte identical output. There is therefore no
timestamp, no random id, no set iteration and no float rounding - field order is
the literal construction order below.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.core.converter import ConversionReport, ConversionResult
from app.core.errors import OutputNotWritableError

SCORE_SCHEMA_VERSION = 1


def report_to_dict(report: ConversionReport) -> dict[str, Any]:
    """Stable, explicit serialisation of the conversion report."""
    return {
        "input_note_count": report.input_note_count,
        "output_note_count": report.output_note_count,
        "transpose": report.transpose,
        "auto_transpose": report.auto_transpose,
        "range_mode": report.range_mode,
        "original_pitch_min": report.original_pitch_min,
        "original_pitch_max": report.original_pitch_max,
        "final_pitch_min": report.final_pitch_min,
        "final_pitch_max": report.final_pitch_max,
        "dropped_count": report.dropped_count,
        "octave_fold_count": report.octave_fold_count,
        "nearest_adjusted_count": report.nearest_adjusted_count,
        "semitone_count": report.semitone_count,
        "octave_shift_count": report.octave_shift_count,
        "modifier_count": report.modifier_count,
        "natural_count": report.natural_count,
        "playable_ratio": report.playable_ratio,
    }


def build_score_document(
    result: ConversionResult,
    source_segment: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """Build the score document as plain, JSON-ready Python objects.

    ``source_segment`` records the original media range when the score was
    cropped with ``--start`` / ``--end``; the note times themselves are always
    re-based to the segment start. The key is always present (``null`` for a
    full-song score) so consumers can rely on a uniform schema.
    """
    profile = result.profile
    segment_value: dict[str, float] | None = None
    if source_segment is not None:
        segment_start, segment_end = source_segment
        segment_value = {"start": float(segment_start), "end": float(segment_end)}
    return {
        "version": SCORE_SCHEMA_VERSION,
        "profile": profile.name,
        "lanes": list(profile.base_keys),
        "symbols": {
            "semitone": profile.semitone_symbol,
            "octave_up": profile.octave_up_symbol,
            "octave_down": profile.octave_down_symbol,
        },
        "duration": result.duration,
        "source_segment": segment_value,
        "conversion": {
            "transpose": result.report.transpose,
            "auto_transpose": result.report.auto_transpose,
            "range_mode": result.report.range_mode,
        },
        "report": report_to_dict(result.report),
        "notes": [
            {
                "id": index,
                "start": note.start,
                "duration": note.duration,
                "source_pitch": note.source_pitch,
                "pitch": note.pitch,
                "lane": note.lane,
                "key": note.key_label,
                "semitone": note.semitone,
                "octave": note.octave_offset,
                "confidence": note.confidence,
            }
            for index, note in enumerate(result.notes)
        ],
    }


def score_json(result: ConversionResult, source_segment: tuple[float, float] | None = None) -> str:
    """Serialise the score to a deterministic JSON string (UTF-8, no escaping)."""
    return json.dumps(build_score_document(result, source_segment), indent=2, ensure_ascii=False) + "\n"


def write_score_json(
    result: ConversionResult,
    output_path: str | Path,
    source_segment: tuple[float, float] | None = None,
) -> Path:
    """Write ``score.json`` and return the resolved path.

    ``newline="\\n"`` keeps the bytes identical to :func:`score_json` on every
    platform (no CRLF translation), which is what makes the export reproducible.
    """
    target = Path(output_path)
    try:
        parent = target.parent
        if parent and not parent.exists():
            parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.is_dir():
            raise IsADirectoryError(f"{target} is a directory")
        target.write_text(score_json(result, source_segment), encoding="utf-8", newline="\n")
    except OSError as exc:
        raise OutputNotWritableError(f"cannot write score JSON to {target}: {exc}") from exc

    return target


__all__ = [
    "SCORE_SCHEMA_VERSION",
    "report_to_dict",
    "build_score_document",
    "score_json",
    "write_score_json",
]
