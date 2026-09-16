"""Time-range parsing, normalisation and note cropping.

Pure Python, deterministic, no I/O. Used by the CLI to support
``--start`` / ``--end`` so users can convert only one part of a song (e.g. the
chorus) instead of the whole file.

The output timeline is re-based: after cropping, the first note starts at
``0``. The *original* segment boundaries travel separately as metadata
(``source_segment`` in ``score.json``, ``Segment:`` in the PNG header).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence, TypeVar

from app.core.errors import PianoScoreError
from app.core.models import NoteEvent


class InvalidTimeRangeError(PianoScoreError):
    """A ``--start`` / ``--end`` value or the resulting range is invalid."""


def parse_time_value(value: str | None) -> float | None:
    """Parse a CLI time value.

    Supported formats::

        None          -> None          (option not given)
        ""            -> None
        "48"          -> 48.0          pure seconds
        "82.5"        -> 82.5
        "00:48"       -> 48.0          MM:SS
        "01:22"       -> 82.0
        "01:22.5"     -> 82.5          MM:SS.sss
        "00:01:22"    -> 82.0          HH:MM:SS
        "01:02:03.5"  -> 3723.5
    """
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None

    parts = text.split(":")
    if len(parts) > 3:
        raise InvalidTimeRangeError(
            f"invalid time {value!r}: expected seconds, MM:SS or HH:MM:SS"
        )

    seconds = 0.0
    try:
        for index, part in enumerate(parts):
            # A leading '-' only on the very first component; negatives are
            # rejected later by normalize_time_range with a clearer message.
            body = part[1:] if index == 0 and part.startswith("-") else part
            if not body or any(char not in "0123456789." for char in body):
                raise ValueError(part)
            amount = float(part)
            if index < len(parts) - 1 and abs(amount) >= 60:
                raise ValueError(part)  # minutes/hours must stay below 60
            seconds = seconds * 60.0 + amount
    except ValueError:
        raise InvalidTimeRangeError(
            f"invalid time {value!r}: expected seconds, MM:SS or HH:MM:SS"
        ) from None
    return seconds


def format_clock(seconds: float) -> str:
    """Human-facing clock: ``48`` -> ``00:48``, ``3723`` -> ``01:02:03``."""
    total = max(int(round(seconds)), 0)
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


@dataclass(frozen=True)
class TimeRange:
    """A normalised ``[start, end)`` range in seconds."""

    start: float
    end: float
    #: ``True`` when *end* was clamped down to the media duration.
    end_clamped: bool = False

    @property
    def duration(self) -> float:
        return self.end - self.start


def normalize_time_range(
    start: float | None,
    end: float | None,
    total_duration: float | None,
) -> TimeRange:
    """Apply the ``--start`` / ``--end`` defaults and validate the result.

    - ``start`` defaults to ``0.0``
    - ``end`` defaults to ``total_duration`` (error when unknown)
    - ``end`` beyond the media duration is clamped to it (flagged)
    """
    if start is None:
        start = 0.0
    if start < 0:
        raise InvalidTimeRangeError(f"start must be >= 0, got {start}")
    if total_duration is not None and start >= total_duration:
        raise InvalidTimeRangeError(
            f"Start time {format_clock(start)} is beyond the source duration "
            f"({format_clock(total_duration)})."
        )

    if end is None:
        if total_duration is None:
            raise InvalidTimeRangeError(
                "cannot determine the media duration; please pass --end explicitly"
            )
        end = float(total_duration)

    if end <= start:
        raise InvalidTimeRangeError(
            f"end ({end:g}) must be greater than start ({start:g})"
        )

    clamped = False
    if total_duration is not None and end > total_duration:
        end = float(total_duration)
        clamped = True

    return TimeRange(start=start, end=end, end_clamped=clamped)


_Note = TypeVar("_Note")


def _crop_one(note: _Note, start_time: float, end_time: float) -> _Note | None:
    """Clip a single ``start``/``duration``-shaped note into the range."""
    note_start = float(getattr(note, "start"))
    duration = float(getattr(note, "duration"))
    note_end = note_start + duration

    if note_end <= start_time or note_start >= end_time:
        return None  # completely outside

    clipped_start = max(note_start, start_time)
    clipped_end = min(note_end, end_time)
    new_duration = clipped_end - clipped_start
    if new_duration <= 1e-9:
        return None  # nothing left after clipping

    new_start = clipped_start - start_time  # re-base the timeline to 0
    return replace(note, start=new_start, duration=new_duration)  # type: ignore[arg-type,return-value]


def crop_note_events(
    notes: Sequence[NoteEvent], start_time: float, end_time: float
) -> tuple[NoteEvent, ...]:
    """Crop *notes* to ``[start_time, end_time)`` and re-base to 0.

    - notes fully outside the range are removed
    - notes crossing a boundary are truncated into it
    - a note spanning the whole range becomes ``[start, end]``
    - pitch / velocity / confidence and every other field are preserved
    - the input sequence is never modified; the result is deterministic
      (sorted by start time, strictly positive durations)
    """
    if end_time <= start_time:
        raise InvalidTimeRangeError(
            f"end ({end_time:g}) must be greater than start ({start_time:g})"
        )

    cropped: list[NoteEvent] = []
    for note in sorted(notes, key=lambda item: (item.start, item.pitch)):
        clipped = _crop_one(note, start_time, end_time)
        if clipped is not None:
            cropped.append(clipped)
    return tuple(cropped)


__all__ = [
    "InvalidTimeRangeError",
    "TimeRange",
    "parse_time_value",
    "normalize_time_range",
    "crop_note_events",
    "format_clock",
]
