"""Onset grouping, overlap resolution and polyphony statistics.

Pure Python on purpose: this module must stay importable and testable without
torch, TransKun, librosa, the CLI or any renderer. The only data type it knows is
:class:`~app.core.models.NoteEvent`, and nothing here ever mutates its input.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, replace

from app.core.models import NoteEvent

DEFAULT_ONSET_EPSILON = 0.03
DEFAULT_OVERLAP_FRAGMENT_THRESHOLD = 0.03


def note_sort_key(note: NoteEvent) -> tuple[float, int, float]:
    """Canonical deterministic ordering used by every function in this module."""
    return (note.start, note.pitch, note.end)


# --------------------------------------------------------------------------- #
# onset grouping
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class OnsetGroup:
    """Notes that start at (almost) the same instant.

    ``anchor`` is the onset of the *first* note of the group; every member
    satisfies ``note.start - anchor <= epsilon``.
    """

    anchor: float
    notes: tuple[NoteEvent, ...]

    @property
    def size(self) -> int:
        return len(self.notes)

    @property
    def pitches(self) -> tuple[int, ...]:
        return tuple(note.pitch for note in self.notes)


def group_by_onset(
    notes: Iterable[NoteEvent],
    epsilon: float = DEFAULT_ONSET_EPSILON,
) -> tuple[OnsetGroup, ...]:
    """Group notes whose onset lies within *epsilon* seconds of the group anchor.

    Transcribed onsets are floats and never compare equal, so almost-simultaneous
    notes have to be grouped. The anchor of a group is fixed to the onset of its
    first note and a note is admitted only when ``note.start - anchor <= epsilon``.

    Comparing against the anchor rather than against the previously admitted note
    is what keeps the grouping from *chain merging*: with the chained rule, notes
    spaced ``epsilon / 2`` apart would collapse into a single unbounded group.

    The result is deterministic: notes are ordered by ``(start, pitch, end)``
    before grouping, so the same input always yields the same groups.
    """
    if epsilon < 0.0 or not math.isfinite(epsilon):
        raise ValueError(f"epsilon must be a finite value >= 0, got {epsilon!r}")

    ordered = sorted(notes, key=note_sort_key)
    groups: list[OnsetGroup] = []
    anchor = 0.0
    current: list[NoteEvent] = []

    for note in ordered:
        if not current or note.start - anchor > epsilon:
            if current:
                groups.append(OnsetGroup(anchor=anchor, notes=tuple(current)))
            anchor = note.start
            current = [note]
        else:
            current.append(note)

    if current:
        groups.append(OnsetGroup(anchor=anchor, notes=tuple(current)))

    return tuple(groups)


# --------------------------------------------------------------------------- #
# overlap resolution
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class OverlapResolution:
    """Result of forcing a note sequence to be strictly non-overlapping."""

    notes: tuple[NoteEvent, ...]
    truncated_count: int
    dropped_fragment_count: int


def resolve_overlaps(
    notes: Iterable[NoteEvent],
    fragment_threshold: float = DEFAULT_OVERLAP_FRAGMENT_THRESHOLD,
) -> OverlapResolution:
    """Force strictly non-overlapping output using ``truncate_previous``.

    Notes are visited in ``(start, pitch, end)`` order. When a note begins before
    the previously kept note ends, the **previous** note is truncated to that
    onset (``previous.end = next.start``).

    If the shortened note would be shorter than *fragment_threshold* it is dropped
    instead of being emitted as an inaudible fragment. This threshold is about
    fragments *created by truncation* only - it is deliberately independent from
    the filter that removes originally short transcribed notes.

    Counters: ``truncated_count`` is the number of notes whose end was shortened
    (including the ones that were then discarded), ``dropped_fragment_count`` is
    how many of those shortened notes fell below the threshold.
    """
    if fragment_threshold < 0.0 or not math.isfinite(fragment_threshold):
        raise ValueError(f"fragment_threshold must be a finite value >= 0, got {fragment_threshold!r}")

    kept: list[NoteEvent] = []
    truncated_count = 0
    dropped_fragment_count = 0

    for note in sorted(notes, key=note_sort_key):
        while kept and kept[-1].end > note.start:
            previous = kept.pop()
            remaining = note.start - previous.start
            truncated_count += 1
            # Never emit a zero/negative-length note; that would be invalid data.
            if remaining >= fragment_threshold and remaining > 0.0:
                kept.append(replace(previous, duration=remaining))
            else:
                dropped_fragment_count += 1
        kept.append(note)

    kept.sort(key=note_sort_key)
    return OverlapResolution(
        notes=tuple(kept),
        truncated_count=truncated_count,
        dropped_fragment_count=dropped_fragment_count,
    )


def is_monophonic(notes: Iterable[NoteEvent], tolerance: float = 0.0) -> bool:
    """``True`` when no two notes overlap by more than *tolerance* seconds.

    Back-to-back notes (``previous.end == next.start``) are monophonic, which is
    exactly what :func:`resolve_overlaps` produces.
    """
    ordered = sorted(notes, key=note_sort_key)
    return all(
        current.start >= previous.end - tolerance
        for previous, current in zip(ordered, ordered[1:])
    )


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PolyphonyStats:
    """How many notes sound at the same time."""

    maximum: int
    mean: float

    @property
    def is_monophonic(self) -> bool:
        return self.maximum <= 1


def polyphony_stats(notes: Iterable[NoteEvent]) -> PolyphonyStats:
    """Return the maximum and the time-weighted mean simultaneous note count."""
    note_list = list(notes)
    if not note_list:
        return PolyphonyStats(maximum=0, mean=0.0)

    points: list[tuple[float, int]] = []
    for note in note_list:
        points.append((note.start, 1))
        points.append((note.end, -1))
    # At an identical timestamp an ending note (-1) is processed before a
    # starting one (+1), so back-to-back notes do not count as an overlap.
    points.sort()

    current = 0
    maximum = 0
    area = 0.0
    previous = points[0][0]
    for timestamp, delta in points:
        area += current * (timestamp - previous)
        previous = timestamp
        current += delta
        maximum = max(maximum, current)

    span = points[-1][0] - points[0][0]
    mean = area / span if span > 0 else 0.0
    return PolyphonyStats(maximum=maximum, mean=mean)


__all__ = [
    "DEFAULT_ONSET_EPSILON",
    "DEFAULT_OVERLAP_FRAGMENT_THRESHOLD",
    "OnsetGroup",
    "OverlapResolution",
    "PolyphonyStats",
    "note_sort_key",
    "group_by_onset",
    "resolve_overlaps",
    "is_monophonic",
    "polyphony_stats",
]
