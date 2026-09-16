"""Make every melody pitch playable on the instrument.

Playability is decided against :attr:`InstrumentProfile.playable_pitch_set`, never
against a min/max window, so a profile with holes inside its range is handled
correctly.

Three modes are supported:

``octave_fold`` (default)
    Move the note by whole octaves until it lands on a playable pitch with the
    same pitch class. Nearest octave first, lower pitch wins a tie. A pitch with
    no playable octave at all is dropped - ``octave_fold`` never silently falls
    back to ``nearest_note``.
``nearest_note``
    Snap to the playable pitch with the smallest distance; lower pitch wins a tie.
``drop``
    Remove unplayable notes outright.

``source_pitch`` is never touched: only :attr:`MelodyNote.pitch` changes.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from app.core.instrument_profile import InstrumentProfile
from app.core.models import MAX_MIDI_PITCH, MIN_MIDI_PITCH, MelodyNote

RANGE_MODES: tuple[str, ...] = ("octave_fold", "nearest_note", "drop")
DEFAULT_RANGE_MODE = "octave_fold"

_OCTAVE = 12
_MAX_OCTAVE_DISTANCE = MAX_MIDI_PITCH // _OCTAVE


@dataclass(frozen=True)
class RangeFitResult:
    """The fitted notes plus a count of what had to be done to them."""

    notes: tuple[MelodyNote, ...]
    input_note_count: int
    dropped_count: int
    octave_fold_count: int
    nearest_adjusted_count: int

    @property
    def adjusted_count(self) -> int:
        return self.octave_fold_count + self.nearest_adjusted_count


def fold_candidates(pitch: int, profile: InstrumentProfile) -> tuple[int, ...]:
    """Playable pitches with the same pitch class as *pitch*, nearest octave first.

    Ties (an equally distant octave below and above) keep the lower pitch first.
    """
    candidates: list[int] = []
    for distance in range(1, _MAX_OCTAVE_DISTANCE + 1):
        for shifted in (pitch - _OCTAVE * distance, pitch + _OCTAVE * distance):
            if MIN_MIDI_PITCH <= shifted <= MAX_MIDI_PITCH and profile.is_playable(shifted):
                candidates.append(shifted)
        if candidates:
            break
    return tuple(candidates)


def nearest_playable(pitch: int, profile: InstrumentProfile) -> int:
    """Closest playable pitch; the lower pitch wins an equal distance."""
    return min(profile.playable_pitches, key=lambda candidate: (abs(candidate - pitch), candidate))


def fit_range(
    notes: Iterable[MelodyNote],
    profile: InstrumentProfile,
    mode: str = DEFAULT_RANGE_MODE,
) -> RangeFitResult:
    """Return the notes with every pitch made playable according to *mode*."""
    if mode not in RANGE_MODES:
        raise ValueError(f"range mode must be one of {RANGE_MODES}, got {mode!r}")

    source = list(notes)
    fitted: list[MelodyNote] = []
    dropped = 0
    folded = 0
    snapped = 0

    for note in source:
        if profile.is_playable(note.pitch):
            fitted.append(note)
            continue

        if mode == "drop":
            dropped += 1
            continue

        if mode == "nearest_note":
            snapped += 1
            fitted.append(note.with_pitch(nearest_playable(note.pitch, profile)))
            continue

        candidates = fold_candidates(note.pitch, profile)
        if not candidates:
            # No playable octave shares this pitch class: fold gives up rather
            # than pretending a different note is the same one.
            dropped += 1
            continue
        folded += 1
        fitted.append(note.with_pitch(candidates[0]))

    return RangeFitResult(
        notes=tuple(fitted),
        input_note_count=len(source),
        dropped_count=dropped,
        octave_fold_count=folded,
        nearest_adjusted_count=snapped,
    )


def fit_pitches(
    pitches: Sequence[int],
    profile: InstrumentProfile,
    mode: str = DEFAULT_RANGE_MODE,
) -> list[int]:
    """Convenience helper: fit bare pitches (used by tests and diagnostics)."""
    notes = (
        MelodyNote(source_pitch=pitch, pitch=pitch, start=index * 0.5, duration=0.5, velocity=80)
        for index, pitch in enumerate(pitches)
    )
    return [note.pitch for note in fit_range(notes, profile, mode).notes]


__all__ = [
    "RANGE_MODES",
    "DEFAULT_RANGE_MODE",
    "RangeFitResult",
    "fold_candidates",
    "nearest_playable",
    "fit_range",
    "fit_pitches",
]
