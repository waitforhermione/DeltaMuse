"""Transposition of a melody into the instrument's range.

Two modes:

``manual_transpose``
    Shift by a fixed number of semitones, limited to ``-12..+12``. The result must
    stay inside the MIDI range, otherwise the request is rejected.
``auto_transpose``
    Search ``-12..+12`` and rank the candidates. The ranking is deliberately
    ordered so that *playability wins over convenience*:

    1. fewest pitches pushed outside MIDI ``0..127``,
    2. fewest pitches that are not directly playable,
    3. fewest octave folds,
    4. lowest modifier complexity (semitone and octave-shift modifiers),
    5. smallest ``abs(transpose)``,
    6. deterministic tie-break (smallest transposition).

``source_pitch`` is always preserved: transposition only moves ``pitch``.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from app.core.errors import TransposeError
from app.core.instrument_profile import InstrumentProfile
from app.core.mapper import map_notes
from app.core.models import MAX_MIDI_PITCH, MIN_MIDI_PITCH, MelodyNote, NoteEvent
from app.core.range_fitter import DEFAULT_RANGE_MODE, fit_range

MIN_TRANSPOSE = -12
MAX_TRANSPOSE = 12

#: Modifier complexity used by the auto search. A semitone is a small extra, an
#: octave shift is a bigger one - but both are ranked *after* playability, so they
#: can never outweigh it.
SEMITONE_MODIFIER_COST = 1.0
OCTAVE_SHIFT_MODIFIER_COST = 2.0

TRANSPOSE_CANDIDATES: tuple[int, ...] = tuple(range(MIN_TRANSPOSE, MAX_TRANSPOSE + 1))


@dataclass(frozen=True)
class TransposeMetrics:
    """Everything the auto search ranks a candidate by."""

    midi_out_of_range_count: int
    unplayable_count: int
    octave_fold_count: int
    modifier_cost: float
    mapped_note_count: int
    semitone_count: int = 0
    octave_shift_count: int = 0

    def rank_key(self, semitones: int) -> tuple[int, int, int, float, int, int]:
        return (
            self.midi_out_of_range_count,
            self.unplayable_count,
            self.octave_fold_count,
            self.modifier_cost,
            abs(semitones),
            semitones,
        )


@dataclass(frozen=True)
class AutoTransposeResult:
    """The chosen transposition and why it won."""

    semitones: int
    metrics: TransposeMetrics
    candidates: tuple[int, ...] = TRANSPOSE_CANDIDATES


def validate_transpose(semitones: int) -> int:
    """Return *semitones* if it is a legal manual transposition."""
    if isinstance(semitones, bool) or not isinstance(semitones, int):
        raise TransposeError(f"transpose must be an int, got {semitones!r}")
    if not MIN_TRANSPOSE <= semitones <= MAX_TRANSPOSE:
        raise TransposeError(
            f"transpose must be within {MIN_TRANSPOSE}..{MAX_TRANSPOSE} semitones, got {semitones}"
        )
    return semitones


def manual_transpose(notes: Iterable[NoteEvent], semitones: int) -> list[MelodyNote]:
    """Shift the melody by *semitones*, keeping every note's ``source_pitch``."""
    shift = validate_transpose(semitones)

    transposed: list[MelodyNote] = []
    for note in notes:
        source = MelodyNote.from_note_event(note)
        pitch = source.pitch + shift
        # Check before building the note: an out-of-range shift is a user error
        # and must be reported as such, not as a model validation failure.
        if not MIN_MIDI_PITCH <= pitch <= MAX_MIDI_PITCH:
            raise TransposeError(
                f"transposing {note.pitch} by {shift} semitones leaves the MIDI range "
                f"(0..127): {pitch}"
            )
        transposed.append(source.with_pitch(pitch))
    return transposed


def _evaluate(
    notes: Sequence[NoteEvent],
    semitones: int,
    profile: InstrumentProfile,
    range_mode: str,
) -> TransposeMetrics:
    """Score one candidate without ever raising on unplayable input."""
    shifted: list[MelodyNote] = []
    out_of_range = 0
    for note in notes:
        pitch = note.pitch + semitones
        if not MIN_MIDI_PITCH <= pitch <= MAX_MIDI_PITCH:
            out_of_range += 1
            continue
        shifted.append(MelodyNote.from_note_event(note).with_pitch(pitch))

    unplayable = sum(1 for note in shifted if not profile.is_playable(note.pitch))

    fitted = fit_range(shifted, profile, range_mode)
    # fit_range only emits playable pitches, so mapping cannot fail here.
    mapped, stats = map_notes(fitted.notes, profile)
    modifier_cost = (
        stats.semitone_count * SEMITONE_MODIFIER_COST
        + stats.octave_shift_count * OCTAVE_SHIFT_MODIFIER_COST
    )

    return TransposeMetrics(
        midi_out_of_range_count=out_of_range,
        unplayable_count=unplayable,
        octave_fold_count=fitted.octave_fold_count,
        modifier_cost=modifier_cost,
        mapped_note_count=len(mapped),
        semitone_count=stats.semitone_count,
        octave_shift_count=stats.octave_shift_count,
    )


def auto_transpose(
    notes: Iterable[NoteEvent],
    profile: InstrumentProfile,
    range_mode: str = DEFAULT_RANGE_MODE,
) -> AutoTransposeResult:
    """Pick the transposition that makes the melody most playable.

    Ties are broken deterministically: candidates are visited in ascending order
    and only a *strictly* better rank replaces the current best, so the same input
    always yields the same transposition.
    """
    melody = list(notes)
    if not melody:
        return AutoTransposeResult(
            semitones=0,
            metrics=TransposeMetrics(
                midi_out_of_range_count=0,
                unplayable_count=0,
                octave_fold_count=0,
                modifier_cost=0.0,
                mapped_note_count=0,
            ),
        )

    best_semitones = 0
    best_metrics: TransposeMetrics | None = None
    best_rank: tuple[int, int, int, float, int, int] | None = None

    for candidate in TRANSPOSE_CANDIDATES:
        metrics = _evaluate(melody, candidate, profile, range_mode)
        rank = metrics.rank_key(candidate)
        if best_rank is None or rank < best_rank:
            best_semitones = candidate
            best_metrics = metrics
            best_rank = rank

    assert best_metrics is not None  # TRANSPOSE_CANDIDATES is never empty
    return AutoTransposeResult(semitones=best_semitones, metrics=best_metrics)


__all__ = [
    "MIN_TRANSPOSE",
    "MAX_TRANSPOSE",
    "SEMITONE_MODIFIER_COST",
    "OCTAVE_SHIFT_MODIFIER_COST",
    "TRANSPOSE_CANDIDATES",
    "TransposeMetrics",
    "AutoTransposeResult",
    "validate_transpose",
    "manual_transpose",
    "auto_transpose",
]
