"""Melody -> Delta Harmonica playable notes.

The stage order is fixed and must not be shuffled::

    monophonic melody (NoteEvent[])
      -> transpose       (manual or automatic)
      -> range fit       (octave_fold | nearest_note | drop)
      -> map             (canonical key binding)
      -> PlayableNote[]

Range fitting therefore always happens *before* mapping, which is what lets the
mapper refuse anything that is not already playable.

Pure Python: no torch, no librosa, no CLI, no renderer.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from app.core.instrument_profile import InstrumentProfile
from app.core.mapper import MappingStats, map_notes
from app.core.models import MAX_MIDI_PITCH, MIN_MIDI_PITCH, NoteEvent, PlayableNote
from app.core.range_fitter import DEFAULT_RANGE_MODE, RANGE_MODES, RangeFitResult, fit_range
from app.core.transposer import (
    MAX_TRANSPOSE,
    MIN_TRANSPOSE,
    auto_transpose,
    manual_transpose,
)

AUTO_TRANSPOSE = "auto"


@dataclass(frozen=True)
class ConversionConfig:
    """User-facing conversion options."""

    #: ``"auto"`` or a manual shift within ``-12..+12``.
    transpose: int | str = AUTO_TRANSPOSE
    #: One of :data:`app.core.range_fitter.RANGE_MODES`.
    range_mode: str = DEFAULT_RANGE_MODE

    def __post_init__(self) -> None:
        if self.range_mode not in RANGE_MODES:
            raise ValueError(f"range_mode must be one of {RANGE_MODES}, got {self.range_mode!r}")

        transpose = self.transpose
        if isinstance(transpose, str):
            if transpose != AUTO_TRANSPOSE:
                raise ValueError(f"transpose must be {AUTO_TRANSPOSE!r} or an int, got {transpose!r}")
        elif isinstance(transpose, bool) or not isinstance(transpose, int):
            raise ValueError(f"transpose must be {AUTO_TRANSPOSE!r} or an int, got {transpose!r}")
        elif not MIN_TRANSPOSE <= transpose <= MAX_TRANSPOSE:
            raise ValueError(
                f"transpose must be within {MIN_TRANSPOSE}..{MAX_TRANSPOSE} semitones, got {transpose}"
            )

    @property
    def auto(self) -> bool:
        return self.transpose == AUTO_TRANSPOSE


@dataclass(frozen=True)
class ConversionReport:
    """Everything that happened between the melody and the playable notes."""

    input_note_count: int
    output_note_count: int

    transpose: int
    auto_transpose: bool
    range_mode: str

    original_pitch_min: int | None
    original_pitch_max: int | None
    final_pitch_min: int | None
    final_pitch_max: int | None

    dropped_count: int
    octave_fold_count: int
    nearest_adjusted_count: int

    semitone_count: int
    octave_shift_count: int
    modifier_count: int
    natural_count: int

    playable_ratio: float


@dataclass(frozen=True)
class ConversionResult:
    """The playable notes plus the full account of the conversion."""

    notes: tuple[PlayableNote, ...]
    report: ConversionReport
    profile: InstrumentProfile
    config: ConversionConfig
    #: Length of the *input* melody in seconds (the music's own length).
    duration: float


def _pitch_bounds(notes: Sequence[NoteEvent]) -> tuple[int | None, int | None]:
    if not notes:
        return None, None
    pitches = [note.pitch for note in notes]
    return min(pitches), max(pitches)


def _final_pitch_bounds(notes: Sequence[PlayableNote]) -> tuple[int | None, int | None]:
    if not notes:
        return None, None
    pitches = [note.pitch for note in notes]
    return min(pitches), max(pitches)


def convert_melody(
    notes: Iterable[NoteEvent],
    config: ConversionConfig | None = None,
    profile: InstrumentProfile | None = None,
) -> ConversionResult:
    """Convert a monophonic melody into playable notes for *profile*.

    ``source_pitch`` on every output note is the melody pitch as it entered this
    function, no matter how many transpositions or folds happened on the way.
    """
    settings = config if config is not None else ConversionConfig()
    instrument = profile if profile is not None else InstrumentProfile.default()

    # Sorting first makes the result independent of the input ordering.
    melody = sorted(notes, key=lambda note: (note.start, note.pitch, note.end))
    original_min, original_max = _pitch_bounds(melody)

    if settings.auto:
        choice = auto_transpose(melody, instrument, settings.range_mode)
        semitones = choice.semitones
    else:
        semitones = int(settings.transpose)  # type: ignore[arg-type]

    transposed = manual_transpose(melody, semitones)
    fitted: RangeFitResult = fit_range(transposed, instrument, settings.range_mode)
    playable, stats = map_notes(fitted.notes, instrument)

    final_min, final_max = _final_pitch_bounds(playable)
    report = ConversionReport(
        input_note_count=len(melody),
        output_note_count=len(playable),
        transpose=semitones,
        auto_transpose=settings.auto,
        range_mode=settings.range_mode,
        original_pitch_min=original_min,
        original_pitch_max=original_max,
        final_pitch_min=final_min,
        final_pitch_max=final_max,
        dropped_count=fitted.dropped_count,
        octave_fold_count=fitted.octave_fold_count,
        nearest_adjusted_count=fitted.nearest_adjusted_count,
        semitone_count=stats.semitone_count,
        octave_shift_count=stats.octave_shift_count,
        modifier_count=stats.modifier_count,
        natural_count=stats.natural_count,
        playable_ratio=(len(playable) / len(melody)) if melody else 0.0,
    )

    return ConversionResult(
        notes=playable,
        report=report,
        profile=instrument,
        config=settings,
        duration=max((note.end for note in melody), default=0.0),
    )


def mapping_stats(result: ConversionResult) -> MappingStats:
    """Modifier statistics of an already converted result."""
    return MappingStats(
        natural_count=result.report.natural_count,
        semitone_count=result.report.semitone_count,
        octave_shift_count=result.report.octave_shift_count,
    )


__all__ = [
    "AUTO_TRANSPOSE",
    "ConversionConfig",
    "ConversionReport",
    "ConversionResult",
    "convert_melody",
    "mapping_stats",
    "MAX_MIDI_PITCH",
    "MIN_MIDI_PITCH",
]
