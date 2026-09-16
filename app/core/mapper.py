"""Melody pitch -> concrete instrument key.

The mapper does exactly one thing: given a pitch that is *already* playable, it
picks the canonical binding (lane, semitone modifier, octave offset) and produces
a :class:`~app.core.models.PlayableNote`.

It deliberately does **not** transpose, fold octaves, snap to the nearest note or
drop anything - that is :mod:`app.core.range_fitter`'s job, and keeping the two
apart is what makes the conversion explainable. If a pitch cannot be mapped, the
mapper raises instead of silently fixing it.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from app.core.instrument_profile import InstrumentProfile
from app.core.models import MelodyNote, PlayableNote


@dataclass(frozen=True)
class MappingStats:
    """How much work the player has to do on top of the plain keys."""

    natural_count: int
    semitone_count: int
    octave_shift_count: int

    @property
    def modifier_count(self) -> int:
        """Notes that need at least one modifier (semitone and/or octave)."""
        return self.semitone_count + self.octave_shift_count


def map_note(note: MelodyNote, profile: InstrumentProfile) -> PlayableNote:
    """Map a single playable note onto its canonical key.

    Raises
    ------
    UnmappedPitchError
        The pitch is not in the profile's playable pitch set.
    """
    binding = profile.canonical_binding(note.pitch)
    return PlayableNote(
        source_pitch=note.source_pitch,
        pitch=note.pitch,
        start=note.start,
        duration=note.duration,
        lane=binding.lane,
        key_label=binding.key_label,
        semitone=binding.semitone,
        octave_offset=binding.octave_offset,
        confidence=note.confidence,
    )


def map_notes(
    notes: Iterable[MelodyNote],
    profile: InstrumentProfile,
) -> tuple[tuple[PlayableNote, ...], MappingStats]:
    """Map every note, returning the playable notes and their modifier statistics."""
    playable: list[PlayableNote] = []
    natural = 0
    semitones = 0
    octave_shifts = 0

    for note in notes:
        mapped = map_note(note, profile)
        playable.append(mapped)
        if mapped.semitone:
            semitones += 1
        if mapped.octave_offset != 0:
            octave_shifts += 1
        if mapped.is_natural:
            natural += 1

    return tuple(playable), MappingStats(
        natural_count=natural,
        semitone_count=semitones,
        octave_shift_count=octave_shifts,
    )


__all__ = ["MappingStats", "map_note", "map_notes"]
