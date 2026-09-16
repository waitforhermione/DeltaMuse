"""Transposition: manual shifts and the ranked automatic search."""

from __future__ import annotations

import pytest

from app.core.errors import TransposeError
from app.core.instrument_profile import InstrumentProfile
from app.core.mapper import map_notes
from app.core.models import NoteEvent
from app.core.range_fitter import DEFAULT_RANGE_MODE, fit_range
from app.core.transposer import (
    MAX_TRANSPOSE,
    MIN_TRANSPOSE,
    OCTAVE_SHIFT_MODIFIER_COST,
    SEMITONE_MODIFIER_COST,
    TRANSPOSE_CANDIDATES,
    TransposeMetrics,
    auto_transpose,
    manual_transpose,
    validate_transpose,
)

from conftest import make_note, make_profile


def melody(*pitches: int, start: float = 0.0, duration: float = 0.5) -> list[NoteEvent]:
    return [
        make_note(pitch, start=start + index * 0.5, duration=duration)
        for index, pitch in enumerate(pitches)
    ]


def _argmin_semitones(notes: list[NoteEvent], profile: InstrumentProfile) -> int:
    """Recompute the auto search result independently of the implementation."""
    best: tuple[tuple[int, int, int, float, int, int], int] | None = None
    for candidate in TRANSPOSE_CANDIDATES:
        rank = _evaluate(notes, candidate, profile).rank_key(candidate)
        if best is None or rank < best[0]:
            best = (rank, candidate)
    assert best is not None
    return best[1]


def _evaluate(notes: list[NoteEvent], semitones: int, profile: InstrumentProfile):
    from app.core.transposer import _evaluate as evaluate

    return evaluate(notes, semitones, profile, DEFAULT_RANGE_MODE)


class TestManualTransposition:
    def test_zero_shift_is_identity(self, profile: InstrumentProfile) -> None:
        notes = melody(60, 64)
        transposed = manual_transpose(notes, 0)
        assert [note.pitch for note in transposed] == [60, 64]
        assert [note.source_pitch for note in transposed] == [60, 64]

    def test_plus_twelve_and_minus_twelve(self, profile: InstrumentProfile) -> None:
        notes = melody(60)
        assert manual_transpose(notes, 12)[0].pitch == 72
        assert manual_transpose(notes, -12)[0].pitch == 48

    @pytest.mark.parametrize("semitones", [-12, -1, 0, 1, 12])
    def test_source_pitch_is_never_overwritten(
        self, profile: InstrumentProfile, semitones: int
    ) -> None:
        transposed = manual_transpose(melody(72), semitones)
        assert transposed[0].source_pitch == 72
        assert transposed[0].pitch == 72 + semitones

    @pytest.mark.parametrize("semitones", [MIN_TRANSPOSE, MAX_TRANSPOSE])
    def test_boundaries_are_accepted(self, semitones: int) -> None:
        assert validate_transpose(semitones) == semitones

    @pytest.mark.parametrize("semitones", [-13, 13, 100])
    def test_shifts_beyond_the_range_are_rejected(self, semitones: int) -> None:
        with pytest.raises(TransposeError, match="within -12..12 semitones"):
            manual_transpose(melody(60), semitones)

    def test_leaving_the_midi_range_is_rejected(self) -> None:
        with pytest.raises(TransposeError, match="leaves the MIDI range"):
            manual_transpose(melody(120), 12)

    def test_leaving_the_midi_range_below_zero_is_rejected(self) -> None:
        with pytest.raises(TransposeError, match="leaves the MIDI range"):
            manual_transpose(melody(4), -12)

    def test_non_integer_shift_is_rejected(self) -> None:
        with pytest.raises(TransposeError, match="must be an int"):
            validate_transpose("3")  # type: ignore[arg-type]

    def test_booleans_are_not_integers(self) -> None:
        with pytest.raises(TransposeError, match="must be an int"):
            validate_transpose(True)  # type: ignore[arg-type]

    def test_empty_melody(self, profile: InstrumentProfile) -> None:
        assert manual_transpose([], 7) == []

    def test_timing_and_velocity_are_preserved(self, profile: InstrumentProfile) -> None:
        transposed = manual_transpose(melody(72, start=1.5, duration=0.25), -5)
        assert transposed[0].start == 1.5
        assert transposed[0].duration == 0.25
        assert transposed[0].end == 1.75


class TestAutoTransposition:
    def test_zero_when_the_melody_already_sits_in_the_base_octave(
        self, profile: InstrumentProfile
    ) -> None:
        result = auto_transpose(melody(60, 62, 64, 65, 67, 69, 71, 72), profile)
        assert result.semitones == 0
        assert result.metrics.midi_out_of_range_count == 0
        assert result.metrics.unplayable_count == 0
        assert result.metrics.octave_fold_count == 0
        assert result.metrics.modifier_cost == 0.0

    def test_search_covers_minus_twelve_to_plus_twelve(self) -> None:
        assert TRANSPOSE_CANDIDATES == tuple(range(-12, 13))
        assert len(TRANSPOSE_CANDIDATES) == 25

    def test_result_is_deterministic(self, profile: InstrumentProfile) -> None:
        notes = melody(72, 79, 76, 74)
        first = auto_transpose(notes, profile)
        second = auto_transpose(notes, profile)
        assert first == second
        assert first.semitones == second.semitones

    def test_matches_an_independent_argmin(
        self, profile: InstrumentProfile
    ) -> None:
        """The chosen shift must be the lexicographic minimum over all candidates."""
        for pitches in ([72, 79, 76, 74], [40, 42, 44], [120, 122], [60], [84, 85, 86]):
            notes = melody(*pitches)
            expected = _argmin_semitones(notes, profile)
            assert auto_transpose(notes, profile).semitones == expected, pitches

    def test_out_of_range_pitches_are_penalised_before_everything_else(
        self, profile: InstrumentProfile
    ) -> None:
        result = auto_transpose(melody(120, 122), profile)
        assert result.metrics.midi_out_of_range_count == 0
        assert result.semitones == 0

    def test_prefers_playability_over_modifier_convenience(
        self, profile: InstrumentProfile
    ) -> None:
        """A shift that avoids folds wins even if it adds modifiers."""
        result = auto_transpose(melody(40, 42, 44), profile)
        assert result.metrics.unplayable_count == 0
        assert result.metrics.octave_fold_count == 0
        assert result.metrics.modifier_cost < 3 * OCTAVE_SHIFT_MODIFIER_COST + 1e-9

    def test_prefers_naturals_in_a_nearer_octave_over_semitones(
        self, profile: InstrumentProfile
    ) -> None:
        """40/42/44 cannot reach the base octave, so the search weighs
        three naturals one octave down (3 x 2) against three semitone spellings
        in that same octave (3 x (1 + 2)) - the naturals win."""
        result = auto_transpose(melody(40, 42, 44), profile)

        assert result.semitones == 8
        assert result.metrics.semitone_count == 0
        assert result.metrics.octave_shift_count == 3
        assert result.metrics.modifier_cost == pytest.approx(3 * OCTAVE_SHIFT_MODIFIER_COST)

    def test_no_fully_playable_candidate_still_returns_a_choice(self) -> None:
        """A profile with very few playable pitches must not crash the search."""
        tiny = make_profile(base_keys=["Z"], natural_intervals=[0], octave_offsets=[0])
        # 66 and 67 are two semitones apart, so they can never both be playable.
        notes = melody(66, 68)

        first = auto_transpose(notes, tiny)
        second = auto_transpose(notes, tiny)

        assert MIN_TRANSPOSE <= first.semitones <= MAX_TRANSPOSE
        assert first == second
        assert first.metrics.unplayable_count >= 1  # no candidate plays everything
        assert first.metrics.mapped_note_count == 1

    def test_reports_the_modifier_composition(self, profile: InstrumentProfile) -> None:
        """60/63/66 has a gap of 3, which the naturals cannot follow at any shift."""
        result = auto_transpose(melody(60, 63, 66), profile)

        assert result.semitones == 1
        assert result.metrics.semitone_count == 1
        assert result.metrics.octave_shift_count == 0
        assert result.metrics.modifier_cost == pytest.approx(SEMITONE_MODIFIER_COST)

    def test_empty_melody_returns_zero(self, profile: InstrumentProfile) -> None:
        result = auto_transpose([], profile)
        assert result.semitones == 0
        assert result.metrics == TransposeMetrics(0, 0, 0, 0.0, 0)

    def test_candidates_are_reported(self, profile: InstrumentProfile) -> None:
        assert auto_transpose(melody(60), profile).candidates == TRANSPOSE_CANDIDATES


class TestRanking:
    def test_criteria_are_ordered_exactly_as_documented(self) -> None:
        low = TransposeMetrics(0, 0, 0, 0.0, 0)
        assert low.rank_key(0) == (0, 0, 0, 0.0, 0, 0)
        assert low.rank_key(0) < TransposeMetrics(1, 0, 0, 0.0, 0).rank_key(0)
        assert low.rank_key(0) < TransposeMetrics(0, 1, 0, 0.0, 0).rank_key(0)
        assert low.rank_key(0) < TransposeMetrics(0, 0, 1, 0.0, 0).rank_key(0)
        assert low.rank_key(0) < TransposeMetrics(0, 0, 0, 0.5, 0).rank_key(0)
        assert low.rank_key(0) < TransposeMetrics(0, 0, 0, 0.0, 0).rank_key(7)
        assert low.rank_key(0) < low.rank_key(1)

    def test_modifier_cost_weights(self) -> None:
        assert SEMITONE_MODIFIER_COST < OCTAVE_SHIFT_MODIFIER_COST

    def test_modifier_cost_never_outranks_playability(self) -> None:
        configured = TransposeMetrics(0, 1, 1, 0.0, 0)
        convenient = TransposeMetrics(0, 0, 0, 1e6, 0)
        assert convenient.rank_key(0) < configured.rank_key(0)

    def test_the_whole_pipeline_agrees_with_the_ranking(self, profile: InstrumentProfile) -> None:
        """Fitting must happen before mapping for the ranking to be meaningful."""
        notes = melody(90)
        metrics = _evaluate(notes, -6, profile)
        transposed = manual_transpose(notes, -6)
        fitted = fit_range(transposed, profile, DEFAULT_RANGE_MODE)
        mapped, _stats = map_notes(fitted.notes, profile)

        assert metrics.octave_fold_count == fitted.octave_fold_count
        assert metrics.mapped_note_count == len(mapped)
