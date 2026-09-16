"""Range fitting: octave_fold, nearest_note and drop."""

from __future__ import annotations

import pytest

from app.core.models import MelodyNote
from app.core.range_fitter import (
    DEFAULT_RANGE_MODE,
    RANGE_MODES,
    fit_pitches,
    fit_range,
    fold_candidates,
    nearest_playable,
)

from conftest import make_note, make_profile


def note(pitch: int, source_pitch: int | None = None) -> MelodyNote:
    return MelodyNote(
        source_pitch=pitch if source_pitch is None else source_pitch,
        pitch=pitch,
        start=0.0,
        duration=0.5,
        velocity=80,
    )


class TestFoldCandidates:
    def test_folds_up_to_the_nearest_octave(self, profile) -> None:
        assert fold_candidates(40, profile) == (52,)  # 40 + 12
        assert fold_candidates(46, profile) == (58,)

    def test_folds_down_to_the_nearest_octave(self, profile) -> None:
        assert fold_candidates(90, profile) == (78,)  # 90 - 12
        assert fold_candidates(96, profile) == (84,)

    def test_lists_every_octave_candidate_of_the_pitch_class(self, profile) -> None:
        # 66 % 12 == 6 -> 54 and 78 are both one octave away and both playable.
        assert fold_candidates(66, profile) == (54, 78)

    def test_lower_pitch_wins_the_tie(self) -> None:
        """72 has playable octaves at 60 and 84 -> the lower one is chosen first."""
        tied = make_profile(base_keys=["Z", "X"], natural_intervals=[0, 24], octave_offsets=[0])
        assert tied.playable_pitch_set == frozenset({60, 61, 84, 85})
        assert fold_candidates(72, tied) == (60, 84)

    def test_nearest_octave_distance_wins(self, profile) -> None:
        # 88 % 12 == 4 -> 64 and 88-24=64... but 76 is one octave away.
        assert fold_candidates(88, profile) == (76,)
        assert fold_candidates(2, profile)[0] == 50  # three octaves up

    def test_no_candidate_when_the_profile_lacks_the_pitch_class(self) -> None:
        sparse = make_profile(base_keys=["Z"], natural_intervals=[0], octave_offsets=[0])
        assert fold_candidates(66, sparse) == ()

    def test_always_searches_outwards_from_the_pitch(self, profile) -> None:
        for pitch in range(0, 128):
            candidates = fold_candidates(pitch, profile)
            assert list(candidates) == sorted(candidates)
            assert all(abs(candidate - pitch) % 12 == 0 for candidate in candidates)


class TestOctaveFold:
    def test_already_playable_notes_are_untouched(self, profile) -> None:
        result = fit_range([note(60), note(72)], profile, "octave_fold")

        assert [n.pitch for n in result.notes] == [60, 72]
        assert result.dropped_count == 0
        assert result.octave_fold_count == 0
        assert result.nearest_adjusted_count == 0

    def test_folds_up(self, profile) -> None:
        result = fit_range([note(40)], profile, "octave_fold")

        assert [n.pitch for n in result.notes] == [52]
        assert result.octave_fold_count == 1
        assert result.dropped_count == 0

    def test_folds_down(self, profile) -> None:
        result = fit_range([note(90)], profile, "octave_fold")

        assert [n.pitch for n in result.notes] == [78]
        assert result.octave_fold_count == 1

    def test_prefers_the_lower_pitch_on_a_tie(self) -> None:
        tied = make_profile(base_keys=["Z", "X"], natural_intervals=[0, 24], octave_offsets=[0])
        result = fit_range([note(72)], tied, "octave_fold")

        assert [n.pitch for n in result.notes] == [60]
        assert result.octave_fold_count == 1

    def test_drops_a_pitch_without_a_playable_octave(self) -> None:
        sparse = make_profile(base_keys=["Z"], natural_intervals=[0], octave_offsets=[0])
        result = fit_range([note(66), note(60)], sparse, "octave_fold")

        assert [n.pitch for n in result.notes] == [60]
        assert result.dropped_count == 1
        assert result.octave_fold_count == 0

    def test_octave_fold_never_falls_back_to_nearest_note(self) -> None:
        """46 cannot be folded (no playable octave shares its class) -> dropped."""
        sparse = make_profile(base_keys=["Z", "X"], natural_intervals=[0, 4], octave_offsets=[0])
        # playable: 60,61,64,65 -> 46 has no octave candidate, but 47 is nearer.
        result = fit_range([note(46)], sparse, "octave_fold")

        assert result.notes == ()
        assert result.dropped_count == 1

    def test_source_pitch_survives_folding(self, profile) -> None:
        result = fit_range([note(40, source_pitch=84)], profile, "octave_fold")

        assert result.notes[0].pitch == 52
        assert result.notes[0].source_pitch == 84

    def test_counts_are_per_note(self, profile) -> None:
        result = fit_range([note(60), note(40), note(90), note(62)], profile, "octave_fold")

        assert [n.pitch for n in result.notes] == [60, 52, 78, 62]
        assert result.input_note_count == 4
        assert result.octave_fold_count == 2
        assert result.dropped_count == 0
        assert result.adjusted_count == 2


class TestNearestNote:
    def test_snaps_to_the_closest_playable_pitch(self, profile) -> None:
        result = fit_range([note(46)], profile, "nearest_note")
        assert [n.pitch for n in result.notes] == [48]
        assert result.nearest_adjusted_count == 1

    def test_nearest_playable_helper(self, profile) -> None:
        assert nearest_playable(46, profile) == 48
        assert nearest_playable(47, profile) == 48
        assert nearest_playable(86, profile) == 85
        assert nearest_playable(60, profile) == 60

    def test_tie_prefers_the_lower_pitch(self) -> None:
        sparse = make_profile(base_keys=["Z", "X"], natural_intervals=[0, 3], octave_offsets=[0])
        # playable: 60,61,63,64 -> 62 is exactly one semitone from both 61 and 63.
        assert nearest_playable(62, sparse) == 61

        result = fit_range([note(62)], sparse, "nearest_note")
        assert [n.pitch for n in result.notes] == [61]
        assert result.nearest_adjusted_count == 1

    def test_never_folds(self, profile) -> None:
        result = fit_range([note(40)], profile, "nearest_note")
        assert result.octave_fold_count == 0
        assert result.nearest_adjusted_count == 1


class TestDrop:
    def test_unplayable_notes_are_removed_and_counted(self, profile) -> None:
        result = fit_range([note(46), note(60), note(90)], profile, "drop")

        assert [n.pitch for n in result.notes] == [60]
        assert result.dropped_count == 2
        assert result.octave_fold_count == 0
        assert result.nearest_adjusted_count == 0

    def test_playable_notes_are_kept(self, profile) -> None:
        result = fit_range([note(60), note(72)], profile, "drop")
        assert [n.pitch for n in result.notes] == [60, 72]
        assert result.dropped_count == 0


class TestCommon:
    def test_default_mode_is_octave_fold(self) -> None:
        assert DEFAULT_RANGE_MODE == "octave_fold"
        assert RANGE_MODES == ("octave_fold", "nearest_note", "drop")

    def test_invalid_mode_is_rejected(self, profile) -> None:
        with pytest.raises(ValueError, match="range mode"):
            fit_range([note(60)], profile, "squash")

    def test_source_pitch_is_preserved_in_every_mode(self, profile) -> None:
        for mode in RANGE_MODES:
            result = fit_range([note(40, source_pitch=99)], profile, mode)
            if result.notes:
                assert result.notes[0].source_pitch == 99

    def test_timing_is_preserved(self, profile) -> None:
        source = MelodyNote(source_pitch=40, pitch=40, start=1.25, duration=0.75, velocity=90)
        result = fit_range([source], profile, "octave_fold")
        assert result.notes[0].start == 1.25
        assert result.notes[0].duration == 0.75
        assert result.notes[0].velocity == 90

    def test_all_playable_pitches_are_already_fine(self, profile) -> None:
        result = fit_range([note(pitch) for pitch in profile.playable_pitches], profile)
        assert [n.pitch for n in result.notes] == list(profile.playable_pitches)
        assert result.adjusted_count == 0
        assert result.dropped_count == 0

    def test_fit_pitches_helper(self, profile) -> None:
        assert fit_pitches([40, 60, 90], profile) == [52, 60, 78]
        assert fit_pitches([40, 60, 90], profile, "nearest_note") == [48, 60, 85]
        assert fit_pitches([40, 60, 90], profile, "drop") == [60]

    def test_empty_input(self, profile) -> None:
        for mode in RANGE_MODES:
            result = fit_range([], profile, mode)
            assert result.notes == ()
            assert result.input_note_count == 0
