"""Onset grouping, overlap resolution and polyphony statistics."""

from __future__ import annotations

import pytest

from app.core.models import NoteEvent
from app.core.polyphony import (
    PolyphonyStats,
    group_by_onset,
    is_monophonic,
    note_sort_key,
    polyphony_stats,
    resolve_overlaps,
)

from conftest import make_note


class TestOnsetGrouping:
    def test_empty_input_gives_no_groups(self) -> None:
        assert group_by_onset([]) == ()

    def test_single_note_is_one_group(self) -> None:
        groups = group_by_onset([make_note(60)])
        assert len(groups) == 1
        assert groups[0].anchor == 0.0
        assert groups[0].pitches == (60,)
        assert groups[0].size == 1

    def test_identical_onsets_are_grouped(self) -> None:
        notes = [make_note(60), make_note(64), make_note(67)]
        groups = group_by_onset(notes, epsilon=0.03)
        assert len(groups) == 1
        assert groups[0].pitches == (60, 64, 67)

    def test_onsets_inside_epsilon_are_grouped(self) -> None:
        notes = [make_note(60, start=0.0), make_note(67, start=0.02)]
        assert len(group_by_onset(notes, epsilon=0.03)) == 1

    def test_onsets_outside_epsilon_are_split(self) -> None:
        notes = [make_note(60, start=0.0), make_note(67, start=0.04)]
        assert len(group_by_onset(notes, epsilon=0.03)) == 2

    def test_grouping_uses_the_anchor_and_never_chain_merges(self) -> None:
        """Notes spaced epsilon/2 apart must NOT collapse into one giant group."""
        notes = [
            make_note(60, start=0.000),
            make_note(62, start=0.025),
            make_note(64, start=0.050),
            make_note(65, start=0.075),
        ]
        groups = group_by_onset(notes, epsilon=0.03)

        # anchor 0.000 admits 0.025; 0.050 is 0.050 away from the anchor -> new group
        assert len(groups) == 2
        assert [group.size for group in groups] == [2, 2]
        assert [group.anchor for group in groups] == [pytest.approx(0.0), pytest.approx(0.05)]

    def test_groups_are_ordered_by_anchor(self) -> None:
        notes = [make_note(60, start=2.0), make_note(64, start=0.0), make_note(67, start=1.0)]
        groups = group_by_onset(notes, epsilon=0.03)
        assert [group.anchor for group in groups] == [0.0, 1.0, 2.0]

    def test_members_inside_a_group_are_sorted(self) -> None:
        notes = [make_note(72, start=0.0), make_note(60, start=0.0), make_note(67, start=0.0)]
        assert group_by_onset(notes)[0].pitches == (60, 67, 72)

    def test_grouping_is_independent_of_input_order(self) -> None:
        notes = [
            make_note(60, start=0.0, duration=0.5),
            make_note(64, start=0.01, duration=0.4),
            make_note(67, start=1.0, duration=0.5),
            make_note(71, start=1.02, duration=0.3),
        ]
        reference = group_by_onset(notes, epsilon=0.03)
        shuffled = group_by_onset(list(reversed(notes)), epsilon=0.03)
        assert [group.pitches for group in shuffled] == [group.pitches for group in reference]

    def test_zero_epsilon_only_groups_exact_matches(self) -> None:
        notes = [make_note(60, start=0.0), make_note(64, start=0.0), make_note(67, start=0.001)]
        groups = group_by_onset(notes, epsilon=0.0)
        assert [group.size for group in groups] == [2, 1]

    def test_input_notes_are_not_modified(self) -> None:
        notes = [make_note(72, start=0.5, duration=0.25)]
        snapshot = list(notes)
        group_by_onset(notes)
        assert notes == snapshot
        assert notes[0] is snapshot[0]

    def test_negative_epsilon_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="epsilon"):
            group_by_onset([], epsilon=-0.01)

    def test_sort_key_is_stable_and_complete(self) -> None:
        first = make_note(60, start=0.0, duration=1.0)
        second = make_note(60, start=0.0, duration=1.0)
        assert note_sort_key(first) == note_sort_key(second)


class TestOverlapResolution:
    def test_non_overlapping_notes_are_untouched(self) -> None:
        notes = [make_note(60, start=0.0, duration=0.4), make_note(64, start=0.5, duration=0.4)]
        resolution = resolve_overlaps(notes)
        assert [note.pitch for note in resolution.notes] == [60, 64]
        assert resolution.truncated_count == 0
        assert resolution.dropped_fragment_count == 0

    def test_back_to_back_notes_are_not_touched(self) -> None:
        notes = [make_note(60, start=0.0, duration=0.5), make_note(64, start=0.5, duration=0.5)]
        resolution = resolve_overlaps(notes)
        assert resolution.truncated_count == 0
        assert is_monophonic(resolution.notes)

    def test_exact_boundary_is_not_truncated(self) -> None:
        notes = [make_note(60, start=0.0, duration=1.0), make_note(64, start=1.0, duration=1.0)]
        resolution = resolve_overlaps(notes)
        assert resolution.notes[0].duration == pytest.approx(1.0)
        assert resolution.truncated_count == 0

    def test_partial_overlap_truncates_previous(self) -> None:
        notes = [make_note(60, start=0.0, duration=1.0), make_note(64, start=0.6, duration=1.0)]
        resolution = resolve_overlaps(notes)

        assert resolution.truncated_count == 1
        assert resolution.notes[0].pitch == 60
        assert resolution.notes[0].end == pytest.approx(0.6)
        assert resolution.notes[1].duration == pytest.approx(1.0)
        assert is_monophonic(resolution.notes)

    def test_complete_containment_truncates_previous(self) -> None:
        notes = [make_note(60, start=0.0, duration=2.0), make_note(64, start=0.5, duration=1.0)]
        resolution = resolve_overlaps(notes)

        assert resolution.notes[0].end == pytest.approx(0.5)
        assert resolution.notes[1].duration == pytest.approx(1.0)
        assert is_monophonic(resolution.notes)

    def test_three_way_overlap_is_fully_resolved(self) -> None:
        notes = [
            make_note(60, start=0.0, duration=1.0),
            make_note(64, start=0.5, duration=1.0),
            make_note(67, start=1.2, duration=1.0),
        ]
        resolution = resolve_overlaps(notes)

        assert is_monophonic(resolution.notes)
        assert [note.end for note in resolution.notes] == [
            pytest.approx(0.5),
            pytest.approx(1.2),
            pytest.approx(2.2),
        ]

    def test_truncation_fragment_below_threshold_is_dropped(self) -> None:
        # 60 is truncated to [0, 0.02] -> 0.02 s < 0.03 s -> dropped.
        notes = [make_note(60, start=0.0, duration=0.05), make_note(64, start=0.02, duration=0.5)]
        resolution = resolve_overlaps(notes, fragment_threshold=0.03)

        assert [note.pitch for note in resolution.notes] == [64]
        assert resolution.truncated_count == 1
        assert resolution.dropped_fragment_count == 1

    def test_truncation_fragment_above_threshold_is_kept(self) -> None:
        notes = [make_note(60, start=0.0, duration=0.05), make_note(64, start=0.04, duration=0.5)]
        resolution = resolve_overlaps(notes, fragment_threshold=0.03)

        assert [note.pitch for note in resolution.notes] == [60, 64]
        assert resolution.notes[0].duration == pytest.approx(0.04)
        assert resolution.dropped_fragment_count == 0

    def test_originally_short_note_is_not_a_fragment(self) -> None:
        """A short note that never overlapped anything must survive untouched.

        This is the semantic difference between "originally short" notes (filtered
        by ``min_note_duration`` upstream) and fragments *created by truncation*.
        """
        notes = [make_note(60, start=0.0, duration=0.02), make_note(64, start=0.5, duration=0.5)]
        resolution = resolve_overlaps(notes, fragment_threshold=0.03)

        assert [note.pitch for note in resolution.notes] == [60, 64]
        assert resolution.notes[0].duration == pytest.approx(0.02)
        assert resolution.truncated_count == 0
        assert resolution.dropped_fragment_count == 0

    def test_result_is_sorted_and_monophonic(self) -> None:
        notes = [
            make_note(67, start=1.3, duration=0.8),
            make_note(60, start=0.0, duration=0.9),
            make_note(64, start=0.6, duration=0.9),
        ]
        resolution = resolve_overlaps(notes)

        starts = [note.start for note in resolution.notes]
        assert starts == sorted(starts)
        assert is_monophonic(resolution.notes)

    def test_zero_length_fragment_never_survives(self) -> None:
        notes = [make_note(60, start=1.0, duration=1.0), make_note(64, start=1.0, duration=1.0)]
        resolution = resolve_overlaps(notes, fragment_threshold=0.0)
        assert is_monophonic(resolution.notes)
        assert all(note.duration > 0 for note in resolution.notes)

    @pytest.mark.parametrize("threshold", [-0.1, float("nan")])
    def test_invalid_threshold_is_rejected(self, threshold: float) -> None:
        with pytest.raises(ValueError, match="fragment_threshold"):
            resolve_overlaps([], fragment_threshold=threshold)


class TestMonophonic:
    def test_empty_sequence_is_monophonic(self) -> None:
        assert is_monophonic([])

    def test_overlapping_notes_are_detected(self) -> None:
        assert not is_monophonic([make_note(60, duration=1.0), make_note(64, start=0.5, duration=1.0)])

    def test_contained_note_is_detected(self) -> None:
        assert not is_monophonic([make_note(60, duration=2.0), make_note(64, start=0.5, duration=0.2)])

    def test_touching_notes_are_monophonic(self) -> None:
        assert is_monophonic([make_note(60, duration=0.5), make_note(64, start=0.5, duration=0.5)])

    def test_tolerance_allows_small_overlaps(self) -> None:
        notes = [make_note(60, duration=0.5), make_note(64, start=0.49, duration=0.5)]
        assert not is_monophonic(notes)
        assert is_monophonic(notes, tolerance=0.02)


class TestPolyphonyStats:
    def test_empty_notes(self) -> None:
        stats = polyphony_stats([])
        assert stats == PolyphonyStats(maximum=0, mean=0.0)
        assert stats.is_monophonic

    def test_single_note(self) -> None:
        stats = polyphony_stats([make_note(60, duration=2.0)])
        assert stats.maximum == 1
        assert stats.mean == pytest.approx(1.0)

    def test_overlapping_chord(self) -> None:
        notes = [
            make_note(60, start=0.0, duration=1.0),
            make_note(64, start=0.0, duration=1.0),
            make_note(67, start=0.0, duration=1.0),
        ]
        stats = polyphony_stats(notes)
        assert stats.maximum == 3
        assert stats.mean == pytest.approx(3.0)
        assert not stats.is_monophonic

    def test_back_to_back_notes_do_not_count_as_polyphony(self) -> None:
        notes = [
            make_note(60, start=0.0, duration=0.5),
            make_note(64, start=0.5, duration=0.5),
            make_note(67, start=1.0, duration=0.5),
        ]
        assert polyphony_stats(notes).maximum == 1

    def test_mean_is_time_weighted(self) -> None:
        # two notes for 1 s, silence for 1 s, one note for 1 s -> 3/3 = 1.0
        notes = [
            make_note(60, start=0.0, duration=1.0),
            make_note(64, start=0.0, duration=1.0),
            make_note(67, start=2.0, duration=1.0),
        ]
        stats = polyphony_stats(notes)
        assert stats.maximum == 2
        assert stats.mean == pytest.approx(1.0)

    def test_generated_melody_reports_single_voice(self, sample_notes: list[NoteEvent]) -> None:
        assert polyphony_stats(sample_notes).is_monophonic
