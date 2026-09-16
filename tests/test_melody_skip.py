"""M2.5: the SKIP state of the continuity strategy.

``highest`` / ``lowest`` still emit exactly one note per onset group; only
``continuity`` may leave a group empty when that group holds nothing but
accompaniment and forcing a note would drag the melodic path away.
"""

from __future__ import annotations

import pytest

from app.core.melody_extractor import (
    MelodyExtractionConfig,
    _skip_run_allowed,
    extract_melody,
)
from app.core.models import NoteEvent
from app.core.polyphony import is_monophonic

from conftest import make_note


def melody_config(**overrides: object) -> MelodyExtractionConfig:
    return MelodyExtractionConfig(**overrides)  # type: ignore[arg-type]


def pitches(notes: tuple[NoteEvent, ...]) -> list[int]:
    return [note.pitch for note in notes]


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
def skipped_bass_group() -> list[NoteEvent]:
    """C5 (+C2) / G2 only (melody missed by the AMT) / D5 (+C2)."""
    return [
        make_note(72, start=0.0, duration=0.5, velocity=90),
        make_note(36, start=0.0, duration=0.5, velocity=60),
        make_note(43, start=0.5, duration=0.5, velocity=60),
        make_note(74, start=1.0, duration=0.5, velocity=90),
        make_note(36, start=1.0, duration=0.5, velocity=60),
    ]


@pytest.fixture
def two_bar_hole() -> list[NoteEvent]:
    """C5 / two accompaniment-only groups / D5."""
    return [
        make_note(72, start=0.0, duration=0.5, velocity=90),
        make_note(36, start=0.0, duration=0.5, velocity=60),
        make_note(36, start=0.5, duration=0.5, velocity=60),
        make_note(43, start=1.0, duration=0.5, velocity=60),
        make_note(74, start=1.5, duration=0.5, velocity=90),
        make_note(36, start=1.5, duration=0.5, velocity=60),
    ]


@pytest.fixture
def sparse_transcription() -> list[NoteEvent]:
    """A faithful replica of the onset groups TransKun produced for samples/sample.mid.

    The left hand spans 41..48, the right hand 72..79, and five of the twelve
    right-hand notes were never detected - which is exactly the situation that
    dragged the M2 continuity path into the bass.
    """
    table = (
        # (pitch, start, duration, velocity)
        (48, 0.00, 1.15, 42),
        (43, 1.19, 1.17, 39),
        (48, 2.39, 1.18, 46),
        (45, 3.59, 1.16, 44),
        (74, 4.19, 1.76, 57),
        (41, 4.79, 1.19, 33),
        (43, 5.98, 1.18, 41),
        (48, 7.19, 1.18, 44),
        (43, 8.39, 1.16, 38),
        (48, 9.59, 1.17, 39),
        (72, 0.00, 0.56, 63),
        (79, 1.19, 1.17, 72),
        (76, 2.39, 1.77, 61),
        (72, 5.99, 0.57, 66),
        (76, 7.19, 0.81, 64),
        (79, 8.40, 0.58, 64),
    )
    return [make_note(pitch, start=start, duration=duration, velocity=velocity)
            for pitch, start, duration, velocity in table]


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #
class TestSkipConfig:
    def test_defaults(self) -> None:
        config = MelodyExtractionConfig()
        assert config.skip_cost == pytest.approx(2.5)
        assert config.max_skip_groups == 8
        assert config.max_skip_seconds == pytest.approx(2.0)

    @pytest.mark.parametrize("value", [0.0, -1.0])
    def test_a_free_skip_is_rejected(self, value: float) -> None:
        """A free skip would let the optimiser drop the entire piece."""
        with pytest.raises(ValueError, match="skip_cost"):
            MelodyExtractionConfig(skip_cost=value)

    def test_group_cap_is_validated(self) -> None:
        with pytest.raises(ValueError, match="max_skip_groups"):
            MelodyExtractionConfig(max_skip_groups=0)

    @pytest.mark.parametrize("value", [0.0, -1.0])
    def test_seconds_cap_is_validated(self, value: float) -> None:
        with pytest.raises(ValueError, match="max_skip_seconds"):
            MelodyExtractionConfig(max_skip_seconds=value)

    def test_caps_can_be_disabled(self) -> None:
        config = MelodyExtractionConfig(max_skip_groups=None, max_skip_seconds=None)
        assert config.max_skip_groups is None and config.max_skip_seconds is None


class TestSkipWindow:
    """Direct checks of the window decision used by the path search."""

    ANCHORS = (0.0, 0.5, 1.0, 1.5)

    def test_adjacent_groups_need_no_skip(self) -> None:
        config = melody_config()
        assert _skip_run_allowed(self.ANCHORS, -1, 0, config)
        assert _skip_run_allowed(self.ANCHORS, 0, 1, config)
        assert _skip_run_allowed(self.ANCHORS, 3, 4, config)

    def test_group_cap_is_applied(self) -> None:
        config = melody_config(max_skip_groups=2, max_skip_seconds=None)
        assert _skip_run_allowed(self.ANCHORS, 0, 2, config)  # 1 skipped
        assert _skip_run_allowed(self.ANCHORS, 0, 3, config)  # 2 skipped
        assert not _skip_run_allowed(self.ANCHORS, 0, 4, config)  # 3 skipped
        assert not _skip_run_allowed(self.ANCHORS, -1, 4, config)  # 4 skipped

    def test_time_cap_measures_the_skipped_groups(self) -> None:
        config = melody_config(max_skip_groups=None, max_skip_seconds=0.5)
        # A single skipped group spans nothing, however far apart the onsets are.
        assert _skip_run_allowed(self.ANCHORS, 0, 2, config)
        assert _skip_run_allowed(self.ANCHORS, 2, 4, config)
        # Two skipped groups 0.5 s apart span exactly 0.5 s ...
        assert _skip_run_allowed(self.ANCHORS, 0, 3, config)
        # ... three of them span 1.0 s and are too long.
        assert not _skip_run_allowed(self.ANCHORS, 0, 4, config)

    def test_leading_and_trailing_skips_use_the_same_rule(self) -> None:
        config = melody_config(max_skip_groups=None, max_skip_seconds=0.5)
        assert _skip_run_allowed(self.ANCHORS, -1, 2, config)  # leading: groups 0,1
        assert not _skip_run_allowed(self.ANCHORS, -1, 3, config)  # leading: 0,1,2

    def test_caps_can_be_disabled(self) -> None:
        config = melody_config(max_skip_groups=None, max_skip_seconds=None)
        assert _skip_run_allowed(self.ANCHORS, -1, 4, config)


# --------------------------------------------------------------------------- #
# the requested behaviour
# --------------------------------------------------------------------------- #
class TestSkipBehaviour:
    def test_bass_only_group_is_skipped_instead_of_selected(
        self, skipped_bass_group: list[NoteEvent]
    ) -> None:
        result = extract_melody(skipped_bass_group, melody_config())

        assert pitches(result.notes) == [72, 74]
        assert result.selected_group_count == 2
        assert result.skipped_group_count == 1
        assert result.has_rests

    def test_two_consecutive_bass_only_groups_are_skipped(
        self, two_bar_hole: list[NoteEvent]
    ) -> None:
        result = extract_melody(two_bar_hole, melody_config())

        assert pitches(result.notes) == [72, 74]
        assert result.skipped_group_count == 2

    def test_skipping_leaves_a_real_rest(self, two_bar_hole: list[NoteEvent]) -> None:
        result = extract_melody(two_bar_hole, melody_config())

        assert result.notes[0].end == pytest.approx(0.5)
        assert result.notes[1].start == pytest.approx(1.5)
        assert result.notes[1].start > result.notes[0].end

    def test_low_register_melody_is_never_skipped(self) -> None:
        """A real bass-line melody must survive: SKIP must not delete low notes."""
        notes = [make_note(pitch, start=index * 0.5, duration=0.5)
                 for index, pitch in enumerate((48, 50, 52, 53))]
        result = extract_melody(notes, melody_config())

        assert pitches(result.notes) == [48, 50, 52, 53]
        assert result.skipped_group_count == 0

    def test_clear_melody_is_never_skipped(self) -> None:
        notes = [make_note(pitch, start=index * 0.5, duration=0.5)
                 for index, pitch in enumerate((72, 74, 76, 77, 79))]
        result = extract_melody(notes, melody_config())

        assert pitches(result.notes) == [72, 74, 76, 77, 79]
        assert result.skipped_group_count == 0

    def test_short_ornament_does_not_pretend_the_group_is_empty(self) -> None:
        """SKIP must not become a cheap way out of a normal decision."""
        notes = [
            make_note(72, start=0.0, duration=0.8, velocity=90),
            make_note(74, start=1.0, duration=0.8, velocity=90),
            make_note(91, start=1.0, duration=0.08, velocity=70),
            make_note(76, start=2.0, duration=0.8, velocity=90),
        ]
        result = extract_melody(notes, melody_config())

        assert pitches(result.notes) == [72, 74, 76]
        assert result.skipped_group_count == 0

    def test_repeated_pitches_still_survive(self) -> None:
        notes = [make_note(72, start=index * 0.5, duration=0.5) for index in range(3)]
        result = extract_melody(notes, melody_config())

        assert pitches(result.notes) == [72, 72, 72]
        assert result.skipped_group_count == 0

    def test_the_whole_piece_cannot_be_skipped(self) -> None:
        """Skip cost accumulates, so an all-accompaniment input still yields notes."""
        notes = [make_note(pitch, start=index * 0.5, duration=0.5, velocity=55)
                 for index, pitch in enumerate((36, 43, 36, 41, 43, 36))]
        result = extract_melody(notes, melody_config())

        assert result.output_note_count >= 1
        assert result.skipped_group_count < result.total_group_count

    def test_long_accompaniment_run_still_yields_notes(self) -> None:
        notes = [make_note(36, start=index * 0.5, duration=0.5) for index in range(12)]
        result = extract_melody(notes, melody_config(max_skip_seconds=None))

        assert result.output_note_count > 0

    def test_no_notes_are_invented(self, two_bar_hole: list[NoteEvent]) -> None:
        """SKIP means "no melody here", never "synthesise something plausible"."""
        result = extract_melody(two_bar_hole, melody_config())

        for note in result.notes:
            matches = [
                original
                for original in two_bar_hole
                if original.pitch == note.pitch and original.start == pytest.approx(note.start)
            ]
            assert matches, f"melody contains a note that is not in the input: {note}"
            assert note.duration <= max(match.duration for match in matches) + 1e-9


class TestSkipCaps:
    def test_group_cap_limits_the_longest_hole(self, two_bar_hole: list[NoteEvent]) -> None:
        result = extract_melody(two_bar_hole, melody_config(max_skip_groups=1))

        assert result.longest_skip_run <= 1
        assert result.output_note_count > 2  # forced to pick notes inside the run

    def test_seconds_cap_limits_the_longest_hole(self, two_bar_hole: list[NoteEvent]) -> None:
        # Two consecutive skips 0.5 s apart span 0.5 s, which exceeds this cap.
        result = extract_melody(two_bar_hole, melody_config(max_skip_seconds=0.4))

        assert result.longest_skip_run <= 1
        assert result.output_note_count > 2

    def test_disabling_the_caps_allows_the_longer_hole(self, two_bar_hole: list[NoteEvent]) -> None:
        config = melody_config(max_skip_groups=None, max_skip_seconds=None)
        result = extract_melody(two_bar_hole, config)

        assert result.skipped_group_count == 2
        assert pitches(result.notes) == [72, 74]

    def test_a_wide_hole_needs_a_wide_window(self) -> None:
        notes = [
            make_note(72, start=0.0, duration=0.5, velocity=90),
            make_note(36, start=0.5, duration=0.5, velocity=60),
            make_note(36, start=1.0, duration=0.5, velocity=60),
            make_note(36, start=1.5, duration=0.5, velocity=60),
            make_note(74, start=2.0, duration=0.5, velocity=90),
        ]
        wide = extract_melody(notes, melody_config())
        narrow = extract_melody(notes, melody_config(max_skip_groups=1))

        assert wide.skipped_group_count == 3
        assert wide.longest_skip_run == 3
        assert narrow.longest_skip_run <= 1
        assert narrow.skipped_group_count < wide.skipped_group_count

    def test_skip_cost_is_monotone(self, two_bar_hole: list[NoteEvent]) -> None:
        cheap = extract_melody(two_bar_hole, melody_config(skip_cost=0.1))
        expensive = extract_melody(two_bar_hole, melody_config(skip_cost=50.0))

        assert cheap.skipped_group_count >= expensive.skipped_group_count
        assert expensive.skipped_group_count == 0
        assert cheap.skipped_group_count == 2


# --------------------------------------------------------------------------- #
# invariants
# --------------------------------------------------------------------------- #
class TestSkipInvariants:
    @pytest.mark.parametrize("strategy", ["highest", "lowest"])
    def test_baselines_never_skip(
        self, strategy: str, skipped_bass_group: list[NoteEvent]
    ) -> None:
        result = extract_melody(skipped_bass_group, melody_config(strategy=strategy))

        assert result.skipped_group_count == 0
        assert result.selected_group_count == result.total_group_count
        assert result.output_note_count == result.total_group_count

    def test_statistics_add_up(self, sparse_transcription: list[NoteEvent]) -> None:
        result = extract_melody(sparse_transcription, melody_config())

        assert result.total_group_count == result.onset_group_count
        assert result.selected_group_count + result.skipped_group_count == result.total_group_count
        assert result.output_note_count == result.selected_group_count
        assert result.has_rests is True

    def test_selected_group_indices_are_consistent(
        self, sparse_transcription: list[NoteEvent]
    ) -> None:
        result = extract_melody(sparse_transcription, melody_config())

        assert list(result.selected_group_indices) == [0, 1, 2, 4, 6, 7, 8]
        assert list(result.selected_group_indices) == sorted(result.selected_group_indices)
        assert len(set(result.selected_group_indices)) == result.selected_group_count
        assert result.longest_skip_run == 1

    def test_output_stays_monophonic_and_sorted(self, sparse_transcription: list[NoteEvent]) -> None:
        result = extract_melody(sparse_transcription, melody_config())
        notes = list(result.notes)

        assert notes
        assert is_monophonic(notes)
        assert [note.start for note in notes] == sorted(note.start for note in notes)
        assert all(note.duration > 0 for note in notes)

    def test_deterministic_and_independent_of_input_order(
        self, sparse_transcription: list[NoteEvent]
    ) -> None:
        config = melody_config()
        first = extract_melody(sparse_transcription, config)
        second = extract_melody(sparse_transcription, config)
        reversed_input = extract_melody(list(reversed(sparse_transcription)), config)

        assert first.notes == second.notes
        assert first.notes == reversed_input.notes
        assert first.skipped_group_count == reversed_input.skipped_group_count

    def test_input_notes_are_never_modified(self, sparse_transcription: list[NoteEvent]) -> None:
        snapshot = list(sparse_transcription)
        extract_melody(sparse_transcription, melody_config())
        assert sparse_transcription == snapshot

    def test_overlap_resolution_still_applies(self) -> None:
        notes = [
            make_note(72, start=0.0, duration=2.0),  # long C5
            make_note(36, start=0.5, duration=0.5),  # bass only -> skipped
            make_note(74, start=1.0, duration=0.5),  # D5
        ]
        result = extract_melody(notes, melody_config())

        assert pitches(result.notes) == [72, 74]
        assert result.overlap_truncated_count == 1
        assert result.notes[0].end == pytest.approx(1.0)
        assert result.dropped_fragment_count == 0
        assert is_monophonic(result.notes)


# --------------------------------------------------------------------------- #
# the real regression case
# --------------------------------------------------------------------------- #
class TestSparseTranscriptionRegression:
    def test_no_accompaniment_contamination(self, sparse_transcription: list[NoteEvent]) -> None:
        """M2 picked 48 43 48 45 74 41 43 48 43 48 here; M2.5 must pick the melody."""
        result = extract_melody(sparse_transcription, melody_config())

        assert pitches(result.notes) == [72, 79, 76, 74, 72, 76, 79]
        assert all(note.pitch >= 60 for note in result.notes)
        assert all(note.pitch not in {41, 43, 45, 48} for note in result.notes)

    def test_skipped_groups_are_the_accompaniment_only_ones(
        self, sparse_transcription: list[NoteEvent]
    ) -> None:
        result = extract_melody(sparse_transcription, melody_config())

        assert result.total_group_count == 10
        assert result.selected_group_count == 7
        assert result.skipped_group_count == 3

    def test_pitch_range_becomes_the_melody_range(
        self, sparse_transcription: list[NoteEvent]
    ) -> None:
        result = extract_melody(sparse_transcription, melody_config())
        pitches_found = [note.pitch for note in result.notes]

        assert (min(pitches_found), max(pitches_found)) == (72, 79)

    def test_skipping_is_what_fixes_it(self, sparse_transcription: list[NoteEvent]) -> None:
        """With skipping made prohibitively expensive the M2 failure comes back."""
        no_skip = extract_melody(sparse_transcription, melody_config(skip_cost=1e6))
        with_skip = extract_melody(sparse_transcription, melody_config())

        contamination = {41, 43, 45, 48}
        assert any(note.pitch in contamination for note in no_skip.notes)
        assert not any(note.pitch in contamination for note in with_skip.notes)
