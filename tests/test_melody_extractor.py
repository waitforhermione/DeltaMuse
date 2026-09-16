"""Melody extraction: filtering, the three strategies, the cost model and invariants."""

from __future__ import annotations

import pytest

from app.core.errors import PianoScoreError
from app.core.melody_extractor import (
    DEFAULT_INTERVAL_COST_POINTS,
    STRATEGIES,
    MelodyExtractionConfig,
    MelodyExtractor,
    assert_monophonic,
    emission_cost,
    extract_melody,
    extract_melody_notes,
    gap_relaxation_factor,
    interval_cost,
    transition_cost,
)
from app.core.models import NoteEvent
from app.core.polyphony import is_monophonic

from conftest import make_note


def melody_config(**overrides: object) -> MelodyExtractionConfig:
    return MelodyExtractionConfig(**overrides)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
def two_hand_notes() -> list[NoteEvent]:
    """Left-hand arpeggio under a stepwise right-hand melody."""
    bass = (36, 43, 48, 43, 36, 43, 48, 43)
    melody = (64, 65, 67, 69, 71, 72, 71, 69)
    notes: list[NoteEvent] = []
    for index, (low, high) in enumerate(zip(bass, melody)):
        start = index * 0.5
        notes.append(make_note(low, start=start, duration=0.5, velocity=60))
        notes.append(make_note(high, start=start, duration=0.5, velocity=90))
    return notes


@pytest.fixture
def ornament_notes() -> list[NoteEvent]:
    """C5 D5 E5 F5 with a fleeting G6 ornament sharing the E5 onset."""
    return [
        make_note(72, start=0.0, duration=0.8, velocity=90),
        make_note(74, start=1.0, duration=0.8, velocity=90),
        make_note(76, start=2.0, duration=0.8, velocity=90),
        make_note(91, start=2.0, duration=0.08, velocity=70),
        make_note(77, start=3.0, duration=0.8, velocity=90),
    ]


@pytest.fixture
def new_phrase_notes() -> list[NoteEvent]:
    """C5, then a 1.5 s rest, then a leap to C6 next to a short F5.

    The gap is exactly ``gap_relax_full``, so the leap is billed at the relaxed
    floor rate; without that relaxation the leap would be avoided.
    """
    return [
        make_note(72, start=0.0, duration=1.5, velocity=90),
        make_note(77, start=3.0, duration=0.06, velocity=60),
        make_note(84, start=3.0, duration=0.60, velocity=100),
    ]


@pytest.fixture
def dense_polyphonic_notes() -> list[NoteEvent]:
    """Three overlapping voices on every 0.4 s beat."""
    notes: list[NoteEvent] = []
    for index in range(12):
        start = index * 0.4
        notes.append(make_note(48 + (index % 3) * 2, start=start, duration=0.9, velocity=70))
        notes.append(make_note(64 + (index % 4), start=start, duration=0.35, velocity=85))
        notes.append(make_note(67 + (index % 3), start=start + 0.02, duration=0.5, velocity=80))
    return notes


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #
class TestConfig:
    def test_defaults_are_the_documented_ones(self) -> None:
        config = MelodyExtractionConfig()
        assert config.strategy == "continuity"
        assert config.min_note_duration == pytest.approx(0.05)
        assert config.onset_epsilon == pytest.approx(0.03)
        assert config.overlap_fragment_threshold == pytest.approx(0.03)
        assert config.max_candidates_per_group == 16

    def test_for_strategy_helper(self) -> None:
        assert MelodyExtractionConfig.for_strategy("lowest").strategy == "lowest"

    def test_unknown_strategy_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="strategy"):
            MelodyExtractionConfig(strategy="middle")

    @pytest.mark.parametrize("value", [0.0, -0.1])
    def test_min_note_duration_must_be_positive(self, value: float) -> None:
        with pytest.raises(ValueError, match="min_note_duration"):
            MelodyExtractionConfig(min_note_duration=value)

    def test_onset_epsilon_must_not_be_negative(self) -> None:
        with pytest.raises(ValueError, match="onset_epsilon"):
            MelodyExtractionConfig(onset_epsilon=-0.01)

    def test_overlap_threshold_must_not_be_negative(self) -> None:
        with pytest.raises(ValueError, match="overlap_fragment_threshold"):
            MelodyExtractionConfig(overlap_fragment_threshold=-0.01)

    def test_pitch_window_is_validated(self) -> None:
        with pytest.raises(ValueError, match="min_pitch"):
            MelodyExtractionConfig(min_pitch=200)
        with pytest.raises(ValueError, match="exceed max_pitch"):
            MelodyExtractionConfig(min_pitch=80, max_pitch=60)

    def test_weights_must_not_be_negative(self) -> None:
        with pytest.raises(ValueError, match="bass_weight"):
            MelodyExtractionConfig(bass_weight=-1.0)

    def test_register_window_is_validated(self) -> None:
        with pytest.raises(ValueError, match="register_floor"):
            MelodyExtractionConfig(register_floor=90, register_ceiling=70)

    def test_interval_cost_points_are_validated(self) -> None:
        with pytest.raises(ValueError, match="at least two"):
            MelodyExtractionConfig(interval_cost_points=((0.0, 0.0),))
        with pytest.raises(ValueError, match="strictly increasing"):
            MelodyExtractionConfig(interval_cost_points=((0.0, 0.0), (5.0, 1.0), (3.0, 2.0)))
        with pytest.raises(ValueError, match="non-decreasing"):
            MelodyExtractionConfig(interval_cost_points=((0.0, 1.0), (5.0, 0.5)))

    def test_gap_configuration_is_validated(self) -> None:
        with pytest.raises(ValueError, match="gap_relax_full"):
            MelodyExtractionConfig(gap_relax_start=1.0, gap_relax_full=0.5)
        with pytest.raises(ValueError, match="gap_relax_min_factor"):
            MelodyExtractionConfig(gap_relax_min_factor=1.5)

    def test_candidate_guard_is_validated(self) -> None:
        with pytest.raises(ValueError, match="max_candidates_per_group"):
            MelodyExtractionConfig(max_candidates_per_group=0)


# --------------------------------------------------------------------------- #
# cost model
# --------------------------------------------------------------------------- #
class TestCostModel:
    def test_interval_cost_is_monotone_non_decreasing(self) -> None:
        config = MelodyExtractionConfig()
        costs = [interval_cost(distance, config) for distance in range(0, 61)]
        assert costs == sorted(costs)
        assert costs[0] == 0.0
        assert costs[2] < costs[5] < costs[12] < costs[24] < costs[48]

    def test_interval_cost_hits_the_breakpoints(self) -> None:
        config = MelodyExtractionConfig()
        for semitones, expected in DEFAULT_INTERVAL_COST_POINTS:
            assert interval_cost(semitones, config) == pytest.approx(expected)

    def test_interval_cost_interpolates_and_extrapolates(self) -> None:
        config = MelodyExtractionConfig()
        # halfway between the (5, 0.25) and (12, 1.10) breakpoints
        assert interval_cost(8.5, config) == pytest.approx((0.25 + 1.10) / 2)
        # past the last breakpoint the final slope (3.20 - 1.10) / 12 keeps going
        assert interval_cost(36, config) == pytest.approx(3.20 + (3.20 - 1.10))

    def test_interval_cost_is_symmetric_and_never_forbids_a_leap(self) -> None:
        config = MelodyExtractionConfig()
        assert interval_cost(-7, config) == pytest.approx(interval_cost(7, config))
        assert interval_cost(60, config) < float("inf")

    def test_gap_relaxation_is_full_strength_across_short_gaps(self) -> None:
        config = MelodyExtractionConfig()
        assert gap_relaxation_factor(-0.3, config) == 1.0
        assert gap_relaxation_factor(0.0, config) == 1.0
        assert gap_relaxation_factor(config.gap_relax_start, config) == 1.0

    def test_gap_relaxation_decays_to_the_floor(self) -> None:
        config = MelodyExtractionConfig()
        assert gap_relaxation_factor(config.gap_relax_full, config) == pytest.approx(
            config.gap_relax_min_factor
        )
        assert gap_relaxation_factor(30.0, config) == pytest.approx(config.gap_relax_min_factor)

    def test_gap_relaxation_is_monotone_decreasing(self) -> None:
        config = MelodyExtractionConfig()
        factors = [gap_relaxation_factor(step / 10.0, config) for step in range(0, 30)]
        assert factors == sorted(factors, reverse=True)

    def test_transition_cost_shrinks_after_a_rest(self) -> None:
        config = MelodyExtractionConfig()
        previous = make_note(60, start=0.0, duration=0.5)
        attached = make_note(75, start=0.6, duration=0.5)
        after_rest = make_note(75, start=3.0, duration=0.5)

        assert transition_cost(previous, after_rest, config) < transition_cost(previous, attached, config)
        assert transition_cost(previous, after_rest, config) == pytest.approx(
            interval_cost(15, config) * config.gap_relax_min_factor
        )

    def test_emission_cost_penalises_the_bass_more_the_lower_it_goes(self) -> None:
        config = MelodyExtractionConfig()
        assert (
            emission_cost(make_note(36), config)
            > emission_cost(make_note(48), config)
            > emission_cost(make_note(60), config)
        )

    def test_bass_penalty_is_soft_not_a_hard_cut(self) -> None:
        config = MelodyExtractionConfig()
        # Still a finite, survivable cost - never an exclusion.
        assert emission_cost(make_note(24), config) < 10.0
        assert emission_cost(make_note(24), config) > emission_cost(make_note(60), config)

    def test_emission_cost_prefers_healthy_durations(self) -> None:
        config = MelodyExtractionConfig()
        assert emission_cost(make_note(64, duration=0.02), config) > emission_cost(
            make_note(64, duration=0.4), config
        )

    def test_high_register_grace_notes_are_mildly_discouraged(self) -> None:
        config = MelodyExtractionConfig()
        assert emission_cost(make_note(91), config) > emission_cost(make_note(79), config)

    def test_velocity_is_only_a_whisper(self) -> None:
        config = MelodyExtractionConfig()
        loud = emission_cost(make_note(64, duration=0.4, velocity=127), config)
        quiet = emission_cost(make_note(64, duration=0.4, velocity=20), config)
        assert loud < quiet
        assert quiet - loud <= 2.0 * config.velocity_weight + 1e-9

    def test_missing_confidence_costs_nothing(self) -> None:
        config = MelodyExtractionConfig()
        unknown = emission_cost(make_note(64, confidence=None), config)
        perfect = emission_cost(make_note(64, confidence=1.0), config)
        assert unknown == pytest.approx(perfect)

    def test_confidence_is_used_when_a_backend_reports_it(self) -> None:
        config = MelodyExtractionConfig()
        assert emission_cost(make_note(64, confidence=0.1), config) > emission_cost(
            make_note(64, confidence=0.9), config
        )


# --------------------------------------------------------------------------- #
# highest / lowest baselines
# --------------------------------------------------------------------------- #
class TestHighestAndLowest:
    @pytest.mark.parametrize(("strategy", "expected"), [("highest", 67), ("lowest", 60)])
    def test_three_note_chord(self, strategy: str, expected: int) -> None:
        notes = [make_note(60), make_note(64), make_note(67)]
        result = extract_melody(notes, melody_config(strategy=strategy))
        assert [note.pitch for note in result.notes] == [expected]

    @pytest.mark.parametrize(("strategy", "expected"), [("highest", 72), ("lowest", 60)])
    def test_two_note_chord(self, strategy: str, expected: int) -> None:
        notes = [make_note(60, start=0.0), make_note(72, start=0.01)]
        result = extract_melody(notes, melody_config(strategy=strategy))
        assert [note.pitch for note in result.notes] == [expected]

    @pytest.mark.parametrize("strategy", ["highest", "lowest", "continuity"])
    def test_single_note(self, strategy: str) -> None:
        result = extract_melody([make_note(64)], melody_config(strategy=strategy))
        assert [note.pitch for note in result.notes] == [64]

    @pytest.mark.parametrize(("strategy", "expected"), [("highest", 67), ("lowest", 60)])
    def test_onsets_within_epsilon_share_a_group(self, strategy: str, expected: int) -> None:
        notes = [make_note(60, start=0.0), make_note(67, start=0.025)]
        result = extract_melody(notes, melody_config(strategy=strategy, onset_epsilon=0.03))
        assert [note.pitch for note in result.notes] == [expected]

    @pytest.mark.parametrize(("strategy", "expected"), [("highest", 60), ("lowest", 60)])
    def test_onsets_outside_epsilon_become_separate_groups(self, strategy: str, expected: int) -> None:
        notes = [make_note(60, start=0.0, duration=0.5), make_note(72, start=0.04, duration=0.1)]
        result = extract_melody(notes, melody_config(strategy=strategy, onset_epsilon=0.02))
        assert [note.pitch for note in result.notes] == [expected, 72]

    def test_epsilon_controls_grouping(self) -> None:
        notes = [make_note(60, start=0.0, duration=0.5), make_note(72, start=0.04, duration=0.5)]

        tight = extract_melody(notes, melody_config(strategy="highest", onset_epsilon=0.02))
        loose = extract_melody(notes, melody_config(strategy="highest", onset_epsilon=0.05))

        assert [note.pitch for note in tight.notes] == [60, 72]
        assert [note.pitch for note in loose.notes] == [72]

    def test_tie_is_broken_by_longer_duration(self) -> None:
        notes = [make_note(72, start=0.0, duration=0.2), make_note(72, start=0.0, duration=0.8)]
        result = extract_melody(notes, melody_config(strategy="highest"))
        assert result.notes[0].duration == pytest.approx(0.8)

    def test_tie_is_then_broken_by_higher_velocity(self) -> None:
        notes = [
            make_note(72, start=0.0, duration=0.5, velocity=40),
            make_note(72, start=0.0, duration=0.5, velocity=110),
        ]
        result = extract_melody(notes, melody_config(strategy="highest"))
        assert result.notes[0].velocity == 110

    def test_lowest_is_the_mirror_of_highest(self) -> None:
        notes = [make_note(48, start=0.0, duration=0.5), make_note(60, start=0.0, duration=0.2)]
        result = extract_melody(notes, melody_config(strategy="lowest"))
        assert [note.pitch for note in result.notes] == [48]


# --------------------------------------------------------------------------- #
# continuity
# --------------------------------------------------------------------------- #
class TestContinuity:
    def test_follows_the_right_hand_instead_of_the_left_hand(self, two_hand_notes: list[NoteEvent]) -> None:
        result = extract_melody(two_hand_notes, melody_config(strategy="continuity"))

        expected = [64, 65, 67, 69, 71, 72, 71, 69]
        assert [note.pitch for note in result.notes] == expected
        assert result.output_note_count == len(expected)

    def test_highest_fails_where_continuity_succeeds(self, ornament_notes: list[NoteEvent]) -> None:
        highest = extract_melody(ornament_notes, melody_config(strategy="highest"))
        continuity = extract_melody(ornament_notes, melody_config(strategy="continuity"))

        # The ornament wins the highest baseline outright ...
        assert [note.pitch for note in highest.notes] == [72, 74, 91, 77]
        # ... but continuity keeps the melodic path through E5.
        assert [note.pitch for note in continuity.notes] == [72, 74, 76, 77]

    def test_gap_relaxation_lets_a_new_phrase_leap(self, new_phrase_notes: list[NoteEvent]) -> None:
        result = extract_melody(new_phrase_notes, melody_config(strategy="continuity"))
        assert [note.pitch for note in result.notes] == [72, 84]

    def test_without_gap_relaxation_the_leap_is_avoided(self, new_phrase_notes: list[NoteEvent]) -> None:
        """Same input, relaxation disabled: the DP stays glued to the old register."""
        strict = melody_config(strategy="continuity", gap_relax_min_factor=1.0)
        result = extract_melody(new_phrase_notes, strict)
        assert [note.pitch for note in result.notes] == [72, 77]

    def test_repeated_pitches_are_kept_as_separate_notes(self) -> None:
        notes = [
            make_note(72, start=0.0, duration=0.5),
            make_note(72, start=1.0, duration=0.5),
            make_note(72, start=2.0, duration=0.5),
        ]
        result = extract_melody(notes, melody_config(strategy="continuity"))

        assert [note.pitch for note in result.notes] == [72, 72, 72]
        assert result.output_note_count == 3

    def test_works_without_any_confidence(self, two_hand_notes: list[NoteEvent]) -> None:
        assert all(note.confidence is None for note in two_hand_notes)
        result = extract_melody(two_hand_notes, melody_config(strategy="continuity"))
        assert result.output_note_count == 8

    def test_confidence_breaks_a_tie_when_present(self) -> None:
        notes = [
            make_note(72, start=0.0, duration=0.5, confidence=0.05),
            make_note(72, start=0.0, duration=0.5, confidence=0.95),
        ]
        result = extract_melody(notes, melody_config(strategy="continuity"))
        assert result.notes[0].confidence == pytest.approx(0.95)

    def test_low_register_melody_survives_when_it_is_the_only_choice(self) -> None:
        notes = [make_note(40, start=0.0, duration=0.5), make_note(41, start=0.5, duration=0.5)]
        result = extract_melody(notes, melody_config(strategy="continuity"))
        assert [note.pitch for note in result.notes] == [40, 41]

    def test_bass_candidate_loses_to_a_melody_register_candidate(self) -> None:
        notes = [make_note(40, start=0.0, duration=0.5, velocity=110), make_note(64, start=0.0, duration=0.5)]
        result = extract_melody(notes, melody_config(strategy="continuity"))
        assert [note.pitch for note in result.notes] == [64]

    def test_huge_onset_group_is_capped(self) -> None:
        notes = [make_note(48 + pitch, start=0.0, duration=0.5) for pitch in range(24)]
        config = melody_config(strategy="continuity", max_candidates_per_group=8)
        result = extract_melody(notes, config)

        assert result.output_note_count == 1
        assert is_monophonic(result.notes)

    def test_uncapped_huge_onset_group_also_works(self) -> None:
        notes = [make_note(48 + pitch, start=0.0, duration=0.5) for pitch in range(24)]
        config = melody_config(strategy="continuity", max_candidates_per_group=None)
        assert extract_melody(notes, config).output_note_count == 1

    def test_result_is_identical_when_run_again(self, dense_polyphonic_notes: list[NoteEvent]) -> None:
        config = melody_config(strategy="continuity")
        first = extract_melody(dense_polyphonic_notes, config)
        second = extract_melody(dense_polyphonic_notes, config)
        assert first.notes == second.notes


# --------------------------------------------------------------------------- #
# filtering
# --------------------------------------------------------------------------- #
class TestFiltering:
    def test_short_notes_are_filtered_and_counted(self) -> None:
        notes = [make_note(60, start=0.0, duration=0.02), make_note(64, start=0.5, duration=0.5)]
        result = extract_melody(notes, melody_config(min_note_duration=0.05))

        assert result.input_note_count == 2
        assert result.removed_short_count == 1
        assert result.filtered_note_count == 1
        assert [note.pitch for note in result.notes] == [64]

    def test_note_exactly_at_the_minimum_duration_is_kept(self) -> None:
        notes = [make_note(64, start=0.0, duration=0.05)]
        assert extract_melody(notes, melody_config(min_note_duration=0.05)).output_note_count == 1

    def test_pitch_window_is_optional(self) -> None:
        # ``highest`` is used here on purpose: it never skips, so the assertions
        # below isolate the pitch-window filter from the skip logic.
        notes = [make_note(36, start=0.0, duration=0.5), make_note(60, start=0.5, duration=0.5)]

        default = extract_melody(notes, melody_config(strategy="highest"))
        assert default.removed_pitch_range_count == 0
        assert default.filtered_note_count == 2
        assert [note.pitch for note in default.notes] == [36, 60]

        windowed = extract_melody(notes, melody_config(strategy="highest", min_pitch=48))
        assert [note.pitch for note in windowed.notes] == [60]
        assert windowed.removed_pitch_range_count == 1

    def test_everything_filtered_yields_an_empty_result(self) -> None:
        notes = [make_note(60, start=0.0, duration=0.01)]
        result = extract_melody(notes, melody_config(min_note_duration=0.5))
        assert result.notes == ()
        assert result.output_note_count == 0
        assert result.removed_short_count == 1

    def test_empty_input(self) -> None:
        result = extract_melody([])
        assert result.notes == ()
        assert result.input_note_count == 0
        assert result.onset_group_count == 0
        assert result.output_note_count == 0

    def test_non_note_input_is_rejected(self) -> None:
        with pytest.raises(TypeError, match="NoteEvent"):
            extract_melody([("not", "a", "note")])  # type: ignore[list-item]

    def test_input_notes_are_never_modified(self, dense_polyphonic_notes: list[NoteEvent]) -> None:
        snapshot = list(dense_polyphonic_notes)
        extract_melody(dense_polyphonic_notes)
        assert dense_polyphonic_notes == snapshot
        assert all(original is current for original, current in zip(dense_polyphonic_notes, snapshot))


# --------------------------------------------------------------------------- #
# overlap handling reported by the extractor
# --------------------------------------------------------------------------- #
class TestOverlapHandling:
    def test_sustained_note_is_truncated_to_the_next_onset(self) -> None:
        notes = [make_note(60, start=0.0, duration=0.9), make_note(64, start=0.6, duration=0.4)]
        result = extract_melody(notes, melody_config(strategy="continuity"))

        assert result.overlap_truncated_count == 1
        assert result.notes[0].end == pytest.approx(0.6)
        assert is_monophonic(result.notes)

    def test_truncation_fragment_is_dropped_and_reported(self) -> None:
        config = melody_config(onset_epsilon=0.01, min_note_duration=0.05)
        notes = [make_note(60, start=0.0, duration=0.05), make_note(64, start=0.02, duration=0.5)]
        result = extract_melody(notes, config)

        assert result.removed_short_count == 0  # 0.05 s is not "originally short"
        assert result.overlap_truncated_count == 1
        assert result.dropped_fragment_count == 1
        assert [note.pitch for note in result.notes] == [64]

    def test_non_overlapping_input_is_left_alone(self) -> None:
        notes = [make_note(60, start=0.0, duration=0.4), make_note(64, start=0.5, duration=0.4)]
        result = extract_melody(notes)
        assert result.overlap_truncated_count == 0
        assert result.dropped_fragment_count == 0

    def test_fragment_threshold_is_configurable(self) -> None:
        notes = [make_note(60, start=0.0, duration=0.05), make_note(64, start=0.02, duration=0.5)]
        custom = melody_config(onset_epsilon=0.01, overlap_fragment_threshold=0.01)
        result = extract_melody(notes, custom)

        assert result.dropped_fragment_count == 0
        assert [note.pitch for note in result.notes] == [60, 64]


# --------------------------------------------------------------------------- #
# output invariants
# --------------------------------------------------------------------------- #
class TestOutputInvariants:
    @pytest.mark.parametrize("strategy", STRATEGIES)
    def test_strictly_monophonic_and_sorted(
        self, strategy: str, dense_polyphonic_notes: list[NoteEvent]
    ) -> None:
        result = extract_melody(dense_polyphonic_notes, melody_config(strategy=strategy))
        notes = list(result.notes)

        assert notes
        assert is_monophonic(notes)
        assert [note.start for note in notes] == sorted(note.start for note in notes)
        assert all(note.duration > 0.0 for note in notes)
        assert all(0 <= note.pitch <= 127 for note in notes)
        assert result.output_note_count == len(notes)

    @pytest.mark.parametrize("strategy", STRATEGIES)
    def test_counts_add_up(self, strategy: str, dense_polyphonic_notes: list[NoteEvent]) -> None:
        result = extract_melody(dense_polyphonic_notes, melody_config(strategy=strategy))

        assert result.input_note_count == len(dense_polyphonic_notes)
        assert result.removed_note_count == result.removed_short_count + result.removed_pitch_range_count
        assert result.notes_merged_by_grouping == (
            result.filtered_note_count - result.onset_group_count
        )
        assert result.total_group_count == result.onset_group_count
        assert result.selected_group_count + result.skipped_group_count == result.total_group_count
        assert result.output_note_count == result.selected_group_count - result.dropped_fragment_count
        assert result.strategy == strategy

    @pytest.mark.parametrize("strategy", STRATEGIES)
    def test_result_object_carries_the_configuration(
        self, strategy: str, dense_polyphonic_notes: list[NoteEvent]
    ) -> None:
        config = melody_config(strategy=strategy)
        result = MelodyExtractor(config).extract(dense_polyphonic_notes)
        assert result.config is config

    def test_assert_monophonic_helper(self) -> None:
        assert_monophonic([make_note(60, duration=0.5), make_note(64, start=0.5, duration=0.5)])
        with pytest.raises(PianoScoreError, match="overlapping"):
            assert_monophonic([make_note(60, duration=1.0), make_note(64, start=0.5, duration=1.0)])

    def test_extract_melody_notes_returns_a_plain_list(self, two_hand_notes: list[NoteEvent]) -> None:
        notes = extract_melody_notes(two_hand_notes)
        assert isinstance(notes, list)
        assert all(isinstance(note, NoteEvent) for note in notes)
        assert is_monophonic(notes)


# --------------------------------------------------------------------------- #
# scale
# --------------------------------------------------------------------------- #
class TestScale:
    def test_five_hundred_onset_groups_with_six_candidates_each(self) -> None:
        """O(groups x candidates^2) must stay comfortably fast - no exponential search."""
        notes: list[NoteEvent] = []
        for index in range(500):
            start = index * 0.25
            for offset in range(6):
                notes.append(
                    make_note(
                        60 + offset * 3,
                        start=start + offset * 0.004,
                        duration=0.3,
                        velocity=70 + offset,
                    )
                )

        result = extract_melody(notes, melody_config(strategy="continuity", onset_epsilon=0.03))

        assert result.onset_group_count == 500
        assert result.output_note_count == 500
        assert is_monophonic(result.notes)
