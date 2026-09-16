"""Static score layout: sections, geometry, fragments, grid, determinism."""

from __future__ import annotations

import math

import pytest

from app.core.errors import EmptyScoreError, ScoreLayoutError
from app.core.models import PlayableNote
from app.score.layout import (
    DEFAULT_SECTION_DURATION,
    GridLine,
    LayoutHeader,
    LayoutScore,
    ScoreLayoutConfig,
    build_grid,
    build_layout,
    duration_to_width,
    lane_to_y,
    section_index_for_end,
    section_index_for_time,
    section_time_bounds,
    time_to_x,
)

LANES = ("Z", "X", "C", "V", "B", "N", "M", ",")


def playable(
    pitch: int,
    lane: int,
    start: float,
    duration: float,
    *,
    source_pitch: int | None = None,
    key_label: str | None = None,
    semitone: bool = False,
    octave_offset: int = 0,
    confidence: float | None = None,
) -> PlayableNote:
    return PlayableNote(
        source_pitch=pitch if source_pitch is None else source_pitch,
        pitch=pitch,
        start=start,
        duration=duration,
        lane=lane,
        key_label=key_label if key_label is not None else LANES[lane],
        semitone=semitone,
        octave_offset=octave_offset,
        confidence=confidence,
    )


def layout_of(*notes: PlayableNote, **kwargs) -> LayoutScore:
    return build_layout(list(notes), lanes=LANES, **kwargs)


# --------------------------------------------------------------------------- #
# section assignment
# --------------------------------------------------------------------------- #
class TestSectionAssignment:
    def test_helper(self) -> None:
        assert section_index_for_time(0.0, 6.0) == 0
        assert section_index_for_time(5.999, 6.0) == 0
        assert section_index_for_time(6.0, 6.0) == 1
        assert section_index_for_time(6.001, 6.0) == 1
        assert section_index_for_time(11.999, 6.0) == 1
        assert section_index_for_time(12.0, 6.0) == 2

    @pytest.mark.parametrize(
        ("end", "expected"),
        [(6.0, 0), (5.999, 0), (6.001, 1), (12.0, 1), (7.5, 1), (0.5, 0)],
    )
    def test_end_helper_never_spills_on_a_boundary(self, end: float, expected: int) -> None:
        assert section_index_for_end(end, 6.0) == expected

    def test_note_on_the_boundary_belongs_to_the_next_section(self) -> None:
        layout = layout_of(playable(60, 0, 6.0, 1.0))
        assert layout.sections[0].notes == ()
        assert [f.fragment_start for f in layout.sections[1].notes] == [6.0]

    def test_multi_section_score(self, profile) -> None:
        notes = [playable(60 + lane, lane, start, 0.5) for lane, start in enumerate((0.0, 7.0, 13.0))]
        layout = layout_of(*notes)

        assert layout.section_count == 3
        assert [section.index for section in layout.sections] == [0, 1, 2]
        assert [section.start_time for section in layout.sections] == [0.0, 6.0, 12.0]
        assert [len(section.notes) for section in layout.sections] == [1, 1, 1]

    def test_last_section_extends_beyond_the_last_note(self, profile) -> None:
        layout = layout_of(playable(60, 0, 0.0, 0.5))
        assert layout.section_count == 1
        assert layout.sections[0].end_time == pytest.approx(6.0)

    def test_section_time_bounds(self) -> None:
        assert section_time_bounds(0, 6.0) == (0.0, 6.0)
        assert section_time_bounds(2, 6.0) == (12.0, 18.0)


# --------------------------------------------------------------------------- #
# x geometry
# --------------------------------------------------------------------------- #
class TestXGeometry:
    CONFIG = ScoreLayoutConfig(canvas_width=2000, section_duration=6.0)

    def test_helper_edges(self) -> None:
        config = self.CONFIG
        assert time_to_x(0.0, 0.0, 6.0, config) == pytest.approx(config.left_label_width)
        assert time_to_x(3.0, 0.0, 6.0, config) == pytest.approx(config.left_label_width + config.usable_width() / 2)
        assert time_to_x(6.0, 0.0, 6.0, config) == pytest.approx(config.canvas_width - config.right_padding)

    def test_section_start_maps_to_the_left_edge(self) -> None:
        layout = layout_of(playable(60, 0, 0.0, 0.5), config=self.CONFIG)
        assert layout.sections[0].notes[0].x == pytest.approx(80.0)

    def test_section_middle_maps_to_the_middle(self) -> None:
        layout = layout_of(playable(60, 0, 3.0, 0.5), config=self.CONFIG)
        assert layout.sections[0].notes[0].x == pytest.approx(80.0 + 1896.0 / 2)

    def test_section_end_maps_to_the_right_edge(self) -> None:
        config = self.CONFIG
        layout = layout_of(playable(60, 0, 5.999, 0.001), config=config)
        fragment = layout.sections[0].notes[0]

        # 5.999 s is one millisecond left of the section end.
        assert fragment.x == pytest.approx(time_to_x(5.999, 0.0, 6.0, config))
        assert fragment.x + fragment.width >= config.canvas_width - config.right_padding

    def test_the_same_time_maps_identically_in_every_section(self) -> None:
        config = self.CONFIG
        first = time_to_x(1.0, 0.0, 6.0, config)
        second = time_to_x(7.0, 6.0, 6.0, config)
        third = time_to_x(13.0, 12.0, 6.0, config)
        assert first == second == third

    def test_scale_is_uniform_for_the_short_last_section(self, profile) -> None:
        """14.2 s of music keeps the 6 s scale: the last row is not stretched."""
        notes = [playable(60, 0, 12.5, 1.5), playable(62, 1, 0.0, 0.5)]
        layout = layout_of(*notes, config=ScoreLayoutConfig(section_duration=6.0))

        assert layout.section_count == 3
        assert layout.sections[2].start_time == pytest.approx(12.0)
        assert layout.sections[2].end_time == pytest.approx(18.0)
        # 12.5 s is 0.5 s into the section -> same x as 0.5 s in section 0.
        assert layout.sections[2].notes[0].x == pytest.approx(80.0 + 0.5 / 6.0 * 1896.0)


# --------------------------------------------------------------------------- #
# lane geometry
# --------------------------------------------------------------------------- #
class TestLaneGeometry:
    def test_every_lane_has_a_distinct_y(self, profile) -> None:
        config = ScoreLayoutConfig()
        notes = [playable(60 + lane, lane, 0.0, 0.5) for lane in range(8)]
        layout = layout_of(*notes, config=config)

        ys = sorted(fragment.y for fragment in layout.fragments())
        for lower, upper in zip(ys, ys[1:]):
            assert upper - lower >= config.note_height - 1e-9  # no overlap
        assert ys == sorted(
            fragment.y for fragment in sorted(layout.fragments(), key=lambda f: f.lane)
        )

    def test_lane_order_is_data_driven_and_ascending(self, profile) -> None:
        notes = [playable(60, lane, 0.0, 0.5) for lane in range(8)]
        layout = layout_of(*notes)

        by_lane = {fragment.lane: fragment.y for fragment in layout.fragments()}
        assert [by_lane[lane] for lane in range(8)] == sorted(by_lane.values())

    def test_lane_zero_is_at_the_top_of_the_section(self, profile) -> None:
        layout = layout_of(playable(60, 0, 0.0, 0.5))
        section = layout.sections[0]
        fragment = section.notes[0]

        assert fragment.y >= section.y
        assert fragment.y + fragment.height <= section.y + section.height

    def test_lane_seven_is_inside_the_section(self, profile) -> None:
        layout = layout_of(playable(60, 7, 0.0, 0.5))
        section = layout.sections[0]
        fragment = section.notes[0]

        assert fragment.y >= section.y
        assert fragment.y + fragment.height <= section.y + section.height

    def test_lane_to_y_helper_is_exhaustive(self) -> None:
        config = ScoreLayoutConfig()
        section_y = config.sections_top()
        ys = [lane_to_y(lane, section_y, 8, config) for lane in range(8)]

        assert ys == sorted(ys)
        assert len(set(ys)) == 8
        for y in ys:
            assert section_y <= y
            assert y + config.note_height <= section_y + config.section_height

    def test_lane_out_of_range_is_rejected(self) -> None:
        with pytest.raises(ScoreLayoutError, match="lane 8"):
            layout_of(playable(60, 8, 0.0, 0.5, key_label="Z"))

    def test_negative_lane_is_rejected_by_the_model(self) -> None:
        """PlayableNote itself already refuses negative lanes."""
        with pytest.raises(ValueError, match="lane must be >= 0"):
            playable(60, -1, 0.0, 0.5, key_label="Z")


# --------------------------------------------------------------------------- #
# duration -> width
# --------------------------------------------------------------------------- #
class TestWidth:
    def test_short_note_uses_the_minimum_width(self, profile) -> None:
        config = ScoreLayoutConfig(min_note_width=20.0)
        layout = layout_of(playable(60, 0, 0.0, 0.001), config=config)

        assert layout.sections[0].notes[0].width == pytest.approx(20.0)

    def test_normal_note(self) -> None:
        config = ScoreLayoutConfig()
        layout = layout_of(playable(60, 0, 0.0, 3.0), config=config)
        assert layout.sections[0].notes[0].width == pytest.approx(1896.0 / 2)

    def test_long_note(self) -> None:
        layout = layout_of(playable(60, 0, 0.0, 6.0))
        assert layout.sections[0].notes[0].width == pytest.approx(1896.0)

    def test_width_is_never_below_the_minimum(self) -> None:
        config = ScoreLayoutConfig(min_note_width=15.0)
        for duration in (0.0001, 0.01, 0.05, 0.1):
            width = duration_to_width(duration, 6.0, config)
            assert width >= config.min_note_width

    def test_width_grows_with_duration(self) -> None:
        config = ScoreLayoutConfig()
        widths = [duration_to_width(duration, 6.0, config) for duration in (0.5, 1.0, 2.0, 4.0)]
        assert widths == sorted(widths)

    def test_helper_matches_the_fragment(self) -> None:
        config = ScoreLayoutConfig()
        layout = layout_of(playable(60, 0, 0.0, 2.0), config=config)
        fragment = layout.sections[0].notes[0]
        assert fragment.width == pytest.approx(duration_to_width(2.0, 6.0, config))


# --------------------------------------------------------------------------- #
# cross-section fragments
# --------------------------------------------------------------------------- #
class TestCrossSection:
    def test_note_crossing_one_boundary_produces_two_fragments(self) -> None:
        layout = layout_of(playable(60, 0, 5.5, 2.0))  # 5.5 -> 7.5

        assert layout.section_count == 2
        assert layout.fragment_count == 2

        first, second = layout.fragments()

        assert first.section_index == 0
        assert (first.fragment_start, first.fragment_end) == (pytest.approx(5.5), pytest.approx(6.0))
        assert first.fragment_duration == pytest.approx(0.5)
        assert first.is_continuation is False

        assert second.section_index == 1
        assert (second.fragment_start, second.fragment_end) == (pytest.approx(6.0), pytest.approx(7.5))
        assert second.fragment_duration == pytest.approx(1.5)
        assert second.is_continuation is True

    def test_fragments_share_the_source_note(self) -> None:
        layout = layout_of(playable(60, 0, 5.5, 2.0, source_pitch=60))
        fragments = layout.fragments()

        assert {fragment.source_note_id for fragment in fragments} == {0}
        assert {fragment.label for fragment in fragments} == {"Z"}
        assert {fragment.lane for fragment in fragments} == {0}

    def test_fragment_durations_sum_to_the_original(self) -> None:
        layout = layout_of(playable(60, 0, 5.5, 2.0))
        total = sum(fragment.fragment_duration for fragment in layout.fragments())
        assert total == pytest.approx(2.0)

    def test_note_spanning_three_sections(self) -> None:
        # 5.5 -> 13.0 crosses 6.0 and 12.0.
        layout = layout_of(playable(60, 0, 5.5, 7.5))

        assert layout.fragment_count == 3
        fragments = layout.fragments()
        assert [fragment.section_index for fragment in fragments] == [0, 1, 2]
        assert [fragment.fragment_start for fragment in fragments] == [
            pytest.approx(5.5),
            pytest.approx(6.0),
            pytest.approx(12.0),
        ]
        assert [fragment.fragment_duration for fragment in fragments] == [
            pytest.approx(0.5),
            pytest.approx(6.0),
            pytest.approx(1.0),
        ]
        assert sum(f.fragment_duration for f in fragments) == pytest.approx(7.5)
        assert [fragment.is_continuation for fragment in fragments] == [False, True, True]

    def test_only_the_first_fragment_is_not_a_continuation(self) -> None:
        layout = layout_of(playable(60, 0, 5.5, 7.5))
        assert [fragment.is_continuation for fragment in layout.fragments()] == [False, True, True]

    def test_fragments_are_attached_to_their_sections(self) -> None:
        layout = layout_of(playable(60, 0, 5.5, 7.5))

        assert [fragment.section_index for fragment in layout.sections[0].notes] == [0]
        assert [fragment.section_index for fragment in layout.sections[1].notes] == [1]
        assert [fragment.section_index for fragment in layout.sections[2].notes] == [2]

    def test_fragment_ids_are_unique_and_sequential(self) -> None:
        layout = layout_of(playable(60, 0, 5.5, 7.5), playable(64, 2, 1.0, 1.0))
        ids = [fragment.note_id for fragment in layout.fragments()]
        assert ids == list(range(len(ids)))


# --------------------------------------------------------------------------- #
# exact boundaries
# --------------------------------------------------------------------------- #
class TestExactBoundaries:
    def test_note_ending_exactly_at_the_boundary(self) -> None:
        layout = layout_of(playable(60, 0, 0.0, 6.0))

        # The note fills section 0 exactly; no fragment spills into section 1.
        assert layout.section_count == 1
        assert layout.fragment_count == 1
        fragment = layout.sections[0].notes[0]
        assert fragment.fragment_duration == pytest.approx(6.0)
        assert fragment.is_continuation is False

    def test_note_starting_exactly_at_the_boundary(self) -> None:
        layout = layout_of(playable(60, 0, 6.0, 6.0))

        assert layout.section_count == 2
        assert layout.sections[0].notes == ()
        assert layout.sections[1].notes[0].fragment_start == pytest.approx(6.0)
        assert layout.sections[1].notes[0].fragment_duration == pytest.approx(6.0)

    def test_note_zero_to_six(self) -> None:
        layout = layout_of(playable(60, 0, 0.0, 6.0))
        assert [f.fragment_duration for f in layout.fragments()] == [pytest.approx(6.0)]

    def test_note_six_to_twelve(self) -> None:
        layout = layout_of(playable(60, 0, 6.0, 6.0))
        assert layout.sections[0].notes == ()
        assert [f.fragment_duration for f in layout.sections[1].notes] == [pytest.approx(6.0)]

    def test_no_zero_width_fragments_ever(self, profile) -> None:
        notes = [
            playable(60, 0, 0.0, 6.0),
            playable(62, 1, 6.0, 6.0),
            playable(64, 2, 5.5, 7.5),
            playable(67, 3, 0.0, 18.0),
        ]
        layout = layout_of(*notes)
        assert all(fragment.fragment_duration > 0 for fragment in layout.fragments())

    def test_whole_score_back_to_back_notes(self, profile) -> None:
        notes = [playable(60 + lane, lane, lane * 6.0, 6.0) for lane in range(3)]
        layout = layout_of(*notes)

        assert layout.fragment_count == 3
        assert layout.section_count == 3
        for index, section in enumerate(layout.sections):
            assert section.notes[0].fragment_start == pytest.approx(index * 6.0)
            assert section.notes[0].fragment_duration == pytest.approx(6.0)


# --------------------------------------------------------------------------- #
# grid
# --------------------------------------------------------------------------- #
class TestGrid:
    def test_six_second_section_grid(self) -> None:
        grid = build_grid(6.0, ScoreLayoutConfig())

        majors = [line for line in grid if line.kind == "major"]
        minors = [line for line in grid if line.kind == "minor"]

        assert [line.time for line in majors] == [0, 1, 2, 3, 4, 5, 6]
        assert [line.time for line in minors] == [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]

    def test_major_and_minor_never_overlap(self) -> None:
        grid = build_grid(6.0, ScoreLayoutConfig())
        major_times = {round(line.time, 9) for line in grid if line.kind == "major"}
        minor_times = {round(line.time, 9) for line in grid if line.kind == "minor"}
        assert major_times & minor_times == set()

    def test_grid_lines_are_ordered_by_time(self) -> None:
        grid = build_grid(6.0, ScoreLayoutConfig())
        times = [line.time for line in grid]
        assert times == sorted(times)

    def test_grid_x_positions_span_the_usable_width(self) -> None:
        config = ScoreLayoutConfig()
        grid = build_grid(6.0, config)

        assert grid[0].x == pytest.approx(config.left_label_width)
        assert grid[-1].x == pytest.approx(config.canvas_width - config.right_padding)

    def test_grid_is_identical_for_every_section(self, profile) -> None:
        layout = layout_of(playable(60, 0, 0.0, 0.5), playable(62, 1, 6.0, 0.5))
        assert layout.sections[0].grid == layout.sections[1].grid

    def test_grid_is_attached_to_sections(self, profile) -> None:
        layout = layout_of(playable(60, 0, 0.0, 0.5))
        assert isinstance(layout.sections[0].grid[0], GridLine)
        assert all(line.kind in ("major", "minor") for line in layout.sections[0].grid)

    def test_other_section_durations(self) -> None:
        for duration in (4.0, 8.0, 10.0, 12.0):
            grid = build_grid(duration, ScoreLayoutConfig(section_duration=duration))
            majors = [line.time for line in grid if line.kind == "major"]
            assert majors[0] == 0.0
            assert majors[-1] == pytest.approx(duration)
            assert len(majors) == int(duration) + 1

    def test_custom_grid_steps(self) -> None:
        config = ScoreLayoutConfig(section_duration=6.0, major_grid_step=2.0, minor_grid_step=1.0)
        grid = build_grid(6.0, config)
        assert [line.time for line in grid if line.kind == "major"] == [0, 2, 4, 6]
        assert [line.time for line in grid if line.kind == "minor"] == [1, 3, 5]


# --------------------------------------------------------------------------- #
# score level invariants
# --------------------------------------------------------------------------- #
class TestScoreLevel:
    def test_duration_uses_the_note_end_not_the_last_start(self) -> None:
        layout = layout_of(playable(60, 0, 0.0, 5.0))
        assert layout.duration == pytest.approx(5.0)

    def test_duration_keeps_the_score_duration_when_larger(self) -> None:
        layout = layout_of(
            playable(60, 0, 0.0, 1.0),
            header=LayoutHeader(duration=30.0),
        )
        assert layout.duration == pytest.approx(30.0)

    def test_duration_grows_with_the_longest_note(self) -> None:
        layout = layout_of(playable(60, 0, 10.0, 5.0), header=LayoutHeader(duration=8.0))
        assert layout.duration == pytest.approx(15.0)

    def test_canvas_dimensions(self) -> None:
        config = ScoreLayoutConfig(canvas_width=1800, section_duration=6.0)
        layout = layout_of(playable(60, 0, 0.0, 1.0), playable(62, 1, 6.0, 1.0), config=config)

        assert layout.width == 1800.0
        assert layout.height == pytest.approx(config.total_height(2))
        assert layout.height == pytest.approx(16.0 + 120.0 + 2 * 220.0 + 16.0)

    def test_lanes_come_from_the_score(self) -> None:
        layout = layout_of(playable(60, 0, 0.0, 1.0))
        assert layout.lanes == LANES

    def test_header_metadata_is_carried_through(self) -> None:
        header = LayoutHeader(
            title="My Song",
            source="song.mp3",
            profile="delta_harmonica",
            duration=12.0,
            transpose=-7,
            auto_transpose=True,
            range_mode="octave_fold",
            note_count=7,
            playable_ratio=1.0,
        )
        layout = layout_of(playable(65, 3, 0.0, 0.5), header=header)

        assert layout.header is header
        assert layout.header.title == "My Song"
        assert layout.header.transpose == -7

    def test_labels_are_formatted_by_the_layout(self) -> None:
        layout = layout_of(
            playable(60, 0, 0.0, 0.5, key_label="Z"),
            playable(66, 3, 1.0, 0.5, key_label="V", semitone=True),
            playable(84, 7, 2.0, 0.5, key_label=",", octave_offset=1),
        )

        labels = {fragment.label for fragment in layout.fragments()}
        assert labels == {"Z", "V#", ",↑"}

    def test_confidence_flows_into_the_fragments(self) -> None:
        layout = layout_of(playable(60, 0, 0.0, 0.5, confidence=0.25))
        assert layout.fragments()[0].confidence == 0.25

    def test_deterministic(self, profile) -> None:
        notes = [
            playable(60, 0, 0.0, 1.0),
            playable(66, 3, 5.5, 2.0),
            playable(72, 7, 6.0, 6.0),
            playable(84, 7, 12.0, 0.5),
        ]
        first = layout_of(*notes)
        second = layout_of(*list(reversed(notes)))

        assert first.sections == second.sections
        assert first.fragments() == second.fragments()
        assert first.width == second.width
        assert first.height == second.height

    def test_input_notes_are_never_modified(self, profile) -> None:
        notes = [playable(60, 0, 0.0, 1.0), playable(66, 3, 5.5, 2.0)]
        snapshot = list(notes)
        layout_of(*notes)
        assert notes == snapshot

    def test_fragments_inside_each_section_are_ordered_by_time(self, profile) -> None:
        notes = [playable(60 + lane, lane, start, 1.0) for lane, start in enumerate((0.0, 2.0, 4.0))]
        layout = layout_of(*notes)
        starts = [fragment.fragment_start for fragment in layout.sections[0].notes]
        assert starts == sorted(starts)


# --------------------------------------------------------------------------- #
# errors & config
# --------------------------------------------------------------------------- #
class TestErrors:
    def test_empty_score_is_rejected(self) -> None:
        with pytest.raises(EmptyScoreError, match="empty score"):
            build_layout([], lanes=LANES)

    def test_no_lanes_is_rejected(self) -> None:
        with pytest.raises(ScoreLayoutError, match="at least one lane"):
            build_layout([playable(60, 0, 0.0, 0.5)], lanes=())

    def test_lanes_too_tall_for_the_section(self) -> None:
        config = ScoreLayoutConfig(section_height=100.0)  # 8 lanes x 24 px = 192
        with pytest.raises(ScoreLayoutError, match="too small"):
            build_layout([playable(60, 0, 0.0, 0.5)], lanes=LANES, config=config)

    def test_invalid_config_is_rejected(self) -> None:
        with pytest.raises(ScoreLayoutError, match="section_duration"):
            ScoreLayoutConfig(section_duration=0.0)
        with pytest.raises(ScoreLayoutError, match="canvas_width"):
            ScoreLayoutConfig(canvas_width=0)
        with pytest.raises(ScoreLayoutError, match="note_height"):
            ScoreLayoutConfig(note_height=40.0)  # > lane_height
        with pytest.raises(ScoreLayoutError, match="usable width"):
            ScoreLayoutConfig(canvas_width=100, left_label_width=80, right_padding=24)

    def test_negative_paddings_are_rejected(self) -> None:
        with pytest.raises(ScoreLayoutError, match="top_padding"):
            ScoreLayoutConfig(top_padding=-1.0)

    def test_grid_steps_are_validated(self) -> None:
        with pytest.raises(ScoreLayoutError, match="minor_grid_step"):
            ScoreLayoutConfig(major_grid_step=0.5, minor_grid_step=1.0)


class TestConfigDefaults:
    def test_documented_defaults(self) -> None:
        config = ScoreLayoutConfig()
        assert config.canvas_width == 2000
        assert config.section_duration == DEFAULT_SECTION_DURATION == 6.0
        assert config.major_grid_step == 1.0
        assert config.minor_grid_step == 0.5
        assert config.left_label_width == 80.0
        assert config.right_padding == 24.0

    def test_alternative_section_durations_are_accepted(self) -> None:
        for duration in (4.0, 6.0, 8.0, 10.0, 12.0, 2.5):
            assert ScoreLayoutConfig(section_duration=duration).section_duration == duration

    def test_sections_top(self) -> None:
        config = ScoreLayoutConfig()
        assert config.sections_top() == pytest.approx(config.top_padding + config.header_height)
