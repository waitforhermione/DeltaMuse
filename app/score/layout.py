"""Static score layout: PlayableNote[] -> geometry.

Reading direction of the final score:

    x axis = time        y axis = the instrument's lanes
    one row   = one fixed-length section
    sections  = stacked top to bottom

This module produces *only* geometry and metadata. It never imports Pillow,
matplotlib or any GUI toolkit, and it never draws: the renderer (Milestone 5)
consumes a :class:`LayoutScore` and paints it.

Determinism: the same notes and the same config always produce the same sections,
fragments, geometry and grid.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from app.core.errors import EmptyScoreError, ScoreLayoutError
from app.core.models import PlayableNote
from app.score.labels import DEFAULT_SYMBOLS, NoteLabelSymbols, format_note_label

_MAJOR = "major"
_MINOR = "minor"
#: Relative tolerance used when deciding whether a note starts/ends on a section
#: boundary (transcribed times are floats and never compare exactly).
_EPSILON_RELATIVE = 1e-9

DEFAULT_CANVAS_WIDTH = 2000
DEFAULT_SECTION_DURATION = 6.0


@dataclass(frozen=True)
class ScoreLayoutConfig:
    """Every tunable of the static layout, in one place.

    Geometry only - no fonts, no colours, no Pillow.
    """

    canvas_width: int = DEFAULT_CANVAS_WIDTH
    section_duration: float = DEFAULT_SECTION_DURATION

    header_height: float = 120.0
    section_height: float = 220.0

    left_label_width: float = 80.0
    right_padding: float = 24.0
    top_padding: float = 16.0
    bottom_padding: float = 16.0

    lane_height: float = 24.0
    note_height: float = 18.0
    min_note_width: float = 6.0

    major_grid_step: float = 1.0
    minor_grid_step: float = 0.5

    def __post_init__(self) -> None:
        self._require_positive("canvas_width", float(self.canvas_width))
        self._require_positive("section_duration", self.section_duration)
        self._require_positive("header_height", self.header_height)
        self._require_positive("section_height", self.section_height)
        self._require_positive("lane_height", self.lane_height)
        self._require_positive("note_height", self.note_height)
        self._require_positive("min_note_width", self.min_note_width)
        self._require_positive("major_grid_step", self.major_grid_step)
        self._require_positive("minor_grid_step", self.minor_grid_step)

        for name, value in (
            ("left_label_width", self.left_label_width),
            ("right_padding", self.right_padding),
            ("top_padding", self.top_padding),
            ("bottom_padding", self.bottom_padding),
        ):
            if value < 0:
                raise ScoreLayoutError(f"{name} must be >= 0, got {value}")

        if self.note_height > self.lane_height:
            raise ScoreLayoutError(
                f"note_height ({self.note_height}) must not exceed lane_height ({self.lane_height})"
            )
        if self.minor_grid_step > self.major_grid_step:
            raise ScoreLayoutError(
                f"minor_grid_step ({self.minor_grid_step}) must not exceed "
                f"major_grid_step ({self.major_grid_step})"
            )

        usable = self.usable_width()
        if usable <= 0:
            raise ScoreLayoutError(
                f"canvas_width ({self.canvas_width}) minus left_label_width "
                f"({self.left_label_width}) and right_padding ({self.right_padding}) "
                f"leaves no usable width"
            )

    # -- derived geometry -------------------------------------------------- #
    def usable_width(self) -> float:
        """Horizontal space available to the timeline."""
        return self.canvas_width - self.left_label_width - self.right_padding

    def sections_top(self) -> float:
        return self.top_padding + self.header_height

    def section_y(self, index: int) -> float:
        return self.sections_top() + index * self.section_height

    def lane_area_offset(self, lane_count: int) -> float:
        """Vertical gap that centres the lane block inside a section."""
        return (self.section_height - lane_count * self.lane_height) / 2

    def lane_top(self, lane: int, lane_count: int) -> float:
        """Top of *lane*'s band inside a section."""
        return self.lane_area_offset(lane_count) + lane * self.lane_height

    def note_y_offset(self, lane: int, lane_count: int) -> float:
        """Y of a note inside a section, centred vertically in its lane band."""
        return self.lane_top(lane, lane_count) + (self.lane_height - self.note_height) / 2

    def total_height(self, section_count: int) -> float:
        return self.sections_top() + section_count * self.section_height + self.bottom_padding

    # -- internals ---------------------------------------------------------- #
    @staticmethod
    def _require_positive(name: str, value: float) -> None:
        if not math.isfinite(value) or value <= 0:
            raise ScoreLayoutError(f"{name} must be a finite value > 0, got {value!r}")


@dataclass(frozen=True)
class GridLine:
    """One vertical line of a section's time grid."""

    x: float
    time: float
    kind: str  # "major" | "minor"


@dataclass(frozen=True)
class LayoutNote:
    """One drawable fragment of a playable note.

    A note that crosses a section boundary produces several fragments that share
    ``source_note_id``; their ``fragment_duration`` values add up to the original
    ``duration``.
    """

    note_id: int
    source_note_id: int
    section_index: int

    x: float
    y: float
    width: float
    height: float

    label: str
    lane: int

    start: float
    duration: float
    fragment_start: float
    fragment_duration: float

    is_continuation: bool
    confidence: float | None = None

    @property
    def fragment_end(self) -> float:
        return self.fragment_start + self.fragment_duration


@dataclass(frozen=True)
class LayoutSection:
    """One horizontal row of the score."""

    index: int
    start_time: float
    end_time: float

    y: float
    height: float

    notes: tuple[LayoutNote, ...]
    grid: tuple[GridLine, ...] = ()

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


@dataclass(frozen=True)
class LayoutHeader:
    """Metadata rendered above the score (Milestone 5 decides how it looks)."""

    title: str = ""
    source: str | None = None
    profile: str = ""
    duration: float = 0.0
    transpose: int = 0
    auto_transpose: bool = False
    range_mode: str = ""
    note_count: int = 0
    playable_ratio: float = 0.0
    #: Original ``(start, end)`` in the source media when the score was cropped
    #: with ``--start`` / ``--end``; ``None`` for a full-song score.
    segment: tuple[float, float] | None = None


@dataclass(frozen=True)
class LayoutScore:
    """The complete, renderer-ready layout of a score."""

    width: float
    height: float

    header: LayoutHeader
    duration: float
    lanes: tuple[str, ...]
    sections: tuple[LayoutSection, ...]

    @property
    def section_count(self) -> int:
        return len(self.sections)

    @property
    def fragment_count(self) -> int:
        return sum(len(section.notes) for section in self.sections)

    def fragments(self) -> tuple[LayoutNote, ...]:
        return tuple(fragment for section in self.sections for fragment in section.notes)


# --------------------------------------------------------------------------- #
# geometry helpers (individually testable)
# --------------------------------------------------------------------------- #
def section_index_for_time(time: float, section_duration: float) -> int:
    """Index of the section that owns *time*.

    ``floor(time / section_duration)``, guarded against float noise: a time that
    lands exactly on a boundary belongs to the *following* section (6.0 s with
    6 s sections -> section 1, i.e. 6..12).
    """
    if section_duration <= 0:
        raise ScoreLayoutError(f"section_duration must be > 0, got {section_duration!r}")
    index = math.floor(time / section_duration)
    epsilon = section_duration * _EPSILON_RELATIVE
    start = index * section_duration
    if time < start - epsilon:
        index -= 1
    elif time >= start + section_duration + epsilon:
        index += 1
    return index


def section_index_for_end(end: float, section_duration: float) -> int:
    """Index of the last section a note *occupies*.

    A note that ends exactly on a boundary does not spill into the next section
    (notes are half-open intervals ``[start, end)``).
    """
    index = section_index_for_time(end, section_duration)
    boundary = index * section_duration
    if index > 0 and math.isclose(end, boundary, rel_tol=_EPSILON_RELATIVE, abs_tol=_EPSILON_RELATIVE):
        index -= 1
    return index


def time_to_x(
    time: float,
    section_start: float,
    section_duration: float,
    config: ScoreLayoutConfig,
) -> float:
    """Map a time inside a section to its x position on the canvas."""
    local_time = time - section_start
    return config.left_label_width + local_time / section_duration * config.usable_width()


def duration_to_width(
    duration: float,
    section_duration: float,
    config: ScoreLayoutConfig,
) -> float:
    """Map a duration inside a section to its width, never below ``min_note_width``."""
    width = duration / section_duration * config.usable_width()
    return max(width, config.min_note_width)


def lane_to_y(
    lane: int,
    section_y: float,
    lane_count: int,
    config: ScoreLayoutConfig,
) -> float:
    """Map a lane index to the y of its note inside a section."""
    if not 0 <= lane < lane_count:
        raise ScoreLayoutError(f"lane {lane} is outside 0..{lane_count - 1}")
    return section_y + config.note_y_offset(lane, lane_count)


def build_grid(
    section_duration: float,
    config: ScoreLayoutConfig,
) -> tuple[GridLine, ...]:
    """Time grid of one section: majors every *major_grid_step*, minors in between."""
    major_count = int(round(section_duration / config.major_grid_step))
    minor_count = int(math.floor(section_duration / config.minor_grid_step + 1e-9))

    major_times = [index * config.major_grid_step for index in range(major_count + 1)]
    minor_times = [
        index * config.minor_grid_step
        for index in range(minor_count + 1)
        if index * config.minor_grid_step < section_duration
    ]

    lines: list[GridLine] = []
    for time in major_times:
        lines.append(GridLine(x=time_to_x(time, 0.0, section_duration, config), time=time, kind=_MAJOR))
    for time in minor_times:
        if any(math.isclose(time, major, rel_tol=_EPSILON_RELATIVE, abs_tol=_EPSILON_RELATIVE) for major in major_times):
            continue
        lines.append(GridLine(x=time_to_x(time, 0.0, section_duration, config), time=time, kind=_MINOR))
    lines.sort(key=lambda line: (line.time, 0 if line.kind == _MAJOR else 1))
    return tuple(lines)


def section_time_bounds(index: int, section_duration: float) -> tuple[float, float]:
    """Start and end time of section *index* (the end may exceed the score)."""
    return index * section_duration, (index + 1) * section_duration


# --------------------------------------------------------------------------- #
# layout
# --------------------------------------------------------------------------- #
def _split_fragments(
    note: PlayableNote,
    section_duration: float,
) -> list[tuple[int, float, float]]:
    """Return the ``(section_index, fragment_start, fragment_end)`` pieces of *note*."""
    first = section_index_for_time(note.start, section_duration)
    last = section_index_for_end(note.end, section_duration)

    fragments: list[tuple[int, float, float]] = []
    for index in range(first, last + 1):
        section_start, section_end = section_time_bounds(index, section_duration)
        fragment_start = max(note.start, section_start)
        fragment_end = min(note.end, section_end)
        if fragment_end - fragment_start <= 0:
            continue  # never emit a zero-width fragment
        fragments.append((index, fragment_start, fragment_end))
    return fragments


def build_layout(
    notes: Iterable[PlayableNote],
    *,
    lanes: Sequence[str],
    config: ScoreLayoutConfig | None = None,
    header: LayoutHeader | None = None,
    symbols: NoteLabelSymbols = DEFAULT_SYMBOLS,
) -> LayoutScore:
    """Lay out *notes* into sections, fragments and grid lines.

    Raises
    ------
    EmptyScoreError
        ``notes`` is empty - there is nothing to lay out.
    ScoreLayoutError
        The configuration cannot accommodate the lanes, or a lane index is out
        of range.
    """
    settings = config if config is not None else ScoreLayoutConfig()
    metadata = header if header is not None else LayoutHeader()

    playable = sorted(notes, key=lambda note: (note.start, note.pitch, note.lane))
    if not playable:
        raise EmptyScoreError("cannot lay out an empty score: no playable notes")

    lane_count = len(lanes)
    if lane_count == 0:
        raise ScoreLayoutError("the score must define at least one lane")
    if settings.section_height < lane_count * settings.lane_height:
        raise ScoreLayoutError(
            f"section_height ({settings.section_height}) is too small for "
            f"{lane_count} lanes of {settings.lane_height} px"
        )

    section_duration = settings.section_duration
    last_section = max(
        section_index_for_end(note.start + note.duration, section_duration) for note in playable
    )
    section_count = last_section + 1

    sections: list[LayoutSection] = []
    next_note_id = 0
    buckets: list[list[LayoutNote]] = [[] for _ in range(section_count)]

    for source_id, note in enumerate(playable):
        if note.lane < 0 or note.lane >= lane_count:
            raise ScoreLayoutError(
                f"note {source_id} has lane {note.lane}, but the score defines {lane_count} lane(s)"
            )
        label = format_note_label(note, symbols)
        for fragment_index, (section_index, fragment_start, fragment_end) in enumerate(
            _split_fragments(note, section_duration)
        ):
            fragment_duration = fragment_end - fragment_start
            buckets[section_index].append(
                LayoutNote(
                    note_id=next_note_id,
                    source_note_id=source_id,
                    section_index=section_index,
                    x=time_to_x(
                        fragment_start,
                        section_time_bounds(section_index, section_duration)[0],
                        section_duration,
                        settings,
                    ),
                    y=lane_to_y(
                        note.lane,
                        settings.section_y(section_index),
                        lane_count,
                        settings,
                    ),
                    width=duration_to_width(fragment_duration, section_duration, settings),
                    height=settings.note_height,
                    label=label,
                    lane=note.lane,
                    start=note.start,
                    duration=note.duration,
                    fragment_start=fragment_start,
                    fragment_duration=fragment_duration,
                    is_continuation=fragment_index > 0,
                    confidence=note.confidence,
                )
            )
            next_note_id += 1

    for index in range(section_count):
        section_start, section_end = section_time_bounds(index, section_duration)
        sections.append(
            LayoutSection(
                index=index,
                start_time=section_start,
                end_time=section_end,
                y=settings.section_y(index),
                height=settings.section_height,
                notes=tuple(sorted(buckets[index], key=lambda fragment: fragment.fragment_start)),
                grid=build_grid(section_duration, settings),
            )
        )

    max_note_end = max(note.end for note in playable)
    layout_duration = max(metadata.duration, max_note_end)

    return LayoutScore(
        width=float(settings.canvas_width),
        height=settings.total_height(section_count),
        header=metadata,
        duration=layout_duration,
        lanes=tuple(lanes),
        sections=tuple(sections),
    )


__all__ = [
    "ScoreLayoutConfig",
    "GridLine",
    "LayoutNote",
    "LayoutSection",
    "LayoutHeader",
    "LayoutScore",
    "section_index_for_time",
    "section_index_for_end",
    "time_to_x",
    "duration_to_width",
    "lane_to_y",
    "build_grid",
    "section_time_bounds",
    "build_layout",
]
