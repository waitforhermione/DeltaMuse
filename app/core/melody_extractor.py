"""Polyphonic piano notes -> strictly monophonic melody.

Pure Python, standard library only: no torch, no TransKun, no librosa, no CLI, no
renderer. Input and output are :class:`~app.core.models.NoteEvent` sequences, so
this stage works identically for transcribed audio and for imported MIDI.

Pipeline::

    Sequence[NoteEvent]
      -> filter             (minimum duration, optional pitch range)
      -> onset grouping     (anchor-based, deterministic)
      -> note or SKIP per group (highest | lowest | continuity)
      -> overlap resolution (truncate_previous)
      -> strictly monophonic melody (may contain rests)

``highest`` and ``lowest`` always emit exactly one note per onset group. The
``continuity`` strategy can additionally **skip** an onset group: when the melody
is missing from the transcription, the group may hold nothing but accompaniment,
and forcing a note out of it drags the whole melodic path into the left hand.
A skipped group costs ``skip_cost`` and produces a rest instead of a note.
Nothing is ever synthesised to fill a skip - skipping means "no melody here".

``confidence`` is optional everywhere: TransKun 2.0.1 reports no per-note
confidence, so ``NoteEvent.confidence is None`` for every note, and the cost model
simply contributes zero for that term. Nothing is ever inferred or faked.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from app.core.errors import PianoScoreError
from app.core.models import MAX_MIDI_PITCH, MIN_MIDI_PITCH, NoteEvent
from app.core.polyphony import (
    DEFAULT_ONSET_EPSILON,
    DEFAULT_OVERLAP_FRAGMENT_THRESHOLD,
    OnsetGroup,
    group_by_onset,
    is_monophonic,
    resolve_overlaps,
)

STRATEGIES: tuple[str, ...] = ("highest", "lowest", "continuity")
DEFAULT_STRATEGY = "continuity"
DEFAULT_MIN_NOTE_DURATION = 0.05

#: Breakpoints of the pitch-interval cost: ``(semitones, cost)``, ascending in
#: both axes so the cost is monotone. Values in between are linearly interpolated
#: and the last slope is extrapolated beyond the final breakpoint.
DEFAULT_INTERVAL_COST_POINTS: tuple[tuple[float, float], ...] = (
    (0.0, 0.00),
    (2.0, 0.05),
    (5.0, 0.25),
    (12.0, 1.10),
    (24.0, 3.20),
)


@dataclass(frozen=True)
class MelodyExtractionConfig:
    """Everything tunable about melody extraction.

    The defaults are tuned for solo/arranged piano where the melody usually lives
    in the middle-high register above an arpeggiated left hand.
    """

    strategy: str = DEFAULT_STRATEGY

    # -- filtering (before grouping) ------------------------------------- #
    #: Originally short transcribed notes are dropped. Distinct from
    #: ``overlap_fragment_threshold``, which only concerns truncation fragments.
    min_note_duration: float = DEFAULT_MIN_NOTE_DURATION
    min_pitch: int | None = None
    max_pitch: int | None = None

    # -- grouping --------------------------------------------------------- #
    onset_epsilon: float = DEFAULT_ONSET_EPSILON

    # -- overlap resolution ----------------------------------------------- #
    overlap_fragment_threshold: float = DEFAULT_OVERLAP_FRAGMENT_THRESHOLD

    # -- continuity: register / bass / duration / velocity ----------------- #
    #: Pitches below this are mildly discouraged as melody candidates.
    register_floor: int = 60
    #: Pitches above this are mildly discouraged too (grace notes, ornaments).
    register_ceiling: int = 84
    register_low_weight: float = 0.60
    register_high_weight: float = 0.25
    #: Left-hand suppression: a soft, gradually growing penalty below this pitch.
    #: Never a hard cut - real melodies do reach into the lower register.
    bass_pivot: int = 48
    bass_weight: float = 1.20
    #: Fragments shorter than this pay a penalty; longer notes earn a small bonus.
    short_note_reference: float = 0.15
    short_note_weight: float = 0.30
    long_note_reference: float = 0.40
    long_note_bonus: float = 0.08
    #: Velocity is only a whisper of a hint (0..1 mapping is negated, see cost).
    velocity_weight: float = 0.05
    #: Only used when a backend actually reports confidence; ``None`` costs zero.
    confidence_weight: float = 0.50

    # -- continuity: transitions ------------------------------------------ #
    interval_cost_points: tuple[tuple[float, float], ...] = DEFAULT_INTERVAL_COST_POINTS
    #: Below this gap the interval cost is applied at full strength.
    gap_relax_start: float = 0.60
    #: At/after this gap the interval cost is scaled down to the floor factor.
    gap_relax_full: float = 1.50
    #: Floor of the gap relaxation factor (must stay > 0 so big leaps are legal).
    gap_relax_min_factor: float = 0.15

    # -- continuity: skipping --------------------------------------------- #
    #: Cost of leaving one onset group without a melody note (a rest).
    #:
    #: Must be > 0: a free skip would let the optimiser drop the whole piece.
    #: It is deliberately larger than the emission cost of any single note, so
    #: that a skip is driven by *context* (avoiding a pointless register detour)
    #: rather than by "this note looks slightly expensive on its own". With the
    #: default 2.5 the optimiser prefers skipping over a detour of roughly 16+
    #: semitones in and out, and never skips a plausible neighbouring note.
    skip_cost: float = 2.5
    #: Maximum number of consecutive groups that may be skipped. ``None`` = no cap.
    max_skip_groups: int | None = 8
    #: Maximum onset span of a run of skipped groups, in seconds. ``None`` = no cap.
    #:
    #: Measured over the skipped groups themselves, so a run of 5 groups spaced
    #: 0.4 s apart spans 1.6 s. Only applies when at least one group is actually
    #: skipped, so genuine rests (time ranges with no onsets at all) are never
    #: affected. Like :attr:`max_skip_groups` it is a safety guard that bounds the
    #: search window and stops the melody from staying silent for too long -
    #: accompaniment-only stretches longer than this force a note again.
    max_skip_seconds: float | None = 2.0

    #: Guard against pathological onset groups. ``None`` disables the cap.
    max_candidates_per_group: int | None = 16

    def __post_init__(self) -> None:
        if self.strategy not in STRATEGIES:
            raise ValueError(f"strategy must be one of {STRATEGIES}, got {self.strategy!r}")

        _require_positive("min_note_duration", self.min_note_duration)
        _require_non_negative("onset_epsilon", self.onset_epsilon)
        _require_non_negative("overlap_fragment_threshold", self.overlap_fragment_threshold)

        for name in ("min_pitch", "max_pitch"):
            value = getattr(self, name)
            if value is not None and not MIN_MIDI_PITCH <= value <= MAX_MIDI_PITCH:
                raise ValueError(f"{name} must be within [{MIN_MIDI_PITCH}, {MAX_MIDI_PITCH}], got {value}")
        if self.min_pitch is not None and self.max_pitch is not None and self.min_pitch > self.max_pitch:
            raise ValueError(f"min_pitch ({self.min_pitch}) must not exceed max_pitch ({self.max_pitch})")

        for name in (
            "register_low_weight",
            "register_high_weight",
            "bass_weight",
            "short_note_weight",
            "long_note_bonus",
            "velocity_weight",
            "confidence_weight",
        ):
            _require_non_negative(name, getattr(self, name))

        if not MIN_MIDI_PITCH <= self.register_floor <= MAX_MIDI_PITCH:
            raise ValueError(f"register_floor out of MIDI range: {self.register_floor}")
        if not MIN_MIDI_PITCH <= self.register_ceiling <= MAX_MIDI_PITCH:
            raise ValueError(f"register_ceiling out of MIDI range: {self.register_ceiling}")
        if self.register_floor > self.register_ceiling:
            raise ValueError(
                f"register_floor ({self.register_floor}) must not exceed register_ceiling ({self.register_ceiling})"
            )
        if not MIN_MIDI_PITCH <= self.bass_pivot <= MAX_MIDI_PITCH:
            raise ValueError(f"bass_pivot out of MIDI range: {self.bass_pivot}")

        _require_positive("short_note_reference", self.short_note_reference)
        _require_positive("long_note_reference", self.long_note_reference)

        _require_non_negative("gap_relax_start", self.gap_relax_start)
        if self.gap_relax_full <= self.gap_relax_start:
            raise ValueError(
                f"gap_relax_full ({self.gap_relax_full}) must exceed gap_relax_start ({self.gap_relax_start})"
            )
        if not 0.0 <= self.gap_relax_min_factor <= 1.0:
            raise ValueError(f"gap_relax_min_factor must be within [0, 1], got {self.gap_relax_min_factor}")

        _validate_interval_cost_points(self.interval_cost_points)

        # A free skip would let the optimiser drop everything, so it must cost.
        _require_positive("skip_cost", self.skip_cost)
        if self.max_skip_groups is not None and self.max_skip_groups < 1:
            raise ValueError(f"max_skip_groups must be >= 1 or None, got {self.max_skip_groups}")
        if self.max_skip_seconds is not None:
            _require_positive("max_skip_seconds", self.max_skip_seconds)

        if self.max_candidates_per_group is not None and self.max_candidates_per_group < 1:
            raise ValueError(
                f"max_candidates_per_group must be >= 1 or None, got {self.max_candidates_per_group}"
            )

    @classmethod
    def for_strategy(cls, strategy: str) -> "MelodyExtractionConfig":
        """Default configuration for one strategy."""
        return cls(strategy=strategy)


def _require_positive(name: str, value: float) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a finite value > 0, got {value!r}")


def _require_non_negative(name: str, value: float) -> None:
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be a finite value >= 0, got {value!r}")


def _validate_interval_cost_points(points: Sequence[tuple[float, float]]) -> None:
    if len(points) < 2:
        raise ValueError("interval_cost_points needs at least two breakpoints")
    previous_x = -math.inf
    previous_y = -math.inf
    for x, y in points:
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError(f"interval_cost_points must be finite, got {(x, y)!r}")
        if x <= previous_x:
            raise ValueError("interval_cost_points must be strictly increasing in semitones")
        if y < previous_y:
            raise ValueError("interval_cost_points costs must be non-decreasing")
        if y < 0.0:
            raise ValueError("interval_cost_points costs must be >= 0")
        previous_x, previous_y = x, y


# --------------------------------------------------------------------------- #
# cost model (pure functions, individually testable)
# --------------------------------------------------------------------------- #
def interval_cost(semitones: float, config: MelodyExtractionConfig) -> float:
    """Cost of moving ``semitones`` between two consecutive melody notes.

    Piecewise-linear between the configured breakpoints, then extrapolated with
    the last slope. Monotone non-decreasing and always finite, so large leaps are
    *expensive but never forbidden*.
    """
    distance = abs(float(semitones))
    points = config.interval_cost_points
    if distance <= points[0][0]:
        return points[0][1]
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if distance <= x1:
            return y0 + (y1 - y0) * (distance - x0) / (x1 - x0)
    (x0, y0), (x1, y1) = points[-2], points[-1]
    slope = (y1 - y0) / (x1 - x0)
    return y1 + slope * (distance - x1)


def gap_relaxation_factor(gap: float, config: MelodyExtractionConfig) -> float:
    """How much of the interval cost survives across a *gap* of silence.

    ``gap`` is the silence between the end of a candidate and the onset of the
    next one (negative means the two overlap). Short gaps apply the interval cost
    at full strength; after a long rest the cost fades towards
    ``gap_relax_min_factor``, which is what lets a new phrase start with a leap
    instead of being glued to the previous phrase's last pitch.
    """
    if gap <= config.gap_relax_start:
        return 1.0
    if gap >= config.gap_relax_full:
        return config.gap_relax_min_factor
    ratio = (gap - config.gap_relax_start) / (config.gap_relax_full - config.gap_relax_start)
    return 1.0 - ratio * (1.0 - config.gap_relax_min_factor)


def transition_cost(previous: NoteEvent, current: NoteEvent, config: MelodyExtractionConfig) -> float:
    """Cost of placing *current* right after *previous* in the melody."""
    gap = current.start - previous.end
    return gap_relaxation_factor(gap, config) * interval_cost(current.pitch - previous.pitch, config)


def emission_cost(note: NoteEvent, config: MelodyExtractionConfig) -> float:
    """How plausible a single note is as *the* melody note, ignoring context.

    All terms are soft: nothing here can veto a note outright.
    """
    cost = 0.0

    # Register preference: the melody usually sits in the middle-high register.
    if note.pitch < config.register_floor:
        cost += config.register_low_weight * (config.register_floor - note.pitch) / 12.0
    elif note.pitch > config.register_ceiling:
        cost += config.register_high_weight * (note.pitch - config.register_ceiling) / 12.0

    # Left-hand (bass) suppression: grows gradually, never a hard cut.
    if note.pitch < config.bass_pivot:
        cost += config.bass_weight * (config.bass_pivot - note.pitch) / 12.0

    # Very short fragments are discouraged; healthy note lengths get a tiny bonus.
    if note.duration < config.short_note_reference:
        cost += config.short_note_weight * (
            (config.short_note_reference - note.duration) / config.short_note_reference
        )
    else:
        sustained = min(1.0, (note.duration - config.short_note_reference) / config.long_note_reference)
        cost -= config.long_note_bonus * sustained

    # Velocity is a very weak hint - louder is *slightly* more likely to be melody.
    cost -= config.velocity_weight * (note.velocity / 127.0 * 2.0 - 1.0)

    # Confidence only exists for backends that report it; None contributes zero.
    if note.confidence is not None:
        cost += config.confidence_weight * (1.0 - note.confidence)

    return cost


# --------------------------------------------------------------------------- #
# strategies
# --------------------------------------------------------------------------- #
def _highest_note(notes: Sequence[NoteEvent]) -> NoteEvent:
    """Highest pitch, ties broken by longer duration, then higher velocity."""
    return min(notes, key=lambda note: (-note.pitch, -note.duration, -note.velocity, note.start, note.end))


def _lowest_note(notes: Sequence[NoteEvent]) -> NoteEvent:
    """Lowest pitch, ties broken by longer duration, then higher velocity."""
    return min(notes, key=lambda note: (note.pitch, -note.duration, -note.velocity, note.start, note.end))


def _candidate_order(note: NoteEvent) -> tuple[float, int, int, float, float]:
    """Candidate order inside a continuity onset group.

    Also acts as the deterministic tie-break of the path search: longer notes are
    preferred, then louder ones, then the lower pitch.
    """
    return (-note.duration, -note.velocity, note.pitch, note.start, note.end)


def _skip_run_allowed(
    anchors: Sequence[float],
    from_index: int,
    to_index: int,
    config: MelodyExtractionConfig,
) -> bool:
    """May the groups strictly between two selected groups be skipped?

    ``from_index`` is the last selected group (``-1`` for the virtual start) and
    ``to_index`` the next selected group (``len(groups)`` for the virtual end).

    ``max_skip_seconds`` is the onset span *of the skipped groups themselves*
    (first skipped onset to last skipped onset). Measuring it this way keeps the
    rule symmetric - skipping at the beginning, in the middle or at the end of the
    piece behaves identically - and means a genuine rest (a time range with no
    onsets at all) has nothing to skip and is therefore never affected.
    """
    skipped = to_index - from_index - 1
    if skipped <= 0:
        return True
    if config.max_skip_groups is not None and skipped > config.max_skip_groups:
        return False
    if config.max_skip_seconds is not None:
        first_skipped = from_index + 1
        last_skipped = to_index - 1
        if anchors[last_skipped] - anchors[first_skipped] > config.max_skip_seconds:
            return False
    return True


def _candidate_lists(groups: Sequence[OnsetGroup], config: MelodyExtractionConfig) -> list[list[NoteEvent]]:
    lists: list[list[NoteEvent]] = []
    for group in groups:
        ordered = sorted(group.notes, key=_candidate_order)
        if config.max_candidates_per_group is not None:
            ordered = ordered[: config.max_candidates_per_group]
        lists.append(ordered)
    return lists


def _continuity_path(
    groups: Sequence[OnsetGroup],
    config: MelodyExtractionConfig,
) -> tuple[list[NoteEvent], list[int]]:
    """Minimum-cost path over onset groups, where a group may also be skipped.

    The whole sequence is optimised at once instead of picking greedily at each
    step: a locally cheaper note is given up when it would drag the rest of the
    melody off course.

    Formulation
    -----------
    Each candidate note is a node ``(group_index, candidate_slot)`` in a DAG; two
    extra virtual nodes stand for "before the first onset" and "after the last
    one". An edge ``(i0, j0) -> (i, j)`` means *note j of group i is selected
    directly after note j0 of group i0*, and every group in between is skipped::

        edge cost = (i - i0 - 1) * skip_cost + transition_cost(note_i0j0, note_ij)

    Because the edge belongs to a pair of *real* notes, the interval continuity
    and the gap relaxation are computed against the last actually selected melody
    note, not against a skipped placeholder - a skip therefore cannot fake
    continuity. The remaining groups of the skipped run simply add ``skip_cost``
    each, so a long run is not free, and edges are bounded by ``max_skip_groups``
    and ``max_skip_seconds`` (the latter also bounds the search window, since
    spans grow monotonically with distance).

    Skipping is therefore *not* a way to generate notes: a skipped group yields a
    rest. Nothing is interpolated or invented.

    Complexity is ``O(groups x window x candidates^2)`` with a small constant
    window, so it stays practical for real piano pieces.

    Ties are resolved deterministically: edges are examined from the nearest
    previous group outwards (so a tie prefers the least skipping) and ``<`` keeps
    the first minimum.
    """
    candidates = _candidate_lists(groups, config)
    anchors = [group.anchor for group in groups]
    count = len(candidates)
    start_node = (-1, -1)

    # best[i][j]  = cheapest cost of a path ending with candidate j of group i selected
    # back[i][j]  = (previous group, previous candidate) or the virtual start
    best: list[list[float]] = []
    back: list[list[tuple[int, int]]] = []

    for index in range(count):
        group_best: list[float] = []
        group_back: list[tuple[int, int]] = []
        for note in candidates[index]:
            cheapest = math.inf
            source = start_node

            for previous_index in range(index - 1, -1, -1):
                if not _skip_run_allowed(anchors, previous_index, index, config):
                    # Going further back only skips more and spans longer, so it
                    # cannot become legal again.
                    break
                skipped = index - previous_index - 1
                skip_penalty = skipped * config.skip_cost
                for slot, previous_note in enumerate(candidates[previous_index]):
                    total = (
                        best[previous_index][slot]
                        + skip_penalty
                        + transition_cost(previous_note, note, config)
                    )
                    if total < cheapest:
                        cheapest = total
                        source = (previous_index, slot)

            # Alternative: this is the first selected note of the melody, with
            # every earlier group skipped.
            if _skip_run_allowed(anchors, -1, index, config):
                fresh_start = index * config.skip_cost
                if fresh_start < cheapest:
                    cheapest = fresh_start
                    source = start_node

            group_best.append(emission_cost(note, config) + cheapest)
            group_back.append(source)
        best.append(group_best)
        back.append(group_back)

    # The melody may also end before the last onset group (trailing skip).
    end_cost = math.inf
    end_node = (count - 1, 0)
    for index in range(count - 1, -1, -1):
        if not _skip_run_allowed(anchors, index, count, config):
            break
        trailing = (count - 1 - index) * config.skip_cost
        for slot, cost in enumerate(best[index]):
            if cost + trailing < end_cost:
                end_cost = cost + trailing
                end_node = (index, slot)

    # Backtracking can only ever stop at the virtual start, and at least one note
    # is always selected, so the path is non-empty.
    notes: list[NoteEvent] = []
    selected_groups: list[int] = []
    index, slot = end_node
    while index >= 0:
        notes.append(candidates[index][slot])
        selected_groups.append(index)
        index, slot = back[index][slot]
    notes.reverse()
    selected_groups.reverse()
    return notes, selected_groups


def _select_notes(
    groups: Sequence[OnsetGroup],
    config: MelodyExtractionConfig,
) -> tuple[list[NoteEvent], list[int]]:
    """Return the selected notes and the indices of the groups they came from."""
    if config.strategy == "highest":
        return [_highest_note(group.notes) for group in groups], list(range(len(groups)))
    if config.strategy == "lowest":
        return [_lowest_note(group.notes) for group in groups], list(range(len(groups)))
    return _continuity_path(groups, config)


# --------------------------------------------------------------------------- #
# result & extractor
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MelodyExtractionResult:
    """The melody plus a full account of what happened on the way."""

    notes: tuple[NoteEvent, ...]
    strategy: str

    input_note_count: int
    filtered_note_count: int
    onset_group_count: int
    selected_group_indices: tuple[int, ...]
    output_note_count: int

    removed_short_count: int
    removed_pitch_range_count: int
    overlap_truncated_count: int
    dropped_fragment_count: int

    config: MelodyExtractionConfig = field(
        default_factory=MelodyExtractionConfig, repr=False, compare=False
    )

    @property
    def removed_note_count(self) -> int:
        """Notes dropped by the pre-grouping filters."""
        return self.input_note_count - self.filtered_note_count

    @property
    def notes_merged_by_grouping(self) -> int:
        """Notes that shared an onset with another note and were not selected."""
        return self.filtered_note_count - self.onset_group_count

    @property
    def total_group_count(self) -> int:
        """Alias of :attr:`onset_group_count` (every group has to be decided)."""
        return self.onset_group_count

    @property
    def selected_group_count(self) -> int:
        """Onset groups that contributed a melody note."""
        return len(self.selected_group_indices)

    @property
    def skipped_group_count(self) -> int:
        """Groups that produced a rest instead of a melody note.

        Always ``0`` for the ``highest`` and ``lowest`` strategies - only
        ``continuity`` can skip.
        """
        return self.onset_group_count - self.selected_group_count

    @property
    def has_rests(self) -> bool:
        """``True`` when the melody contains a real rest (a skipped group)."""
        return self.skipped_group_count > 0

    @property
    def longest_skip_run(self) -> int:
        """Length of the longest run of consecutive skipped groups."""
        longest = 0
        previous = -1
        for index in self.selected_group_indices:
            longest = max(longest, index - previous - 1)
            previous = index
        return max(longest, self.onset_group_count - previous - 1)


class MelodyExtractor:
    """Turns a polyphonic :class:`NoteEvent` sequence into a monophonic melody."""

    def __init__(self, config: MelodyExtractionConfig | None = None) -> None:
        self.config = config if config is not None else MelodyExtractionConfig()

    def extract(self, notes: Iterable[NoteEvent]) -> MelodyExtractionResult:
        """Extract the melody; the input events are never modified."""
        config = self.config
        input_notes = [self._validate(note) for note in notes]

        # 1) filter originally short notes, then apply the optional pitch window.
        after_duration = [note for note in input_notes if note.duration >= config.min_note_duration]
        removed_short = len(input_notes) - len(after_duration)

        filtered = [
            note
            for note in after_duration
            if (config.min_pitch is None or note.pitch >= config.min_pitch)
            and (config.max_pitch is None or note.pitch <= config.max_pitch)
        ]
        removed_pitch_range = len(after_duration) - len(filtered)

        # 2) group onsets, 3) select one note per group (continuity may skip).
        groups = group_by_onset(filtered, config.onset_epsilon)
        selected, selected_groups = _select_notes(groups, config) if groups else ([], [])

        # 4) overlap resolution creates only fragments - handled with its own threshold.
        resolution = resolve_overlaps(selected, config.overlap_fragment_threshold)

        return MelodyExtractionResult(
            notes=resolution.notes,
            strategy=config.strategy,
            input_note_count=len(input_notes),
            filtered_note_count=len(filtered),
            onset_group_count=len(groups),
            selected_group_indices=tuple(selected_groups),
            output_note_count=len(resolution.notes),
            removed_short_count=removed_short,
            removed_pitch_range_count=removed_pitch_range,
            overlap_truncated_count=resolution.truncated_count,
            dropped_fragment_count=resolution.dropped_fragment_count,
            config=config,
        )

    @staticmethod
    def _validate(note: NoteEvent) -> NoteEvent:
        if not isinstance(note, NoteEvent):
            raise TypeError(f"expected NoteEvent instances, got {type(note).__name__}")
        return note


def extract_melody(
    notes: Iterable[NoteEvent],
    config: MelodyExtractionConfig | None = None,
) -> MelodyExtractionResult:
    """Convenience wrapper around :class:`MelodyExtractor`."""
    return MelodyExtractor(config).extract(notes)


def extract_melody_notes(
    notes: Iterable[NoteEvent],
    config: MelodyExtractionConfig | None = None,
) -> list[NoteEvent]:
    """Same as :func:`extract_melody` but returns only the monophonic notes."""
    return list(extract_melody(notes, config).notes)


def assert_monophonic(notes: Iterable[NoteEvent]) -> None:
    """Raise :class:`PianoScoreError` if *notes* is not strictly monophonic.

    Used as a cheap self-check before the melody is written out, so a regression in
    the overlap logic can never silently produce an unplayable score.
    """
    note_list = list(notes)
    if not is_monophonic(note_list):
        raise PianoScoreError("internal error: melody extraction produced overlapping notes")


__all__ = [
    "STRATEGIES",
    "DEFAULT_STRATEGY",
    "DEFAULT_MIN_NOTE_DURATION",
    "DEFAULT_INTERVAL_COST_POINTS",
    "MelodyExtractionConfig",
    "MelodyExtractionResult",
    "MelodyExtractor",
    "extract_melody",
    "extract_melody_notes",
    "assert_monophonic",
    "interval_cost",
    "gap_relaxation_factor",
    "transition_cost",
    "emission_cost",
]
