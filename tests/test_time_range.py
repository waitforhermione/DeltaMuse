"""Time-range parsing, normalisation and cropping tests."""

from __future__ import annotations

from dataclasses import replace

import pytest

from app.core.models import NoteEvent
from app.core.time_range import (
    InvalidTimeRangeError,
    TimeRange,
    crop_note_events,
    normalize_time_range,
    parse_time_value,
)


class TestParseTimeValue:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (None, None),
            ("", None),
            ("  ", None),
            ("48", 48.0),
            ("82.5", 82.5),
            ("0", 0.0),
            ("00:48", 48.0),
            ("01:22", 82.0),
            ("01:22.5", 82.5),
            ("00:01:22", 82.0),
            ("01:02:03.5", 3723.5),
            (" 00:48 ", 48.0),
        ],
    )
    def test_supported_formats(self, value: str | None, expected: float | None) -> None:
        assert parse_time_value(value) == expected

    @pytest.mark.parametrize(
        "value",
        ["abc", "xyz", "1:2:3:4", "1:-2", "1:a", ":48", "48:", "75:00", "1:75:00"],
    )
    def test_invalid_values_raise(self, value: str) -> None:
        with pytest.raises(InvalidTimeRangeError, match="invalid time"):
            parse_time_value(value)

    def test_negative_values_parse_but_are_rejected_by_normalization(self) -> None:
        assert parse_time_value("-1") == -1.0
        with pytest.raises(InvalidTimeRangeError, match="start must be >= 0"):
            normalize_time_range(-1.0, 10.0, 100.0)


class TestNormalizeTimeRange:
    def test_defaults_cover_the_whole_file(self) -> None:
        assert normalize_time_range(None, None, 103.2) == TimeRange(0.0, 103.2)

    def test_only_start_runs_to_the_end(self) -> None:
        assert normalize_time_range(48.0, None, 103.2) == TimeRange(48.0, 103.2)

    def test_only_end_starts_at_zero(self) -> None:
        assert normalize_time_range(None, 30.0, 103.2) == TimeRange(0.0, 30.0)

    def test_end_beyond_the_media_is_clamped(self) -> None:
        rng = normalize_time_range(100.0, 500.0, 103.2)
        assert rng == TimeRange(100.0, 103.2, end_clamped=True)

    def test_unknown_duration_requires_an_explicit_end(self) -> None:
        with pytest.raises(InvalidTimeRangeError, match="duration"):
            normalize_time_range(10.0, None, None)

    def test_unknown_duration_with_both_values_is_fine(self) -> None:
        assert normalize_time_range(10.0, 20.0, None) == TimeRange(10.0, 20.0)

    @pytest.mark.parametrize(
        ("start", "end", "total"),
        [(-1.0, 10.0, 100.0), (10.0, 10.0, 100.0), (20.0, 10.0, 100.0), (150.0, 200.0, 100.0), (100.0, None, 100.0)],
    )
    def test_invalid_ranges_raise(self, start: float, end: float | None, total: float) -> None:
        with pytest.raises(InvalidTimeRangeError):
            normalize_time_range(start, end, total)


def note(pitch: int, start: float, duration: float) -> NoteEvent:
    return NoteEvent(pitch=pitch, start=start, duration=duration, velocity=80, confidence=0.5)


class TestCropNoteEvents:
    def test_notes_inside_are_kept_and_rebased(self) -> None:
        notes = [note(60, 50.0, 1.0), note(64, 52.0, 1.0)]
        cropped = crop_note_events(notes, 48.0, 60.0)

        assert [(n.pitch, n.start, n.duration) for n in cropped] == [(60, 2.0, 1.0), (64, 4.0, 1.0)]

    def test_notes_outside_are_removed(self) -> None:
        notes = [
            note(60, 0.0, 2.0),  # ends before the boundary -> removed
            note(62, 10.0, 2.0),  # starts at the boundary -> removed
            note(64, 7.0, 1.0),  # inside -> kept
        ]
        cropped = crop_note_events(notes, 5.0, 10.0)

        assert [n.pitch for n in cropped] == [64]

    def test_left_boundary_truncation(self) -> None:
        cropped = crop_note_events([note(60, 46.0, 4.0)], 48.0, 60.0)
        assert len(cropped) == 1
        assert cropped[0].start == pytest.approx(0.0)
        assert cropped[0].duration == pytest.approx(2.0)

    def test_right_boundary_truncation(self) -> None:
        cropped = crop_note_events([note(60, 57.0, 4.0)], 48.0, 60.0)
        assert cropped[0].start == pytest.approx(9.0)
        assert cropped[0].duration == pytest.approx(3.0)

    def test_note_spanning_the_whole_range(self) -> None:
        cropped = crop_note_events([note(60, 40.0, 50.0)], 48.0, 82.0)
        assert cropped[0].start == pytest.approx(0.0)
        assert cropped[0].duration == pytest.approx(34.0)

    def test_other_fields_are_preserved(self) -> None:
        cropped = crop_note_events([note(60, 49.0, 1.0)], 48.0, 60.0)
        assert cropped[0].pitch == 60
        assert cropped[0].velocity == 80
        assert cropped[0].confidence == 0.5

    def test_input_is_not_mutated(self) -> None:
        original = note(60, 46.0, 4.0)
        crop_note_events([original], 48.0, 60.0)
        assert original == note(60, 46.0, 4.0)

    def test_output_is_sorted_by_start(self) -> None:
        notes = [note(67, 55.0, 1.0), note(60, 50.0, 1.0)]
        cropped = crop_note_events(notes, 48.0, 60.0)
        assert [n.pitch for n in cropped] == [60, 67]

    def test_no_zero_or_negative_durations(self) -> None:
        notes = [note(60, 48.0, 1e-12), note(62, 47.0, 1.0)]
        for cropped_note in crop_note_events(notes, 48.0, 60.0):
            assert cropped_note.duration > 0.0

    def test_cropping_is_deterministic(self) -> None:
        notes = [note(60, 46.0, 4.0), note(64, 55.0, 1.0)]
        first = crop_note_events(notes, 48.0, 60.0)
        second = crop_note_events(list(reversed(notes)), 48.0, 60.0)
        assert first == second

    def test_invalid_range_is_rejected(self) -> None:
        with pytest.raises(InvalidTimeRangeError):
            crop_note_events([note(60, 0.0, 1.0)], 10.0, 10.0)

    def test_replace_still_works_with_frozen_notes(self) -> None:
        # crop uses dataclasses.replace internally; sanity-check it on a note
        # with extra fields as well.
        enriched = replace(note(60, 49.0, 1.0))
        cropped = crop_note_events([enriched], 48.0, 60.0)
        assert cropped[0].start == pytest.approx(1.0)
