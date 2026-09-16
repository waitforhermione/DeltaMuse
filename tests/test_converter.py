"""Converter: melody -> transpose -> range fit -> map -> PlayableNote[]."""

from __future__ import annotations

import pytest

from app.core.converter import (
    AUTO_TRANSPOSE,
    ConversionConfig,
    ConversionResult,
    convert_melody,
)
from app.core.errors import UnmappedPitchError
from app.core.models import NoteEvent
from app.core.polyphony import is_monophonic

from conftest import make_note, make_profile


def melody(*pitches: int, start: float = 0.0, duration: float = 0.5) -> list[NoteEvent]:
    return [
        make_note(pitch, start=start + index * 0.5, duration=duration)
        for index, pitch in enumerate(pitches)
    ]


class TestConfig:
    def test_defaults(self) -> None:
        config = ConversionConfig()
        assert config.transpose == AUTO_TRANSPOSE
        assert config.range_mode == "octave_fold"
        assert config.auto is True

    def test_manual_transpose(self) -> None:
        config = ConversionConfig(transpose=-2)
        assert config.auto is False

    @pytest.mark.parametrize("mode", ["octave_fold", "nearest_note", "drop"])
    def test_every_range_mode_is_accepted(self, mode: str) -> None:
        assert ConversionConfig(range_mode=mode).range_mode == mode

    def test_unknown_range_mode_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="range_mode"):
            ConversionConfig(range_mode="squash")

    def test_unknown_transpose_mode_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="transpose"):
            ConversionConfig(transpose="best")

    @pytest.mark.parametrize("transpose", [-13, 13])
    def test_manual_transpose_out_of_range_is_rejected(self, transpose: int) -> None:
        with pytest.raises(ValueError, match="-12..12"):
            ConversionConfig(transpose=transpose)

    def test_boolean_transpose_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="transpose"):
            ConversionConfig(transpose=True)  # type: ignore[arg-type]


class TestPipeline:
    def test_end_to_end_without_any_adjustment(self, profile) -> None:
        result = convert_melody(melody(60, 64, 67), ConversionConfig(transpose=0), profile)

        assert [note.pitch for note in result.notes] == [60, 64, 67]
        assert [note.key_label for note in result.notes] == ["Z", "C", "B"]
        assert result.report.transpose == 0
        assert result.report.playable_ratio == 1.0

    def test_order_is_transpose_then_fit_then_map(self, profile) -> None:
        """87 is only playable after transposing; mapping must see the fitted pitch."""
        result = convert_melody(melody(84), ConversionConfig(transpose=3), profile)

        assert [note.pitch for note in result.notes] == [75]
        assert result.report.octave_fold_count == 1
        assert result.report.transpose == 3

    def test_provenance_survives_transpose_fold_and_mapping(self, profile) -> None:
        """source_pitch = 84, transpose = +3, folded pitch = 75 (X# one octave up)."""
        result = convert_melody(melody(84), ConversionConfig(transpose=3), profile)

        note = result.notes[0]
        assert note.source_pitch == 84
        assert note.pitch == 75
        assert note.lane == 1
        assert note.key_label == "X"
        assert note.semitone is True
        assert note.octave_offset == 1

    def test_provenance_survives_octave_folding_in_both_directions(self, profile) -> None:
        up = convert_melody(melody(40), ConversionConfig(transpose=0), profile)
        down = convert_melody(melody(90), ConversionConfig(transpose=0), profile)

        assert up.notes[0].source_pitch == 40 and up.notes[0].pitch == 52
        assert down.notes[0].source_pitch == 90 and down.notes[0].pitch == 78

    def test_provenance_survives_nearest_note(self, profile) -> None:
        result = convert_melody(
            melody(46), ConversionConfig(transpose=0, range_mode="nearest_note"), profile
        )

        assert result.notes[0].source_pitch == 46
        assert result.notes[0].pitch == 48
        assert result.report.nearest_adjusted_count == 1

    def test_output_is_strictly_monophonic_and_sorted(self, profile) -> None:
        result = convert_melody(melody(60, 63, 66, 69, 72), ConversionConfig(transpose=0), profile)

        notes = list(result.notes)
        assert is_monophonic(notes)
        assert [note.start for note in notes] == sorted(note.start for note in notes)
        assert all(note.duration > 0 for note in notes)

    def test_result_is_independent_of_the_input_order(self, profile) -> None:
        notes = melody(60, 63, 66, 69)
        first = convert_melody(notes, ConversionConfig(transpose=0), profile)
        second = convert_melody(list(reversed(notes)), ConversionConfig(transpose=0), profile)

        assert first.notes == second.notes
        assert first.report == second.report

    def test_duration_is_the_input_melody_length(self, profile) -> None:
        result = convert_melody(melody(60, 64, duration=2.0), ConversionConfig(transpose=0), profile)
        assert result.duration == pytest.approx(2.5)

    def test_duration_survives_dropping(self, profile) -> None:
        result = convert_melody(
            melody(46, 60, duration=1.0), ConversionConfig(transpose=0, range_mode="drop"), profile
        )

        assert result.report.dropped_count == 1
        assert result.duration == pytest.approx(1.5)  # the melody's own length


class TestReport:
    def test_report_fields_for_a_clean_conversion(self, profile) -> None:
        result = convert_melody(melody(60, 64, 67), ConversionConfig(transpose=0), profile)
        report = result.report

        assert report.input_note_count == 3
        assert report.output_note_count == 3
        assert report.transpose == 0
        assert report.auto_transpose is False
        assert report.range_mode == "octave_fold"
        assert (report.original_pitch_min, report.original_pitch_max) == (60, 67)
        assert (report.final_pitch_min, report.final_pitch_max) == (60, 67)
        assert report.dropped_count == 0
        assert report.octave_fold_count == 0
        assert report.nearest_adjusted_count == 0
        assert report.semitone_count == 0
        assert report.octave_shift_count == 0
        assert report.modifier_count == 0
        assert report.playable_ratio == 1.0

    def test_auto_transpose_is_reported_as_such(self, profile) -> None:
        result = convert_melody(melody(72, 79, 76, 74), ConversionConfig(), profile)

        assert result.report.auto_transpose is True
        assert result.report.transpose == result.report.transpose  # stable field

    def test_manual_transpose_is_reported(self, profile) -> None:
        result = convert_melody(melody(72, 79, 76, 74), ConversionConfig(transpose=-7), profile)

        assert result.report.auto_transpose is False
        assert result.report.transpose == -7

    def test_counts_are_consistent(self, profile) -> None:
        result = convert_melody(
            melody(46, 60, 66, 90),
            ConversionConfig(transpose=0, range_mode="drop"),
            profile,
        )
        report = result.report

        assert report.input_note_count == 4
        assert report.output_note_count == 2
        assert report.dropped_count == 2
        assert report.playable_ratio == pytest.approx(0.5)
        assert report.semitone_count + report.octave_shift_count == report.modifier_count
        assert report.output_note_count + report.dropped_count == report.input_note_count

    def test_modifier_counts(self, profile) -> None:
        result = convert_melody(melody(48, 60, 66, 84), ConversionConfig(transpose=0), profile)
        report = result.report

        assert report.natural_count == 1
        assert report.semitone_count == 1
        assert report.octave_shift_count == 2
        assert report.modifier_count == 3

    def test_pitch_bounds(self, profile) -> None:
        result = convert_melody(melody(48, 84), ConversionConfig(transpose=0), profile)

        assert (result.report.original_pitch_min, result.report.original_pitch_max) == (48, 84)
        assert (result.report.final_pitch_min, result.report.final_pitch_max) == (48, 84)


class TestEdgeCases:
    def test_empty_melody(self, profile) -> None:
        result = convert_melody([], ConversionConfig(), profile)

        assert result.notes == ()
        assert result.duration == 0.0
        assert result.report.input_note_count == 0
        assert result.report.output_note_count == 0
        assert result.report.playable_ratio == 0.0
        assert result.report.original_pitch_min is None
        assert result.report.final_pitch_max is None

    def test_everything_dropped_is_reported_not_raised(self, profile) -> None:
        sparse = make_profile(base_keys=["Z"], natural_intervals=[0], octave_offsets=[0])
        result = convert_melody(
            melody(66, 68), ConversionConfig(transpose=0, range_mode="drop"), sparse
        )

        assert result.notes == ()
        assert result.report.dropped_count == 2
        assert result.report.playable_ratio == 0.0

    def test_never_leaks_an_unmapped_pitch_error(self, profile) -> None:
        """Range fitting runs first, so the mapper can never see a bad pitch."""
        for mode in ("octave_fold", "nearest_note", "drop"):
            result = convert_melody(melody(46, 90), ConversionConfig(transpose=0, range_mode=mode), profile)
            assert all(note.pitch in profile.playable_pitch_set for note in result.notes)

    def test_custom_profile_is_used(self) -> None:
        sparse = make_profile(
            name="narrow",
            base_keys=["A", "B"],
            natural_intervals=[0, 12],
            octave_offsets=[0],
        )
        result = convert_melody(melody(60), ConversionConfig(transpose=0), sparse)

        assert result.profile.name == "narrow"
        assert result.notes[0].lane == 0
        assert result.notes[0].key_label == "A"

    def test_confidence_flows_through(self, profile) -> None:
        notes = [make_note(60, start=0.0, duration=0.5, confidence=0.42)]
        result = convert_melody(notes, ConversionConfig(transpose=0), profile)

        assert result.notes[0].confidence == 0.42

    def test_is_a_frozen_public_result(self, profile) -> None:
        result = convert_melody(melody(60), ConversionConfig(transpose=0), profile)
        assert isinstance(result, ConversionResult)

        with pytest.raises(Exception):
            result.notes = ()  # type: ignore[misc]
