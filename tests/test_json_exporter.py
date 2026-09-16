"""score.json: schema, determinism and encoding."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from app.core.converter import ConversionConfig, ConversionResult, convert_melody
from app.core.models import NoteEvent
from app.export.json_exporter import SCORE_SCHEMA_VERSION, build_score_document, score_json, write_score_json

from conftest import make_note

EXPECTED_NOTE_KEYS = {
    "id",
    "start",
    "duration",
    "source_pitch",
    "pitch",
    "lane",
    "key",
    "semitone",
    "octave",
    "confidence",
}
EXPECTED_REPORT_KEYS = {
    "input_note_count",
    "output_note_count",
    "transpose",
    "auto_transpose",
    "range_mode",
    "original_pitch_min",
    "original_pitch_max",
    "final_pitch_min",
    "final_pitch_max",
    "dropped_count",
    "octave_fold_count",
    "nearest_adjusted_count",
    "semitone_count",
    "octave_shift_count",
    "modifier_count",
    "natural_count",
    "playable_ratio",
}


@pytest.fixture
def result() -> ConversionResult:
    notes = [
        make_note(72, start=0.0, duration=0.5, velocity=90, confidence=0.8),
        make_note(66, start=0.5, duration=0.25, velocity=80),
        make_note(84, start=0.75, duration=1.0, velocity=110),
        make_note(90, start=1.75, duration=0.5, velocity=90),  # folds down to 78
    ]
    return convert_melody(notes, ConversionConfig(transpose=0), None)


class TestDocument:
    def test_top_level_schema(self, result: ConversionResult) -> None:
        document = build_score_document(result)

        assert document["version"] == SCORE_SCHEMA_VERSION == 1
        assert document["profile"] == "delta_harmonica"
        assert document["lanes"] == ["Z", "X", "C", "V", "B", "N", "M", ","]
        assert document["symbols"] == {"semitone": "#", "octave_up": "↑", "octave_down": "↓"}
        assert document["duration"] == pytest.approx(2.25)
        assert document["source_segment"] is None
        document_with_segment = build_score_document(result, source_segment=(48.0, 82.0))
        assert document_with_segment["source_segment"] == {"start": 48.0, "end": 82.0}
        assert document["conversion"] == {
            "transpose": 0,
            "auto_transpose": False,
            "range_mode": "octave_fold",
        }
        assert set(document["report"]) == EXPECTED_REPORT_KEYS
        assert set(document) == {
            "version",
            "profile",
            "lanes",
            "symbols",
            "duration",
            "source_segment",
            "conversion",
            "report",
            "notes",
        }

    def test_notes_schema(self, result: ConversionResult) -> None:
        document = build_score_document(result)
        notes = document["notes"]

        assert len(notes) == 4
        assert [note["id"] for note in notes] == [0, 1, 2, 3]
        for note in notes:
            assert set(note) == EXPECTED_NOTE_KEYS

        assert notes[0] == {
            "id": 0,
            "start": 0.0,
            "duration": 0.5,
            "source_pitch": 72,
            "pitch": 72,
            "lane": 7,
            "key": ",",
            "semitone": False,
            "octave": 0,
            "confidence": 0.8,
        }

    def test_every_note_is_fully_expressible(self, result: ConversionResult) -> None:
        """When / which key / semitone / octave must all be present per note."""
        for note in build_score_document(result)["notes"]:
            assert isinstance(note["start"], float) and note["start"] >= 0
            assert isinstance(note["duration"], float) and note["duration"] > 0
            assert note["key"] in result.profile.base_keys
            assert isinstance(note["semitone"], bool)
            assert note["octave"] in result.profile.octave_offsets
            assert 0 <= note["lane"] < result.profile.lane_count

    def test_semitone_and_octave_modifiers_are_recorded(self, result: ConversionResult) -> None:
        notes = build_score_document(result)["notes"]

        semitone = notes[1]
        assert semitone["pitch"] == 66 and semitone["key"] == "V" and semitone["semitone"] is True

        folded = notes[3]
        assert folded["source_pitch"] == 90
        assert folded["pitch"] == 78
        assert folded["key"] == "V"
        assert folded["semitone"] is True
        assert folded["octave"] == 1

    def test_report_values(self, result: ConversionResult) -> None:
        report = build_score_document(result)["report"]

        assert report["input_note_count"] == 4
        assert report["output_note_count"] == 4
        assert report["dropped_count"] == 0
        assert report["octave_fold_count"] == 1
        assert report["semitone_count"] == 2
        assert report["octave_shift_count"] == 2
        assert report["modifier_count"] == 4
        assert report["playable_ratio"] == 1.0
        assert (report["original_pitch_min"], report["original_pitch_max"]) == (66, 90)
        assert (report["final_pitch_min"], report["final_pitch_max"]) == (66, 84)


class TestDeterminism:
    def test_repeated_export_is_byte_identical(self, result: ConversionResult) -> None:
        first = score_json(result)
        second = score_json(result)
        assert first == second
        assert hashlib.sha256(first.encode("utf-8")).hexdigest() == hashlib.sha256(
            second.encode("utf-8")
        ).hexdigest()

    def test_input_order_does_not_matter(self, profile) -> None:
        notes = [make_note(pitch, start=index * 0.5) for index, pitch in enumerate((60, 66, 72))]
        forward = convert_melody(notes, ConversionConfig(transpose=0), profile)
        backward = convert_melody(list(reversed(notes)), ConversionConfig(transpose=0), profile)

        assert score_json(forward) == score_json(backward)

    def test_no_timestamps_or_random_fields(self, result: ConversionResult) -> None:
        document = build_score_document(result)
        forbidden = {"generated_at", "created", "timestamp", "now", "uuid", "id_seed"}
        assert not (set(document) & forbidden)
        assert not (set(document["report"]) & forbidden)


class TestWriting:
    def test_writes_utf8_json(self, result: ConversionResult, tmp_path: Path) -> None:
        path = write_score_json(result, tmp_path / "score.json")

        assert path == tmp_path / "score.json"
        raw = path.read_bytes()
        document = json.loads(raw.decode("utf-8"))

        assert document["symbols"]["octave_up"] == "↑"
        assert document["symbols"]["octave_down"] == "↓"
        assert "↑".encode("utf-8") in raw  # real UTF-8 bytes, not escapes

    def test_written_file_matches_the_in_memory_export(self, result: ConversionResult, tmp_path: Path) -> None:
        path = write_score_json(result, tmp_path / "score.json")

        assert path.read_text(encoding="utf-8") == score_json(result)

    def test_writing_twice_is_byte_identical(self, result: ConversionResult, tmp_path: Path) -> None:
        first = write_score_json(result, tmp_path / "a.json")
        second = write_score_json(result, tmp_path / "b.json")

        assert first.read_bytes() == second.read_bytes()

    def test_creates_missing_directories(self, result: ConversionResult, tmp_path: Path) -> None:
        path = write_score_json(result, tmp_path / "nested" / "deep" / "score.json")
        assert path.is_file()

    def test_rejects_a_directory_target(self, result: ConversionResult, tmp_path: Path) -> None:
        from app.core.errors import OutputNotWritableError

        with pytest.raises(OutputNotWritableError, match="cannot write score JSON"):
            write_score_json(result, tmp_path)

    def test_rejects_a_target_under_a_file(self, result: ConversionResult, tmp_path: Path) -> None:
        from app.core.errors import OutputNotWritableError

        blocker = tmp_path / "blocker.txt"
        blocker.write_text("not a directory", encoding="utf-8")

        with pytest.raises(OutputNotWritableError, match="cannot write score JSON"):
            write_score_json(result, blocker / "score.json")


class TestSchemaStability:
    def test_field_order_is_stable(self, result: ConversionResult) -> None:
        """The JSON key order is part of the schema and must not drift."""
        text = score_json(result)

        assert text.index('"version"') < text.index('"profile"')
        assert text.index('"profile"') < text.index('"lanes"')
        assert text.index('"lanes"') < text.index('"duration"')
        assert text.index('"report"') < text.index('"notes"')
        assert text.index('"source_pitch"') < text.index('"pitch"')
        assert text.index('"pitch"') < text.index('"lane"')
        assert text.index('"lane"') < text.index('"key"')

    def test_document_survives_a_round_trip_through_json(self, result: ConversionResult) -> None:
        document = json.loads(score_json(result))
        assert document == build_score_document(result)

    def test_export_of_a_transposed_score(self, profile) -> None:
        notes: list[NoteEvent] = [make_note(72, start=0.0, duration=0.5)]
        result = convert_melody(notes, ConversionConfig(transpose="auto"), profile)

        document = build_score_document(result)
        assert document["conversion"]["auto_transpose"] is True
        assert document["notes"][0]["source_pitch"] == 72
