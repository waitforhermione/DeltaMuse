"""score.json importer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.core.errors import ScoreDocumentError
from app.export.score_loader import load_score_document, parse_score_document
from app.score.labels import DEFAULT_SYMBOLS

MINIMAL_DOCUMENT = {
    "version": 1,
    "profile": "delta_harmonica",
    "lanes": ["Z", "X"],
    "symbols": {"semitone": "#", "octave_up": "↑", "octave_down": "↓"},
    "duration": 2.0,
    "conversion": {"transpose": -2, "auto_transpose": True, "range_mode": "octave_fold"},
    "report": {"input_note_count": 1, "output_note_count": 1, "playable_ratio": 1.0},
    "notes": [
        {
            "id": 0,
            "start": 0.5,
            "duration": 1.0,
            "source_pitch": 62,
            "pitch": 60,
            "lane": 0,
            "key": "Z",
            "semitone": False,
            "octave": 0,
            "confidence": None,
        }
    ],
}


def write_document(tmp_path: Path, data: object, name: str = "score.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return path


class TestLoading:
    def test_loads_a_valid_document(self, tmp_path: Path) -> None:
        document = load_score_document(write_document(tmp_path, MINIMAL_DOCUMENT))

        assert document.version == 1
        assert document.profile_name == "delta_harmonica"
        assert document.lanes == ("Z", "X")
        assert document.symbols == DEFAULT_SYMBOLS
        assert document.duration == pytest.approx(2.0)
        assert document.transpose == -2
        assert document.auto_transpose is True
        assert document.range_mode == "octave_fold"
        assert document.note_count == 1
        assert document.report["playable_ratio"] == 1.0

    def test_notes_round_trip_into_playable_notes(self, tmp_path: Path) -> None:
        document = load_score_document(write_document(tmp_path, MINIMAL_DOCUMENT))
        note = document.notes[0]

        assert note.source_pitch == 62
        assert note.pitch == 60
        assert note.start == pytest.approx(0.5)
        assert note.duration == pytest.approx(1.0)
        assert note.lane == 0
        assert note.key_label == "Z"
        assert note.semitone is False
        assert note.octave_offset == 0
        assert note.confidence is None

    def test_optional_confidence_is_read(self, tmp_path: Path) -> None:
        data = json.loads(json.dumps(MINIMAL_DOCUMENT))
        data["notes"][0]["confidence"] = 0.75
        document = load_score_document(write_document(tmp_path, data))

        assert document.notes[0].confidence == pytest.approx(0.75)

    def test_missing_confidence_is_tolerated(self, tmp_path: Path) -> None:
        data = json.loads(json.dumps(MINIMAL_DOCUMENT))
        del data["notes"][0]["confidence"]

        document = load_score_document(write_document(tmp_path, data))
        assert document.notes[0].confidence is None

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(ScoreDocumentError, match="not found"):
            load_score_document(tmp_path / "nope.json")

    def test_invalid_json(self, tmp_path: Path) -> None:
        path = tmp_path / "score.json"
        path.write_text("{oops", encoding="utf-8")
        with pytest.raises(ScoreDocumentError, match="not valid JSON"):
            load_score_document(path)

    def test_non_object(self, tmp_path: Path) -> None:
        with pytest.raises(ScoreDocumentError, match="JSON object"):
            load_score_document(write_document(tmp_path, [], "score.json"))


class TestValidation:
    @pytest.mark.parametrize("field", ["version", "profile", "lanes", "duration", "notes"])
    def test_missing_required_field(self, tmp_path: Path, field: str) -> None:
        data = json.loads(json.dumps(MINIMAL_DOCUMENT))
        del data[field]
        with pytest.raises(ScoreDocumentError, match=f"missing required field.*{field}"):
            load_score_document(write_document(tmp_path, data, "score.json"))

    def test_unsupported_version(self, tmp_path: Path) -> None:
        data = json.loads(json.dumps(MINIMAL_DOCUMENT))
        data["version"] = 99
        with pytest.raises(ScoreDocumentError, match="unsupported score version"):
            load_score_document(write_document(tmp_path, data, "score.json"))

    def test_empty_lanes(self, tmp_path: Path) -> None:
        data = json.loads(json.dumps(MINIMAL_DOCUMENT))
        data["lanes"] = []
        with pytest.raises(ScoreDocumentError, match="lanes must be a non-empty"):
            load_score_document(write_document(tmp_path, data, "score.json"))

    def test_lane_out_of_range(self, tmp_path: Path) -> None:
        data = json.loads(json.dumps(MINIMAL_DOCUMENT))
        data["notes"][0]["lane"] = 5
        with pytest.raises(ScoreDocumentError, match="outside 0..1"):
            load_score_document(write_document(tmp_path, data, "score.json"))

    def test_notes_must_be_an_array(self, tmp_path: Path) -> None:
        data = json.loads(json.dumps(MINIMAL_DOCUMENT))
        data["notes"] = {}
        with pytest.raises(ScoreDocumentError, match="notes must be a JSON array"):
            load_score_document(write_document(tmp_path, data, "score.json"))

    def test_note_missing_fields(self, tmp_path: Path) -> None:
        data = json.loads(json.dumps(MINIMAL_DOCUMENT))
        del data["notes"][0]["key"]
        with pytest.raises(ScoreDocumentError, match="note #0 is missing.*key"):
            load_score_document(write_document(tmp_path, data, "score.json"))

    @pytest.mark.parametrize(
        ("field", "value"),
        [("pitch", 200), ("source_pitch", -1), ("lane", "0"), ("octave", 0.5), ("semitone", 1)],
    )
    def test_invalid_note_values(self, tmp_path: Path, field: str, value: object) -> None:
        data = json.loads(json.dumps(MINIMAL_DOCUMENT))
        data["notes"][0][field] = value
        with pytest.raises(ScoreDocumentError, match=f"{field}"):
            load_score_document(write_document(tmp_path, data, "score.json"))

    def test_non_positive_duration(self, tmp_path: Path) -> None:
        data = json.loads(json.dumps(MINIMAL_DOCUMENT))
        data["notes"][0]["duration"] = 0
        with pytest.raises(ScoreDocumentError, match="duration"):
            load_score_document(write_document(tmp_path, data, "score.json"))

    def test_parse_score_document_without_a_path(self) -> None:
        document = parse_score_document(MINIMAL_DOCUMENT)
        assert document.note_count == 1
