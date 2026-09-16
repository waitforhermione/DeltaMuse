"""Layout debug JSON export."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.core.errors import OutputNotWritableError
from app.export.layout_exporter import (
    LAYOUT_SCHEMA_VERSION,
    build_layout_document,
    layout_json,
    write_layout_json,
)
from app.score.layout import LayoutHeader, ScoreLayoutConfig, build_layout

from conftest import LANES, playable_note as playable


def make_layout():
    notes = [
        playable(60, 0, 0.0, 0.5, confidence=0.5),
        playable(66, 3, 5.5, 2.0, key_label="V", semitone=True),
    ]
    return build_layout(
        notes,
        lanes=LANES,
        config=ScoreLayoutConfig(),
        header=LayoutHeader(title="Demo", source="demo.mp3", profile="delta_harmonica", duration=6.0),
    )


class TestDocument:
    def test_schema(self) -> None:
        document = build_layout_document(make_layout())

        assert document["version"] == LAYOUT_SCHEMA_VERSION == 1
        assert document["canvas"] == {"width": 2000.0, "height": 592.0}
        assert document["lanes"] == list(LANES)
        assert set(document) == {
            "version",
            "canvas",
            "duration",
            "lanes",
            "config",
            "header",
            "sections",
        }

    def test_sections_and_fragments(self) -> None:
        document = build_layout_document(make_layout())

        assert len(document["sections"]) == 2
        first = document["sections"][0]
        assert first["index"] == 0
        assert (first["start_time"], first["end_time"]) == (0.0, 6.0)
        assert len(first["notes"]) == 2  # the C5 fragment + the first half of the long note

        long_fragments = [
            note
            for section in document["sections"]
            for note in section["notes"]
            if note["source_note_id"] == 1
        ]
        assert len(long_fragments) == 2
        assert sum(note["fragment_duration"] for note in long_fragments) == pytest.approx(2.0)
        assert [note["is_continuation"] for note in long_fragments] == [False, True]

    def test_grid_is_exported(self) -> None:
        document = build_layout_document(make_layout())
        grid = document["sections"][0]["grid"]

        majors = [line for line in grid if line["kind"] == "major"]
        assert [line["time"] for line in majors] == [0, 1, 2, 3, 4, 5, 6]
        assert all({"x", "time", "kind"} == set(line) for line in grid)

    def test_confidence_is_exported(self) -> None:
        document = build_layout_document(make_layout())
        confidence = [
            note["confidence"] for section in document["sections"] for note in section["notes"]
        ]
        assert confidence == [0.5, None, None]

    def test_header_is_exported(self) -> None:
        document = build_layout_document(make_layout())
        assert document["header"]["title"] == "Demo"
        assert document["header"]["profile"] == "delta_harmonica"

    def test_config_is_exported(self) -> None:
        document = build_layout_document(make_layout(), ScoreLayoutConfig(section_duration=4.0))
        assert document["config"]["section_duration"] == 4.0


class TestWriting:
    def test_write_and_read_back(self, tmp_path: Path) -> None:
        layout = make_layout()
        path = write_layout_json(layout, tmp_path / "layout.json")

        document = json.loads(path.read_text(encoding="utf-8"))
        assert document == build_layout_document(layout)

    def test_deterministic_bytes(self, tmp_path: Path) -> None:
        layout = make_layout()
        first = write_layout_json(layout, tmp_path / "a.json")
        second = write_layout_json(layout, tmp_path / "b.json")

        assert first.read_bytes() == second.read_bytes()

    def test_rejects_a_directory_target(self, tmp_path: Path) -> None:
        with pytest.raises(OutputNotWritableError, match="cannot write layout JSON"):
            write_layout_json(make_layout(), tmp_path)
