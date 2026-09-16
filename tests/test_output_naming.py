"""Default output naming tests."""

from __future__ import annotations

from pathlib import Path

from app.core.output_naming import default_output_path, format_segment_tag, unique_path


class TestSegmentTag:
    def test_minutes_and_seconds(self) -> None:
        assert format_segment_tag(48.0, 82.0) == "00m48s-01m22s"

    def test_rounds(self) -> None:
        assert format_segment_tag(48.4, 82.6) == "00m48s-01m23s"

    def test_zero(self) -> None:
        assert format_segment_tag(0.0, 6.0) == "00m00s-00m06s"

    def test_hours(self) -> None:
        assert format_segment_tag(3723.0, 7322.0) == "01h02m03s-02h02m02s"

    def test_windows_safe(self) -> None:
        tag = format_segment_tag(48.0, 82.0)
        assert ":" not in tag
        assert not set(tag) & set('\\/:*?"<>|')


class TestDefaultOutputPath:
    def test_full_song_png(self, tmp_path: Path) -> None:
        path = default_output_path(tmp_path / "song.mp3", ".png")
        assert path == tmp_path / "song_score.png"

    def test_segment_png(self, tmp_path: Path) -> None:
        path = default_output_path(tmp_path / "song.mp3", ".png", segment=(48.0, 82.0))
        assert path == tmp_path / "song_00m48s-01m22s_score.png"

    def test_score_json(self, tmp_path: Path) -> None:
        assert default_output_path(tmp_path / "song.mp3", ".score.json") == tmp_path / "song.score.json"
        assert default_output_path(
            tmp_path / "song.mp3", ".score.json", segment=(12.0, 36.0)
        ) == tmp_path / "song_00m12s-00m36s.score.json"

    def test_lives_next_to_the_input(self, tmp_path: Path) -> None:
        source = tmp_path / "a" / "b" / "song.mp3"
        source.parent.mkdir(parents=True)
        assert default_output_path(source, ".png").parent == source.parent

    def test_deterministic(self, tmp_path: Path) -> None:
        first = default_output_path(tmp_path / "song.mp3", ".png", segment=(48.0, 82.0))
        second = default_output_path(tmp_path / "song.mp3", ".png", segment=(48.0, 82.0))
        assert first == second


class TestUniquePath:
    def test_free_name_unchanged(self, tmp_path: Path) -> None:
        path = tmp_path / "song_score.png"
        assert unique_path(path) == path

    def test_existing_name_gets_a_counter(self, tmp_path: Path) -> None:
        (tmp_path / "song_score.png").write_bytes(b"x")
        assert unique_path(tmp_path / "song_score.png") == tmp_path / "song_score_2.png"

    def test_counters_continue(self, tmp_path: Path) -> None:
        for name in ("a.png", "a_2.png", "a_3.png"):
            (tmp_path / name).write_bytes(b"x")
        assert unique_path(tmp_path / "a.png") == tmp_path / "a_4.png"

    def test_deterministic(self, tmp_path: Path) -> None:
        (tmp_path / "a.png").write_bytes(b"x")
        assert unique_path(tmp_path / "a.png") == unique_path(tmp_path / "a.png")
