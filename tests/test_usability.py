"""Usability phase tests: segment transcription, naming, settings priority, output."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from app.cli import main
from app.config.user_settings import (
    UserSettings,
    load_settings,
    save_settings,
    settings_path,
    with_updated_field,
)
from app.core.models import NoteEvent

from conftest import MONO_SAMPLE_RATE, FakeTranscriber, write_wav


def make_long_wav(tmp_path: Path, seconds: float = 10.0) -> Path:
    samples = (
        0.3
        * np.sin(
            2.0 * np.pi * 220.0 * np.arange(int(seconds * MONO_SAMPLE_RATE)) / MONO_SAMPLE_RATE
        )
    ).astype(np.float32)
    return write_wav(tmp_path / "song.wav", samples, MONO_SAMPLE_RATE)


class SegmentStub(FakeTranscriber):
    """Emits notes spread over whatever buffer it receives."""

    name = "segment-stub"

    def transcribe(self, source: object) -> list[NoteEvent]:
        self.calls.append(source)
        if self.error is not None:
            raise self.error
        duration = float(getattr(source, "duration", 0.0))
        count = max(int(duration / 0.5), 1)
        return [
            NoteEvent(pitch=60 + (index % 5), start=index * 0.5, duration=0.4, velocity=80)
            for index in range(count)
        ]


def install_spy(monkeypatch: pytest.MonkeyPatch, seen: list[float]) -> None:
    class Spy(SegmentStub):
        def transcribe(self, source: object) -> list[NoteEvent]:
            seen.append(float(getattr(source, "duration", 0.0)))
            return super().transcribe(source)

    monkeypatch.setattr("app.cli.create_transcriber", lambda name, **kwargs: Spy([]))


class TestSegmentTranscriptionTimes:
    def test_segment_notes_are_mapped_back_and_rebased(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """local t -> absolute (t + window.start) -> crop [start, end] -> rebased 0."""
        source = make_long_wav(tmp_path)
        install_spy(monkeypatch, [])

        output = tmp_path / "seg.score.json"
        exit_code = main(["map-melody", str(source), "--start", "4", "--end", "6", "-o", str(output)])
        assert exit_code == 0

        document = json.loads(output.read_text(encoding="utf-8"))
        assert document["source_segment"] == {"start": 4.0, "end": 6.0}
        # Survivors: absolute 4.0-5.5 s (4 notes x 0.4 s); the last ends at 1.9 s.
        assert document["duration"] == pytest.approx(1.9, abs=0.01)

        # window = [2, 8]; a local note at 2.5 s -> absolute 4.5 s -> rebased 0.5 s.
        starts = [note["start"] for note in document["notes"]]
        assert starts, "expected notes inside the cropped segment"
        assert all(start >= 0.0 for start in starts)
        assert all(note["start"] + note["duration"] <= 2.0 + 1e-6 for note in document["notes"])
        assert any(start == pytest.approx(0.5) for start in starts)

    def test_the_model_only_sees_the_segment(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        source = make_long_wav(tmp_path)
        seen: list[float] = []
        install_spy(monkeypatch, seen)

        exit_code = main(
            ["map-melody", str(source), "--start", "4", "--end", "6", "-o", str(tmp_path / "s.json")]
        )
        assert exit_code == 0
        # requested 2 s + 2 s context, not the full 10 s
        assert seen and seen[0] == pytest.approx(6.0, abs=0.1)

    def test_full_track_is_untouched_without_options(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        source = make_long_wav(tmp_path)
        seen: list[float] = []
        install_spy(monkeypatch, seen)

        output = tmp_path / "full.score.json"
        assert main(["map-melody", str(source), "-o", str(output)]) == 0
        assert seen and seen[0] == pytest.approx(10.0, abs=0.1)

        document = json.loads(output.read_text(encoding="utf-8"))
        assert document["source_segment"] is None


class TestDefaultOutputNames:
    def test_convert_without_output_writes_a_png(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path, tone_wav: Path, use_backend: Any
    ) -> None:
        use_backend()
        assert main(["convert", str(tone_wav)]) == 0
        stdout = capsys.readouterr().out

        expected = tone_wav.parent / "tone_score.png"
        assert expected.is_file()
        assert expected.name in stdout

    def test_convert_segment_output_name(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path, tone_wav: Path, use_backend: Any
    ) -> None:
        use_backend()
        assert main(["convert", str(tone_wav), "--start", "0", "--end", "0.4"]) == 0
        capsys.readouterr()

        assert (tone_wav.parent / "tone_00m00s-00m00s_score.png").is_file()

    def test_map_melody_default_stays_score_json(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path, tone_wav: Path, use_backend: Any
    ) -> None:
        use_backend()
        assert main(["map-melody", str(tone_wav)]) == 0
        capsys.readouterr()
        assert (tone_wav.parent / "tone.score.json").is_file()

    def test_existing_output_is_never_overwritten(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path, tone_wav: Path, use_backend: Any
    ) -> None:
        use_backend()
        existing = tmp_path / "custom.png"
        existing.write_bytes(b"old")
        before = existing.read_bytes()

        exit_code = main(["convert", str(tone_wav), "-o", str(existing)])
        stdout = capsys.readouterr().out

        assert exit_code == 0
        assert existing.read_bytes() == before  # untouched
        assert (tmp_path / "custom_2.png").is_file()
        assert "already exists" in stdout


class TestSettingsPriority:
    def _write_settings(self, style: str) -> None:
        save_settings(
            with_updated_field(UserSettings(), "default_style", style), settings_path()
        )

    def test_settings_style_is_used_when_cli_is_silent(
        self, tmp_path: Path, tone_wav: Path, use_backend: Any
    ) -> None:
        self._write_settings("compact")
        use_backend()
        png = tmp_path / "out.png"
        assert main(["convert", str(tone_wav), "-o", str(png)]) == 0

        from PIL import Image

        with Image.open(png) as image:
            assert image.size[0] == 2000  # compact

    def test_cli_beats_settings(
        self, tmp_path: Path, tone_wav: Path, use_backend: Any
    ) -> None:
        self._write_settings("compact")
        use_backend()
        png = tmp_path / "out.png"
        assert main(["convert", str(tone_wav), "--style", "practice", "-o", str(png)]) == 0

        from PIL import Image

        with Image.open(png) as image:
            assert image.size[0] == 2400  # practice from the CLI wins

    def test_corrupt_settings_fall_back_with_a_warning(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path, tone_wav: Path, use_backend: Any
    ) -> None:
        path = settings_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{corrupt", encoding="utf-8")

        use_backend()
        exit_code = main(["convert", str(tone_wav), "-o", str(tmp_path / "out.png")])
        stdout = capsys.readouterr().out

        assert exit_code == 0
        assert "Warning:" in stdout

    def test_config_command_round_trip(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["config", "--set", "default_style", "compact"]) == 0
        capsys.readouterr()

        settings, _warnings = load_settings()
        assert settings.default_style == "compact"

        assert main(["config", "--reset"]) == 0
        capsys.readouterr()
        settings, _warnings = load_settings()
        assert settings == UserSettings()
