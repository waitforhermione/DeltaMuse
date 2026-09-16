"""Application service tests (Qt-free)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from app.application.conversion_service import (
    ConversionRequest,
    ConversionService,
    STAGES,
)
from app.core.errors import PianoScoreError

from conftest import MONO_SAMPLE_RATE, write_wav


def make_long_wav(tmp_path: Path, seconds: float = 10.0) -> Path:
    import numpy as np

    samples = (
        0.3
        * np.sin(2.0 * np.pi * 220.0 * np.arange(int(seconds * MONO_SAMPLE_RATE)) / MONO_SAMPLE_RATE)
    ).astype(np.float32)
    return write_wav(tmp_path / "song.wav", samples, MONO_SAMPLE_RATE)


def install_stub(monkeypatch: pytest.MonkeyPatch) -> list:
    from app.core.models import NoteEvent

    seen: list = []

    class Stub:
        name = "stub"

        def transcribe(self, source: object) -> list[NoteEvent]:
            seen.append(source)
            duration = float(getattr(source, "duration", 0.0))
            count = max(int(duration / 0.5), 1)
            return [
                NoteEvent(pitch=60 + (index % 5), start=index * 0.5, duration=0.4, velocity=80)
                for index in range(count)
            ]

    monkeypatch.setattr(
        "app.application.conversion_service.create_transcriber", lambda name, **kwargs: Stub()
    )
    return seen


class _EmptyStub:
    name = "empty"

    def transcribe(self, source: object) -> list:
        return []


def test_service_is_qt_free() -> None:
    """The service module must be importable in a clean process without Qt."""
    import subprocess

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, app.application.conversion_service as m; "
            "assert 'PySide6' not in sys.modules, 'service pulled in Qt'; print('ok')",
        ],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parents[1]),
    )
    assert result.returncode == 0, result.stderr


class TestInspect:
    def test_audio_file(self, tone_wav: Path) -> None:
        info = ConversionService().inspect(tone_wav)
        assert info.supported and info.kind == "audio"
        assert info.duration == pytest.approx(0.5, abs=0.05)

    def test_midi_file(self, tmp_path: Path) -> None:
        from app.core.models import NoteEvent
        from app.midi.midi_writer import write_midi

        path = write_midi(
            [NoteEvent(pitch=60, start=0.0, duration=1.0, velocity=80)], tmp_path / "song.mid"
        )
        info = ConversionService().inspect(path)
        assert info.supported and info.kind == "midi" and info.duration

    def test_unsupported_and_missing(self, tmp_path: Path) -> None:
        path = tmp_path / "notes.txt"
        path.write_text("x")
        assert not ConversionService().inspect(path).supported
        assert ConversionService().inspect(tmp_path / "nope.mp3").kind == "missing"


class TestRun:
    def test_full_track_conversion(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        source = make_long_wav(tmp_path)
        seen = install_stub(monkeypatch)
        outcome = ConversionService().run(ConversionRequest(input_path=source, style="practice"))

        assert isinstance(outcome.output, Path) and outcome.output.is_file()
        assert outcome.output.name == "song_score.png"
        assert outcome.notes > 0
        assert outcome.segment is None
        assert seen[0].duration == pytest.approx(10.0, abs=0.1)

    def test_segment_conversion(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        source = make_long_wav(tmp_path)
        seen = install_stub(monkeypatch)
        outcome = ConversionService().run(
            ConversionRequest(input_path=source, style="practice", start="4", end="6")
        )

        assert outcome.segment == (4.0, 6.0)
        assert outcome.output.name == "song_00m04s-00m06s_score.png"
        # the model only saw 2 s + 2 s context
        assert seen[0].duration == pytest.approx(6.0, abs=0.1)

        from PIL import Image

        with Image.open(outcome.output) as image:
            assert image.format == "PNG"

    def test_progress_order(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        source = make_long_wav(tmp_path)
        install_stub(monkeypatch)
        stages: list[int] = []
        ConversionService().run(
            ConversionRequest(input_path=source),
            on_progress=lambda stage, text: stages.append(stage),
        )
        assert stages == sorted(stages)
        assert stages == [stage for stage, _text in STAGES]

    def test_overwrite_protection(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        source = make_long_wav(tmp_path)
        install_stub(monkeypatch)
        output = tmp_path / "custom.png"
        output.write_bytes(b"old")

        outcome = ConversionService().run(ConversionRequest(input_path=source, output_path=output))
        assert outcome.output.name == "custom_2.png"
        assert output.read_bytes() == b"old"

    def test_missing_file_error(self, tmp_path: Path) -> None:
        with pytest.raises(PianoScoreError, match="not found"):
            ConversionService().run(ConversionRequest(input_path=tmp_path / "nope.mp3"))

    def test_no_notes_error(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from app.core.errors import NoNotesError

        source = make_long_wav(tmp_path)
        monkeypatch.setattr(
            "app.application.conversion_service.create_transcriber",
            lambda name, **kwargs: _EmptyStub(),
        )
        with pytest.raises(NoNotesError, match="No piano notes"):
            ConversionService().run(ConversionRequest(input_path=source, start="2", end="3"))
