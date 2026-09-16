"""CLI behaviour: commands, reports and error handling."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pretty_midi
import pytest

from app.cli import main
from app.core.errors import TranscriptionError
from app.core.models import NoteEvent
from app.midi.midi_writer import write_midi
from app.score.layout import LayoutHeader, build_layout
from app.score.styles import get_preset

from conftest import FakeTranscriber  # noqa: F401  (kept for direct imports in tests)


@pytest.fixture
def midi_input(tmp_path: Path, sample_notes: list[NoteEvent]) -> Path:
    return write_midi(sample_notes, tmp_path / "song.mid")


class TestAnalyzeAudio:
    def test_reports_an_audio_file(
        self, capsys: pytest.CaptureFixture[str], tone_wav: Path, use_backend: Any
    ) -> None:
        use_backend()
        exit_code = main(["analyze", str(tone_wav)])
        stdout = capsys.readouterr().out

        assert exit_code == 0
        assert "Input type" in stdout and "audio" in stdout
        assert "Notes" in stdout and "4" in stdout
        assert "Pitch range" in stdout and "60-72" in stdout
        assert "Polyphony" in stdout
        assert "Confidence" in stdout

    def test_writes_a_json_report(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path, tone_wav: Path, use_backend: Any
    ) -> None:
        use_backend()
        report_path = tmp_path / "report.json"
        exit_code = main(["analyze", str(tone_wav), "--json", str(report_path)])
        capsys.readouterr()

        assert exit_code == 0
        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert report["input_type"] == "audio"
        assert report["note_count"] == 4
        assert report["pitch_min"] == 60 and report["pitch_max"] == 72
        assert report["mean_confidence"] == pytest.approx(0.75)
        assert len(report["notes"]) == 4
        assert set(report["notes"][0]) == {"pitch", "start", "duration", "velocity", "confidence"}

    def test_polyphony_is_computed(
        self, capsys: pytest.CaptureFixture[str], tone_wav: Path, use_backend: Any
    ) -> None:
        overlapping = [
            NoteEvent(pitch=60, start=0.0, duration=1.0, velocity=80),
            NoteEvent(pitch=64, start=0.0, duration=1.0, velocity=80),
        ]
        use_backend(overlapping)
        assert main(["analyze", str(tone_wav)]) == 0
        assert "max 2" in capsys.readouterr().out

    def test_empty_transcription_is_an_error(
        self, capsys: pytest.CaptureFixture[str], tone_wav: Path, use_backend: Any
    ) -> None:
        use_backend([])
        exit_code = main(["analyze", str(tone_wav)])
        captured = capsys.readouterr()

        assert exit_code == 1
        assert captured.err.startswith("ERROR: ")
        assert "no notes" in captured.err

    def test_backend_failures_surface_as_errors(
        self, capsys: pytest.CaptureFixture[str], tone_wav: Path, use_backend: Any
    ) -> None:
        use_backend(error=TranscriptionError("model exploded"))
        exit_code = main(["analyze", str(tone_wav)])

        assert exit_code == 1
        assert "ERROR: model exploded" in capsys.readouterr().err

    def test_unknown_backend_is_rejected(
        self, capsys: pytest.CaptureFixture[str], tone_wav: Path
    ) -> None:
        exit_code = main(["analyze", str(tone_wav), "--backend", "ghost"])

        assert exit_code == 1
        assert "unknown backend" in capsys.readouterr().err


class TestAnalyzeMidi:
    def test_reports_a_midi_file(self, capsys: pytest.CaptureFixture[str], midi_input: Path) -> None:
        exit_code = main(["analyze", str(midi_input)])
        stdout = capsys.readouterr().out

        assert exit_code == 0
        assert "midi" in stdout
        assert "Tracks" in stdout
        assert "Acoustic Grand Piano" in stdout
        assert "Notes" in stdout

    def test_midi_notes_have_no_confidence(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path, midi_input: Path, sample_notes: list[NoteEvent]
    ) -> None:
        report_path = tmp_path / "midi.json"
        assert main(["analyze", str(midi_input), "--json", str(report_path)]) == 0
        stdout = capsys.readouterr().out

        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert report["mean_confidence"] is None
        assert all(note["confidence"] is None for note in report["notes"])
        assert "n/a (MIDI input)" in stdout


class TestTranscribe:
    def test_writes_a_readable_midi_file(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path, tone_wav: Path, use_backend: Any
    ) -> None:
        use_backend()
        output = tmp_path / "out.mid"
        exit_code = main(["transcribe", str(tone_wav), "--output", str(output)])
        stdout = capsys.readouterr().out

        assert exit_code == 0
        assert output.exists()
        assert "Wrote MIDI" in stdout
        reloaded = pretty_midi.PrettyMIDI(str(output))
        assert len(reloaded.instruments[0].notes) == 4

    def test_default_output_name(
        self, capsys: pytest.CaptureFixture[str], tone_wav: Path, use_backend: Any
    ) -> None:
        use_backend()
        assert main(["transcribe", str(tone_wav)]) == 0
        capsys.readouterr()
        assert (tone_wav.parent / "tone_transcribed.mid").exists()

    def test_midi_input_is_rejected_with_a_hint(
        self, capsys: pytest.CaptureFixture[str], midi_input: Path
    ) -> None:
        exit_code = main(["transcribe", str(midi_input)])
        err = capsys.readouterr().err

        assert exit_code == 1
        assert "already a MIDI file" in err
        assert "analyze" in err

    def test_empty_transcription_is_an_error(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path, tone_wav: Path, use_backend: Any
    ) -> None:
        use_backend([])
        exit_code = main(["transcribe", str(tone_wav), "-o", str(tmp_path / "out.mid")])

        assert exit_code == 1
        assert "no notes" in capsys.readouterr().err


class TestExtractMelody:
    def test_extracts_a_melody_from_midi(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path, midi_input: Path
    ) -> None:
        output = tmp_path / "melody.mid"
        exit_code = main(["extract-melody", str(midi_input), "-o", str(output)])
        stdout = capsys.readouterr().out

        assert exit_code == 0
        assert "Strategy" in stdout and "continuity" in stdout
        assert "Melody notes" in stdout
        assert "max 1" in stdout
        assert output.exists()

        reloaded = pretty_midi.PrettyMIDI(str(output))
        assert len(reloaded.instruments[0].notes) == 4

    def test_default_output_name(
        self, capsys: pytest.CaptureFixture[str], midi_input: Path
    ) -> None:
        assert main(["extract-melody", str(midi_input)]) == 0
        capsys.readouterr()
        assert (midi_input.parent / "song_melody.mid").exists()

    def test_strategy_is_selectable(
        self, capsys: pytest.CaptureFixture[str], midi_input: Path
    ) -> None:
        assert main(["extract-melody", str(midi_input), "--strategy", "highest"]) == 0
        assert "highest" in capsys.readouterr().out

    def test_audio_input_goes_through_the_backend(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path, tone_wav: Path, use_backend: Any
    ) -> None:
        use_backend()
        output = tmp_path / "melody.mid"
        exit_code = main(["extract-melody", str(tone_wav), "-o", str(output)])
        capsys.readouterr()

        assert exit_code == 0
        assert len(pretty_midi.PrettyMIDI(str(output)).instruments[0].notes) == 4

    def test_short_notes_are_filtered_and_reported(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        notes = [
            NoteEvent(pitch=60, start=0.0, duration=0.02, velocity=80),
            NoteEvent(pitch=64, start=0.0, duration=0.5, velocity=80),
        ]
        path = write_midi(notes, tmp_path / "short.mid")
        output = tmp_path / "melody.mid"

        exit_code = main(["extract-melody", str(path), "-o", str(output)])
        stdout = capsys.readouterr().out

        assert exit_code == 0
        assert "Filtered out" in stdout and "1 short" in stdout
        assert len(pretty_midi.PrettyMIDI(str(output)).instruments[0].notes) == 1

    def test_chordal_input_yields_a_monophonic_file(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        chords: list[NoteEvent] = []
        for index in range(4):
            start = index * 0.5
            for pitch in (48, 55, 64 + index):
                chords.append(NoteEvent(pitch=pitch, start=start, duration=0.9, velocity=80))
        path = write_midi(chords, tmp_path / "chords.mid")
        output = tmp_path / "melody.mid"

        assert main(["extract-melody", str(path), "-o", str(output)]) == 0
        capsys.readouterr()

        written = sorted(pretty_midi.PrettyMIDI(str(output)).instruments[0].notes, key=lambda n: n.start)
        assert len(written) == 4
        for previous, current in zip(written, written[1:]):
            assert current.start >= previous.end - 1e-6

    def test_writes_a_json_report(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path, midi_input: Path
    ) -> None:
        report_path = tmp_path / "melody.json"
        exit_code = main(
            [
                "extract-melody",
                str(midi_input),
                "-o",
                str(tmp_path / "melody.mid"),
                "--json",
                str(report_path),
            ]
        )
        capsys.readouterr()

        assert exit_code == 0
        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert report["strategy"] == "continuity"
        assert report["input_note_count"] == 4
        assert report["output_note_count"] == 4
        assert report["polyphony_max"] == 1
        assert len(report["notes"]) == 4
        assert report["total_group_count"] == 4
        assert report["selected_group_count"] == 4
        assert report["skipped_group_count"] == 0

    def test_reports_skipped_groups(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        """A group holding only accompaniment is skipped and reported as such."""
        notes = [
            NoteEvent(pitch=72, start=0.0, duration=0.5, velocity=90),
            NoteEvent(pitch=36, start=0.0, duration=0.5, velocity=60),
            NoteEvent(pitch=43, start=0.5, duration=0.5, velocity=60),
            NoteEvent(pitch=74, start=1.0, duration=0.5, velocity=90),
        ]
        path = write_midi(notes, tmp_path / "sparse.mid")
        output = tmp_path / "melody.mid"
        report_path = tmp_path / "melody.json"

        exit_code = main(["extract-melody", str(path), "-o", str(output), "--json", str(report_path)])
        stdout = capsys.readouterr().out

        assert exit_code == 0
        assert "Onset groups" in stdout
        assert "Selected groups" in stdout
        assert "Skipped groups" in stdout

        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert report["total_group_count"] == 3
        assert report["selected_group_count"] == 2
        assert report["skipped_group_count"] == 1
        assert [note["pitch"] for note in report["notes"]] == [72, 74]

        written = sorted(pretty_midi.PrettyMIDI(str(output)).instruments[0].notes, key=lambda n: n.start)
        assert [note.pitch for note in written] == [72, 74]

    def test_baselines_report_no_skips(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        notes = [
            NoteEvent(pitch=72, start=0.0, duration=0.5, velocity=90),
            NoteEvent(pitch=43, start=0.5, duration=0.5, velocity=60),
            NoteEvent(pitch=74, start=1.0, duration=0.5, velocity=90),
        ]
        path = write_midi(notes, tmp_path / "sparse.mid")
        report_path = tmp_path / "highest.json"

        assert (
            main(
                [
                    "extract-melody",
                    str(path),
                    "--strategy",
                    "highest",
                    "-o",
                    str(tmp_path / "melody.mid"),
                    "--json",
                    str(report_path),
                ]
            )
            == 0
        )
        capsys.readouterr()

        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert report["skipped_group_count"] == 0
        assert report["selected_group_count"] == report["total_group_count"] == 3

    def test_everything_filtered_is_an_error(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        path = write_midi(
            [NoteEvent(pitch=60, start=0.0, duration=0.02, velocity=80)], tmp_path / "tiny.mid"
        )
        exit_code = main(["extract-melody", str(path), "-o", str(tmp_path / "melody.mid")])

        assert exit_code == 1
        assert "no notes" in capsys.readouterr().err

    def test_missing_input_file(self, capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
        exit_code = main(["extract-melody", str(tmp_path / "ghost.mid")])

        assert exit_code == 1
        assert "not found" in capsys.readouterr().err

    def test_invalid_strategy_is_rejected(self, midi_input: Path) -> None:
        with pytest.raises(SystemExit) as excinfo:
            main(["extract-melody", str(midi_input), "--strategy", "middle"])
        assert excinfo.value.code == 2


class TestHarmonicaConversion:
    """map-melody / convert: melody -> PlayableNote[] -> score.json."""

    def test_map_melody_from_a_midi_file(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path, midi_input: Path
    ) -> None:
        output = tmp_path / "melody.score.json"
        exit_code = main(["map-melody", str(midi_input), "-o", str(output)])
        stdout = capsys.readouterr().out

        assert exit_code == 0
        assert "Notes" in stdout and "Playable" in stdout
        assert output.is_file()

        document = json.loads(output.read_text(encoding="utf-8"))
        assert document["profile"] == "delta_harmonica"
        assert document["lanes"] == ["Z", "X", "C", "V", "B", "N", "M", ","]
        assert len(document["notes"]) == 4
        assert [note["id"] for note in document["notes"]] == [0, 1, 2, 3]

    def test_default_output_name(self, capsys: pytest.CaptureFixture[str], midi_input: Path) -> None:
        assert main(["map-melody", str(midi_input)]) == 0
        capsys.readouterr()
        assert (midi_input.parent / "song.score.json").is_file()

    def test_every_note_is_fully_expressible(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path, midi_input: Path
    ) -> None:
        output = tmp_path / "score.json"
        assert main(["map-melody", str(midi_input), "-o", str(output)]) == 0
        capsys.readouterr()

        document = json.loads(output.read_text(encoding="utf-8"))
        for note in document["notes"]:
            assert set(note) == {
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
            assert note["key"] in document["lanes"]
            assert isinstance(note["semitone"], bool)

    def test_manual_transpose_is_applied(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path, midi_input: Path
    ) -> None:
        output = tmp_path / "score.json"
        assert main(["map-melody", str(midi_input), "--transpose", "-2", "-o", str(output)]) == 0
        capsys.readouterr()

        document = json.loads(output.read_text(encoding="utf-8"))
        assert document["conversion"] == {
            "transpose": -2,
            "auto_transpose": False,
            "range_mode": "octave_fold",
        }
        assert document["report"]["original_pitch_min"] == 60
        assert document["report"]["final_pitch_min"] == 58

    def test_drop_mode_is_reported(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        notes = [
            NoteEvent(pitch=84, start=0.0, duration=0.5, velocity=80),
            NoteEvent(pitch=90, start=0.5, duration=0.5, velocity=80),
        ]
        path = write_midi(notes, tmp_path / "sparse.mid")
        output = tmp_path / "score.json"

        exit_code = main(
            ["map-melody", str(path), "--transpose", "0", "--range", "drop", "--verbose", "-o", str(output)]
        )
        stdout = capsys.readouterr().out

        assert exit_code == 0
        assert "Dropped" in stdout or "dropped" in stdout
        document = json.loads(output.read_text(encoding="utf-8"))
        assert document["report"]["dropped_count"] == 1
        assert document["report"]["playable_ratio"] == pytest.approx(0.5)
        assert len(document["notes"]) == 1

    def test_map_melody_is_deterministic(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path, midi_input: Path
    ) -> None:
        first = tmp_path / "a.score.json"
        second = tmp_path / "b.score.json"

        assert main(["map-melody", str(midi_input), "-o", str(first)]) == 0
        assert main(["map-melody", str(midi_input), "-o", str(second)]) == 0
        capsys.readouterr()

        assert first.read_bytes() == second.read_bytes()

    def test_convert_rejects_a_midi_input(
        self, capsys: pytest.CaptureFixture[str], midi_input: Path
    ) -> None:
        exit_code = main(["convert", str(midi_input)])
        error = capsys.readouterr().err

        assert exit_code == 1
        assert "already a MIDI file" in error
        assert "map-melody" in error

    def test_convert_runs_the_whole_pipeline_from_audio(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path, tone_wav: Path, use_backend: Any
    ) -> None:
        use_backend()
        output = tmp_path / "song.score.json"

        exit_code = main(["convert", str(tone_wav), "-o", str(output)])
        stdout = capsys.readouterr().out

        assert exit_code == 0
        assert "Converting" in stdout and "tone.wav" in stdout
        assert "Writing score JSON..." in stdout
        assert output.is_file()

        document = json.loads(output.read_text(encoding="utf-8"))
        assert document["profile"] == "delta_harmonica"
        assert len(document["notes"]) == 4
        assert document["report"]["playable_ratio"] == 1.0

    def test_invalid_transpose_is_rejected(self, midi_input: Path) -> None:
        with pytest.raises(SystemExit) as excinfo:
            main(["map-melody", str(midi_input), "--transpose", "20"])
        assert excinfo.value.code == 2

    def test_invalid_range_mode_is_rejected(self, midi_input: Path) -> None:
        with pytest.raises(SystemExit) as excinfo:
            main(["map-melody", str(midi_input), "--range", "squash"])
        assert excinfo.value.code == 2

    def test_missing_profile_is_reported(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path, midi_input: Path
    ) -> None:
        exit_code = main(["map-melody", str(midi_input), "--profile", str(tmp_path / "nope.json")])

        assert exit_code == 1
        assert "ERROR:" in capsys.readouterr().err

    def test_everything_dropped_is_an_error(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        profile = tmp_path / "narrow.json"
        profile.write_text(
            json.dumps(
                {
                    "name": "narrow",
                    "base_pitch": 60,
                    "base_keys": ["Z"],
                    "natural_intervals": [0],
                    "octave_offsets": [0],
                    "semitone_symbol": "#",
                    "octave_up_symbol": "↑",
                    "octave_down_symbol": "↓",
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        path = write_midi(
            [NoteEvent(pitch=66, start=0.0, duration=0.5, velocity=80)], tmp_path / "tiny.mid"
        )

        exit_code = main(
            [
                "map-melody",
                str(path),
                "--transpose",
                "0",
                "--range",
                "drop",
                "--profile",
                str(profile),
                "-o",
                str(tmp_path / "s.json"),
            ]
        )
        error = capsys.readouterr().err

        assert exit_code == 1
        assert "dropped" in error


class TestLayoutScore:
    """layout-score: score.json -> static layout (debug JSON)."""

    def test_lays_out_a_score_document(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path, midi_input: Path
    ) -> None:
        score = tmp_path / "song.score.json"
        layout_output = tmp_path / "song.layout.json"

        assert main(["map-melody", str(midi_input), "-o", str(score)]) == 0
        capsys.readouterr()
        exit_code = main(["layout-score", str(score), "--section-duration", "6", "-o", str(layout_output)])
        stdout = capsys.readouterr().out

        assert exit_code == 0
        assert "Sections" in stdout
        assert "Fragments" in stdout
        assert "Canvas" in stdout
        assert layout_output.is_file()

        document = json.loads(layout_output.read_text(encoding="utf-8"))
        assert document["version"] == 1
        assert document["lanes"] == ["Z", "X", "C", "V", "B", "N", "M", ","]
        assert len(document["sections"]) >= 1
        assert all(len(section["grid"]) > 0 for section in document["sections"])

        fragments = [note for section in document["sections"] for note in section["notes"]]
        assert {fragment["source_note_id"] for fragment in fragments} == set(range(4))

    def test_default_output_name(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path, midi_input: Path
    ) -> None:
        score = tmp_path / "song.score.json"
        assert main(["map-melody", str(midi_input), "-o", str(score)]) == 0
        capsys.readouterr()

        assert main(["layout-score", str(score)]) == 0
        capsys.readouterr()
        assert (tmp_path / "song.layout.json").is_file()

    def test_section_duration_is_configurable(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path, midi_input: Path
    ) -> None:
        score = tmp_path / "song.score.json"
        assert main(["map-melody", str(midi_input), "-o", str(score)]) == 0
        capsys.readouterr()

        short = tmp_path / "short.layout.json"
        long = tmp_path / "long.layout.json"
        assert main(["layout-score", str(score), "--section-duration", "2", "-o", str(short)]) == 0
        assert main(["layout-score", str(score), "--section-duration", "12", "-o", str(long)]) == 0
        capsys.readouterr()

        short_document = json.loads(short.read_text(encoding="utf-8"))
        long_document = json.loads(long.read_text(encoding="utf-8"))
        assert len(short_document["sections"]) > len(long_document["sections"])

    def test_every_fragment_is_inside_its_section(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path, midi_input: Path
    ) -> None:
        score = tmp_path / "song.score.json"
        layout_output = tmp_path / "song.layout.json"
        assert main(["map-melody", str(midi_input), "-o", str(score)]) == 0
        assert main(["layout-score", str(score), "-o", str(layout_output)]) == 0
        capsys.readouterr()

        document = json.loads(layout_output.read_text(encoding="utf-8"))
        canvas_width = document["canvas"]["width"]
        for section in document["sections"]:
            for note in section["notes"]:
                assert note["x"] >= document["config"]["left_label_width"] - 1e-9
                assert note["y"] >= section["y"] - 1e-9
                assert note["y"] + note["height"] <= section["y"] + section["height"] + 1e-9
                assert note["x"] + note["width"] <= canvas_width + 1e-9

    def test_missing_score_file(self, capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
        exit_code = main(["layout-score", str(tmp_path / "nope.score.json")])
        error = capsys.readouterr().err

        assert exit_code == 1
        assert "ERROR:" in error

    def test_invalid_section_duration(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path, midi_input: Path
    ) -> None:
        score = tmp_path / "song.score.json"
        assert main(["map-melody", str(midi_input), "-o", str(score)]) == 0
        capsys.readouterr()

        exit_code = main(["layout-score", str(score), "--section-duration", "0"])
        error = capsys.readouterr().err

        assert exit_code == 1
        assert "ERROR:" in error


class TestRenderScore:
    """render-score: score.json -> PNG."""

    def test_renders_a_png(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path, midi_input: Path
    ) -> None:
        score = tmp_path / "song.score.json"
        png = tmp_path / "song.png"
        assert main(["map-melody", str(midi_input), "-o", str(score)]) == 0
        capsys.readouterr()

        exit_code = main(["render-score", str(score), "--output", str(png)])
        stdout = capsys.readouterr().out

        assert exit_code == 0
        assert "Canvas" in stdout and "Output" in stdout
        assert png.is_file()

        from PIL import Image

        with Image.open(png) as image:
            # the default style is practice -> 2400 px canvas
            assert image.size[0] == 2400
            assert image.size[1] > 0

    def test_default_output_name(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path, midi_input: Path
    ) -> None:
        score = tmp_path / "song.score.json"
        assert main(["map-melody", str(midi_input), "-o", str(score)]) == 0
        capsys.readouterr()

        assert main(["render-score", str(score)]) == 0
        capsys.readouterr()
        assert (tmp_path / "song.png").is_file()

    def test_missing_score_file(self, capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
        exit_code = main(["render-score", str(tmp_path / "ghost.score.json")])
        assert exit_code == 1
        assert "ERROR:" in capsys.readouterr().err


class TestConvertToPng:
    """convert / map-melody with a .png output run the whole pipeline."""

    def test_convert_renders_a_png_from_audio(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path, tone_wav: Path, use_backend: Any
    ) -> None:
        use_backend()
        png = tmp_path / "song_score.png"

        exit_code = main(["convert", str(tone_wav), "-o", str(png)])
        stdout = capsys.readouterr().out

        assert exit_code == 0
        assert "Saved score:" in stdout
        assert png.is_file()

        from PIL import Image

        with Image.open(png) as image:
            assert image.format == "PNG"
            assert image.size[0] == 2400  # default style: practice

    def test_map_melody_renders_a_png_from_midi(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path, midi_input: Path
    ) -> None:
        png = tmp_path / "song.png"

        exit_code = main(["map-melody", str(midi_input), "-o", str(png)])
        capsys.readouterr()

        assert exit_code == 0
        assert png.is_file()

    def test_json_output_still_writes_score_json(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path, midi_input: Path
    ) -> None:
        output = tmp_path / "song.score.json"
        assert main(["map-melody", str(midi_input), "-o", str(output)]) == 0
        capsys.readouterr()

        document = json.loads(output.read_text(encoding="utf-8"))
        assert document["version"] == 1

    def test_section_duration_reaches_the_render(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path, midi_input: Path
    ) -> None:
        from PIL import Image

        short = tmp_path / "short.png"
        assert main(["map-melody", str(midi_input), "--section-duration", "2", "-o", str(short)]) == 0
        capsys.readouterr()

        # A 2 s section row is 1/3 of the height of the default 6 s rows.
        long = tmp_path / "long.png"
        assert main(["map-melody", str(midi_input), "--section-duration", "6", "-o", str(long)]) == 0
        capsys.readouterr()

        with Image.open(short) as a, Image.open(long) as b:
            assert a.size[1] > b.size[1]


class TestTimeRangeOptions:
    """--start / --end across convert, map-melody and extract-melody."""

    def _midi(self, tmp_path: Path) -> Path:
        notes = [
            NoteEvent(pitch=60, start=0.0, duration=1.0, velocity=80),
            NoteEvent(pitch=64, start=3.0, duration=4.0, velocity=80),  # spans 5..8
            NoteEvent(pitch=67, start=7.0, duration=1.0, velocity=80),
            NoteEvent(pitch=72, start=11.0, duration=0.5, velocity=80),
        ]
        return write_midi(notes, tmp_path / "phrase.mid")

    def test_convert_to_score_json_with_a_segment(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        source = self._midi(tmp_path)
        output = tmp_path / "phrase.score.json"

        exit_code = main(["map-melody", str(source), "--start", "2", "--end", "8", "-o", str(output)])
        stdout = capsys.readouterr().out

        assert exit_code == 0
        assert "Segment" in stdout and "00:02" in stdout and "00:08" in stdout
        document = json.loads(output.read_text(encoding="utf-8"))
        assert document["source_segment"] == {"start": 2.0, "end": 8.0}
        assert document["duration"] == pytest.approx(6.0)
        # The timeline is re-based: the note at 3 s becomes 1 s in the segment.
        starts = [note["start"] for note in document["notes"]]
        assert starts[0] == pytest.approx(1.0)
        assert all(note["start"] >= 0.0 for note in document["notes"])
        assert all(note["start"] + note["duration"] <= 6.0 + 1e-9 for note in document["notes"])

    def test_convert_to_png_with_a_segment(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        source = self._midi(tmp_path)
        output = tmp_path / "phrase.png"

        exit_code = main(["map-melody", str(source), "--start", "00:02", "--end", "00:08", "-o", str(output)])
        capsys.readouterr()

        assert exit_code == 0
        from PIL import Image

        with Image.open(output) as image:
            assert image.format == "PNG"

    def test_convert_time_formats(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        source = self._midi(tmp_path)
        output = tmp_path / "phrase.score.json"

        exit_code = main(["map-melody", str(source), "--start", "00:00:02", "--end", "00:08", "-o", str(output)])
        capsys.readouterr()

        assert exit_code == 0
        document = json.loads(output.read_text(encoding="utf-8"))
        assert document["source_segment"] == {"start": 2.0, "end": 8.0}

    def test_png_header_shows_the_segment(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        from PIL import Image

        from app.export.score_loader import load_score_document
        from app.score.renderer_png import render_score
        from app.score.styles import get_preset

        source = self._midi(tmp_path)
        score = tmp_path / "phrase.score.json"
        assert main(["map-melody", str(source), "--start", "00:00:02", "--end", "00:08", "-o", str(score)]) == 0
        capsys.readouterr()

        document = load_score_document(score)
        assert document.source_segment == (2.0, 8.0)

        preset = get_preset("compact")
        layout = build_layout(
            document.notes,
            lanes=document.lanes,
            config=preset.layout,
            header=LayoutHeader(segment=document.source_segment),
            symbols=document.symbols,
        )

        class Sink:
            def __init__(self) -> None:
                self.texts: list[str] = []

            def rectangle(self, xy, **kwargs) -> None: ...
            def rounded_rectangle(self, xy, **kwargs) -> None: ...
            def line(self, xy, **kwargs) -> None: ...
            def text(self, xy, text, **kwargs) -> None:
                self.texts.append(text)
            def text_length(self, text, font=None) -> float:
                return len(text) * 8.0

        sink = Sink()
        render_score(layout, style=preset.style, config=preset.layout, sink_factory=lambda _img: sink)
        joined = " | ".join(sink.texts)
        assert "Segment: 00:02–00:08" in joined
        # Section titles restart at zero.
        assert "00:00–00:06" in joined
        assert "00:08" not in joined.replace("Segment: 00:02–00:08", "")

    def test_end_beyond_the_media_is_clamped_with_a_warning(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        source = self._midi(tmp_path)
        output = tmp_path / "phrase.score.json"

        exit_code = main(["map-melody", str(source), "--start", "5", "--end", "999", "-o", str(output)])
        stdout = capsys.readouterr().out

        assert exit_code == 0
        assert "clamped" in stdout
        document = json.loads(output.read_text(encoding="utf-8"))
        assert document["source_segment"]["end"] == pytest.approx(11.5)

    def test_extract_melody_supports_a_segment(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        source = self._midi(tmp_path)
        output = tmp_path / "phrase_melody.mid"

        exit_code = main(
            ["extract-melody", str(source), "--strategy", "highest", "--start", "2", "--end", "8", "-o", str(output)]
        )
        stdout = capsys.readouterr().out

        assert exit_code == 0
        assert "Segment" in stdout
        melody_notes = pretty_midi.PrettyMIDI(str(output)).instruments[0].notes
        assert melody_notes
        assert all(note.start >= 0.0 for note in melody_notes)

    def test_no_options_keep_the_old_behaviour(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        source = self._midi(tmp_path)
        output = tmp_path / "full.score.json"

        exit_code = main(["map-melody", str(source), "-o", str(output)])
        capsys.readouterr()

        assert exit_code == 0
        document = json.loads(output.read_text(encoding="utf-8"))
        assert document["source_segment"] is None

    @pytest.mark.parametrize(
        ("options", "message"),
        [
            (["--start", "abc"], "invalid time"),
            (["--end", "xyz"], "invalid time"),
            (["--start", "-1"], "start must be >= 0"),
            (["--start", "8", "--end", "2"], "greater than start"),
            (["--start", "999"], "beyond the source duration"),
        ],
    )
    def test_invalid_segment_options(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path, options: list[str], message: str
    ) -> None:
        source = self._midi(tmp_path)
        exit_code = main(["map-melody", str(source), *options, "-o", str(tmp_path / "x.score.json")])

        assert exit_code == 1
        assert message in capsys.readouterr().err

    def test_segment_with_no_notes_is_an_error(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        source = self._midi(tmp_path)  # first note starts at 0 and ends at 1
        exit_code = main(["map-melody", str(source), "--start", "9.5", "--end", "10.5", "-o", str(tmp_path / "x.score.json")])

        assert exit_code == 1
        assert "no notes" in capsys.readouterr().err

    def test_convert_audio_with_a_segment_and_style(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path, tone_wav: Path, use_backend: Any
    ) -> None:
        use_backend()
        output = tmp_path / "chorus.png"

        exit_code = main(
            ["convert", str(tone_wav), "--start", "0", "--end", "2.5", "--style", "practice", "-o", str(output)]
        )
        capsys.readouterr()

        assert exit_code == 0
        from PIL import Image

        with Image.open(output) as image:
            # practice preset canvas
            assert image.size[0] == 2400


class TestErrorHandling:
    def test_missing_input_file(self, capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
        exit_code = main(["analyze", str(tmp_path / "ghost.wav")])
        captured = capsys.readouterr()

        assert exit_code == 1
        assert captured.err.startswith("ERROR: ")
        assert "not found" in captured.err
        assert "Traceback" not in captured.err

    def test_unsupported_input_type(self, capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
        path = tmp_path / "notes.txt"
        path.write_text("hello")
        exit_code = main(["analyze", str(path)])

        assert exit_code == 1
        stderr = capsys.readouterr().err
        assert "Unsupported input format" in stderr
        assert ".txt" in stderr and "Supported" in stderr

    def test_corrupt_audio(self, capsys: pytest.CaptureFixture[str], corrupt_wav: Path) -> None:
        exit_code = main(["analyze", str(corrupt_wav)])

        assert exit_code == 1
        assert capsys.readouterr().err.startswith("ERROR: ")

    def test_unexpected_errors_hide_the_traceback(
        self, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, tone_wav: Path
    ) -> None:
        def boom(name: str, **kwargs: object) -> None:
            raise ValueError("kaboom")

        monkeypatch.setattr("app.cli.create_transcriber", boom)
        exit_code = main(["analyze", str(tone_wav)])
        captured = capsys.readouterr()

        assert exit_code == 1
        assert "unexpected failure" in captured.err
        assert "Traceback" not in captured.err
        assert "--debug" in captured.err

    def test_debug_flag_shows_the_traceback(
        self, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, tone_wav: Path
    ) -> None:
        def boom(name: str, **kwargs: object) -> None:
            raise ValueError("kaboom")

        monkeypatch.setattr("app.cli.create_transcriber", boom)
        exit_code = main(["analyze", str(tone_wav), "--debug"])
        captured = capsys.readouterr()

        assert exit_code == 1
        assert "Traceback" in captured.err

    def test_a_subcommand_is_required(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit) as excinfo:
            main([])
        assert excinfo.value.code == 2
