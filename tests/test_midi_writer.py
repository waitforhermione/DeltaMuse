"""NoteEvent[] -> MIDI, verified with pretty_midi *and* mido."""

from __future__ import annotations

from pathlib import Path

import mido
import pretty_midi
import pytest

from app.core.errors import OutputNotWritableError
from app.core.models import NoteEvent
from app.midi.midi_writer import notes_to_pretty_midi, sorted_notes, write_midi


class TestInMemory:
    def test_builds_one_piano_track(self, sample_notes: list[NoteEvent]) -> None:
        midi = notes_to_pretty_midi(sample_notes)
        assert len(midi.instruments) == 1
        assert len(midi.instruments[0].notes) == len(sample_notes)
        assert not midi.instruments[0].is_drum

    def test_is_empty_safe(self) -> None:
        assert notes_to_pretty_midi([]).instruments[0].notes == []

    def test_notes_are_sorted_deterministically(self) -> None:
        unordered = [
            NoteEvent(pitch=67, start=1.0, duration=0.5, velocity=60),
            NoteEvent(pitch=60, start=0.0, duration=0.5, velocity=60),
            NoteEvent(pitch=64, start=0.0, duration=0.25, velocity=60),
        ]
        assert [(n.pitch, n.start) for n in sorted_notes(unordered)] == [(60, 0.0), (64, 0.0), (67, 1.0)]

    def test_rejects_objects_that_are_not_notes(self) -> None:
        with pytest.raises(TypeError, match="NoteEvent"):
            sorted_notes([("not", "a", "note")])  # type: ignore[list-item]


class TestWriting:
    def test_round_trips_through_pretty_midi(self, tmp_path: Path, sample_notes: list[NoteEvent]) -> None:
        path = write_midi(sample_notes, tmp_path / "out.mid")
        assert path.exists() and path.stat().st_size > 0

        reloaded = pretty_midi.PrettyMIDI(str(path))
        written = sorted(reloaded.instruments[0].notes, key=lambda note: note.start)

        assert len(written) == len(sample_notes)
        for original, restored in zip(sample_notes, written):
            assert restored.pitch == original.pitch
            assert restored.velocity == original.velocity
            assert restored.start == pytest.approx(original.start, abs=0.002)
            assert restored.end == pytest.approx(original.end, abs=0.002)

    def test_is_readable_by_an_independent_library(
        self, tmp_path: Path, sample_notes: list[NoteEvent]
    ) -> None:
        path = write_midi(sample_notes, tmp_path / "out.mid")

        midi = mido.MidiFile(str(path))
        note_ons = [
            message
            for track in midi.tracks
            for message in track
            if message.type == "note_on" and message.velocity > 0
        ]

        assert len(note_ons) == len(sample_notes)
        assert sorted(message.note for message in note_ons) == sorted(note.pitch for note in sample_notes)

    def test_creates_missing_parent_directories(self, tmp_path: Path, sample_notes: list[NoteEvent]) -> None:
        path = write_midi(sample_notes, tmp_path / "nested" / "deep" / "out.mid")
        assert path.exists()

    def test_respects_custom_tempo_and_resolution(self, tmp_path: Path, sample_notes: list[NoteEvent]) -> None:
        path = write_midi(sample_notes, tmp_path / "out.mid", tempo=90.0, resolution=480)
        midi = mido.MidiFile(str(path))
        assert midi.ticks_per_beat == 480

    def test_fails_on_a_directory_target(self, tmp_path: Path, sample_notes: list[NoteEvent]) -> None:
        with pytest.raises(OutputNotWritableError, match="cannot write MIDI"):
            write_midi(sample_notes, tmp_path)

    def test_fails_when_the_parent_is_a_file(self, tmp_path: Path, sample_notes: list[NoteEvent]) -> None:
        blocker = tmp_path / "blocker.txt"
        blocker.write_text("not a directory")
        with pytest.raises(OutputNotWritableError, match="cannot write MIDI"):
            write_midi(sample_notes, blocker / "out.mid")
