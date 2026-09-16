"""MIDI file -> NoteEvent[] (the "existing MIDI -> score" entry point)."""

from __future__ import annotations

from pathlib import Path

import pretty_midi
import pytest

from app.core.errors import MidiError
from app.core.models import NoteEvent
from app.midi.midi_loader import load_midi, load_midi_notes
from app.midi.midi_writer import write_midi


@pytest.fixture
def midi_path(tmp_path: Path, sample_notes: list[NoteEvent]) -> Path:
    return write_midi(sample_notes, tmp_path / "song.mid")


class TestLoading:
    def test_round_trips_written_notes(self, midi_path: Path, sample_notes: list[NoteEvent]) -> None:
        document = load_midi(midi_path)
        assert document.note_count == len(sample_notes)
        assert [note.pitch for note in document.notes] == [note.pitch for note in sample_notes]
        assert all(note.confidence is None for note in document.notes)

    def test_reports_track_metadata(self, midi_path: Path) -> None:
        document = load_midi(midi_path)
        assert len(document.tracks) == 1
        assert document.tracks[0].index == 0
        assert document.tracks[0].instrument == "Acoustic Grand Piano"
        assert document.tracks[0].note_count == 4

    def test_reports_duration_and_tempo(self, midi_path: Path) -> None:
        document = load_midi(midi_path)
        assert document.duration == pytest.approx(2.5, abs=0.01)
        assert document.tempo == pytest.approx(120.0)

    def test_load_midi_notes_returns_a_plain_list(self, midi_path: Path) -> None:
        notes = load_midi_notes(midi_path)
        assert isinstance(notes, list)
        assert all(isinstance(note, NoteEvent) for note in notes)

    def test_single_track_selection(self, tmp_path: Path) -> None:
        midi = pretty_midi.PrettyMIDI(initial_tempo=120.0)
        for pitch in (60, 72):
            instrument = pretty_midi.Instrument(program=0)
            instrument.notes.append(pretty_midi.Note(velocity=80, pitch=pitch, start=0.0, end=1.0))
            midi.instruments.append(instrument)
        path = tmp_path / "two_tracks.mid"
        midi.write(str(path))

        assert len(load_midi(path).notes) == 2
        assert [note.pitch for note in load_midi(path, track_index=1).notes] == [72]

    def test_drum_tracks_are_ignored_by_default(self, tmp_path: Path) -> None:
        midi = pretty_midi.PrettyMIDI(initial_tempo=120.0)
        piano = pretty_midi.Instrument(program=0)
        piano.notes.append(pretty_midi.Note(velocity=80, pitch=60, start=0.0, end=1.0))
        drums = pretty_midi.Instrument(program=0, is_drum=True)
        drums.notes.append(pretty_midi.Note(velocity=100, pitch=38, start=0.0, end=0.1))
        midi.instruments.extend([piano, drums])
        path = tmp_path / "drums.mid"
        midi.write(str(path))

        document = load_midi(path)
        assert [note.pitch for note in document.notes] == [60]
        assert len(document.tracks) == 2


class TestFailures:
    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(MidiError, match="not found"):
            load_midi(tmp_path / "nope.mid")

    def test_directory_instead_of_file(self, tmp_path: Path) -> None:
        with pytest.raises(MidiError, match="directory"):
            load_midi(tmp_path)

    def test_unsupported_extension(self, tmp_path: Path) -> None:
        path = tmp_path / "song.txt"
        path.write_text("nope")
        with pytest.raises(MidiError, match="unsupported MIDI extension"):
            load_midi(path)

    def test_empty_file(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.mid"
        path.touch()
        with pytest.raises(MidiError, match="empty"):
            load_midi(path)

    def test_corrupt_content(self, tmp_path: Path) -> None:
        path = tmp_path / "broken.mid"
        path.write_bytes(b"MThd\x00\x00\x00\x06 definitely not a real header")
        with pytest.raises(MidiError, match="could not read MIDI file"):
            load_midi(path)

    def test_track_index_out_of_range(self, midi_path: Path) -> None:
        with pytest.raises(MidiError, match="out of range"):
            load_midi(midi_path, track_index=7)
