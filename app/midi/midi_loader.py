"""Standard MIDI File -> ``NoteEvent[]``.

Transcription and harmonica conversion are fully decoupled, so the same
downstream pipeline must accept an existing ``.mid``/``.midi`` file as input:
``audio -> MIDI -> score`` and ``MIDI -> score`` are both valid entry points.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pretty_midi

from app.core.errors import MidiError
from app.core.models import NoteEvent

SUPPORTED_MIDI_EXTENSIONS = frozenset({".mid", ".midi"})


@dataclass(frozen=True)
class MidiTrackInfo:
    """Metadata of one MIDI track."""

    index: int
    name: str
    instrument: str
    is_drum: bool
    note_count: int


@dataclass(frozen=True)
class MidiDocument:
    """Everything the pipeline needs to know about an imported MIDI file."""

    path: Path
    notes: tuple[NoteEvent, ...]
    tracks: tuple[MidiTrackInfo, ...]
    duration: float
    tempo: float

    @property
    def note_count(self) -> int:
        return len(self.notes)


def _validate_midi_path(path: str | Path) -> Path:
    resolved = Path(path)
    if not resolved.exists():
        raise MidiError(f"MIDI file not found: {resolved}")
    if resolved.is_dir():
        raise MidiError(f"expected a MIDI file but got a directory: {resolved}")
    if resolved.suffix.lower() not in SUPPORTED_MIDI_EXTENSIONS:
        raise MidiError(
            f"unsupported MIDI extension {resolved.suffix or '<none>'!r}. "
            f"Supported: {', '.join(sorted(SUPPORTED_MIDI_EXTENSIONS))}"
        )
    if resolved.stat().st_size == 0:
        raise MidiError(f"MIDI file is empty: {resolved}")
    return resolved


def _initial_tempo(midi: pretty_midi.PrettyMIDI) -> float:
    try:
        _, tempi = midi.get_tempo_changes()
    except Exception:  # pragma: no cover - malformed tempo maps
        return 120.0
    return float(tempi[0]) if len(tempi) else 120.0


def _track_info(index: int, instrument: pretty_midi.Instrument) -> MidiTrackInfo:
    try:
        instrument_name = pretty_midi.program_to_instrument_name(instrument.program)
    except Exception:  # pragma: no cover - exotic program numbers
        instrument_name = f"program {instrument.program}"
    if instrument.is_drum:
        instrument_name = instrument_name or "Drums"
    return MidiTrackInfo(
        index=index,
        name=instrument.name or "",
        instrument=instrument_name,
        is_drum=bool(instrument.is_drum),
        note_count=len(instrument.notes),
    )


def load_midi(
    path: str | Path,
    *,
    track_index: int | None = None,
    ignore_drums: bool = True,
) -> MidiDocument:
    """Read a MIDI file into a :class:`MidiDocument`.

    Parameters
    ----------
    path:
        Input ``.mid`` / ``.midi`` file.
    track_index:
        Restrict the import to a single track. ``None`` merges all tracks.
    ignore_drums:
        Skip drum tracks when merging.
    """
    resolved = _validate_midi_path(path)

    try:
        midi = pretty_midi.PrettyMIDI(str(resolved))
    except Exception as exc:
        raise MidiError(f"could not read MIDI file {resolved.name}: {exc}") from exc

    tracks: list[pretty_midi.Instrument] = list(midi.instruments)
    if track_index is not None:
        if not 0 <= track_index < len(tracks):
            raise MidiError(
                f"track index {track_index} is out of range: {resolved.name} has {len(tracks)} track(s)"
            )
        selected = [(track_index, tracks[track_index])]
    else:
        selected = [
            (index, instrument)
            for index, instrument in enumerate(tracks)
            if not (ignore_drums and instrument.is_drum)
        ]

    notes: list[NoteEvent] = []
    for index, instrument in selected:
        for raw in instrument.notes:
            if raw.end <= raw.start:
                # Degenerate note (zero or negative length) - nothing to play.
                continue
            try:
                notes.append(
                    NoteEvent.from_seconds(
                        pitch=int(raw.pitch),
                        start=max(float(raw.start), 0.0),
                        end=float(raw.end),
                        velocity=int(raw.velocity),
                        confidence=None,
                    )
                )
            except (TypeError, ValueError):
                continue

    notes.sort(key=lambda note: (note.start, note.pitch, note.end))

    return MidiDocument(
        path=resolved,
        notes=tuple(notes),
        tracks=tuple(_track_info(index, instrument) for index, instrument in enumerate(tracks)),
        duration=float(midi.get_end_time()),
        tempo=_initial_tempo(midi),
    )


def load_midi_notes(path: str | Path, **kwargs: object) -> list[NoteEvent]:
    """Convenience wrapper returning only the imported notes."""
    return list(load_midi(path, **kwargs).notes)  # type: ignore[arg-type]


__all__ = [
    "SUPPORTED_MIDI_EXTENSIONS",
    "MidiTrackInfo",
    "MidiDocument",
    "load_midi",
    "load_midi_notes",
]
