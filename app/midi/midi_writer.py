"""``NoteEvent[]`` -> Standard MIDI File.

MIDI is an *intermediate* representation here: it exists so the transcription can
be listened to, inspected and debugged. The final product is a static score image,
never a MIDI file.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import pretty_midi

from app.core.errors import OutputNotWritableError
from app.core.models import NoteEvent

DEFAULT_TEMPO = 120.0
DEFAULT_RESOLUTION = 960
DEFAULT_PROGRAM = 0  # Acoustic Grand Piano
DEFAULT_TRACK_NAME = "transcription"


def sorted_notes(notes: Iterable[NoteEvent]) -> list[NoteEvent]:
    """Return the notes in a deterministic playback order."""
    ordered: list[NoteEvent] = []
    for note in notes:
        if not isinstance(note, NoteEvent):
            raise TypeError(f"expected NoteEvent instances, got {type(note).__name__}")
        ordered.append(note)
    ordered.sort(key=lambda note: (note.start, note.pitch, note.end))
    return ordered


def notes_to_pretty_midi(
    notes: Sequence[NoteEvent],
    *,
    tempo: float = DEFAULT_TEMPO,
    resolution: int = DEFAULT_RESOLUTION,
    program: int = DEFAULT_PROGRAM,
    track_name: str = DEFAULT_TRACK_NAME,
) -> pretty_midi.PrettyMIDI:
    """Build an in-memory :class:`pretty_midi.PrettyMIDI` holding one piano track."""
    ordered = sorted_notes(notes)
    midi = pretty_midi.PrettyMIDI(initial_tempo=float(tempo), resolution=int(resolution))
    instrument = pretty_midi.Instrument(program=int(program), name=track_name, is_drum=False)
    for note in ordered:
        instrument.notes.append(
            pretty_midi.Note(
                velocity=int(note.velocity),
                pitch=int(note.pitch),
                start=float(note.start),
                end=float(note.end),
            )
        )
    midi.instruments.append(instrument)
    return midi


def write_midi(
    notes: Sequence[NoteEvent],
    output_path: str | Path,
    *,
    tempo: float = DEFAULT_TEMPO,
    resolution: int = DEFAULT_RESOLUTION,
    program: int = DEFAULT_PROGRAM,
    track_name: str = DEFAULT_TRACK_NAME,
) -> Path:
    """Write *notes* to *output_path* and return the resolved path.

    Missing parent directories are created; any write failure is reported as
    :class:`~app.core.errors.OutputNotWritableError`.
    """
    target = Path(output_path)
    midi = notes_to_pretty_midi(
        notes,
        tempo=tempo,
        resolution=resolution,
        program=program,
        track_name=track_name,
    )

    try:
        parent = target.parent
        if parent and not parent.exists():
            parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.is_dir():
            raise IsADirectoryError(f"{target} is a directory")
        midi.write(str(target))
    except OSError as exc:
        raise OutputNotWritableError(f"cannot write MIDI to {target}: {exc}") from exc
    except Exception as exc:  # pretty_midi surfaces encoder problems as generic errors
        raise OutputNotWritableError(f"cannot write MIDI to {target}: {exc}") from exc

    return target


__all__ = [
    "DEFAULT_TEMPO",
    "DEFAULT_RESOLUTION",
    "DEFAULT_PROGRAM",
    "sorted_notes",
    "notes_to_pretty_midi",
    "write_midi",
]
