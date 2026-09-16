"""Render a MIDI file (e.g. a melody produced by extract-melody) to a WAV preview.

This is a *manual listening aid only* (Milestone 2 spec §30): it reuses the
additive "piano" synthesiser from :mod:`generate_sample_audio`, so no MIDI
synthesiser, soundfont or network is needed, and nothing in the test suite
depends on it. The result is a rough preview, not a polished piano rendering.

Usage::

    python samples/render_midi_preview.py samples/canon_in_d_melody.mid
    # -> samples/canon_in_d_melody.wav
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

from generate_sample_audio import SAMPLE_RATE, synthesize_note, write_stereo


def load_notes(path: Path) -> list[tuple[int, float, float]]:
    """Return ``(pitch, start_seconds, duration_seconds)`` for every note."""
    import pretty_midi

    midi = pretty_midi.PrettyMIDI(str(path))
    notes = [
        (note.pitch, float(note.start), float(note.end - note.start))
        for instrument in midi.instruments
        for note in instrument.notes
    ]
    notes.sort(key=lambda item: (item[1], item[0]))
    return notes


def render_seconds(
    notes: list[tuple[int, float, float]],
    tail: float = 1.5,
    sample_rate: int = SAMPLE_RATE,
    amplitude: float = 0.7,
) -> np.ndarray:
    """Mix ``(pitch, start s, duration s)`` notes into one mono buffer."""
    if not notes:
        raise SystemExit("ERROR: the MIDI file contains no notes")
    duration = max(start + length for _, start, length in notes) + tail
    buffer = np.zeros(int(round(duration * sample_rate)), dtype=np.float64)

    for pitch, start, length in notes:
        begin = int(round(start * sample_rate))
        rendered = synthesize_note(pitch, length, sample_rate, amplitude)
        end = min(begin + rendered.shape[0], buffer.shape[0])
        if end > begin:
            buffer[begin:end] += rendered[: end - begin]

    peak = float(np.max(np.abs(buffer))) or 1.0
    buffer *= 0.92 / peak

    fade = int(0.01 * sample_rate)
    if fade > 0:
        envelope = np.ones_like(buffer)
        envelope[:fade] = np.linspace(0.0, 1.0, fade)
        envelope[-fade:] = np.linspace(1.0, 0.0, fade)
        buffer *= envelope
    return buffer.astype(np.float32)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("midi", help="input .mid / .midi file")
    parser.add_argument("-o", "--output", default=None, help="output WAV path (default: <input>.wav)")
    parser.add_argument("--amplitude", type=float, default=0.7, help="per-note amplitude (default: 0.7)")
    args = parser.parse_args(argv)

    midi_path = Path(args.midi)
    if not midi_path.is_file():
        raise SystemExit(f"ERROR: {midi_path} does not exist")

    notes = load_notes(midi_path)
    mono = render_seconds(notes, amplitude=args.amplitude)

    output = Path(args.output) if args.output else midi_path.with_suffix(".wav")
    write_stereo(output, mono, SAMPLE_RATE)
    print(
        f"wrote {output}  ({mono.shape[0] / SAMPLE_RATE:.2f} s, {len(notes)} notes, "
        f"{SAMPLE_RATE} Hz stereo)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
