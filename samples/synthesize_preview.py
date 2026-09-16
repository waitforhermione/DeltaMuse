"""Render a MIDI file to a WAV preview for human listening checks.

Purely a *listening aid* for verifying that an extracted melody sounds right. It
is deliberately not part of the library, no test depends on it, and it uses
pretty_midi's bundled plain-sine synthesiser so that no soundfont or heavyweight
dependency is required.

Usage::

    python samples/synthesize_preview.py samples/melody_continuity.mid
    python samples/synthesize_preview.py samples/melody_highest.mid -o out.wav
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pretty_midi
import soundfile as sf

SAMPLE_RATE = 44100


def render(midi_path: str | Path, output_path: str | Path | None = None) -> Path:
    """Synthesise *midi_path* into a WAV file and return the written path."""
    source = Path(midi_path)
    if not source.is_file():
        raise FileNotFoundError(f"MIDI file not found: {source}")

    midi = pretty_midi.PrettyMIDI(str(source))
    audio = np.asarray(midi.synthesize(fs=SAMPLE_RATE), dtype=np.float32)

    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 0.0:
        audio = audio * (0.9 / peak)

    target = Path(output_path) if output_path is not None else source.with_suffix(".wav")
    target.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(target), audio, SAMPLE_RATE, subtype="PCM_16")
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render a MIDI file to a WAV preview.")
    parser.add_argument("midi", nargs="+", help="input .mid/.midi file(s)")
    parser.add_argument("-o", "--output", default=None, help="output WAV (only for a single input)")
    args = parser.parse_args(argv)

    if args.output is not None and len(args.midi) > 1:
        print("ERROR: --output can only be used with a single input file", file=sys.stderr)
        return 1

    for midi_path in args.midi:
        try:
            written = render(midi_path, args.output)
        except (OSError, ValueError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        print(f"wrote {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
