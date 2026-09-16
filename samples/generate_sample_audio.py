"""Generate the synthetic "piano" samples used for manual acceptance runs.

Real recordings are not redistributable, so this script synthesises a short
piano-like phrase (right-hand melody + left-hand bass) with an additive
synthesiser: harmonic stack, hammer attack, exponential decay. It is good enough
for the AMT model to find onsets and pitches, and it needs no network access.

Usage::

    python samples/generate_sample_audio.py
    # -> samples/sample.wav, samples/sample.mp3, samples/sample_short.wav
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

SAMPLE_RATE = 44100
#: (harmonic number, relative amplitude) - a rough piano spectrum.
_HARMONICS: tuple[tuple[int, float], ...] = (
    (1, 1.0),
    (2, 0.55),
    (3, 0.30),
    (4, 0.17),
    (5, 0.10),
    (6, 0.06),
    (7, 0.04),
    (8, 0.02),
)

#: 100 BPM -> 0.6 s per beat.
SECONDS_PER_BEAT = 0.6

#: (midi pitch, start beat, duration in beats)
MELODY: tuple[tuple[int, float, float], ...] = (
    (72, 0.0, 1.0),   # C5
    (76, 1.0, 1.0),   # E5
    (79, 2.0, 2.0),   # G5
    (76, 4.0, 1.0),   # E5
    (77, 5.0, 1.0),   # F5
    (76, 6.0, 1.0),   # E5
    (74, 7.0, 3.0),   # D5
    (72, 10.0, 1.0),  # C5
    (74, 11.0, 1.0),  # D5
    (76, 12.0, 2.0),  # E5
    (79, 14.0, 1.0),  # G5
    (84, 15.0, 3.0),  # C6
)

#: Left-hand accompaniment, half notes.
BASS: tuple[tuple[int, float, float], ...] = (
    (48, 0.0, 2.0),   # C3
    (43, 2.0, 2.0),   # G2
    (48, 4.0, 2.0),   # C3
    (45, 6.0, 2.0),   # A2
    (41, 8.0, 2.0),   # F2
    (43, 10.0, 2.0),  # G2
    (48, 12.0, 2.0),  # C3
    (43, 14.0, 2.0),  # G2
    (48, 16.0, 2.0),  # C3
)


def midi_to_hz(pitch: int) -> float:
    """Equal-tempered MIDI pitch -> frequency in Hz."""
    return 440.0 * (2.0 ** ((pitch - 69) / 12.0))


def synthesize_note(pitch: int, duration: float, sample_rate: int, amplitude: float) -> np.ndarray:
    """Render one piano-like note."""
    length = max(int(round(duration * sample_rate)), 1)
    time = np.arange(length, dtype=np.float64) / sample_rate
    frequency = midi_to_hz(pitch)

    tone = np.zeros(length, dtype=np.float64)
    total = 0.0
    for harmonic, weight in _HARMONICS:
        partial = frequency * harmonic
        if partial >= sample_rate * 0.45:  # stay below Nyquist
            break
        tone += weight * np.sin(2.0 * np.pi * partial * time + 0.03 * harmonic)
        total += weight
    tone /= max(total, 1e-9)

    attack = np.clip(time / 0.006, 0.0, 1.0)
    decay = np.exp(-time * (1.4 + frequency / 800.0))
    release = np.clip((duration - time) / 0.04, 0.0, 1.0)
    return amplitude * tone * attack * decay * release


def render_phrase(
    notes: tuple[tuple[int, float, float], ...],
    duration: float,
    sample_rate: int = SAMPLE_RATE,
    amplitude: float = 0.7,
) -> np.ndarray:
    """Mix a set of ``(pitch, start beat, length in beats)`` notes into a buffer."""
    buffer = np.zeros(int(round(duration * sample_rate)) + 1, dtype=np.float64)
    for pitch, start_beat, length_beats in notes:
        start = int(round(start_beat * SECONDS_PER_BEAT * sample_rate))
        rendered = synthesize_note(pitch, length_beats * SECONDS_PER_BEAT, sample_rate, amplitude)
        end = min(start + rendered.shape[0], buffer.shape[0])
        if end > start:
            buffer[start:end] += rendered[: end - start]
    return buffer


def build_sample(sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Right-hand melody over a left-hand bass line, ~11 seconds long."""
    total_beats = max(start + length for _, start, length in MELODY + BASS)
    duration = total_beats * SECONDS_PER_BEAT
    mix = render_phrase(MELODY, duration, sample_rate, amplitude=0.70)
    mix += render_phrase(BASS, duration, sample_rate, amplitude=0.55)

    peak = float(np.max(np.abs(mix))) or 1.0
    mix = mix * (0.92 / peak)
    # Fade the very edges so the file does not click.
    fade = int(0.01 * sample_rate)
    if fade > 0:
        envelope = np.ones_like(mix)
        envelope[:fade] = np.linspace(0.0, 1.0, fade)
        envelope[-fade:] = np.linspace(1.0, 0.0, fade)
        mix *= envelope
    return mix.astype(np.float32)


def build_short_sample(seconds: float = 3.5, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """A short C major arpeggio - the tiny integration-test input.

    The phrase ends half a second before the buffer does, so the model can also
    see the offset of the last note instead of running into the end of the file.
    """
    notes = ((60, 0.0, 1.0), (64, 1.0, 1.0), (67, 2.0, 1.0), (72, 3.0, 1.0), (76, 4.0, 1.0))
    mix = render_phrase(notes, seconds, sample_rate, amplitude=0.8)
    peak = float(np.max(np.abs(mix))) or 1.0
    return (mix * (0.9 / peak)).astype(np.float32)


def write_stereo(path: Path, mono: np.ndarray, sample_rate: int) -> None:
    """Write *mono* as a stereo file (channels differ by a hair of decorrelation)."""
    delay = 11  # ~0.25 ms
    left = mono
    right = np.concatenate([np.zeros(delay, dtype=mono.dtype), mono[:-delay]]) if delay < mono.size else mono
    stereo = np.stack([left, right], axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), stereo, sample_rate, subtype="PCM_16")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent),
        help="directory that receives sample.wav / sample.mp3 / sample_short.wav",
    )
    parser.add_argument("--sample-rate", type=int, default=SAMPLE_RATE)
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    sample = build_sample(args.sample_rate)

    wav_path = output_dir / "sample.wav"
    write_stereo(wav_path, sample, args.sample_rate)
    print(f"wrote {wav_path}  ({sample.shape[0] / args.sample_rate:.2f} s)")

    mp3_path = output_dir / "sample.mp3"
    try:
        mp3_data = np.stack([sample, sample], axis=1)
        sf.write(str(mp3_path), mp3_data, args.sample_rate, format="MP3")
        print(f"wrote {mp3_path}")
    except Exception as exc:  # MP3 encoding needs libsndfile >= 1.1
        print(f"skipped MP3 export: {exc}")

    short_path = output_dir / "sample_short.wav"
    short = build_short_sample(sample_rate=args.sample_rate)
    write_stereo(short_path, short, args.sample_rate)
    print(f"wrote {short_path}  ({short.shape[0] / args.sample_rate:.2f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
