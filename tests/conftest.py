"""Shared test fixtures.

Everything here is generated locally with numpy + soundfile: the suite must never
need the network, a real recording, or a downloaded model.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from app.core.instrument_profile import InstrumentProfile
from app.core.models import NoteEvent, PlayableNote

MONO_SAMPLE_RATE = 22050
MP3_SAMPLE_RATE = 44100

#: The shipped Delta Harmonica configuration, used as the base for profile tests.
PROFILE_DATA: dict[str, object] = {
    "name": "delta_harmonica",
    "base_pitch": 60,
    "base_keys": ["Z", "X", "C", "V", "B", "N", "M", ","],
    "natural_intervals": [0, 2, 4, 5, 7, 9, 11, 12],
    "octave_offsets": [-1, 0, 1],
    "semitone_symbol": "#",
    "octave_up_symbol": "↑",
    "octave_down_symbol": "↓",
}


def make_profile(**overrides: object) -> InstrumentProfile:
    """Build a profile from the shipped config with individual fields replaced."""
    data = dict(PROFILE_DATA)
    data.update(overrides)
    return InstrumentProfile.from_dict(data)


@pytest.fixture(autouse=True)
def _isolated_user_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the user-settings directory at a per-test temp dir.

    Without this, a developer's real ``settings.json`` would leak preferences
    into the suite and make the tests order-dependent.
    """
    monkeypatch.delenv("APPDATA", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))


@pytest.fixture
def use_backend(monkeypatch: pytest.MonkeyPatch, sample_notes: list[NoteEvent]) -> Any:
    """Install a fake transcriber in place of the real backend."""

    def install(notes: list[NoteEvent] | None = None, error: Exception | None = None) -> FakeTranscriber:
        fake = FakeTranscriber(sample_notes if notes is None else notes, error)
        monkeypatch.setattr("app.cli.create_transcriber", lambda name, **kwargs: fake)
        return fake

    return install


@pytest.fixture
def sample_notes() -> list[NoteEvent]:
    """A four-note monophonic C major arpeggio."""
    return [
        NoteEvent(pitch=60, start=0.0, duration=0.5, velocity=80, confidence=0.9),
        NoteEvent(pitch=64, start=0.5, duration=0.5, velocity=90, confidence=0.8),
        NoteEvent(pitch=67, start=1.0, duration=0.5, velocity=100, confidence=0.7),
        NoteEvent(pitch=72, start=1.5, duration=1.0, velocity=110, confidence=0.6),
    ]


@pytest.fixture(scope="session")
def profile() -> InstrumentProfile:
    return InstrumentProfile.default()


LANES = ("Z", "X", "C", "V", "B", "N", "M", ",")


def playable_note(
    pitch: int,
    lane: int,
    start: float,
    duration: float,
    *,
    source_pitch: int | None = None,
    key_label: str | None = None,
    semitone: bool = False,
    octave_offset: int = 0,
    confidence: float | None = None,
) -> PlayableNote:
    """Terse :class:`PlayableNote` constructor for layout tests."""
    return PlayableNote(
        source_pitch=pitch if source_pitch is None else source_pitch,
        pitch=pitch,
        start=start,
        duration=duration,
        lane=lane,
        key_label=key_label if key_label is not None else LANES[lane],
        semitone=semitone,
        octave_offset=octave_offset,
        confidence=confidence,
    )

_MP3_SUPPORTED = "MP3" in sf.available_formats()


def decaying_tone(
    pitch: int = 69,
    seconds: float = 0.5,
    sample_rate: int = MONO_SAMPLE_RATE,
    amplitude: float = 0.6,
) -> np.ndarray:
    """A short piano-ish decaying tone (fundamental + one octave)."""
    time = np.arange(int(round(seconds * sample_rate)), dtype=np.float64) / sample_rate
    frequency = 440.0 * (2.0 ** ((pitch - 69) / 12.0))
    tone = np.sin(2.0 * np.pi * frequency * time) + 0.4 * np.sin(4.0 * np.pi * frequency * time)
    envelope = np.clip(time / 0.005, 0.0, 1.0) * np.exp(-time * 2.0)
    return (amplitude * tone * envelope).astype(np.float32)


def write_wav(path: Path, samples: np.ndarray, sample_rate: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), samples, sample_rate, subtype="PCM_16")
    return path


def make_note(
    pitch: int,
    start: float = 0.0,
    duration: float = 0.5,
    velocity: int = 80,
    confidence: float | None = None,
) -> NoteEvent:
    """Terse :class:`NoteEvent` constructor for melody/polyphony tests."""
    return NoteEvent(
        pitch=pitch, start=start, duration=duration, velocity=velocity, confidence=confidence
    )


class FakeTranscriber:
    """A :class:`~app.audio.transcriber.PianoTranscriber` that needs no model."""

    name = "fake"

    def __init__(self, notes: list[NoteEvent] | None = None, error: Exception | None = None) -> None:
        self.notes = list(notes or [])
        self.error = error
        self.calls: list[object] = []

    def transcribe(self, source: object) -> list[NoteEvent]:
        self.calls.append(source)
        if self.error is not None:
            raise self.error
        return list(self.notes)

    def describe(self) -> dict[str, object]:
        return {"backend": self.name, "device": "cpu", "reports_confidence": True}


# --------------------------------------------------------------------------- #
# audio fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
def tone_wav(tmp_path: Path) -> Path:
    """0.5 s mono tone at 22.05 kHz."""
    return write_wav(tmp_path / "tone.wav", decaying_tone(), MONO_SAMPLE_RATE)


@pytest.fixture
def stereo_wav(tmp_path: Path) -> Path:
    """0.5 s stereo tone whose right channel is silent."""
    mono = decaying_tone()
    stereo = np.stack([mono, np.zeros_like(mono)], axis=1)
    return write_wav(tmp_path / "stereo.wav", stereo, MONO_SAMPLE_RATE)


@pytest.fixture
def silent_wav(tmp_path: Path) -> Path:
    """0.2 s of digital silence."""
    return write_wav(tmp_path / "silence.wav", np.zeros(int(0.2 * MONO_SAMPLE_RATE), np.float32), MONO_SAMPLE_RATE)


@pytest.fixture
def corrupt_wav(tmp_path: Path) -> Path:
    """A ``.wav`` file that is not audio at all."""
    path = tmp_path / "corrupt.wav"
    path.write_bytes(b"this is definitely not a RIFF wave file")
    return path


@pytest.fixture
def mp3_file(tmp_path: Path) -> Path:
    """0.5 s MP3 (skipped when libsndfile has no MPEG support)."""
    if not _MP3_SUPPORTED:
        pytest.skip("libsndfile was built without MP3 support")
    mono = decaying_tone(seconds=0.5, sample_rate=MP3_SAMPLE_RATE)
    path = tmp_path / "tone.mp3"
    try:
        sf.write(str(path), np.stack([mono, mono], axis=1), MP3_SAMPLE_RATE, format="MP3")
    except Exception as exc:  # pragma: no cover - depends on the local libsndfile
        pytest.skip(f"MP3 encoding unavailable: {exc}")
    return path


# --------------------------------------------------------------------------- #
# notes fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
def sample_notes() -> list[NoteEvent]:
    """A four-note monophonic C major arpeggio."""
    return [
        NoteEvent(pitch=60, start=0.0, duration=0.5, velocity=80, confidence=0.9),
        NoteEvent(pitch=64, start=0.5, duration=0.5, velocity=90, confidence=0.8),
        NoteEvent(pitch=67, start=1.0, duration=0.5, velocity=100, confidence=0.7),
        NoteEvent(pitch=72, start=1.5, duration=1.0, velocity=110, confidence=0.6),
    ]


@pytest.fixture
def fake_transcriber_cls() -> type[FakeTranscriber]:
    return FakeTranscriber
