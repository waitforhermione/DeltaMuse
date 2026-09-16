"""Layer-agnostic data models.

The pipeline is strictly layered::

    Audio -> Transcription -> NoteEvent -> Melody Processing
          -> PlayableNote -> Score Layout -> Renderer

:class:`NoteEvent` is the single currency handed from the transcription layer to
every later stage, regardless of whether it came from audio or from a MIDI file.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

MIN_MIDI_PITCH = 0
MAX_MIDI_PITCH = 127
MAX_VELOCITY = 127

_PITCH_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")

_INT_TYPES = (int, np.integer)
_FLOAT_TYPES = (int, float, np.integer, np.floating)


def _require_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, _INT_TYPES):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    return int(value)


def _require_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, _FLOAT_TYPES):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}")
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return value


def midi_pitch_to_name(pitch: int) -> str:
    """Convert a MIDI pitch number to scientific pitch notation (60 -> ``C4``)."""
    pitch = _require_int(pitch, "pitch")
    if not MIN_MIDI_PITCH <= pitch <= MAX_MIDI_PITCH:
        raise ValueError(f"pitch must be within [{MIN_MIDI_PITCH}, {MAX_MIDI_PITCH}], got {pitch}")
    return f"{_PITCH_NAMES[pitch % 12]}{pitch // 12 - 1}"


@dataclass(frozen=True)
class NoteEvent:
    """A single note coming out of transcription or a MIDI file.

    Attributes
    ----------
    pitch:
        MIDI pitch number in ``0..127``.
    start:
        Onset in seconds from the beginning of the piece, ``>= 0``.
    duration:
        Sounding length in seconds, ``> 0``.
    velocity:
        MIDI velocity in ``0..127``.
    confidence:
        Audio transcription confidence in ``[0, 1]``. ``None`` when the note came
        from a MIDI file, or from a backend that does not report per-note
        confidence.
    """

    pitch: int
    start: float
    duration: float
    velocity: int
    confidence: float | None = None

    def __post_init__(self) -> None:
        pitch = _require_int(self.pitch, "pitch")
        velocity = _require_int(self.velocity, "velocity")
        start = _require_float(self.start, "start")
        duration = _require_float(self.duration, "duration")

        if not MIN_MIDI_PITCH <= pitch <= MAX_MIDI_PITCH:
            raise ValueError(f"pitch must be within [{MIN_MIDI_PITCH}, {MAX_MIDI_PITCH}], got {pitch}")
        if start < 0.0:
            raise ValueError(f"start must be >= 0, got {start}")
        if duration <= 0.0:
            raise ValueError(f"duration must be > 0, got {duration}")
        if not 0 <= velocity <= MAX_VELOCITY:
            raise ValueError(f"velocity must be within [0, {MAX_VELOCITY}], got {velocity}")

        confidence = self.confidence
        if confidence is not None:
            confidence = _require_float(confidence, "confidence")
            if not 0.0 <= confidence <= 1.0:
                raise ValueError(f"confidence must be within [0, 1], got {confidence}")

        object.__setattr__(self, "pitch", pitch)
        object.__setattr__(self, "velocity", velocity)
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "duration", duration)
        object.__setattr__(self, "confidence", confidence)

    # -- derived ---------------------------------------------------------- #
    @property
    def end(self) -> float:
        """Offset in seconds (``start + duration``)."""
        return self.start + self.duration

    @property
    def name(self) -> str:
        """Scientific pitch notation of :attr:`pitch`."""
        return midi_pitch_to_name(self.pitch)

    # -- construction helpers --------------------------------------------- #
    @classmethod
    def from_seconds(
        cls,
        *,
        pitch: int,
        start: float,
        end: float,
        velocity: int,
        confidence: float | None = None,
    ) -> "NoteEvent":
        """Build a note from an onset/offset pair, as most backends report it."""
        return cls(
            pitch=pitch,
            start=start,
            duration=float(end) - float(start),
            velocity=velocity,
            confidence=confidence,
        )

    # -- serialisation ----------------------------------------------------- #
    def to_dict(self) -> dict[str, Any]:
        return {
            "pitch": self.pitch,
            "start": self.start,
            "duration": self.duration,
            "velocity": self.velocity,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NoteEvent":
        return cls(
            pitch=data["pitch"],
            start=data["start"],
            duration=data["duration"],
            velocity=data["velocity"],
            confidence=data.get("confidence"),
        )


@dataclass(frozen=True)
class AudioBuffer:
    """Decoded, mono, float PCM audio ready for a transcription backend.

    Produced exclusively by :mod:`app.audio.loader` so that backends never have to
    decode the same file twice.
    """

    samples: np.ndarray
    sample_rate: int
    duration: float

    def __post_init__(self) -> None:
        samples = self.samples
        if not isinstance(samples, np.ndarray):
            raise TypeError(f"samples must be a numpy.ndarray, got {type(samples).__name__}")
        if samples.ndim != 1:
            raise ValueError(f"samples must be a 1-D mono array, got shape {samples.shape}")
        if samples.size == 0:
            raise ValueError("samples must not be empty")
        if not np.issubdtype(samples.dtype, np.floating):
            raise TypeError(f"samples must have a floating dtype, got {samples.dtype}")

        sample_rate = _require_int(self.sample_rate, "sample_rate")
        if sample_rate <= 0:
            raise ValueError(f"sample_rate must be > 0, got {sample_rate}")

        duration = _require_float(self.duration, "duration")
        if duration <= 0.0:
            raise ValueError(f"duration must be > 0, got {duration}")

        object.__setattr__(self, "sample_rate", sample_rate)
        object.__setattr__(self, "duration", duration)

    @property
    def n_samples(self) -> int:
        return int(self.samples.shape[0])

    @property
    def channels(self) -> int:
        return 1


@dataclass(frozen=True)
class MelodyNote:
    """A melody note travelling through the harmonica conversion stages.

    ``source_pitch`` is the pitch exactly as melody extraction produced it, before
    any transposition or range fitting, and it is *never* overwritten. That
    provenance survives every later stage and is what
    :class:`PlayableNote` reports back to the user.
    """

    source_pitch: int
    pitch: int
    start: float
    duration: float
    velocity: int
    confidence: float | None = None

    def __post_init__(self) -> None:
        source_pitch = _require_int(self.source_pitch, "source_pitch")
        pitch = _require_int(self.pitch, "pitch")
        velocity = _require_int(self.velocity, "velocity")
        start = _require_float(self.start, "start")
        duration = _require_float(self.duration, "duration")

        for name, value in (("source_pitch", source_pitch), ("pitch", pitch)):
            if not MIN_MIDI_PITCH <= value <= MAX_MIDI_PITCH:
                raise ValueError(f"{name} must be within [{MIN_MIDI_PITCH}, {MAX_MIDI_PITCH}], got {value}")
        if start < 0.0:
            raise ValueError(f"start must be >= 0, got {start}")
        if duration <= 0.0:
            raise ValueError(f"duration must be > 0, got {duration}")
        if not 0 <= velocity <= MAX_VELOCITY:
            raise ValueError(f"velocity must be within [0, {MAX_VELOCITY}], got {velocity}")

        confidence = self.confidence
        if confidence is not None:
            confidence = _require_float(confidence, "confidence")
            if not 0.0 <= confidence <= 1.0:
                raise ValueError(f"confidence must be within [0, 1], got {confidence}")

        object.__setattr__(self, "source_pitch", source_pitch)
        object.__setattr__(self, "pitch", pitch)
        object.__setattr__(self, "velocity", velocity)
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "duration", duration)
        object.__setattr__(self, "confidence", confidence)

    @property
    def end(self) -> float:
        return self.start + self.duration

    @property
    def name(self) -> str:
        return midi_pitch_to_name(self.pitch)

    @classmethod
    def from_note_event(cls, note: NoteEvent) -> "MelodyNote":
        """Start the conversion with ``source_pitch == pitch``."""
        return cls(
            source_pitch=note.pitch,
            pitch=note.pitch,
            start=note.start,
            duration=note.duration,
            velocity=note.velocity,
            confidence=note.confidence,
        )

    def with_pitch(self, pitch: int) -> "MelodyNote":
        """Return a copy with a new pitch, keeping ``source_pitch`` and timing."""
        return MelodyNote(
            source_pitch=self.source_pitch,
            pitch=pitch,
            start=self.start,
            duration=self.duration,
            velocity=self.velocity,
            confidence=self.confidence,
        )


@dataclass(frozen=True)
class PlayableNote:
    """A melody note mapped onto one concrete key of the instrument.

    ``pitch`` is the sounding pitch that will actually be played; ``source_pitch``
    is where the note came from in the original melody. ``semitone`` and
    ``octave_offset`` are the modifiers the player has to apply on top of the
    lane's key (``key_label``).
    """

    source_pitch: int
    pitch: int
    start: float
    duration: float
    lane: int
    key_label: str
    semitone: bool
    octave_offset: int
    confidence: float | None = None

    def __post_init__(self) -> None:
        source_pitch = _require_int(self.source_pitch, "source_pitch")
        pitch = _require_int(self.pitch, "pitch")
        lane = _require_int(self.lane, "lane")
        octave_offset = _require_int(self.octave_offset, "octave_offset")
        start = _require_float(self.start, "start")
        duration = _require_float(self.duration, "duration")

        for name, value in (("source_pitch", source_pitch), ("pitch", pitch)):
            if not MIN_MIDI_PITCH <= value <= MAX_MIDI_PITCH:
                raise ValueError(f"{name} must be within [{MIN_MIDI_PITCH}, {MAX_MIDI_PITCH}], got {value}")
        if lane < 0:
            raise ValueError(f"lane must be >= 0, got {lane}")
        if not isinstance(self.key_label, str) or not self.key_label:
            raise ValueError(f"key_label must be a non-empty string, got {self.key_label!r}")
        if not isinstance(self.semitone, bool):
            raise TypeError(f"semitone must be a bool, got {type(self.semitone).__name__}")
        if start < 0.0:
            raise ValueError(f"start must be >= 0, got {start}")
        if duration <= 0.0:
            raise ValueError(f"duration must be > 0, got {duration}")

        confidence = self.confidence
        if confidence is not None:
            confidence = _require_float(confidence, "confidence")
            if not 0.0 <= confidence <= 1.0:
                raise ValueError(f"confidence must be within [0, 1], got {confidence}")

        object.__setattr__(self, "source_pitch", source_pitch)
        object.__setattr__(self, "pitch", pitch)
        object.__setattr__(self, "lane", lane)
        object.__setattr__(self, "octave_offset", octave_offset)
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "duration", duration)
        object.__setattr__(self, "confidence", confidence)

    @property
    def end(self) -> float:
        return self.start + self.duration

    @property
    def name(self) -> str:
        return midi_pitch_to_name(self.pitch)

    @property
    def is_natural(self) -> bool:
        return not self.semitone and self.octave_offset == 0
