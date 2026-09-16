"""NoteEvent / AudioBuffer validation."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from app.core.models import AudioBuffer, NoteEvent, midi_pitch_to_name


class TestNoteEvent:
    def test_accepts_a_valid_note(self) -> None:
        note = NoteEvent(pitch=60, start=0.0, duration=0.5, velocity=100, confidence=0.75)
        assert note.pitch == 60
        assert note.end == pytest.approx(0.5)

    def test_accepts_missing_confidence(self) -> None:
        note = NoteEvent(pitch=60, start=1.0, duration=1.0, velocity=64)
        assert note.confidence is None

    @pytest.mark.parametrize("pitch", [-1, 128, 1000])
    def test_rejects_out_of_range_pitch(self, pitch: int) -> None:
        with pytest.raises(ValueError, match="pitch"):
            NoteEvent(pitch=pitch, start=0.0, duration=0.5, velocity=64)

    @pytest.mark.parametrize("pitch", [0, 127])
    def test_accepts_boundary_pitches(self, pitch: int) -> None:
        assert NoteEvent(pitch=pitch, start=0.0, duration=0.5, velocity=64).pitch == pitch

    def test_rejects_non_integer_pitch(self) -> None:
        with pytest.raises(TypeError, match="pitch"):
            NoteEvent(pitch="60", start=0.0, duration=0.5, velocity=64)  # type: ignore[arg-type]

    def test_rejects_boolean_pitch(self) -> None:
        with pytest.raises(TypeError, match="pitch"):
            NoteEvent(pitch=True, start=0.0, duration=0.5, velocity=64)  # type: ignore[arg-type]

    @pytest.mark.parametrize("start", [-0.001, -5.0])
    def test_rejects_negative_start(self, start: float) -> None:
        with pytest.raises(ValueError, match="start"):
            NoteEvent(pitch=60, start=start, duration=0.5, velocity=64)

    @pytest.mark.parametrize("duration", [0.0, -0.5])
    def test_rejects_non_positive_duration(self, duration: float) -> None:
        with pytest.raises(ValueError, match="duration"):
            NoteEvent(pitch=60, start=0.0, duration=duration, velocity=64)

    @pytest.mark.parametrize("velocity", [-1, 128])
    def test_rejects_out_of_range_velocity(self, velocity: int) -> None:
        with pytest.raises(ValueError, match="velocity"):
            NoteEvent(pitch=60, start=0.0, duration=0.5, velocity=velocity)

    @pytest.mark.parametrize("confidence", [-0.01, 1.01, 42.0])
    def test_rejects_out_of_range_confidence(self, confidence: float) -> None:
        with pytest.raises(ValueError, match="confidence"):
            NoteEvent(pitch=60, start=0.0, duration=0.5, velocity=64, confidence=confidence)

    @pytest.mark.parametrize("confidence", [0.0, 1.0])
    def test_accepts_boundary_confidence(self, confidence: float) -> None:
        note = NoteEvent(pitch=60, start=0.0, duration=0.5, velocity=64, confidence=confidence)
        assert note.confidence == confidence

    def test_rejects_non_finite_start(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            NoteEvent(pitch=60, start=float("nan"), duration=0.5, velocity=64)

    def test_is_immutable(self) -> None:
        note = NoteEvent(pitch=60, start=0.0, duration=0.5, velocity=64)
        with pytest.raises(FrozenInstanceError):
            note.pitch = 61  # type: ignore[misc]

    def test_from_seconds_computes_duration(self) -> None:
        note = NoteEvent.from_seconds(pitch=62, start=1.25, end=1.75, velocity=70)
        assert note.duration == pytest.approx(0.5)
        assert note.end == pytest.approx(1.75)

    def test_name_uses_scientific_pitch_notation(self) -> None:
        assert NoteEvent(pitch=60, start=0.0, duration=1.0, velocity=64).name == "C4"
        assert midi_pitch_to_name(69) == "A4"
        assert midi_pitch_to_name(0) == "C-1"

    def test_dict_round_trip(self) -> None:
        note = NoteEvent(pitch=60, start=0.5, duration=0.25, velocity=99, confidence=0.5)
        assert NoteEvent.from_dict(note.to_dict()) == note


class TestAudioBuffer:
    def test_accepts_mono_float_audio(self) -> None:
        buffer = AudioBuffer(samples=np.zeros(100, dtype=np.float32), sample_rate=1000, duration=0.1)
        assert buffer.n_samples == 100
        assert buffer.channels == 1

    def test_rejects_empty_samples(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            AudioBuffer(samples=np.zeros(0, dtype=np.float32), sample_rate=1000, duration=0.1)

    def test_rejects_multichannel_samples(self) -> None:
        with pytest.raises(ValueError, match="1-D"):
            AudioBuffer(samples=np.zeros((10, 2), dtype=np.float32), sample_rate=1000, duration=0.1)

    def test_rejects_integer_samples(self) -> None:
        with pytest.raises(TypeError, match="floating"):
            AudioBuffer(samples=np.zeros(10, dtype=np.int16), sample_rate=1000, duration=0.1)

    @pytest.mark.parametrize("sample_rate", [0, -44100])
    def test_rejects_invalid_sample_rate(self, sample_rate: int) -> None:
        with pytest.raises(ValueError, match="sample_rate"):
            AudioBuffer(samples=np.zeros(10, dtype=np.float32), sample_rate=sample_rate, duration=0.1)

    @pytest.mark.parametrize("duration", [0.0, -1.0])
    def test_rejects_invalid_duration(self, duration: float) -> None:
        with pytest.raises(ValueError, match="duration"):
            AudioBuffer(samples=np.zeros(10, dtype=np.float32), sample_rate=1000, duration=duration)


def test_note_event_invariants_hold_for_random_notes() -> None:
    """Sweep a wide range to make sure the model never yields illegal values."""
    rng = np.random.default_rng(20260916)
    for _ in range(500):
        note = NoteEvent(
            pitch=int(rng.integers(0, 128)),
            start=float(rng.uniform(0.0, 60.0)),
            duration=float(rng.uniform(0.001, 5.0)),
            velocity=int(rng.integers(0, 128)),
            confidence=float(rng.uniform(0.0, 1.0)),
        )
        assert 0 <= note.pitch <= 127
        assert note.start >= 0.0
        assert note.duration > 0.0
        assert 0 <= note.velocity <= 127
        assert 0.0 <= note.confidence <= 1.0
