"""Transcriber protocol, backend factory and the TransKun adapter.

No model is downloaded and no network access is used: the adapter is exercised
against a stub torch/model pair. A real end-to-end model run is available behind
the ``PIANO_SCORE_RUN_MODEL_TESTS=1`` environment flag.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from app.audio.transcriber import (
    BACKENDS,
    DEFAULT_BACKEND,
    PianoTranscriber,
    available_backends,
    create_transcriber,
    resolve_audio_buffer,
)
from app.audio.backends.transkun_backend import TransKunBackend
from app.core.errors import BackendUnavailableError, TranscriptionError
from app.core.models import AudioBuffer, NoteEvent

from conftest import MONO_SAMPLE_RATE, FakeTranscriber, decaying_tone

_RUN_MODEL_TESTS = os.environ.get("PIANO_SCORE_RUN_MODEL_TESTS") == "1"


# --------------------------------------------------------------------------- #
# stand-ins for torch and the model
# --------------------------------------------------------------------------- #
class _FakeTensor:
    def __init__(self, array: np.ndarray) -> None:
        self.array = array
        self.device: str | None = None

    def reshape(self, *shape: int) -> "_FakeTensor":
        return _FakeTensor(self.array.reshape(*shape))

    def to(self, device: str) -> "_FakeTensor":
        self.device = device
        return self


class _FakeTorch:
    def __init__(self) -> None:
        self.grad_enabled = True

    def from_numpy(self, array: np.ndarray) -> _FakeTensor:
        return _FakeTensor(np.asarray(array, dtype=np.float32))

    def is_grad_enabled(self) -> bool:
        return self.grad_enabled

    def set_grad_enabled(self, value: bool) -> None:
        self.grad_enabled = bool(value)


class _FakeEvent:
    def __init__(self, start: float, end: float, pitch: int, velocity: int) -> None:
        self.start = start
        self.end = end
        self.pitch = pitch
        self.velocity = velocity


class _FakeModel:
    def __init__(self, events: list[object], fs: int = 16000, error: Exception | None = None) -> None:
        self.fs = fs
        self.error = error
        self._events = events
        self.received: _FakeTensor | None = None
        self.kwargs: dict[str, object] | None = None

    def transcribe(self, tensor: _FakeTensor, **kwargs: object) -> list[object]:
        self.received = tensor
        self.kwargs = kwargs
        if self.error is not None:
            raise self.error
        return list(self._events)


def _backend_using(monkeypatch: pytest.MonkeyPatch, model: object, torch: object | None = None) -> TransKunBackend:
    """Attach a stub model to a backend so no checkpoint is ever loaded."""
    backend = TransKunBackend()
    fake_torch = torch if torch is not None else _FakeTorch()

    def fake_ensure_model(self: TransKunBackend) -> object:
        self._torch = fake_torch
        self._resolved_device = "cpu"
        self._model = model
        return model

    monkeypatch.setattr(TransKunBackend, "_ensure_model", fake_ensure_model)
    return backend


@pytest.fixture
def buffer() -> AudioBuffer:
    samples = decaying_tone(seconds=0.25, sample_rate=MONO_SAMPLE_RATE)
    return AudioBuffer(
        samples=samples, sample_rate=MONO_SAMPLE_RATE, duration=samples.shape[0] / MONO_SAMPLE_RATE
    )


# --------------------------------------------------------------------------- #
# protocol & factory
# --------------------------------------------------------------------------- #
class TestProtocolAndFactory:
    def test_fake_backend_satisfies_the_protocol(self) -> None:
        assert isinstance(FakeTranscriber(), PianoTranscriber)

    def test_transkun_backend_satisfies_the_protocol(self) -> None:
        assert isinstance(TransKunBackend(), PianoTranscriber)

    def test_transkun_is_the_default_backend(self) -> None:
        assert DEFAULT_BACKEND == "transkun"
        assert "transkun" in available_backends()

    def test_create_transcriber_rejects_unknown_names(self) -> None:
        with pytest.raises(BackendUnavailableError, match="unknown backend"):
            create_transcriber("does-not-exist")

    def test_create_transcriber_reports_missing_dependencies(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(BACKENDS, "ghost", "definitely_missing_module:Ghost")
        with pytest.raises(BackendUnavailableError, match="not installed"):
            create_transcriber("ghost")

    def test_create_transcriber_builds_the_transkun_adapter(self) -> None:
        backend = create_transcriber("transkun", device="cpu")
        assert isinstance(backend, TransKunBackend)
        assert backend.name == "transkun"

    def test_is_available_returns_a_boolean(self) -> None:
        assert isinstance(TransKunBackend.is_available(), bool)


class TestAudioBufferResolution:
    def test_passes_an_audio_buffer_through(self, buffer: AudioBuffer) -> None:
        assert resolve_audio_buffer(buffer) is buffer

    def test_loads_a_path(self, tone_wav: Path) -> None:
        resolved = resolve_audio_buffer(tone_wav)
        assert isinstance(resolved, AudioBuffer)
        assert resolved.sample_rate == MONO_SAMPLE_RATE

    def test_rejects_other_types(self) -> None:
        with pytest.raises(TypeError, match="AudioBuffer"):
            resolve_audio_buffer(1234)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# TransKun adapter
# --------------------------------------------------------------------------- #
class TestBackendConfiguration:
    def test_rejects_hop_not_smaller_than_segment(self) -> None:
        with pytest.raises(ValueError, match="smaller than"):
            TransKunBackend(segment_hop_size=20.0, segment_size=10.0)

    def test_rejects_unknown_device(self) -> None:
        with pytest.raises(ValueError, match="device"):
            TransKunBackend(device="tpu")

    def test_defaults_defer_to_the_checkpoint_config(self) -> None:
        backend = TransKunBackend()
        assert backend.segment_hop_size is None
        assert backend.segment_size is None
        assert backend.describe()["reports_confidence"] is False

    def test_missing_checkpoint_fails_cleanly(self, buffer: AudioBuffer) -> None:
        backend = TransKunBackend(weight_path="definitely/missing/2.0.pt")
        with pytest.raises(BackendUnavailableError, match="asset not found"):
            backend.transcribe(buffer)


class TestEventConversion:
    def test_converts_a_valid_event(self) -> None:
        note = TransKunBackend._to_note_event(_FakeEvent(0.5, 1.25, 64, 90))
        assert note == NoteEvent(pitch=64, start=0.5, duration=0.75, velocity=90, confidence=None)

    def test_skips_control_change_events(self) -> None:
        assert TransKunBackend._to_note_event(_FakeEvent(0.0, 1.0, -64, 100)) is None

    def test_skips_degenerate_events(self) -> None:
        assert TransKunBackend._to_note_event(_FakeEvent(1.0, 1.0, 60, 64)) is None
        assert TransKunBackend._to_note_event(_FakeEvent(2.0, 1.0, 60, 64)) is None

    def test_skips_unusable_objects(self) -> None:
        assert TransKunBackend._to_note_event(object()) is None
        assert TransKunBackend._to_note_event(None) is None

    def test_clamps_velocity_into_the_midi_range(self) -> None:
        assert TransKunBackend._to_note_event(_FakeEvent(0.0, 1.0, 60, 0)).velocity == 1
        assert TransKunBackend._to_note_event(_FakeEvent(0.0, 1.0, 60, 999)).velocity == 127

    def test_clamps_negative_onsets(self) -> None:
        note = TransKunBackend._to_note_event(_FakeEvent(-0.05, 0.1, 60, 64))
        assert note is not None
        assert note.start == 0.0
        assert note.duration == pytest.approx(0.1)


class TestTranscribeWithStubModel:
    def test_returns_sorted_note_events(
        self, monkeypatch: pytest.MonkeyPatch, buffer: AudioBuffer
    ) -> None:
        model = _FakeModel([_FakeEvent(0.2, 0.4, 67, 80), _FakeEvent(0.0, 0.2, 60, 90)])
        backend = _backend_using(monkeypatch, model)

        notes = backend.transcribe(buffer)

        assert [note.pitch for note in notes] == [60, 67]
        assert all(isinstance(note, NoteEvent) for note in notes)
        assert [note.start for note in notes] == [0.0, 0.2]

    def test_resamples_to_the_model_sample_rate(
        self, monkeypatch: pytest.MonkeyPatch, buffer: AudioBuffer
    ) -> None:
        model = _FakeModel([], fs=16000)
        backend = _backend_using(monkeypatch, model)

        backend.transcribe(buffer)

        assert model.received is not None
        assert model.received.array.shape == (int(round(buffer.duration * 16000)), 1)
        assert model.received.device == "cpu"

    def test_passes_segment_overrides_to_the_model(
        self, monkeypatch: pytest.MonkeyPatch, buffer: AudioBuffer
    ) -> None:
        model = _FakeModel([])
        backend = TransKunBackend(segment_hop_size=4.0, segment_size=8.0)
        fake_torch = _FakeTorch()

        def fake_ensure_model(self: TransKunBackend) -> object:
            self._torch = fake_torch
            self._resolved_device = "cpu"
            self._model = model
            return model

        monkeypatch.setattr(TransKunBackend, "_ensure_model", fake_ensure_model)
        backend.transcribe(buffer)

        assert model.kwargs is not None
        assert model.kwargs["stepInSecond"] == 4.0
        assert model.kwargs["segmentSizeInSecond"] == 8.0
        assert model.kwargs["discardSecondHalf"] is False

    def test_gradients_are_restored_after_inference(
        self, monkeypatch: pytest.MonkeyPatch, buffer: AudioBuffer
    ) -> None:
        fake_torch = _FakeTorch()
        backend = _backend_using(monkeypatch, _FakeModel([]), torch=fake_torch)

        backend.transcribe(buffer)

        assert fake_torch.grad_enabled is True

    def test_model_failures_become_transcription_errors(
        self, monkeypatch: pytest.MonkeyPatch, buffer: AudioBuffer
    ) -> None:
        model = _FakeModel([], error=RuntimeError("CUDA out of memory"))
        backend = _backend_using(monkeypatch, model)

        with pytest.raises(TranscriptionError, match="TransKun failed"):
            backend.transcribe(buffer)

    def test_empty_model_output_is_not_an_error(
        self, monkeypatch: pytest.MonkeyPatch, buffer: AudioBuffer
    ) -> None:
        backend = _backend_using(monkeypatch, _FakeModel([]))
        assert backend.transcribe(buffer) == []

    def test_accepts_a_path_and_decodes_it(
        self, monkeypatch: pytest.MonkeyPatch, tone_wav: Path
    ) -> None:
        model = _FakeModel([_FakeEvent(0.0, 0.1, 60, 70)])
        backend = _backend_using(monkeypatch, model)

        notes = backend.transcribe(tone_wav)

        assert len(notes) == 1
        assert notes[0].pitch == 60


# --------------------------------------------------------------------------- #
# optional real-model integration test
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not _RUN_MODEL_TESTS, reason="set PIANO_SCORE_RUN_MODEL_TESTS=1 to run the real model")
@pytest.mark.skipif(not TransKunBackend.is_available(), reason="the transkun backend is not installed")
def test_real_transkun_transcribes_a_short_piano_clip(tmp_path: Path) -> None:
    """Real integration test: WAV -> TransKun -> NoteEvent[] (slow, opt-in).

    Uses the same piano-like synthesiser as the shipped sample generator, because
    a pure sine wave carries none of the spectral cues the AMT model was trained on.
    """
    import soundfile as sf

    from generate_sample_audio import build_short_sample

    sample_rate = 44100
    mix = build_short_sample(seconds=3.5, sample_rate=sample_rate)
    path = tmp_path / "arpeggio.wav"
    sf.write(str(path), mix, sample_rate)

    notes = TransKunBackend(device="cpu").transcribe(path)

    assert notes, "the real model returned no notes for a clear arpeggio"
    assert all(isinstance(note, NoteEvent) for note in notes)
    assert all(0 <= note.pitch <= 127 for note in notes)
    assert all(note.start >= 0.0 and note.duration > 0.0 for note in notes)
    # The arpeggio is C4 E4 G4 C5 E5; accept octave errors, reject a wrong scale degree.
    detected = {note.pitch % 12 for note in notes}
    assert detected <= {0, 4, 7}, f"unexpected pitch classes: {sorted(detected)}"
    assert len(notes) >= 3
