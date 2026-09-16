"""Audio loader + preprocessing behaviour."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from app.audio.loader import (
    decode_audio_file,
    find_ffmpeg,
    is_audio_path,
    load_audio,
    validate_audio_path,
)
from app.audio.preprocessing import peak_normalize, preprocess, resample, to_mono
from app.core.errors import AudioLoadError, UnsupportedFormatError
from app.core.models import AudioBuffer

from conftest import MONO_SAMPLE_RATE, decaying_tone, write_wav


class TestLoading:
    def test_reads_a_wav_file(self, tone_wav: Path) -> None:
        buffer = load_audio(tone_wav)
        assert isinstance(buffer, AudioBuffer)
        assert buffer.sample_rate == MONO_SAMPLE_RATE
        assert buffer.duration == pytest.approx(0.5, abs=0.01)
        assert buffer.samples.dtype == np.float32
        assert buffer.samples.ndim == 1

    def test_reads_an_mp3_file(self, mp3_file: Path) -> None:
        buffer = load_audio(mp3_file)
        assert buffer.sample_rate == 44100
        assert buffer.duration == pytest.approx(0.5, abs=0.05)
        assert np.max(np.abs(buffer.samples)) > 0.0

    def test_decode_audio_file_keeps_channels(self, stereo_wav: Path) -> None:
        samples, sample_rate = decode_audio_file(stereo_wav)
        assert samples.ndim == 2
        assert samples.shape[1] == 2
        assert sample_rate == MONO_SAMPLE_RATE

    def test_stereo_is_downmixed_to_mono(self, stereo_wav: Path) -> None:
        raw, _ = sf.read(str(stereo_wav), dtype="float32", always_2d=True)
        buffer = load_audio(stereo_wav, normalize=False)
        assert buffer.samples.ndim == 1
        assert np.allclose(buffer.samples, raw.mean(axis=1), atol=1e-6)

    def test_duration_matches_the_loaded_samples(self, tone_wav: Path) -> None:
        buffer = load_audio(tone_wav)
        assert buffer.duration == pytest.approx(buffer.n_samples / buffer.sample_rate, rel=1e-9)


class TestNormalisation:
    def test_peak_normalisation_reaches_the_target(self, tone_wav: Path) -> None:
        buffer = load_audio(tone_wav, normalize=True)
        assert float(np.max(np.abs(buffer.samples))) == pytest.approx(0.95, abs=1e-4)

    def test_normalisation_can_be_disabled(self, tone_wav: Path) -> None:
        buffer = load_audio(tone_wav, normalize=False)
        assert float(np.max(np.abs(buffer.samples))) < 0.95

    def test_digital_silence_does_not_blow_up(self, silent_wav: Path) -> None:
        buffer = load_audio(silent_wav)
        assert np.all(buffer.samples == 0.0)
        assert np.all(np.isfinite(buffer.samples))


class TestResampling:
    def test_resamples_to_the_requested_rate(self, tone_wav: Path) -> None:
        original = load_audio(tone_wav)
        resampled = load_audio(tone_wav, sample_rate=44100)
        assert resampled.sample_rate == 44100
        assert resampled.duration == pytest.approx(original.duration, abs=0.02)
        assert resampled.n_samples > original.n_samples

    def test_resampling_helper_preserves_duration(self) -> None:
        source = decaying_tone(seconds=1.0, sample_rate=8000, amplitude=0.5)
        target = resample(source, 8000, 16000)
        assert target.shape[0] == pytest.approx(source.shape[0] * 2, rel=0.01)

    def test_same_rate_returns_the_input_unchanged(self) -> None:
        source = decaying_tone(seconds=0.1, sample_rate=8000)
        assert np.array_equal(resample(source, 8000, 8000), source)

    def test_invalid_rates_are_rejected(self) -> None:
        with pytest.raises(AudioLoadError, match="sample rate"):
            resample(np.zeros(10, dtype=np.float32), 0, 16000)


class TestPreprocessing:
    def test_to_mono_averages_channels(self) -> None:
        stereo = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        assert np.allclose(to_mono(stereo), [0.5, 0.5])

    def test_to_mono_passes_through_mono_audio(self) -> None:
        mono = np.array([0.1, 0.2], dtype=np.float32)
        assert np.array_equal(to_mono(mono), mono)

    def test_to_mono_rejects_3d_input(self) -> None:
        with pytest.raises(AudioLoadError, match="1-D or 2-D"):
            to_mono(np.zeros((2, 2, 2), dtype=np.float32))

    def test_peak_normalize_scales_to_the_target(self) -> None:
        normalized = peak_normalize(np.array([0.1, -0.2], dtype=np.float32), peak=0.5)
        assert float(np.max(np.abs(normalized))) == pytest.approx(0.5)

    def test_preprocess_nan_is_replaced_by_silence(self) -> None:
        buffer = preprocess(np.array([np.nan, 0.5, -0.5], dtype=np.float32), 1000, normalize=False)
        assert np.all(np.isfinite(buffer.samples))

    def test_preprocess_outputs_a_valid_buffer(self) -> None:
        buffer = preprocess(np.zeros(2000, dtype=np.float32), 1000)
        assert buffer.sample_rate == 1000
        assert buffer.duration == pytest.approx(2.0)


class TestFailures:
    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(AudioLoadError, match="not found"):
            load_audio(tmp_path / "nope.wav")

    def test_directory_instead_of_file(self, tmp_path: Path) -> None:
        with pytest.raises(AudioLoadError, match="directory"):
            load_audio(tmp_path)

    def test_unsupported_extension(self, tmp_path: Path) -> None:
        path = tmp_path / "song.txt"
        path.write_text("not audio")
        with pytest.raises(UnsupportedFormatError, match="unsupported audio format"):
            load_audio(path)

    def test_empty_file(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.wav"
        path.touch()
        with pytest.raises(AudioLoadError, match="empty"):
            load_audio(path)

    def test_corrupt_file(self, corrupt_wav: Path) -> None:
        with pytest.raises(AudioLoadError):
            load_audio(corrupt_wav)

    def test_validate_returns_the_path_for_good_input(self, tone_wav: Path) -> None:
        assert validate_audio_path(tone_wav) == tone_wav


class TestHelpers:
    @pytest.mark.parametrize("name", ["a.mp3", "a.WAV", "b/wave.wave", "x.flac"])
    def test_is_audio_path_accepts_supported_files(self, name: str) -> None:
        assert is_audio_path(name)

    @pytest.mark.parametrize("name", ["a.mid", "a.txt", "a"])
    def test_is_audio_path_rejects_other_files(self, name: str) -> None:
        assert not is_audio_path(name)

    def test_find_ffmpeg_returns_a_path_or_none(self) -> None:
        result = find_ffmpeg()
        assert result is None or isinstance(result, str)


def test_generated_short_wav_round_trip(tmp_path: Path) -> None:
    """The tiny integration fixture: write a WAV, read it back, get notes-ready audio."""
    samples = decaying_tone(seconds=0.25, sample_rate=16000, amplitude=0.4)
    path = write_wav(tmp_path / "short.wav", samples, 16000)
    buffer = load_audio(path)
    assert buffer.sample_rate == 16000
    assert buffer.duration == pytest.approx(0.25, abs=0.005)
    assert float(np.max(np.abs(buffer.samples))) == pytest.approx(0.95, abs=1e-4)
