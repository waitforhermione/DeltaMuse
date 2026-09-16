"""Segment-aware audio decoding tests."""

from __future__ import annotations

import numpy as np
import pytest

from app.audio.loader import (
    DEFAULT_SEGMENT_CONTEXT_AFTER,
    DEFAULT_SEGMENT_CONTEXT_BEFORE,
    SegmentDecodeError,
    load_audio,
    load_audio_segment,
    probe_audio_duration,
)

from conftest import MONO_SAMPLE_RATE, write_wav


@pytest.fixture
def long_wav(tmp_path):
    """A 10 s tone (the content itself does not matter for these tests)."""
    sample_rate = MONO_SAMPLE_RATE
    total = np.zeros(10 * sample_rate, dtype=np.float64)
    for second in range(10):
        frequency = 220.0 * (2 ** (second / 12.0))
        time = np.arange(sample_rate, dtype=np.float64) / sample_rate
        total[second * sample_rate : (second + 1) * sample_rate] += 0.5 * np.sin(
            2.0 * np.pi * frequency * time
        )
    peak = float(np.max(np.abs(total))) or 1.0
    return write_wav(tmp_path / "long.wav", (total * (0.9 / peak)).astype(np.float32), sample_rate)


class TestProbeDuration:
    def test_probe_returns_the_duration(self, long_wav) -> None:
        assert probe_audio_duration(long_wav) == pytest.approx(10.0, abs=0.01)

    def test_probe_never_raises(self, tmp_path) -> None:
        garbage = tmp_path / "garbage.wav"
        garbage.write_bytes(b"nope")
        assert probe_audio_duration(garbage) is None


class TestLoadAudioSegment:
    def test_returns_only_the_window_plus_context(self, long_wav) -> None:
        buffer, window = load_audio_segment(
            long_wav, 4.0, 6.0, context_before=2.0, context_after=2.0
        )

        assert window.requested_start == 4.0
        assert window.requested_end == 6.0
        assert window.start == pytest.approx(2.0, abs=0.01)
        assert window.end == pytest.approx(8.0, abs=0.01)
        assert buffer.duration == pytest.approx(6.0, abs=0.02)

    def test_default_context_is_two_seconds(self, long_wav) -> None:
        _buffer, window = load_audio_segment(long_wav, 4.0, 6.0)
        assert window.context_before == DEFAULT_SEGMENT_CONTEXT_BEFORE == 2.0
        assert window.context_after == DEFAULT_SEGMENT_CONTEXT_AFTER == 2.0

    def test_context_is_clamped_at_the_file_start(self, long_wav) -> None:
        _buffer, window = load_audio_segment(long_wav, 1.0, 3.0)
        assert window.start == pytest.approx(0.0, abs=0.01)

    def test_context_is_clamped_at_the_file_end(self, long_wav) -> None:
        _buffer, window = load_audio_segment(long_wav, 8.0, 9.5)
        assert window.end == pytest.approx(10.0, abs=0.01)

    def test_segment_is_shorter_than_the_full_track(self, long_wav) -> None:
        segment, _window = load_audio_segment(long_wav, 4.0, 6.0)
        full = load_audio(long_wav)
        assert segment.duration < full.duration
        assert full.duration == pytest.approx(10.0, abs=0.01)

    def test_content_matches_the_full_decode(self, long_wav) -> None:
        """The segment (t=0 at window.start) tracks the full decode, up to the
        per-buffer peak normalisation."""
        segment, window = load_audio_segment(long_wav, 4.0, 6.0)
        full = load_audio(long_wav)

        count = MONO_SAMPLE_RATE  # compare one second of audio
        offset = 0.5  # well inside the segment
        seg_slice = segment.samples[int(offset * segment.sample_rate) :][:count]
        full_slice = full.samples[int((window.start + offset) * full.sample_rate) :][:count]

        scale = float(np.max(np.abs(seg_slice))) / float(np.max(np.abs(full_slice)))
        np.testing.assert_allclose(seg_slice, full_slice * scale, atol=1e-3)

    def test_time_offset_round_trip(self, long_wav) -> None:
        """local + window.start must reproduce the absolute source time."""
        _buffer, window = load_audio_segment(long_wav, 4.0, 6.0)
        assert window.start == pytest.approx(2.0, abs=0.01)
        for local, expected in ((0.0, 2.0), (0.5, 2.5), (3.0, 5.0), (5.9, 7.9)):
            assert local + window.start == pytest.approx(expected, abs=0.02)

    def test_zero_context_is_possible_but_not_the_default(self, long_wav) -> None:
        buffer, window = load_audio_segment(long_wav, 4.0, 6.0, context_before=0.0, context_after=0.0)
        assert window.start == pytest.approx(4.0, abs=0.01)
        assert buffer.duration == pytest.approx(2.0, abs=0.02)

    def test_unseekable_container_raises_segment_error(self, tmp_path) -> None:
        # .m4a needs FFmpeg; without seeking support it must raise the typed
        # error so the CLI can fall back to full-track decoding.
        path = tmp_path / "song.m4a"
        path.write_bytes(b"not really audio")
        with pytest.raises(SegmentDecodeError):
            load_audio_segment(path, 1.0, 2.0)

    def test_range_outside_the_file_is_rejected(self, long_wav) -> None:
        with pytest.raises(SegmentDecodeError, match="outside"):
            load_audio_segment(long_wav, 95.0, 100.0)

    def test_inverted_range_is_rejected(self, long_wav) -> None:
        with pytest.raises(SegmentDecodeError, match="after"):
            load_audio_segment(long_wav, 6.0, 4.0)
