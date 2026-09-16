"""Audio decoding entry point.

Responsibilities, in this order: **decode -> mono -> normalise -> resample**.

A backend must never decode a file itself: the pipeline hands every backend an
:class:`~app.core.models.AudioBuffer` produced here, and a backend that needs a
specific sample rate resamples it inside its own adapter.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.audio.preprocessing import preprocess
from app.core.errors import (
    AudioLoadError,
    FfmpegNotFoundError,
    SegmentDecodeError,
    UnsupportedFormatError,
)
from app.core.models import AudioBuffer

#: Containers libsndfile (bundled with ``soundfile``) can decode directly.
LIBSNDFILE_EXTENSIONS = frozenset(
    {".wav", ".wave", ".flac", ".ogg", ".oga", ".mp3", ".opus", ".aiff", ".aif"}
)

#: Containers that require an external FFmpeg binary.
FFMPEG_EXTENSIONS = frozenset({".m4a", ".mp4", ".aac", ".wma", ".webm", ".amr", ".mp2"})

SUPPORTED_AUDIO_EXTENSIONS = LIBSNDFILE_EXTENSIONS | FFMPEG_EXTENSIONS

_F32LE_DTYPE = np.dtype("<f4")


def is_audio_path(path: str | Path) -> bool:
    """Return ``True`` when *path* looks like an audio file we accept."""
    return Path(path).suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS


def find_ffmpeg() -> str | None:
    """Return the FFmpeg executable path, or ``None`` when it is not installed."""
    return shutil.which("ffmpeg")


def require_ffmpeg(reason: str) -> str:
    """Return the FFmpeg path or raise a helpful :class:`FfmpegNotFoundError`."""
    ffmpeg = find_ffmpeg()
    if ffmpeg is None:
        raise FfmpegNotFoundError(
            f"{reason} Install FFmpeg and make sure it is on your PATH, "
            "or convert the file to .wav/.mp3 with libsndfile support first."
        )
    return ffmpeg


def validate_audio_path(path: str | Path) -> Path:
    """Validate that *path* points at a readable, non-empty audio file."""
    resolved = Path(path)
    if not resolved.exists():
        raise AudioLoadError(f"audio file not found: {resolved}")
    if resolved.is_dir():
        raise AudioLoadError(f"expected an audio file but got a directory: {resolved}")
    if resolved.suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_AUDIO_EXTENSIONS))
        raise UnsupportedFormatError(
            f"unsupported audio format {resolved.suffix or '<none>'!r}. Supported: {supported}"
        )
    try:
        size = resolved.stat().st_size
    except OSError as exc:  # pragma: no cover - very unusual on desktop OSes
        raise AudioLoadError(f"cannot stat audio file {resolved}: {exc}") from exc
    if size == 0:
        raise AudioLoadError(f"audio file is empty: {resolved}")
    return resolved


def decode_with_libsndfile(path: Path) -> tuple[np.ndarray, int]:
    """Decode *path* with ``soundfile``; return ``(samples(frames, channels), sr)``."""
    import soundfile as sf

    try:
        data, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    except Exception as exc:  # soundfile raises a zoo of exception types
        raise AudioLoadError(f"could not decode {path.name} with libsndfile: {exc}") from exc

    if data.size == 0 or sample_rate <= 0:
        raise AudioLoadError(f"decoded audio is empty: {path}")
    return np.asarray(data, dtype=np.float32), int(sample_rate)


def decode_with_ffmpeg(path: Path) -> tuple[np.ndarray, int]:
    """Decode *path* by transcoding it to a temporary WAV through FFmpeg."""
    ffmpeg = require_ffmpeg(f"{path.suffix or 'this'} files need FFmpeg to be decoded.")

    with tempfile.TemporaryDirectory(prefix="piano_to_harmonica_") as tmpdir:
        wav_path = Path(tmpdir) / "decoded.wav"
        command = [
            ffmpeg,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(path),
            "-c:a",
            "pcm_f32le",
            str(wav_path),
        ]
        try:
            completed = subprocess.run(command, capture_output=True, text=True, check=False)
        except OSError as exc:
            raise FfmpegNotFoundError(f"failed to run FFmpeg: {exc}") from exc

        if completed.returncode != 0 or not wav_path.exists():
            detail = (completed.stderr or "").strip() or "unknown FFmpeg failure"
            raise AudioLoadError(f"FFmpeg could not decode {path.name}: {detail}")
        return decode_with_libsndfile(wav_path)


def decode_audio_file(path: str | Path) -> tuple[np.ndarray, int]:
    """Decode any supported audio file into ``(samples, sample_rate)``.

    Returns a ``(frames, channels)`` float32 array without touching the pipeline
    mono/normalise/resample steps.
    """
    resolved = validate_audio_path(path)

    if resolved.suffix.lower() in LIBSNDFILE_EXTENSIONS:
        try:
            return decode_with_libsndfile(resolved)
        except AudioLoadError as exc:
            try:
                return decode_with_ffmpeg(resolved)
            except FfmpegNotFoundError:
                raise exc from None
    return decode_with_ffmpeg(resolved)


def load_audio(
    path: str | Path,
    *,
    sample_rate: int | None = None,
    normalize: bool = True,
    peak: float = 0.95,
) -> AudioBuffer:
    """Decode *path* into a mono :class:`AudioBuffer`.

    Parameters
    ----------
    path:
        Input ``.mp3`` / ``.wav`` (or any other supported container).
    sample_rate:
        Target sample rate. ``None`` keeps the file's native rate.
    normalize:
        Peak-normalise the signal to *peak*.
    peak:
        Target peak amplitude used when *normalize* is true.
    """
    samples, source_rate = decode_audio_file(path)
    return preprocess(
        samples,
        source_rate,
        target_sample_rate=sample_rate,
        normalize=normalize,
        peak=peak,
    )


# --------------------------------------------------------------------------- #
# segment-aware decoding
# --------------------------------------------------------------------------- #
DEFAULT_SEGMENT_CONTEXT_BEFORE = 2.0
DEFAULT_SEGMENT_CONTEXT_AFTER = 2.0


def probe_audio_duration(path: str | Path) -> float | None:
    """Return the media duration in seconds, or ``None`` when it cannot be probed.

    Cheap (header read only, no full decode).
    """
    import soundfile as sf

    resolved = Path(path)
    try:
        info = sf.info(str(resolved))
        duration = float(info.frames) / float(info.samplerate)
        return duration if duration > 0 else None
    except Exception:  # noqa: BLE001 - probing must never be fatal
        return None


@dataclass(frozen=True)
class SegmentWindow:
    """The actual decoded window on the *source* timeline, in seconds."""

    start: float
    end: float
    requested_start: float
    requested_end: float
    context_before: float
    context_after: float

    @property
    def duration(self) -> float:
        return self.end - self.start


def load_audio_segment(
    path: str | Path,
    start_time: float,
    end_time: float,
    *,
    context_before: float = DEFAULT_SEGMENT_CONTEXT_BEFORE,
    context_after: float = DEFAULT_SEGMENT_CONTEXT_AFTER,
    sample_rate: int | None = None,
    normalize: bool = True,
    peak: float = 0.95,
) -> tuple[AudioBuffer, SegmentWindow]:
    """Decode only ``[start_time, end_time)`` plus a little context.

    The context keeps the AMT model from starting its analysis window exactly on
    the requested boundary (a cut in the middle of a sustain or an onset). The
    extra audio is later removed by the strict ``crop_note_events`` step, which
    maps transcription times back: ``absolute = local + window.start``.

    Raises :class:`SegmentDecodeError` when the container cannot be seeked (the
    caller should fall back to :func:`load_audio` on the whole file).
    """
    import soundfile as sf

    resolved = validate_audio_path(path)
    if end_time <= start_time:
        raise SegmentDecodeError(f"segment end ({end_time:g}) must be after start ({start_time:g})")
    if resolved.suffix.lower() not in LIBSNDFILE_EXTENSIONS:
        # FFmpeg-only containers would need a process per segment; not worth it.
        raise SegmentDecodeError(
            f"seek-based decoding is not supported for {resolved.suffix} files"
        )

    try:
        with sf.SoundFile(str(resolved)) as handle:
            source_rate = int(handle.samplerate)
            total = handle.frames / source_rate
            window_start = max(0.0, min(start_time - max(context_before, 0.0), total))
            window_end = min(end_time + max(context_after, 0.0), total)
            if window_end <= window_start:
                raise SegmentDecodeError(
                    f"requested segment {start_time:g}-{end_time:g} s is outside "
                    f"the file ({total:.2f} s)"
                )
            start_frame = int(round(window_start * source_rate))
            frame_count = int(round((window_end - window_start) * source_rate))
            try:
                handle.seek(start_frame)
            except (RuntimeError, ValueError) as exc:
                raise SegmentDecodeError(f"cannot seek in {resolved.name}: {exc}") from exc
            try:
                data = handle.read(frames=frame_count, dtype="float32", always_2d=True)
            except Exception as exc:  # noqa: BLE001
                raise SegmentDecodeError(f"cannot read the segment of {resolved.name}: {exc}") from exc
    except SegmentDecodeError:
        raise
    except Exception as exc:  # noqa: BLE001 - soundfile raises a zoo of types
        raise SegmentDecodeError(f"could not open {resolved.name} for segment decoding: {exc}") from exc

    if data.size == 0:
        raise SegmentDecodeError(f"the requested segment of {resolved.name} is empty")

    # Clamp the real window to what was actually decoded (seek may be coarse).
    actual_start = start_frame / source_rate
    actual_end = actual_start + data.shape[0] / source_rate
    window = SegmentWindow(
        start=actual_start,
        end=actual_end,
        requested_start=start_time,
        requested_end=end_time,
        context_before=max(context_before, 0.0),
        context_after=max(context_after, 0.0),
    )
    buffer = preprocess(
        np.asarray(data, dtype=np.float32),
        source_rate,
        target_sample_rate=sample_rate,
        normalize=normalize,
        peak=peak,
    )
    return buffer, window


__all__ = [
    "SUPPORTED_AUDIO_EXTENSIONS",
    "LIBSNDFILE_EXTENSIONS",
    "FFMPEG_EXTENSIONS",
    "DEFAULT_SEGMENT_CONTEXT_BEFORE",
    "DEFAULT_SEGMENT_CONTEXT_AFTER",
    "is_audio_path",
    "find_ffmpeg",
    "require_ffmpeg",
    "validate_audio_path",
    "decode_with_libsndfile",
    "decode_with_ffmpeg",
    "decode_audio_file",
    "load_audio",
    "probe_audio_duration",
    "SegmentWindow",
    "load_audio_segment",
]
