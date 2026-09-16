"""Minimal, explicit audio preprocessing.

Exactly three operations are considered "necessary" for the first version:

1. down-mix to mono,
2. peak normalisation,
3. resampling.

Harmonic/percussive separation and other tricks are deliberately *not* applied by
default: the chosen piano AMT backend is trained on normal audio, and extra
processing usually hurts more than it helps. A backend that needs a specific
sample rate does the resampling inside its own adapter.
"""

from __future__ import annotations

from math import gcd

import numpy as np

from app.core.errors import AudioLoadError
from app.core.models import AudioBuffer

DEFAULT_PEAK = 0.95


def to_mono(samples: np.ndarray) -> np.ndarray:
    """Down-mix ``(frames, channels)`` or 1-D input to a 1-D mono array."""
    data = np.asarray(samples, dtype=np.float32)
    if data.ndim == 1:
        return data
    if data.ndim != 2:
        raise AudioLoadError(f"expected 1-D or 2-D audio, got shape {data.shape}")
    return data.mean(axis=1, dtype=np.float32)


def peak_normalize(samples: np.ndarray, peak: float = DEFAULT_PEAK) -> np.ndarray:
    """Scale *samples* so the loudest sample sits at *peak* (skip silence)."""
    data = np.asarray(samples, dtype=np.float32)
    current_peak = float(np.max(np.abs(data))) if data.size else 0.0
    if current_peak <= 0.0:
        # Digital silence: nothing to scale, keep the signal as-is.
        return data
    return (data * (float(peak) / current_peak)).astype(np.float32)


def resample(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Resample a 1-D signal with soxr (through librosa) or scipy as a fallback."""
    source_rate = int(source_rate)
    target_rate = int(target_rate)
    if source_rate <= 0 or target_rate <= 0:
        raise AudioLoadError(f"invalid sample rate conversion {source_rate} -> {target_rate}")
    if source_rate == target_rate:
        return np.asarray(samples, dtype=np.float32)

    data = np.asarray(samples, dtype=np.float32)
    try:
        import librosa
    except ImportError:  # pragma: no cover - librosa is a hard dependency
        librosa = None

    if librosa is not None:
        try:
            return np.asarray(
                librosa.resample(
                    data, orig_sr=source_rate, target_sr=target_rate, res_type="soxr_hq", axis=0
                ),
                dtype=np.float32,
            )
        except Exception:  # pragma: no cover - fall through to scipy
            pass

    from scipy.signal import resample_poly

    divisor = gcd(source_rate, target_rate)
    return np.asarray(
        resample_poly(data, target_rate // divisor, source_rate // divisor, axis=0),
        dtype=np.float32,
    )


def preprocess(
    samples: np.ndarray,
    sample_rate: int,
    *,
    target_sample_rate: int | None = None,
    normalize: bool = True,
    peak: float = DEFAULT_PEAK,
) -> AudioBuffer:
    """Down-mix, normalise and resample raw PCM into an :class:`AudioBuffer`."""
    data = to_mono(samples)
    if data.size == 0:
        raise AudioLoadError("audio contains no samples after down-mixing")
    if not np.any(np.isfinite(data)):
        raise AudioLoadError("audio contains no finite samples")

    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    if normalize:
        data = peak_normalize(data, peak=peak)

    rate = int(sample_rate)
    if target_sample_rate is not None:
        rate = int(target_sample_rate)
        data = resample(data, int(sample_rate), rate)

    if data.size == 0:
        raise AudioLoadError("resampling produced an empty signal")

    return AudioBuffer(
        samples=np.ascontiguousarray(data, dtype=np.float32),
        sample_rate=rate,
        duration=float(data.shape[0]) / float(rate),
    )


__all__ = ["DEFAULT_PEAK", "to_mono", "peak_normalize", "resample", "preprocess"]
