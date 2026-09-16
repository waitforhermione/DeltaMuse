"""Typed, user-facing errors for the whole pipeline.

Every *expected* failure raises a :class:`PianoScoreError` subclass. The CLI only
has to catch that single base class to print ``ERROR: ...`` and return a non-zero
exit code without leaking a traceback to the user.
"""

from __future__ import annotations


class PianoScoreError(Exception):
    """Base class for every expected, user-facing error."""

    exit_code = 1


# --------------------------------------------------------------------------- #
# Audio layer
# --------------------------------------------------------------------------- #
class AudioLoadError(PianoScoreError):
    """The audio file does not exist, cannot be decoded, or is empty/corrupt."""


class SegmentDecodeError(AudioLoadError):
    """A seek-based partial decode is unavailable for this file.

    Callers should fall back to full-track decoding when they catch this.
    """


class UnsupportedFormatError(AudioLoadError):
    """The file extension is not a supported audio container."""


class FfmpegNotFoundError(AudioLoadError):
    """The codec was missing from libsndfile and FFmpeg is not installed."""


# --------------------------------------------------------------------------- #
# Transcription layer
# --------------------------------------------------------------------------- #
class BackendUnavailableError(PianoScoreError):
    """The requested backend / model is not installed or cannot be loaded."""


class TranscriptionError(PianoScoreError):
    """The backend raised while transcribing an otherwise valid audio file."""


class NoNotesError(PianoScoreError):
    """The pipeline produced no usable note (empty input or everything filtered)."""


# --------------------------------------------------------------------------- #
# MIDI layer
# --------------------------------------------------------------------------- #
class MidiError(PianoScoreError):
    """A MIDI file could not be read or written."""


class OutputNotWritableError(PianoScoreError):
    """The output path is not writable (missing directory, permission, ...)."""


# --------------------------------------------------------------------------- #
# Harmonica conversion layer
# --------------------------------------------------------------------------- #
class ProfileError(PianoScoreError):
    """The instrument profile is missing, unreadable or internally inconsistent."""


class UnmappedPitchError(PianoScoreError):
    """A pitch could not be mapped onto the instrument.

    The mapper never transposes, folds or drops: range fitting must have made the
    pitch playable before it reaches this point.
    """


class TransposeError(PianoScoreError):
    """A transposition request is out of range or would leave the MIDI range."""


# --------------------------------------------------------------------------- #
# Score layout layer
# --------------------------------------------------------------------------- #
class ScoreLayoutError(PianoScoreError):
    """The layout configuration or geometry is invalid."""


class EmptyScoreError(PianoScoreError):
    """There are no notes to lay out, so no score can be produced."""


class ScoreDocumentError(PianoScoreError):
    """A ``score.json`` file is missing, unreadable or does not match the schema."""


class ScoreRenderError(PianoScoreError):
    """A score image could not be rendered or written."""
