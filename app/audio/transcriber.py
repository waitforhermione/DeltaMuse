"""Backend-agnostic piano transcription interface.

The rest of the pipeline only knows this protocol, never a concrete model. Adding
Aria-AMT or ByteDance Piano later means adding one module under
``app/audio/backends`` and one entry in :data:`BACKENDS` - nothing else changes.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Protocol, runtime_checkable

from app.audio.loader import load_audio
from app.core.errors import BackendUnavailableError
from app.core.models import AudioBuffer, NoteEvent

DEFAULT_BACKEND = "transkun"

#: ``name -> "module.path:ClassName"``. Imported lazily so that heavy ML
#: dependencies are only touched when a backend is actually requested.
BACKENDS: dict[str, str] = {
    "transkun": "app.audio.backends.transkun_backend:TransKunBackend",
}


@runtime_checkable
class PianoTranscriber(Protocol):
    """Anything that can turn a piano recording into :class:`NoteEvent` objects."""

    name: str

    def transcribe(self, source: "AudioBuffer | str | Path") -> list[NoteEvent]:
        """Transcribe *source*.

        ``source`` is either an already decoded
        :class:`~app.core.models.AudioBuffer` (preferred - this is what the
        pipeline passes, so no layer decodes a file twice) or a path, in which
        case the backend loads it through :func:`app.audio.loader.load_audio`.
        """
        ...


def resolve_audio_buffer(source: AudioBuffer | str | Path) -> AudioBuffer:
    """Normalise the :class:`PianoTranscriber` input into an :class:`AudioBuffer`."""
    if isinstance(source, AudioBuffer):
        return source
    if isinstance(source, (str, Path)):
        return load_audio(source)
    raise TypeError(f"expected an AudioBuffer or a path, got {type(source).__name__}")


def available_backends() -> tuple[str, ...]:
    """Names of the transcription backends this build knows how to import."""
    return tuple(BACKENDS)


def create_transcriber(name: str = DEFAULT_BACKEND, **kwargs: object) -> PianoTranscriber:
    """Instantiate a backend by name.

    Raises
    ------
    BackendUnavailableError
        The name is unknown, or the backend's dependencies are not importable.
    """
    try:
        target = BACKENDS[name]
    except KeyError:
        known = ", ".join(sorted(BACKENDS)) or "<none>"
        raise BackendUnavailableError(f"unknown backend {name!r}. Available backends: {known}") from None

    module_name, _, attribute = target.partition(":")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise BackendUnavailableError(
            f"backend {name!r} is not installed ({exc}). Install its dependencies to use it."
        ) from exc

    backend_class = getattr(module, attribute)
    return backend_class(**kwargs)  # type: ignore[no-any-return]


__all__ = [
    "DEFAULT_BACKEND",
    "BACKENDS",
    "PianoTranscriber",
    "resolve_audio_buffer",
    "available_backends",
    "create_transcriber",
]
