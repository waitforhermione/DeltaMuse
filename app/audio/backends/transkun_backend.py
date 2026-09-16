"""TransKun V2 piano transcription backend (adapter).

Wraps the ``transkun`` pip package (Yujia Yan, *Scoring intervals using
non-hierarchical transformer for automatic piano transcription*, ISMIR 2024).
The bundled checkpoint is *TransKun V2 (no pedal extension)* and is trained for
44.1 kHz mono piano audio.

Everything framework specific - torch tensors, checkpoints, sample-rate
conversion - stops at this module. Callers only ever see
:class:`~app.core.models.NoteEvent` objects.
"""

from __future__ import annotations

import importlib.resources as importlib_resources
from pathlib import Path
from typing import Any

import numpy as np

from app.audio.preprocessing import resample
from app.audio.transcriber import resolve_audio_buffer
from app.core.errors import BackendUnavailableError, TranscriptionError
from app.core.models import AudioBuffer, NoteEvent

PACKAGE_NAME = "transkun"
DEFAULT_WEIGHT_FILENAME = "2.0.pt"
DEFAULT_CONF_FILENAME = "2.0.conf"

#: ``None`` means "use the values stored inside the shipped checkpoint config"
#: (segmentHopSizeInSecond = 8 s, segmentSizeInSecond = 16 s for TransKun 2.0.1).
DEFAULT_SEGMENT_HOP_SIZE: float | None = None
DEFAULT_SEGMENT_SIZE: float | None = None

INSTALL_HINT = 'pip install "transkun==2.0.1"'
_VALID_DEVICES = ("auto", "cpu", "cuda", "mps")


class TransKunBackend:
    """Piano-only audio -> note transcription through TransKun V2."""

    name = "transkun"

    def __init__(
        self,
        *,
        device: str = "auto",
        weight_path: str | Path | None = None,
        conf_path: str | Path | None = None,
        segment_hop_size: float | None = None,
        segment_size: float | None = None,
    ) -> None:
        if segment_hop_size is not None and segment_size is not None and segment_hop_size >= segment_size:
            raise ValueError(
                f"segment_hop_size ({segment_hop_size}) must be smaller than segment_size ({segment_size})"
            )

        self.device = str(device or "auto").lower()
        if self.device not in _VALID_DEVICES:
            raise ValueError(f"device must be one of {_VALID_DEVICES}, got {device!r}")

        self.weight_path = Path(weight_path) if weight_path is not None else None
        self.conf_path = Path(conf_path) if conf_path is not None else None
        self.segment_hop_size = DEFAULT_SEGMENT_HOP_SIZE if segment_hop_size is None else float(segment_hop_size)
        self.segment_size = DEFAULT_SEGMENT_SIZE if segment_size is None else float(segment_size)

        self._model: Any | None = None
        self._torch: Any | None = None
        self._resolved_device: str | None = None

    # ------------------------------------------------------------------ #
    # availability
    # ------------------------------------------------------------------ #
    @staticmethod
    def is_available() -> bool:
        """``True`` when torch, moduleconf and transkun are all importable."""
        import importlib

        for module_name in ("torch", "moduleconf", PACKAGE_NAME):
            try:
                importlib.import_module(module_name)
            except Exception:
                return False
        return True

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #
    def transcribe(self, source: AudioBuffer | str | Path) -> list[NoteEvent]:
        """Transcribe *source* into a list of :class:`NoteEvent`.

        ``source`` may be an already decoded
        :class:`~app.core.models.AudioBuffer` (avoids decoding twice) or a path.
        """
        buffer = resolve_audio_buffer(source)
        model = self._ensure_model()
        torch = self._torch

        target_rate = int(getattr(model, "fs", buffer.sample_rate))
        samples = buffer.samples
        if buffer.sample_rate != target_rate:
            samples = resample(samples, buffer.sample_rate, target_rate)
        if samples.size == 0:
            raise TranscriptionError("nothing to transcribe: the audio signal is empty")

        tensor = torch.from_numpy(np.ascontiguousarray(samples, dtype=np.float32))
        tensor = tensor.reshape(-1, 1).to(self._resolved_device)

        try:
            grad_enabled = torch.is_grad_enabled()
            torch.set_grad_enabled(False)
            try:
                events = model.transcribe(
                    tensor,
                    stepInSecond=self.segment_hop_size,
                    segmentSizeInSecond=self.segment_size,
                    discardSecondHalf=False,
                )
            finally:
                torch.set_grad_enabled(grad_enabled)
        except Exception as exc:
            raise TranscriptionError(f"TransKun failed to transcribe the audio: {exc}") from exc

        notes: list[NoteEvent] = []
        for event in events or ():
            note = self._to_note_event(event)
            if note is not None:
                notes.append(note)
        notes.sort(key=lambda note: (note.start, note.pitch))
        return notes

    def describe(self) -> dict[str, Any]:
        """Short, human-readable description of this backend instance."""
        return {
            "backend": self.name,
            "device": self._resolved_device or self.device,
            "weight": str(self.weight_path) if self.weight_path else "<bundled 2.0.pt>",
            "segment_hop_size": self.segment_hop_size,
            "segment_size": self.segment_size,
            "reports_confidence": False,
        }

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #
    @staticmethod
    def _to_note_event(event: Any) -> NoteEvent | None:
        """Convert one TransKun event into a :class:`NoteEvent`.

        Returns ``None`` for events that are not playable notes: the model emits
        negative "pitch" values to encode sustain-pedal control changes.
        """
        try:
            pitch = int(event.pitch)
            start = float(event.start)
            end = float(event.end)
        except (AttributeError, TypeError, ValueError):
            return None

        if pitch <= 0 or pitch > 127:
            return None
        if end <= start:
            return None

        try:
            velocity = int(getattr(event, "velocity", 0) or 0)
        except (TypeError, ValueError):
            velocity = 0
        # MIDI velocity 0 is a note-off, so clamp model output into 1..127.
        velocity = min(max(velocity, 1), 127)

        try:
            return NoteEvent.from_seconds(
                pitch=pitch,
                start=max(start, 0.0),
                end=end,
                velocity=velocity,
                # TransKun 2.0.1 does not expose a per-note confidence.
                confidence=None,
            )
        except (TypeError, ValueError):
            return None

    def _import_runtime(self) -> tuple[Any, Any]:
        import importlib

        try:
            torch = importlib.import_module("torch")
        except ImportError as exc:
            raise BackendUnavailableError(
                f"PyTorch is required by the transkun backend ({exc}). Install it with: "
                f'{INSTALL_HINT} (or "pip install torch torchaudio")'
            ) from exc

        try:
            moduleconf = importlib.import_module("moduleconf")
        except ImportError as exc:
            raise BackendUnavailableError(
                f"the 'moduleconf' package is required to read the TransKun model config ({exc}). "
                f"Install it with: {INSTALL_HINT}"
            ) from exc

        return torch, moduleconf

    def _resolve_asset(self, custom: Path | None, filename: str) -> Path:
        if custom is not None:
            path = Path(custom)
            if not path.is_file():
                raise BackendUnavailableError(f"TransKun asset not found: {path}")
            return path

        try:
            package_root = importlib_resources.files(PACKAGE_NAME)
        except (ImportError, ModuleNotFoundError, TypeError) as exc:
            raise BackendUnavailableError(
                f"the transkun package is not installed ({exc}). Install it with: {INSTALL_HINT}"
            ) from exc

        path = Path(str(package_root.joinpath("pretrained", filename)))
        if not path.is_file():
            raise BackendUnavailableError(
                f"TransKun ships no bundled asset {filename!r} at {path}. "
                f"Reinstall the package with: {INSTALL_HINT}"
            )
        return path

    def _resolve_device(self, torch: Any) -> str:
        if self.device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            mps = getattr(getattr(torch, "backends", None), "mps", None)
            if mps is not None and getattr(mps, "is_available", lambda: False)():
                return "mps"
            return "cpu"
        if self.device == "cuda" and not torch.cuda.is_available():
            raise BackendUnavailableError("device 'cuda' was requested but no CUDA device is available")
        return self.device

    @staticmethod
    def _load_checkpoint(torch: Any, weight_path: Path, device: str) -> Any:
        """Load the checkpoint, tolerating both old and new ``torch.load`` APIs."""
        try:
            return torch.load(str(weight_path), map_location=device, weights_only=False)
        except TypeError:  # pragma: no cover - torch < 2.0 has no weights_only kwarg
            return torch.load(str(weight_path), map_location=device)

    def _ensure_model(self) -> Any:
        """Build and cache the TransKun model (heavy: called at most once)."""
        if self._model is not None:
            return self._model

        torch, moduleconf = self._import_runtime()
        weight_path = self._resolve_asset(self.weight_path, DEFAULT_WEIGHT_FILENAME)
        conf_path = self._resolve_asset(self.conf_path, DEFAULT_CONF_FILENAME)
        device = self._resolve_device(torch)

        try:
            manager = moduleconf.parseFromFile(str(conf_path))
            model_class = manager["Model"].module.TransKun
            conf = manager["Model"].config
        except Exception as exc:
            raise BackendUnavailableError(
                f"could not read the TransKun model configuration {conf_path.name}: {exc}"
            ) from exc

        try:
            checkpoint = self._load_checkpoint(torch, weight_path, device)
        except Exception as exc:
            raise BackendUnavailableError(
                f"could not load the TransKun checkpoint {weight_path.name}: {exc}"
            ) from exc

        try:
            model = model_class(conf=conf).to(device)
            state_dict = None
            if isinstance(checkpoint, dict):
                state_dict = checkpoint.get("best_state_dict") or checkpoint.get("state_dict")
            model.load_state_dict(state_dict if state_dict is not None else checkpoint, strict=False)
            model.eval()
        except Exception as exc:
            raise BackendUnavailableError(f"could not build the TransKun model: {exc}") from exc

        self._torch = torch
        self._resolved_device = device
        self._model = model
        return model


__all__ = ["TransKunBackend"]
