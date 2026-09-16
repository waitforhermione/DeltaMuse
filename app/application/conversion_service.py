"""Shared, Qt-free application service used by both the CLI and the GUI.

``ConversionService.run(request, on_progress) -> ConversionOutcome`` composes
the same public building blocks as the CLI (loader, melody extractor,
converter, layout, renderers); it contains no music logic of its own, prints
nothing, and never imports PySide6.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from app.audio.loader import load_audio, load_audio_segment, probe_audio_duration
from app.audio.transcriber import DEFAULT_BACKEND, create_transcriber
from app.core.converter import ConversionConfig, convert_melody
from app.core.errors import NoNotesError, PianoScoreError
from app.core.instrument_profile import InstrumentProfile
from app.core.melody_extractor import (
    DEFAULT_MIN_NOTE_DURATION,
    DEFAULT_STRATEGY,
    MelodyExtractionConfig,
    MelodyExtractor,
    assert_monophonic,
)
from app.core.models import NoteEvent
from app.core.output_naming import default_output_path, unique_path
from app.core.time_range import (
    TimeRange,
    crop_note_events,
    format_clock,
    normalize_time_range,
    parse_time_value,
)
from app.export.image_exporter import export_score_png
from app.export.json_exporter import write_score_json
from app.midi.midi_loader import SUPPORTED_MIDI_EXTENSIONS, load_midi
from app.score.labels import NoteLabelSymbols
from app.score.layout import LayoutHeader, build_layout
from app.score.styles import get_preset

STAGES: tuple[tuple[int, str], ...] = (
    (1, "Loading audio..."),
    (2, "Transcribing piano..."),
    (3, "Extracting melody..."),
    (4, "Mapping harmonica notes..."),
    (5, "Rendering score..."),
)

ProgressCallback = Callable[[int, str], None]

SUPPORTED_INPUT_EXTENSIONS = SUPPORTED_MIDI_EXTENSIONS | {".mp3", ".wav"}


@dataclass(frozen=True)
class InputFileInfo:
    """Light-weight file inspection (never runs the model)."""

    path: Path
    kind: str  # "audio" | "midi" | "unsupported" | "missing"
    duration: float | None = None
    error: str | None = None

    @property
    def supported(self) -> bool:
        return self.kind in ("audio", "midi")


@dataclass(frozen=True)
class ConversionRequest:
    """Everything the service needs; mirrors the CLI options."""

    input_path: Path
    output_path: Path | None = None
    style: str = "practice"
    start: str | None = None
    end: str | None = None
    transpose: str | int = "auto"
    range_mode: str = "octave_fold"
    section_duration: float | None = None
    context_before: float = 2.0
    context_after: float = 2.0
    strategy: str = DEFAULT_STRATEGY
    min_note_duration: float = DEFAULT_MIN_NOTE_DURATION
    backend: str = DEFAULT_BACKEND
    device: str = "auto"

    @property
    def is_segment(self) -> bool:
        return self.start is not None or self.end is not None


@dataclass
class ConversionOutcome:
    """The user-facing result of a successful conversion."""

    output: Path
    notes: int = 0
    transpose: str = "0"
    playable: float = 1.0
    elapsed: float = 0.0
    timings: dict[str, float] = field(default_factory=dict)
    segment: tuple[float, float] | None = None
    transcribed_seconds: float = 0.0


class ConversionService:
    """Facade over the conversion pipeline. One instance can be reused."""

    def __init__(self, settings: UserSettings | None = None) -> None:
        self._settings = settings

    def inspect(self, path: str | Path) -> InputFileInfo:
        """Cheap metadata check; never runs the model."""
        resolved = Path(path)
        if not resolved.is_file():
            return InputFileInfo(path=resolved, kind="missing", error="Input file not found.")
        suffix = resolved.suffix.lower()
        if suffix in SUPPORTED_MIDI_EXTENSIONS:
            try:
                document = load_midi(resolved)
            except PianoScoreError as exc:
                return InputFileInfo(path=resolved, kind="midi", error=str(exc))
            return InputFileInfo(path=resolved, kind="midi", duration=document.duration)
        if suffix in {".mp3", ".wav", ".flac", ".ogg"}:
            duration = probe_audio_duration(resolved)
            if duration is None:
                return InputFileInfo(path=resolved, kind="audio", error="Cannot read this audio file.")
            return InputFileInfo(path=resolved, kind="audio", duration=duration)
        return InputFileInfo(
            path=resolved,
            kind="unsupported",
            error="Unsupported file type. Supported: .mp3, .wav, .mid, .midi",
        )

    def run(
        self, request: ConversionRequest, on_progress: ProgressCallback | None = None
    ) -> ConversionOutcome:
        """Run the full pipeline. Raises :class:`PianoScoreError` on failure."""
        started = time.perf_counter()
        timings: dict[str, float] = {}
        path = Path(request.input_path)
        if not path.is_file():
            raise PianoScoreError(f"Input file not found: {path.resolve()}")

        time_range = self._prepare_time_range(request, path)
        segment_value = (time_range.start, time_range.end) if time_range is not None else None

        self._progress(on_progress, 1)
        t0 = time.perf_counter()
        notes, transcribed = self._load_and_transcribe(request, path, time_range, on_progress)
        timings["decode+transcribe"] = time.perf_counter() - t0

        if time_range is not None:
            notes = list(crop_note_events(notes, time_range.start, time_range.end))
        if not notes:
            where = (
                f" in segment {format_clock(time_range.start)}-{format_clock(time_range.end)}"
                if time_range is not None
                else ""
            )
            raise NoNotesError(
                f"No piano notes were detected in {path.name}{where}. "
                "Try choosing a different segment or a cleaner piano recording."
            )

        self._progress(on_progress, 3)
        t0 = time.perf_counter()
        melody = MelodyExtractor(
            MelodyExtractionConfig(
                strategy=request.strategy, min_note_duration=request.min_note_duration
            )
        ).extract(notes)
        timings["melody"] = time.perf_counter() - t0
        if not melody.notes:
            raise NoNotesError(
                "No usable melody notes remained after melody extraction. "
                "Try a different strategy or a cleaner piano recording."
            )
        assert_monophonic(melody.notes)

        self._progress(on_progress, 4)
        t0 = time.perf_counter()
        result = convert_melody(
            melody.notes,
            ConversionConfig(transpose=request.transpose, range_mode=request.range_mode),
            InstrumentProfile.default(),
        )
        timings["mapping"] = time.perf_counter() - t0
        if not result.notes:
            raise NoNotesError(f"Every melody note was dropped by range mode {request.range_mode!r}.")

        output = self._resolve_output(request, path, segment_value)

        self._progress(on_progress, 5)
        t0 = time.perf_counter()
        if output.suffix.lower() == ".png":
            written = self._render_png(request, result, path, segment_value, output)
        else:
            written = write_score_json(result, output, source_segment=segment_value)
        timings["render"] = time.perf_counter() - t0

        report = result.report
        transpose = f"{report.transpose:+d}" if report.transpose else "0"
        return ConversionOutcome(
            output=written,
            notes=report.output_note_count,
            transpose=transpose,
            playable=report.playable_ratio,
            elapsed=time.perf_counter() - started,
            timings=timings,
            segment=segment_value,
            transcribed_seconds=transcribed,
        )

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #
    @staticmethod
    def _progress(on_progress: ProgressCallback | None, stage: int) -> None:
        if on_progress is not None:
            on_progress(stage, dict(STAGES).get(stage, ""))

    def _prepare_time_range(self, request: ConversionRequest, path: Path) -> TimeRange | None:
        if not request.is_segment:
            return None
        total = probe_audio_duration(path)
        if total is None and path.suffix.lower() in SUPPORTED_MIDI_EXTENSIONS:
            total = load_midi(path).duration
        return normalize_time_range(
            parse_time_value(request.start), parse_time_value(request.end), total
        )

    def _load_and_transcribe(
        self,
        request: ConversionRequest,
        path: Path,
        time_range: TimeRange | None,
        on_progress: ProgressCallback | None,
    ) -> tuple[list[NoteEvent], float]:
        if path.suffix.lower() in SUPPORTED_MIDI_EXTENSIONS:
            document = load_midi(path)
            return list(document.notes), document.duration

        transcriber = create_transcriber(request.backend, device=request.device)
        window = None
        if time_range is not None:
            buffer, window = load_audio_segment(
                path,
                time_range.start,
                time_range.end,
                context_before=request.context_before,
                context_after=request.context_after,
            )
        else:
            buffer = load_audio(path)

        self._progress(on_progress, 2)
        raw_notes = transcriber.transcribe(buffer)
        if window is None:
            return list(raw_notes), buffer.duration

        # Map local segment times back onto the absolute source timeline.
        offset = window.start
        notes = [
            NoteEvent(
                pitch=n.pitch,
                start=n.start + offset,
                duration=n.duration,
                velocity=n.velocity,
                confidence=n.confidence,
            )
            for n in raw_notes
        ]
        return notes, window.end - window.start

    def _resolve_output(
        self,
        request: ConversionRequest,
        path: Path,
        segment_value: tuple[float, float] | None,
    ) -> Path:
        if request.output_path is not None:
            return unique_path(Path(request.output_path))
        return unique_path(default_output_path(path, ".png", segment=segment_value))

    def _render_png(self, request: ConversionRequest, result, path: Path, segment_value, output: Path) -> Path:
        from dataclasses import replace as _replace

        preset = get_preset(request.style)
        layout_config = preset.layout
        if request.section_duration is not None:
            layout_config = _replace(layout_config, section_duration=request.section_duration)
        layout = build_layout(
            result.notes,
            lanes=result.profile.base_keys,
            config=layout_config,
            header=LayoutHeader(
                title=f"{result.profile.name} score",
                source=str(path),
                profile=result.profile.name,
                duration=result.duration,
                transpose=result.report.transpose,
                auto_transpose=result.report.auto_transpose,
                range_mode=result.report.range_mode,
                note_count=result.report.output_note_count,
                playable_ratio=result.report.playable_ratio,
                segment=segment_value,
            ),
            symbols=NoteLabelSymbols.from_profile(result.profile),
        )
        return export_score_png(layout, output, style=preset.style, config=layout_config)


__all__ = [
    "ConversionOutcome",
    "ConversionRequest",
    "ConversionService",
    "InputFileInfo",
    "ProgressCallback",
    "STAGES",
    "SUPPORTED_INPUT_EXTENSIONS",
]

