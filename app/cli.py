"""Command line interface.

The first version has no GUI: everything is driven by
``python main.py analyze|transcribe|extract-melody ...``. Expected failures
surface as ``ERROR: ...`` plus a non-zero exit code; tracebacks are only shown
with ``--debug``.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
import warnings
from dataclasses import replace as replace_dataclass
from pathlib import Path
from typing import Any, Sequence

from app import __version__
from app.audio.loader import (
    DEFAULT_SEGMENT_CONTEXT_AFTER,
    DEFAULT_SEGMENT_CONTEXT_BEFORE,
    SUPPORTED_AUDIO_EXTENSIONS,
    SegmentDecodeError,
    is_audio_path,
    load_audio,
    load_audio_segment,
    probe_audio_duration,
)
from app.audio.transcriber import DEFAULT_BACKEND, available_backends, create_transcriber
from app.config.user_settings import (
    UserSettings,
    load_settings,
    parse_setting_value,
    save_settings,
    settings_path,
    with_updated_field,
)
from app.core.converter import AUTO_TRANSPOSE, ConversionConfig, ConversionResult, convert_melody
from app.core.errors import NoNotesError, PianoScoreError
from app.core.instrument_profile import InstrumentProfile
from app.core.time_range import (
    TimeRange,
    crop_note_events,
    format_clock,
    normalize_time_range,
    parse_time_value,
)
from app.export.image_exporter import export_score_png
from app.export.layout_exporter import write_layout_json
from app.export.score_loader import load_score_document
from app.score.labels import NoteLabelSymbols
from app.score.layout import (
    DEFAULT_CANVAS_WIDTH,
    DEFAULT_SECTION_DURATION,
    LayoutHeader,
    ScoreLayoutConfig,
    build_layout,
)
from app.score.renderer_png import render_score
from app.score.styles import DEFAULT_PRESET_NAME, PRESETS, VisualPreset, get_preset
from app.core.melody_extractor import (
    DEFAULT_MIN_NOTE_DURATION,
    DEFAULT_STRATEGY,
    STRATEGIES,
    MelodyExtractionConfig,
    MelodyExtractionResult,
    MelodyExtractor,
    assert_monophonic,
)
from app.core.models import NoteEvent, midi_pitch_to_name
from app.core.output_naming import default_output_path, unique_path
from app.core.polyphony import DEFAULT_ONSET_EPSILON, polyphony_stats
from app.core.range_fitter import DEFAULT_RANGE_MODE, RANGE_MODES
from app.config.user_settings import (
    UserSettings,
    load_settings,
    parse_setting_value,
    save_settings,
    settings_path,
    with_updated_field,
)
from app.export.json_exporter import write_score_json
from app.midi.midi_loader import SUPPORTED_MIDI_EXTENSIONS, load_midi
from app.midi.midi_writer import write_midi

_LABEL_WIDTH = 17


# --------------------------------------------------------------------------- #
# argument parsing
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    backends = ", ".join(available_backends())
    parser = argparse.ArgumentParser(
        prog="python main.py",
        description=(
            "Piano audio -> Delta Harmonica static score generator. "
            "Quick start: python main.py convert song.mp3"
        ),
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="show detailed diagnostics (backend, timings, intermediate counts)",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    config = subparsers.add_parser("config", help="show or change user preferences")
    config.add_argument(
        "--set",
        nargs=2,
        metavar=("KEY", "VALUE"),
        action="append",
        default=None,
        help="set one preference, e.g. --set default_style practice",
    )
    config.add_argument("--reset", action="store_true", help="restore all defaults")
    config.set_defaults(handler=_command_config)

    analyze = subparsers.add_parser("analyze", help="report what was found in an audio or MIDI file")
    analyze.add_argument("input", help="input .mp3/.wav audio, or a .mid/.midi file")
    analyze.add_argument("--json", dest="json_path", default=None, help="also write the report as JSON")
    _add_verbosity(analyze)
    _add_backend_options(analyze)
    analyze.set_defaults(handler=_command_analyze)

    transcribe = subparsers.add_parser("transcribe", help="transcribe audio into a MIDI file")
    transcribe.add_argument("input", help="input .mp3/.wav audio file")
    transcribe.add_argument(
        "-o",
        "--output",
        default=None,
        help="output MIDI path (default: <input>_transcribed.mid)",
    )
    _add_verbosity(transcribe)
    _add_backend_options(transcribe)
    transcribe.set_defaults(handler=_command_transcribe)

    extract = subparsers.add_parser(
        "extract-melody",
        help="reduce polyphonic piano notes to a strictly monophonic melody",
    )
    extract.add_argument("input", help="input .mp3/.wav audio, or a .mid/.midi file")
    extract.add_argument(
        "--strategy",
        default=DEFAULT_STRATEGY,
        choices=STRATEGIES,
        help=f"melody selection strategy (default: {DEFAULT_STRATEGY})",
    )
    extract.add_argument(
        "--min-note-duration",
        type=float,
        default=DEFAULT_MIN_NOTE_DURATION,
        help=f"drop originally transcribed notes shorter than this, in seconds "
        f"(default: {DEFAULT_MIN_NOTE_DURATION})",
    )
    extract.add_argument(
        "--onset-epsilon",
        type=float,
        default=DEFAULT_ONSET_EPSILON,
        help=f"treat onsets within this many seconds as simultaneous "
        f"(default: {DEFAULT_ONSET_EPSILON})",
    )
    extract.add_argument(
        "-o",
        "--output",
        default=None,
        help="output MIDI path (default: <input>_melody.mid)",
    )
    extract.add_argument("--json", dest="json_path", default=None, help="also write the report as JSON")
    _add_verbosity(extract)
    _add_segment_options(extract)
    _add_backend_options(extract)
    extract.set_defaults(handler=_command_extract_melody)

    map_melody = subparsers.add_parser(
        "map-melody",
        help="convert a melody into playable Delta Harmonica notes (score.json)",
    )
    map_melody.add_argument("input", help="melody input: .mid/.midi, or audio")
    map_melody.add_argument(
        "-o",
        "--output",
        default=None,
        help="output score JSON path (default: <input>[_segment].score.json)",
    )
    _add_verbosity(map_melody)
    _add_conversion_options(map_melody)
    _add_layout_options(map_melody)
    _add_melody_options(map_melody)
    _add_segment_options(map_melody)
    _add_style_option(map_melody)
    _add_backend_options(map_melody)
    map_melody.set_defaults(handler=_command_map_melody)

    convert = subparsers.add_parser(
        "convert",
        help="audio -> transcription -> melody -> playable notes -> score.json or PNG",
    )
    convert.add_argument("input", help="input .mp3/.wav audio file")
    convert.add_argument(
        "-o",
        "--output",
        default=None,
        help="output path; the extension picks the format: .png (default) or .json",
    )
    _add_verbosity(convert)
    _add_conversion_options(convert)
    _add_layout_options(convert)
    _add_melody_options(convert)
    _add_segment_options(convert)
    _add_style_option(convert)
    _add_backend_options(convert)
    convert.set_defaults(handler=_command_convert)

    layout = subparsers.add_parser(
        "layout-score",
        help="compute the static score layout of a score.json (debug JSON, no image)",
    )
    layout.add_argument("input", help="input score.json produced by map-melody / convert")
    layout.add_argument(
        "-o",
        "--output",
        default=None,
        help="output layout JSON path (default: <input>.layout.json)",
    )
    layout.add_argument(
        "--section-duration",
        type=float,
        default=DEFAULT_SECTION_DURATION,
        help=f"seconds per score row (default: {DEFAULT_SECTION_DURATION})",
    )
    layout.add_argument(
        "--canvas-width",
        type=int,
        default=DEFAULT_CANVAS_WIDTH,
        help=f"canvas width in pixels (default: {DEFAULT_CANVAS_WIDTH})",
    )
    layout.add_argument("--title", default=None, help="score title for the header metadata")
    _add_verbosity(layout)
    layout.set_defaults(handler=_command_layout_score)

    render = subparsers.add_parser(
        "render-score",
        help="render a score.json into a static PNG score sheet",
    )
    render.add_argument("input", help="input score.json produced by map-melody / convert")
    render.add_argument(
        "-o",
        "--output",
        default=None,
        help="output PNG path (default: <input stem>.png)",
    )
    _add_layout_options(render)
    _add_style_option(render)
    _add_verbosity(render)
    render.set_defaults(handler=_command_render_score)

    return parser


def _add_layout_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--section-duration",
        type=float,
        default=None,
        help=f"seconds per score row (default: preset value, {DEFAULT_SECTION_DURATION:g} s for compact)",
    )
    parser.add_argument(
        "--canvas-width",
        type=int,
        default=None,
        help="canvas width in pixels (default: preset value)",
    )
    parser.add_argument("--title", default=None, help="score title for the header metadata")


def _add_verbosity(parser: argparse.ArgumentParser) -> None:
    # SUPPRESS keeps the main-level --verbose value when the flag is only given
    # before the subcommand, while still allowing it after the subcommand.
    parser.add_argument("--verbose", action="store_true", default=argparse.SUPPRESS)


def _add_segment_options(parser: argparse.ArgumentParser) -> None:
    """``--start`` / ``--end``: convert only one part of the source media."""
    parser.add_argument(
        "--start",
        default=None,
        metavar="TIME",
        help="start of the segment to convert: seconds (48, 82.5), MM:SS (00:48) "
        "or HH:MM:SS (default: beginning)",
    )
    parser.add_argument(
        "--end",
        default=None,
        metavar="TIME",
        help="end of the segment to convert, same formats (default: end of file)",
    )


def _add_style_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--style",
        default=None,
        choices=sorted(PRESETS),
        help="visual preset for PNG output: compact = overview, practice = easier "
        "sight-reading (default: user setting or practice)",
    )


def _default_layout_output(path: Path) -> Path:
    """``song.score.json`` -> ``song.layout.json`` (fall back to ``<stem>.layout.json``)."""
    stem = path.stem
    if path.suffix == ".json" and stem.endswith(".score"):
        stem = stem[: -len(".score")]
    return path.with_name(f"{stem}.layout.json")


def _transpose_argument(value: str) -> int | str:
    """``--transpose`` accepts ``auto`` or a semitone shift in -12..+12."""
    text = value.strip().lower()
    if text == AUTO_TRANSPOSE:
        return AUTO_TRANSPOSE
    try:
        number = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected 'auto' or an integer, got {value!r}") from None
    if not -12 <= number <= 12:
        raise argparse.ArgumentTypeError(f"manual transpose must be within -12..+12, got {number}")
    return number


def _add_conversion_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--transpose",
        type=_transpose_argument,
        default=None,
        help="automatic transposition or a manual shift in -12..+12 (default: auto)",
    )
    parser.add_argument(
        "--range",
        dest="range_mode",
        default=None,
        choices=RANGE_MODES,
        help=f"how to fit unplayable pitches (default: {DEFAULT_RANGE_MODE})",
    )
    parser.add_argument(
        "--profile",
        default=None,
        help="instrument profile JSON (default: the bundled delta_harmonica profile)",
    )


def _add_melody_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--strategy",
        default=DEFAULT_STRATEGY,
        choices=STRATEGIES,
        help=f"melody selection strategy (default: {DEFAULT_STRATEGY})",
    )
    parser.add_argument(
        "--min-note-duration",
        type=float,
        default=DEFAULT_MIN_NOTE_DURATION,
        help=f"drop transcribed notes shorter than this, in seconds (default: {DEFAULT_MIN_NOTE_DURATION})",
    )
    parser.add_argument(
        "--onset-epsilon",
        type=float,
        default=DEFAULT_ONSET_EPSILON,
        help=f"treat onsets within this many seconds as simultaneous (default: {DEFAULT_ONSET_EPSILON})",
    )


def _add_backend_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--backend",
        default=DEFAULT_BACKEND,
        help=f"transcription backend (default: {DEFAULT_BACKEND}; available: {', '.join(available_backends())})",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda", "mps"),
        help="inference device for the backend (default: auto)",
    )
    parser.add_argument("--weight", default=None, help="override the backend checkpoint path")
    parser.add_argument(
        "--segment-hop-size",
        type=float,
        default=None,
        help="override the backend segment hop in seconds (default: backend config)",
    )
    parser.add_argument(
        "--segment-size",
        type=float,
        default=None,
        help="override the backend segment size in seconds (default: backend config)",
    )
    parser.add_argument("--debug", action="store_true", help="show full tracebacks and library warnings")


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def _is_midi_path(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_MIDI_EXTENSIONS


def _backend_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "device": args.device,
        "weight_path": args.weight,
        "segment_hop_size": args.segment_hop_size,
        "segment_size": args.segment_size,
    }


def _load_notes(
    path: Path,
    args: argparse.Namespace,
    time_range: TimeRange | None = None,
    progress: Any = None,
) -> tuple[str, list[NoteEvent], dict[str, Any]]:
    """Dispatch on file type and return ``(input_type, notes, info)``.

    Only this function knows that audio has to go through a backend while MIDI
    is read directly; every consumer downstream sees plain :class:`NoteEvent`.

    When *time_range* is given and the file is audio, only the segment plus a
    little context is decoded and transcribed; the resulting note times are
    mapped back onto the absolute source timeline
    (``absolute = local + window.start``) so the caller can crop strictly.
    ``info["timings"]`` carries decode/transcription durations.
    """
    import time as _time

    if not path.exists():
        raise PianoScoreError(f"Input file not found: {path.resolve()}")

    def _report(stage: int) -> None:
        if progress is not None:
            progress(stage)

    if _is_midi_path(path):
        t0 = _time.perf_counter()
        _report(1)
        document = load_midi(path)
        _report(2)
        info: dict[str, Any] = {
            "duration": document.duration,
            "tracks": len(document.tracks),
            "instruments": [track.instrument for track in document.tracks],
            "tempo": document.tempo,
            "timings": {"decode": _time.perf_counter() - t0, "transcription": 0.0},
        }
        return "midi", list(document.notes), info

    if is_audio_path(path):
        transcriber = create_transcriber(args.backend, **_backend_kwargs(args))
        segment_decode = None
        t0 = _time.perf_counter()
        if time_range is not None:
            _report(1)
            try:
                buffer, window = load_audio_segment(
                    path,
                    time_range.start,
                    time_range.end,
                    context_before=getattr(args, "segment_context_before", 2.0),
                    context_after=getattr(args, "segment_context_after", 2.0),
                )
                segment_decode = window
            except SegmentDecodeError:
                _warn(
                    "Segment decode unavailable; falling back to full-track transcription."
                )
                buffer = load_audio(path)
        else:
            _report(1)
            buffer = load_audio(path)
        decode_seconds = _time.perf_counter() - t0

        _report(2)
        t0 = _time.perf_counter()
        raw_notes = transcriber.transcribe(buffer)
        transcription_seconds = _time.perf_counter() - t0

        if segment_decode is not None:
            # Map the local segment times back onto the source timeline.
            offset = segment_decode.start
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
            duration_value = segment_decode.end - segment_decode.start
        else:
            notes = list(raw_notes)
            duration_value = buffer.duration

        info = {
            "duration": duration_value,
            "sample_rate": buffer.sample_rate,
            "backend": getattr(transcriber, "name", args.backend),
            "device": args.device,
            "segment_decode": segment_decode,
            "timings": {"decode": decode_seconds, "transcription": transcription_seconds},
        }
        if hasattr(transcriber, "describe"):
            info["backend_details"] = transcriber.describe()
            if info["backend_details"].get("device"):
                info["device"] = info["backend_details"]["device"]
        return "audio", notes, info

    audio_supported = ", .".join(sorted(ext.lstrip(".") for ext in SUPPORTED_AUDIO_EXTENSIONS))
    midi_supported = ", ".join(sorted(ext.lstrip(".") for ext in SUPPORTED_MIDI_EXTENSIONS))
    raise PianoScoreError(
        f"Unsupported input format: {path.suffix or '<none>'}. "
        f"Supported: {audio_supported} (audio), {midi_supported} (MIDI)."
    )


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #
def _mean_confidence(notes: Sequence[NoteEvent]) -> float | None:
    values = [note.confidence for note in notes if note.confidence is not None]
    if not values:
        return None
    return sum(values) / len(values)


def _pitch_range(notes: Sequence[NoteEvent]) -> tuple[int, int]:
    pitches = [note.pitch for note in notes]
    return min(pitches), max(pitches)


def _build_report(
    path: Path,
    input_type: str,
    notes: Sequence[NoteEvent],
    info: dict[str, Any],
) -> dict[str, Any]:
    lowest, highest = _pitch_range(notes)
    polyphony = polyphony_stats(notes)
    report: dict[str, Any] = {
        "input": str(path),
        "input_type": input_type,
        "duration": float(info.get("duration", 0.0)),
        "note_count": len(notes),
        "pitch_min": lowest,
        "pitch_max": highest,
        "pitch_range": f"{midi_pitch_to_name(lowest)}-{midi_pitch_to_name(highest)}",
        "mean_confidence": _mean_confidence(notes),
        "polyphony_max": polyphony.maximum,
        "polyphony_mean": round(polyphony.mean, 4),
    }
    for key in ("sample_rate", "backend", "device", "tracks", "instruments", "tempo", "backend_details"):
        if key in info:
            report[key] = info[key]
    report["notes"] = [note.to_dict() for note in notes]
    return report


def _row(label: str, value: Any) -> str:
    return f"{label:<{_LABEL_WIDTH}} : {value}"


# --------------------------------------------------------------------------- #
# user settings (CLI > settings > defaults)
# --------------------------------------------------------------------------- #
def _apply_settings(args: argparse.Namespace, settings: UserSettings) -> None:
    """Fill options the user did not pass explicitly from their preferences."""
    def pick(attr: str, value: Any) -> None:
        if getattr(args, attr, None) is None:
            setattr(args, attr, value)

    pick("style", settings.default_style)
    pick("range_mode", settings.default_range_mode)
    pick("transpose", settings.default_transpose)
    if getattr(args, "section_duration", None) is None:
        args.section_duration = settings.default_section_duration
    if not hasattr(args, "segment_context_before"):
        args.segment_context_before = settings.segment_context_before
    if not hasattr(args, "segment_context_after"):
        args.segment_context_after = settings.segment_context_after


def _stage(args: argparse.Namespace, index: int, total: int, text: str) -> None:
    print(f"[{index}/{total}] {text}")


def _warn(text: str) -> None:
    print(f"Warning: {text}")


def _print_report(report: dict[str, Any]) -> None:
    lines = [
        _row("Input", report["input"]),
        _row("Input type", report["input_type"]),
        _row("Duration", f"{report['duration']:.2f} s"),
    ]
    if "sample_rate" in report:
        lines.append(_row("Sample rate", f"{report['sample_rate']} Hz"))
    if "tracks" in report:
        lines.append(_row("Tracks", report["tracks"]))
        lines.append(_row("Instruments", ", ".join(report.get("instruments") or []) or "<none>"))
    if "backend" in report:
        lines.append(_row("Backend", report["backend"]))
        lines.append(_row("Device", report.get("device", "?")))
    lines.append(_row("Notes", report["note_count"]))
    lines.append(_row("Pitch range", f"{report['pitch_min']}-{report['pitch_max']} ({report['pitch_range']})"))
    lines.append(
        _row("Polyphony", f"max {report['polyphony_max']}, mean {report['polyphony_mean']:.2f}")
    )
    confidence = report["mean_confidence"]
    if confidence is not None:
        confidence_text = f"{confidence:.3f} (mean)"
    elif report["input_type"] == "midi":
        confidence_text = "n/a (MIDI input)"
    else:
        confidence_text = "not reported by this backend"
    lines.append(_row("Confidence", confidence_text))
    print("\n".join(lines))


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #
def _command_analyze(args: argparse.Namespace) -> int:
    path = Path(args.input)
    input_type, notes, info = _load_notes(path, args)
    if not notes:
        raise NoNotesError(f"no notes were produced from {path.name}")

    report = _build_report(path, input_type, notes, info)
    _print_report(report)

    if args.json_path:
        target = Path(args.json_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(_row("Report", str(target)))
    return 0


def _command_transcribe(args: argparse.Namespace) -> int:
    path = Path(args.input)
    if _is_midi_path(path):
        raise PianoScoreError(
            f"{path.name} is already a MIDI file, so there is nothing to transcribe. "
            "Use 'analyze' to inspect it instead."
        )

    input_type, notes, info = _load_notes(path, args)
    if not notes:
        raise NoNotesError(f"transcription of {path.name} produced no notes")

    if args.output:
        output = Path(args.output)
    else:
        output = path.with_name(f"{path.stem}_transcribed.mid")
    written = write_midi(notes, output)

    lowest, highest = _pitch_range(notes)
    print(_row("Input", str(path)))
    print(_row("Notes", len(notes)))
    print(_row("Pitch range", f"{lowest}-{highest} ({midi_pitch_to_name(lowest)}-{midi_pitch_to_name(highest)})"))
    if "backend" in info:
        print(_row("Backend", f"{info['backend']} ({info.get('device', '?')})"))
    print(_row("Wrote MIDI", str(written)))
    return 0


def _melody_report(
    path: Path,
    input_type: str,
    result: MelodyExtractionResult,
    output: Path,
    segment: TimeRange | None = None,
) -> dict[str, Any]:
    notes = result.notes
    lowest, highest = _pitch_range(notes)
    polyphony = polyphony_stats(notes)
    return {
        "input": str(path),
        "input_type": input_type,
        "output": str(output),
        "source_segment": None if segment is None else {"start": segment.start, "end": segment.end},
        "strategy": result.strategy,
        "input_note_count": result.input_note_count,
        "filtered_note_count": result.filtered_note_count,
        "onset_group_count": result.onset_group_count,
        "total_group_count": result.total_group_count,
        "selected_group_count": result.selected_group_count,
        "skipped_group_count": result.skipped_group_count,
        "output_note_count": result.output_note_count,
        "removed_short_count": result.removed_short_count,
        "removed_pitch_range_count": result.removed_pitch_range_count,
        "overlap_truncated_count": result.overlap_truncated_count,
        "dropped_fragment_count": result.dropped_fragment_count,
        "pitch_min": lowest,
        "pitch_max": highest,
        "pitch_range": f"{midi_pitch_to_name(lowest)}-{midi_pitch_to_name(highest)}",
        "polyphony_max": polyphony.maximum,
        "polyphony_mean": round(polyphony.mean, 4),
        "notes": [note.to_dict() for note in notes],
    }


def _command_extract_melody(args: argparse.Namespace) -> int:
    path = Path(args.input)
    time_range = _prepare_time_range(args, path)
    input_type, notes, info = _load_notes(path, args, time_range)
    segment, notes = _apply_time_range(args, notes, info, time_range)
    if not notes:
        where = (
            f" in segment {format_clock(segment.start)}-{format_clock(segment.end)}"
            if segment
            else ""
        )
        raise NoNotesError(
            f"No piano notes were detected in {path.name}{where}. "
            "Try choosing a different segment or a cleaner piano recording."
        )

    config = MelodyExtractionConfig(
        strategy=args.strategy,
        min_note_duration=args.min_note_duration,
        onset_epsilon=args.onset_epsilon,
    )
    result = MelodyExtractor(config).extract(notes)
    if not result.notes:
        raise NoNotesError(f"melody extraction produced no notes for {path.name}")
    assert_monophonic(result.notes)

    output = Path(args.output) if args.output else path.with_name(f"{path.stem}_melody.mid")
    written = write_midi(result.notes, output, track_name="melody")
    report = _melody_report(path, input_type, result, written, segment)

    lowest, highest = _pitch_range(result.notes)
    print(_row("Input", str(path)))
    if segment is not None:
        print(_row("Segment", f"{segment.start:g}-{segment.end:g} s ({segment.duration:g} s of the source)"))
    print(_row("Strategy", result.strategy))
    print(_row("Input notes", result.input_note_count))
    print(_row("Filtered", result.filtered_note_count))
    print(_row("Onset groups", result.total_group_count))
    print(_row("Selected groups", result.selected_group_count))
    print(_row("Skipped groups", result.skipped_group_count))
    print(_row("Melody notes", result.output_note_count))
    print(_row("Pitch range", f"{lowest}-{highest} ({midi_pitch_to_name(lowest)}-{midi_pitch_to_name(highest)})"))
    print(_row("Polyphony", f"max {report['polyphony_max']}, mean {report['polyphony_mean']:.2f}"))
    if result.removed_short_count or result.removed_pitch_range_count:
        print(
            _row(
                "Filtered out",
                f"{result.removed_short_count} short, {result.removed_pitch_range_count} out of range",
            )
        )
    if result.overlap_truncated_count or result.dropped_fragment_count:
        print(
            _row(
                "Overlaps",
                f"{result.overlap_truncated_count} truncated, {result.dropped_fragment_count} fragments dropped",
            )
        )
    print(_row("Output", str(written)))

    if args.json_path:
        target = Path(args.json_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(_row("Report", str(target)))
    return 0


# --------------------------------------------------------------------------- #
# harmonica conversion
# --------------------------------------------------------------------------- #
def _pitch_range_label(lowest: int | None, highest: int | None) -> str:
    if lowest is None or highest is None:
        return "n/a"
    return f"{midi_pitch_to_name(lowest)}-{midi_pitch_to_name(highest)}"


def _load_profile(path: str | None) -> InstrumentProfile:
    return InstrumentProfile.default() if path is None else InstrumentProfile.load(path)


def _melody_for_conversion(
    args: argparse.Namespace, notes: Sequence[NoteEvent]
) -> MelodyExtractionResult:
    config = MelodyExtractionConfig(
        strategy=args.strategy,
        min_note_duration=args.min_note_duration,
        onset_epsilon=args.onset_epsilon,
    )
    result = MelodyExtractor(config).extract(notes)
    if not result.notes:
        raise NoNotesError(
            "No usable melody notes remained after melody extraction. "
            "Try a different --strategy or a cleaner piano recording."
        )
    assert_monophonic(result.notes)
    return result


def _command_config(args: argparse.Namespace) -> int:
    path = settings_path()
    settings, warnings = load_settings()
    for note in warnings:
        _warn(note)

    if args.reset:
        save_settings(UserSettings(), path)
        print(f"Reset user settings to defaults: {path}")
        return 0

    if args.set:
        for key, raw in args.set:
            value = parse_setting_value(key, raw)
            settings = with_updated_field(settings, key, value)
        save_settings(settings, path)
        print(f"Saved {len(args.set)} setting(s) to {path}")

    print(_row("Settings file", str(path)))
    for field_name in (
        "default_style",
        "default_section_duration",
        "default_range_mode",
        "default_transpose",
        "segment_context_before",
        "segment_context_after",
    ):
        print(_row(field_name, getattr(settings, field_name)))
    print(_row("Priority", "CLI argument > settings > built-in default"))
    return 0


def _run_conversion(args: argparse.Namespace, path: Path) -> int:
    import time

    timings: dict[str, float] = {}
    t_start = time.perf_counter()

    time_range = _prepare_time_range(args, path)
    print(_row("Converting", path.name))
    if time_range is not None:
        print(
            _row(
                "Segment",
                f"{format_clock(time_range.start)} → {format_clock(time_range.end)} "
                f"({time_range.duration:g} s of the source)",
            )
        )

    # Decide the output format up front so the stage list can match it.
    # convert defaults to the final PNG; map-melody defaults to score.json.
    explicit_output = Path(args.output) if args.output else None
    default_format = ".png" if args.command == "convert" else ".score.json"
    render_png = (
        explicit_output.suffix.lower() == ".png" if explicit_output else default_format == ".png"
    )
    total_stages = 5
    stage_labels = [
        "Loading audio..." if not _is_midi_path(path) else "Loading MIDI...",
        "Transcribing piano...",
        "Extracting melody...",
        "Mapping harmonica notes...",
        "Rendering PNG..." if render_png else "Writing score JSON...",
    ]

    def progress(stage: int) -> None:
        _stage(args, stage, total_stages, stage_labels[stage - 1])

    t0 = time.perf_counter()
    input_type, notes, info = _load_notes(path, args, time_range, progress=progress)
    load_timings: dict[str, float] = info.pop("timings", {})
    timings.update(load_timings)
    segment, notes = _apply_time_range(args, notes, info, time_range)
    if not notes:
        where = (
            f" in segment {format_clock(segment.start)}-{format_clock(segment.end)}"
            if segment
            else ""
        )
        raise NoNotesError(
            f"No piano notes were detected in {path.name}{where}. "
            "Try choosing a different segment or a cleaner piano recording."
        )

    progress(3)
    t0 = time.perf_counter()
    melody = _melody_for_conversion(args, notes)
    timings["melody"] = time.perf_counter() - t0

    progress(4)
    t0 = time.perf_counter()
    profile = _load_profile(args.profile)
    config = ConversionConfig(transpose=args.transpose, range_mode=args.range_mode)
    result = convert_melody(melody.notes, config, profile)
    timings["conversion"] = time.perf_counter() - t0

    if not result.notes:
        raise NoNotesError(
            f"Every melody note was dropped by range mode {config.range_mode!r} "
            f"on profile {profile.name!r}."
        )

    segment_value = (segment.start, segment.end) if segment is not None else None
    if explicit_output is not None:
        output = unique_path(explicit_output)
        if output != explicit_output:
            _warn(f"{explicit_output.name} already exists; writing {output.name} instead.")
    else:
        suffix = ".png" if render_png else ".score.json"
        output = unique_path(default_output_path(path, suffix, segment=segment_value))

    progress(5)
    t0 = time.perf_counter()
    if render_png:
        preset = get_preset(args.style)
        layout, layout_config = _layout_from_conversion(
            result, args, preset, args.title, source=str(path), segment=segment_value
        )
        written = export_score_png(layout, output, style=preset.style, config=layout_config)
    else:
        written = write_score_json(result, output, source_segment=segment_value)
    timings["rendering"] = time.perf_counter() - t0
    elapsed = time.perf_counter() - t_start

    report = result.report
    transpose = f"{report.transpose:+d}" if report.transpose else "0"
    if report.auto_transpose:
        transpose += " (auto)"
    print()
    print(_row("Notes", report.output_note_count))
    print(_row("Transpose", transpose))
    print(_row("Playable", f"{report.playable_ratio * 100:.1f}%"))
    print(_row("Elapsed", f"{elapsed:.1f} s"))

    if args.verbose:
        print(_row("Backend", f"{info.get('backend', '?')} ({info.get('device', '?')})"))
        window = info.get("segment_decode")
        if window is not None:
            print(
                _row(
                    "Decoded window",
                    f"{window.start:.2f}-{window.end:.2f} s "
                    f"(context {window.context_before:.1f} s before / "
                    f"{window.context_after:.1f} s after)",
                )
            )
        print(_row("Transcribed", f"{info.get('duration', 0.0):.1f} s of audio"))
        print(_row("Melody strategy", melody.strategy))
        print(_row("Groups", f"{melody.selected_group_count} selected, "
                            f"{melody.skipped_group_count} skipped"))
        print(_row("Range mode", f"{report.range_mode} ({report.dropped_count} dropped)"))
        print(_row("Modifiers", f"{report.semitone_count} semitone, "
                                f"{report.octave_shift_count} octave"))
        breakdown = "  ".join(f"{key}: {value:.1f} s" for key, value in timings.items())
        print(_row("Timings", breakdown))
        print(_row("Style", args.style))

    print()
    print("Saved score:")
    print(str(written.resolve()))
    return 0


def _command_map_melody(args: argparse.Namespace) -> int:
    return _run_conversion(args, Path(args.input))


def _command_convert(args: argparse.Namespace) -> int:
    path = Path(args.input)
    if _is_midi_path(path):
        raise PianoScoreError(
            f"{path.name} is already a MIDI file; use 'map-melody' to turn it into a score."
        )
    return _run_conversion(args, path)


# --------------------------------------------------------------------------- #
# static score layout
# --------------------------------------------------------------------------- #
def _resolved_layout_config(args: argparse.Namespace, preset: VisualPreset) -> ScoreLayoutConfig:
    """Preset geometry, overridden by explicitly passed CLI values."""
    overrides: dict[str, Any] = {}
    if args.section_duration is not None:
        overrides["section_duration"] = args.section_duration
    if args.canvas_width is not None:
        overrides["canvas_width"] = args.canvas_width
    if not overrides:
        return preset.layout
    return replace_dataclass(preset.layout, **overrides)


def _layout_from_document(
    document: Any,
    args: argparse.Namespace,
    preset: VisualPreset,
    title: str | None,
) -> tuple[Any, ScoreLayoutConfig]:
    """Build the static layout of a loaded ``score.json`` document."""
    config = _resolved_layout_config(args, preset)
    header = LayoutHeader(
        title=title or f"{document.profile_name} score",
        source=document.source,
        profile=document.profile_name,
        duration=document.duration,
        transpose=document.transpose,
        auto_transpose=document.auto_transpose,
        range_mode=document.range_mode,
        note_count=document.note_count,
        playable_ratio=float(document.report.get("playable_ratio", 0.0)),
        segment=document.source_segment,
    )
    layout = build_layout(
        document.notes,
        lanes=document.lanes,
        config=config,
        header=header,
        symbols=document.symbols,
    )
    return layout, config


def _layout_from_conversion(
    result: ConversionResult,
    args: argparse.Namespace,
    preset: VisualPreset,
    title: str | None,
    source: str | None = None,
    segment: tuple[float, float] | None = None,
) -> tuple[Any, ScoreLayoutConfig]:
    """Build the static layout straight out of a conversion result."""
    config = _resolved_layout_config(args, preset)
    header = LayoutHeader(
        title=title or f"{result.profile.name} score",
        source=source,
        profile=result.profile.name,
        duration=result.duration,
        transpose=result.report.transpose,
        auto_transpose=result.report.auto_transpose,
        range_mode=result.report.range_mode,
        note_count=result.report.output_note_count,
        playable_ratio=result.report.playable_ratio,
        segment=segment,
    )
    layout = build_layout(
        result.notes,
        lanes=result.profile.base_keys,
        config=config,
        header=header,
        symbols=NoteLabelSymbols.from_profile(result.profile),
    )
    return layout, config


def _prepare_time_range(args: argparse.Namespace, path: Path) -> TimeRange | None:
    """Parse and validate ``--start`` / ``--end`` before any heavy work."""
    start_text = getattr(args, "start", None)
    end_text = getattr(args, "end", None)
    if start_text is None and end_text is None:
        return None
    if not path.exists():
        raise PianoScoreError(f"Input file not found: {path.resolve()}")

    # Cheap duration probes so range errors surface before transcription.
    total_duration: float | None
    if _is_midi_path(path):
        total_duration = load_midi(path).duration
    elif is_audio_path(path):
        total_duration = probe_audio_duration(path)
    else:
        total_duration = None

    time_range = normalize_time_range(
        parse_time_value(start_text), parse_time_value(end_text), total_duration
    )
    if time_range.end_clamped:
        _warn(f"--end clamped to the media duration ({format_clock(time_range.end)}).")
    return time_range


def _apply_time_range(
    args: argparse.Namespace,
    notes: list[NoteEvent],
    info: dict[str, Any],
    time_range: TimeRange | None,
) -> tuple[TimeRange | None, list[NoteEvent]]:
    """Strictly crop already-normalised notes into *time_range* and re-base."""
    if time_range is None:
        return None, notes
    cropped = crop_note_events(notes, time_range.start, time_range.end)
    return time_range, cropped


def _command_layout_score(args: argparse.Namespace) -> int:
    path = Path(args.input)
    document = load_score_document(path)
    layout, config = _layout_from_document(document, args, get_preset(DEFAULT_PRESET_NAME), args.title)

    output = Path(args.output) if args.output else _default_layout_output(path)
    written = write_layout_json(layout, output, config)

    print(_row("Input", str(path)))
    print(_row("Notes", document.note_count))
    print(_row("Score duration", f"{document.duration:.2f} s"))
    print(_row("Layout duration", f"{layout.duration:.2f} s"))
    print(_row("Section duration", f"{config.section_duration:g} s"))
    print(_row("Sections", layout.section_count))
    print(_row("Fragments", layout.fragment_count))
    print(_row("Canvas", f"{layout.width:g} x {layout.height:g}"))
    print(_row("Output", str(written)))
    return 0


def _command_render_score(args: argparse.Namespace) -> int:
    path = Path(args.input)
    document = load_score_document(path)
    preset = get_preset(args.style)
    layout, config = _layout_from_document(document, args, preset, args.title)

    output = Path(args.output) if args.output else _default_png_output(path)
    written = export_score_png(layout, output, style=preset.style, config=config)

    print(_row("Input", str(path)))
    print(_row("Style", preset.name))
    print(_row("Notes", document.note_count))
    print(_row("Sections", layout.section_count))
    print(_row("Fragments", layout.fragment_count))
    print(_row("Canvas", f"{layout.width:g} x {layout.height:g}"))
    print(_row("Output", str(written)))
    return 0


def _default_png_output(path: Path) -> Path:
    """``song.score.json`` -> ``song.png`` (fall back to ``<stem>.png``)."""
    stem = path.stem
    if path.suffix == ".json" and stem.endswith(".score"):
        stem = stem[: -len(".score")]
    return path.with_name(f"{stem}.png")


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def _configure_warnings(debug: bool) -> None:
    if debug:
        return
    # Keep the CLI output readable: torch/transkun emit a lot of deprecation noise.
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", module=r"^torch\..*")
    warnings.filterwarnings("ignore", module=r"^transkun\..*")
    warnings.filterwarnings("ignore", module=r"^moduleconf.*")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_warnings(getattr(args, "debug", False))

    settings, setting_warnings = load_settings()
    for note in setting_warnings:
        _warn(note)
    _apply_settings(args, settings)

    try:
        return int(args.handler(args))
    except PianoScoreError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return exc.exit_code
    except KeyboardInterrupt:  # pragma: no cover - interactive only
        print("ERROR: interrupted", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001 - last-resort guard for a clean CLI
        if getattr(args, "debug", False):
            traceback.print_exc()
        else:
            print(
                f"ERROR: unexpected failure: {exc} (re-run with --debug for the full traceback)",
                file=sys.stderr,
            )
        return 1


__all__ = ["build_parser", "main"]
