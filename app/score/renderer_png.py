"""PNG renderer for the static score.

This module is a *pure presentation* layer. It only ever sees a
:class:`~app.score.layout.LayoutScore` - never a MIDI file, audio, a
``PlayableNote`` list, TransKun, the mapper or the converter - and it never
recomputes geometry: every rectangle, label and grid line is taken verbatim from
the layout. The :class:`~app.score.layout.ScoreLayoutConfig` that produced the
layout is passed in only to recover the drawing-area edges.

Draw order is fixed so the same layout always paints identically::

    background -> header -> lane rows -> grid lines -> note blocks -> note labels

All drawing goes through a :class:`DrawSink`, which makes the renderer testable
without OCR: tests record the draw calls instead of looking at pixels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

from app.core.errors import EmptyScoreError, ScoreRenderError
from app.score.layout import LayoutNote, LayoutScore, LayoutSection, ScoreLayoutConfig
from app.score.styles import DEFAULT_STYLE, ScoreStyle

#: Fonts are tried in order; the first one that covers the modifier glyphs wins.
_FONT_CANDIDATES: tuple[str, ...] = (
    "segoeui.ttf",  # Windows
    "arial.ttf",  # Windows fallback
    "msyh.ttc",  # Windows, CJK
    "DejaVuSans.ttf",  # Linux
    "calibri.ttf",
    "Ubuntu-R.ttf",
)

#: Bold variants, tried first for lane/note labels (small-size readability).
_FONT_BOLD_CANDIDATES: tuple[str, ...] = (
    "segoeuib.ttf",
    "arialbd.ttf",
    "msyhbd.ttc",
    "DejaVuSans-Bold.ttf",
    "calibrib.ttf",
    "Ubuntu-B.ttf",
)

#: Glyphs a usable font must be able to draw (modifiers, the section dash, sharp).
_REQUIRED_GLYPHS: tuple[str, ...] = ("↑", "↓", "–", "#")

_SECTION_DASH = "–"


class DrawSink(Protocol):
    """The tiny drawing API the renderer needs (Pillow, or a test double)."""

    def rectangle(self, xy: Sequence[float], fill: Any = None, outline: Any = None, width: int = 1) -> None: ...

    def rounded_rectangle(self, xy: Sequence[float], radius: float = 0, fill: Any = None, outline: Any = None, width: int = 1) -> None: ...

    def line(self, xy: Sequence[float], fill: Any = None, width: int = 1) -> None: ...

    def text(self, xy: Sequence[float], text: str, fill: Any = None, font: Any = None) -> None: ...

    def text_length(self, text: str, font: Any = None) -> float: ...


class PillowSink:
    """Adapter around :class:`PIL.ImageDraw.ImageDraw`."""

    def __init__(self, draw: Any) -> None:
        self._draw = draw

    def rectangle(self, xy: Sequence[float], fill: Any = None, outline: Any = None, width: int = 1) -> None:
        self._draw.rectangle([xy[0], xy[1], xy[2], xy[3]], fill=fill, outline=outline, width=width)

    def rounded_rectangle(self, xy: Sequence[float], radius: float = 0, fill: Any = None, outline: Any = None, width: int = 1) -> None:
        self._draw.rounded_rectangle(
            [xy[0], xy[1], xy[2], xy[3]], radius=radius, fill=fill, outline=outline, width=width
        )

    def line(self, xy: Sequence[float], fill: Any = None, width: int = 1) -> None:
        self._draw.line([xy[0], xy[1], xy[2], xy[3]], fill=fill, width=width)

    def text(self, xy: Sequence[float], text: str, fill: Any = None, font: Any = None) -> None:
        self._draw.text([xy[0], xy[1]], text, fill=fill, font=font)

    def text_length(self, text: str, font: Any = None) -> float:
        return float(self._draw.textlength(text, font=font))


# --------------------------------------------------------------------------- #
# fonts
# --------------------------------------------------------------------------- #
def _glyph_available(font: Any, glyph: str) -> bool:
    """``True`` when *font* can actually draw *glyph* (not an empty box)."""
    try:
        mask = font.getmask(glyph)
        return mask.getbbox() is not None
    except Exception:  # noqa: BLE001 - any font failure just means "unusable"
        return False


@lru_cache(maxsize=None)
def load_font(size: int, glyphs: tuple[str, ...] = _REQUIRED_GLYPHS, bold: bool = False) -> Any:
    """Return the first available font that can draw *glyphs*.

    Never raises: if no system font qualifies, Pillow's bitmap default is used so
    rendering still works on a bare system. ``bold=True`` prefers the bold
    variant of each family (used for lane and note labels, which are small).
    """
    if size <= 0:
        raise ScoreRenderError(f"font size must be positive, got {size}")

    from PIL import ImageFont

    families = (_FONT_BOLD_CANDIDATES, _FONT_CANDIDATES) if bold else (_FONT_CANDIDATES,)
    candidates: list[Any] = []
    for family in families:
        for name in family:
            try:
                candidates.append(ImageFont.truetype(name, size))
            except OSError:
                continue
    for font in candidates:
        if all(_glyph_available(font, glyph) for glyph in glyphs):
            return font
    if candidates:
        return candidates[0]
    return ImageFont.load_default()


# --------------------------------------------------------------------------- #
# small helpers (individually testable)
# --------------------------------------------------------------------------- #
def format_time(seconds: float) -> str:
    """``0`` -> ``00:00``, ``6`` -> ``00:06``, ``8.97`` -> ``00:08.97``."""
    if not math.isfinite(seconds) or seconds < 0:
        raise ScoreRenderError(f"time must be a finite value >= 0, got {seconds!r}")
    minutes = int(seconds // 60)
    rest = seconds - minutes * 60
    if math.isclose(rest, round(rest), rel_tol=1e-9, abs_tol=1e-6):
        return f"{minutes:02d}:{round(rest):02d}"
    return f"{minutes:02d}:{rest:05.2f}"


def section_right(layout: LayoutScore, config: ScoreLayoutConfig) -> float:
    """Right edge of the drawing area (notes must never paint past it)."""
    return config.canvas_width - config.right_padding


def clip_note(fragment: LayoutNote, right: float) -> tuple[float, float, float, float]:
    """Return the ``(left, top, right, bottom)`` rectangle for *fragment*.

    ``min_note_width`` can push a fragment slightly past the drawing area, so the
    renderer clips to *right* instead of trusting the layout width.
    """
    left = fragment.x
    clipped_right = min(fragment.x + fragment.width, right)
    if clipped_right <= left:
        raise ScoreRenderError(
            f"note fragment {fragment.note_id} has no visible width after clipping"
        )
    return left, fragment.y, clipped_right, fragment.y + fragment.height


@dataclass(frozen=True)
class _SectionBounds:
    top: float
    bottom: float
    left: float
    right: float
    lane_area_top: float
    lane_area_bottom: float


def _section_bounds(
    layout: LayoutScore, section: LayoutSection, config: ScoreLayoutConfig
) -> _SectionBounds:
    lane_count = len(layout.lanes)
    lane_area_top = section.y + config.lane_area_offset(lane_count)
    return _SectionBounds(
        top=section.y,
        bottom=section.y + section.height,
        left=config.left_label_width,
        right=section_right(layout, config),
        lane_area_top=lane_area_top,
        lane_area_bottom=lane_area_top + lane_count * config.lane_height,
    )


# --------------------------------------------------------------------------- #
# painters (fixed order)
# --------------------------------------------------------------------------- #
def _paint_background(sink: DrawSink, layout: LayoutScore, style: ScoreStyle) -> None:
    sink.rectangle((0.0, 0.0, layout.width, layout.height), fill=style.background)


def _format_transpose(transpose: int) -> str:
    """``3`` -> ``+3``, ``-7`` -> ``-7``, ``0`` -> ``0``."""
    return f"{transpose:+d}" if transpose else "0"


def _paint_header(
    sink: DrawSink, layout: LayoutScore, style: ScoreStyle, config: ScoreLayoutConfig
) -> None:
    header = layout.header
    x = config.left_label_width
    y = config.top_padding
    line_step = style.lane_font_size * 1.5

    title_font = load_font(style.header_font_size)
    meta_font = load_font(style.lane_font_size)

    if header.title:
        sink.text((x, y), header.title, fill=style.text, font=title_font)
        y += style.header_font_size * 1.3

    # Line 1: where the score comes from.
    origin: list[str] = []
    if header.source:
        # Basenames read better on a score sheet than full paths.
        origin.append(f"Source: {Path(header.source).name}")
    if header.profile:
        origin.append(f"Profile: {header.profile}")
    if header.duration:
        origin.append(f"Duration: {header.duration:.1f} s")
    if origin:
        sink.text((x, y), "  ·  ".join(origin), fill=style.text_muted, font=meta_font)
        y += line_step

    # Line 2: how the score was produced.
    facts: list[str] = []
    if header.segment is not None:
        seg_start, seg_end = header.segment
        facts.append(f"Segment: {format_time(seg_start)}–{format_time(seg_end)}")
    transpose = _format_transpose(header.transpose) + (" (auto)" if header.auto_transpose else "")
    facts.append(f"Transpose: {transpose}")
    if header.note_count:
        facts.append(f"Notes: {header.note_count}")
    if header.playable_ratio:
        facts.append(f"Playable: {header.playable_ratio * 100:.1f}%")
    if facts:
        sink.text((x, y), "  ·  ".join(facts), fill=style.text_muted, font=meta_font)


def _paint_section_rule(
    sink: DrawSink, section: LayoutSection, style: ScoreStyle, canvas_width: float
) -> None:
    """A subtle full-width rule marking the start of every section row."""
    sink.line((0.0, section.y, canvas_width, section.y), fill=style.section_rule, width=2)


def _paint_section_title(sink: DrawSink, section: LayoutSection, style: ScoreStyle) -> None:
    label = f"{format_time(section.start_time)}{_SECTION_DASH}{format_time(section.end_time)}"
    sink.text(
        (2.0, section.y + 2), label, fill=style.text_muted, font=load_font(style.section_font_size)
    )


def _paint_lane_rows(
    sink: DrawSink, layout: LayoutScore, bounds: _SectionBounds, style: ScoreStyle
) -> None:
    lane_count = len(layout.lanes)
    lane_height = (bounds.lane_area_bottom - bounds.lane_area_top) / lane_count
    lane_font = load_font(style.lane_font_size, bold=True)

    for lane in range(lane_count):
        band_top = bounds.lane_area_top + lane * lane_height
        band_bottom = band_top + lane_height
        if lane % 2 == 1:
            sink.rectangle(
                (bounds.left, band_top, bounds.right, band_bottom), fill=style.lane_background
            )
        sink.line(
            (bounds.left, band_bottom, bounds.right, band_bottom),
            fill=style.lane_line,
            width=style.lane_line_width,
        )
        centre = band_top + lane_height / 2
        sink.text(
            (2.0, centre - style.lane_font_size / 2),
            layout.lanes[lane],
            fill=style.text,
            font=lane_font,
        )


def _paint_grid(
    sink: DrawSink, section: LayoutSection, bounds: _SectionBounds, style: ScoreStyle
) -> None:
    # Minors first so the more visible majors end up on top of them.
    for kind, colour, width in (
        ("minor", style.grid_minor, style.grid_minor_width),
        ("major", style.grid_major, style.grid_major_width),
    ):
        for line in section.grid:
            if line.kind != kind:
                continue
            sink.line(
                (line.x, bounds.lane_area_top, line.x, bounds.lane_area_bottom),
                fill=colour,
                width=width,
            )


def _paint_notes(
    sink: DrawSink, section: LayoutSection, bounds: _SectionBounds, style: ScoreStyle
) -> None:
    for fragment in section.notes:
        left, top, right, bottom = clip_note(fragment, bounds.right)
        if fragment.is_continuation:
            # Same note as the fragment in the previous section: quieter look,
            # still clearly a bar on this lane.
            fill = style.continuation_fill
            outline = style.continuation_outline
        else:
            fill = style.note_fill
            outline = style.note_outline
        if style.corner_radius > 0 and right - left > style.corner_radius * 2:
            sink.rounded_rectangle(
                (left, top, right, bottom),
                radius=style.corner_radius,
                fill=fill,
                outline=outline,
                width=1,
            )
        else:
            sink.rectangle((left, top, right, bottom), fill=fill, outline=outline, width=1)


def _paint_note_labels(
    sink: DrawSink, section: LayoutSection, bounds: _SectionBounds, style: ScoreStyle
) -> None:
    label_font = load_font(style.note_font_size)
    for fragment in section.notes:
        if fragment.is_continuation:
            # A fragment that continues a note from the previous section is not a
            # new note, so it must not shout its label again.
            continue
        left, top, right, bottom = clip_note(fragment, bounds.right)
        label = fragment.label
        width = sink.text_length(label, label_font)
        if left + style.label_padding * 2 + width > right:
            continue  # too narrow to hold the label without spilling out
        height = bottom - top
        sink.text(
            (left + style.label_padding, top + (height - style.note_font_size) / 2),
            label,
            fill=style.note_label,
            font=label_font,
        )


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def render_score(
    layout: LayoutScore,
    *,
    style: ScoreStyle = DEFAULT_STYLE,
    config: ScoreLayoutConfig | None = None,
    image: Any = None,
    sink_factory: Callable[[Any], DrawSink] | None = None,
) -> Any:
    """Render *layout* into a PIL image (or into *sink_factory*'s recorder).

    Parameters
    ----------
    layout:
        The layout to paint. Geometry is used as-is.
    style:
        Colours and font sizes.
    config:
        The layout configuration that produced *layout*; only its margins are
        needed to recover the drawing-area edges. Defaults to the standard one.
    image:
        Optional pre-created PIL image (useful in tests).
    sink_factory:
        Optional factory returning a :class:`DrawSink` for *image*. Tests use it
        to record draw calls instead of inspecting pixels.
    """
    settings = config if config is not None else ScoreLayoutConfig()
    if not layout.sections:
        raise EmptyScoreError("cannot render a score without sections")
    if layout.width <= 0 or layout.height <= 0:
        raise ScoreRenderError(f"layout has a non-positive canvas: {layout.width} x {layout.height}")

    if image is None:
        from PIL import Image

        image = Image.new(
            "RGB", (int(round(layout.width)), int(round(layout.height))), style.background
        )
    sink = sink_factory(image) if sink_factory is not None else PillowSink(_image_draw(image))

    _paint_background(sink, layout, style)
    _paint_header(sink, layout, style, settings)
    for section in layout.sections:
        bounds = _section_bounds(layout, section, settings)
        _paint_section_rule(sink, section, style, layout.width)
        _paint_section_title(sink, section, style)
        _paint_lane_rows(sink, layout, bounds, style)
        _paint_grid(sink, section, bounds, style)
        _paint_notes(sink, section, bounds, style)
        _paint_note_labels(sink, section, bounds, style)
    return image


def _image_draw(image: Any) -> Any:
    from PIL import ImageDraw

    return ImageDraw.Draw(image)


__all__ = [
    "DrawSink",
    "PillowSink",
    "load_font",
    "format_time",
    "section_right",
    "clip_note",
    "render_score",
]
