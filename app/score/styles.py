"""Visual style of the rendered score.

Every colour, font size and stroke width lives here so the renderer itself
contains no magic numbers. The first palette is deliberately plain: light
background, dark text, medium-contrast note blocks, pale grid - clear and
printer friendly rather than flashy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from app.core.errors import ScoreRenderError
from app.score.layout import ScoreLayoutConfig


@dataclass(frozen=True)
class ScoreStyle:
    """All tunables of the PNG renderer."""

    # -- colours ----------------------------------------------------------- #
    background: str = "#ffffff"
    text: str = "#1a1a1a"
    text_muted: str = "#666666"
    grid_major: str = "#c0c4c8"
    grid_minor: str = "#e3e6e9"
    lane_line: str = "#d4d8dc"
    lane_background: str = "#f3f5f7"
    note_fill: str = "#2f6fde"
    note_outline: str = "#1b4a97"
    note_label: str = "#ffffff"
    #: A fragment that continues a note from the previous section is the *same*
    #: note, so it gets a lighter, quieter look instead of a fresh label.
    continuation_fill: str = "#a9c6ef"
    continuation_outline: str = "#6f9bd8"
    #: Horizontal rule that separates two sections on long scores.
    section_rule: str = "#c2c7cc"

    # -- sizes ------------------------------------------------------------- #
    header_font_size: int = 28
    # Sized so that "00:00–00:06" fits into the left label column.
    section_font_size: int = 15
    lane_font_size: int = 16
    note_font_size: int = 13

    corner_radius: int = 4

    grid_major_width: int = 1
    grid_minor_width: int = 1
    lane_line_width: int = 1
    label_padding: int = 4

    def __post_init__(self) -> None:
        for name, value in (
            ("header_font_size", self.header_font_size),
            ("section_font_size", self.section_font_size),
            ("lane_font_size", self.lane_font_size),
            ("note_font_size", self.note_font_size),
            ("grid_major_width", self.grid_major_width),
            ("grid_minor_width", self.grid_minor_width),
            ("lane_line_width", self.lane_line_width),
            ("label_padding", self.label_padding),
        ):
            if not math.isfinite(float(value)) or value <= 0:
                raise ScoreRenderError(f"{name} must be a positive number, got {value!r}")
        if self.corner_radius < 0:
            raise ScoreRenderError(f"corner_radius must be >= 0, got {self.corner_radius}")

        for name, value in (
            ("background", self.background),
            ("text", self.text),
            ("text_muted", self.text_muted),
            ("grid_major", self.grid_major),
            ("grid_minor", self.grid_minor),
            ("lane_line", self.lane_line),
            ("lane_background", self.lane_background),
            ("note_fill", self.note_fill),
            ("note_outline", self.note_outline),
            ("note_label", self.note_label),
            ("continuation_fill", self.continuation_fill),
            ("continuation_outline", self.continuation_outline),
            ("section_rule", self.section_rule),
        ):
            _validate_colour(name, value)


def _validate_colour(name: str, value: str) -> None:
    if not isinstance(value, str):
        raise ScoreRenderError(f"{name} must be a colour string, got {value!r}")
    body = value[1:] if value.startswith("#") else value
    if len(body) not in (3, 6) or any(c not in "0123456789abcdefABCDEF" for c in body):
        raise ScoreRenderError(f"{name} must be a hex colour like '#rrggbb', got {value!r}")


DEFAULT_STYLE = ScoreStyle()


# --------------------------------------------------------------------------- #
# visual presets
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class VisualPreset:
    """A named bundle of render style + layout geometry.

    The *music semantics* (time -> x, lane -> y, duration -> width) are the same
    in every preset; only scale, spacing and visual weight differ.
    """

    name: str
    style: ScoreStyle
    layout: ScoreLayoutConfig


#: Overview rendering - the original M5 look, good for whole-song exports.
COMPACT = VisualPreset(
    name="compact",
    style=DEFAULT_STYLE,
    layout=ScoreLayoutConfig(),
)

#: Sight-reading preset - taller rows, bigger fonts, stronger separators.
PRACTICE = VisualPreset(
    name="practice",
    style=ScoreStyle(
        header_font_size=40,
        section_font_size=19,
        lane_font_size=22,
        note_font_size=18,
        corner_radius=6,
        label_padding=6,
        grid_major="#b4bac0",
        grid_minor="#dfe3e7",
        lane_line="#c9cfd5",
        lane_background="#eef1f4",
    ),
    layout=ScoreLayoutConfig(
        canvas_width=2400,
        header_height=170.0,
        section_height=330.0,
        left_label_width=96.0,
        right_padding=28.0,
        top_padding=24.0,
        bottom_padding=24.0,
        lane_height=34.0,
        note_height=26.0,
        min_note_width=8.0,
    ),
)

PRESETS: dict[str, VisualPreset] = {"compact": COMPACT, "practice": PRACTICE}
DEFAULT_PRESET_NAME = "compact"


def get_preset(name: str) -> VisualPreset:
    try:
        return PRESETS[name]
    except KeyError:
        known = ", ".join(sorted(PRESETS))
        raise ScoreRenderError(f"unknown visual preset {name!r}; available: {known}") from None


__all__ = [
    "ScoreStyle",
    "DEFAULT_STYLE",
    "VisualPreset",
    "COMPACT",
    "PRACTICE",
    "PRESETS",
    "DEFAULT_PRESET_NAME",
    "get_preset",
]
