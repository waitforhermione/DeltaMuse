"""Debug export of a static score layout as JSON.

Milestone 4 has no renderer, so this file is how a layout gets inspected: it
contains every section, grid line and note fragment with its final geometry.
It is a *debug* artefact - the renderer does not depend on it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.core.errors import OutputNotWritableError
from app.score.layout import LayoutScore, ScoreLayoutConfig

LAYOUT_SCHEMA_VERSION = 1


def build_layout_document(layout: LayoutScore, config: ScoreLayoutConfig | None = None) -> dict[str, Any]:
    """Build the layout document as plain, JSON-ready Python objects."""
    settings = config if config is not None else ScoreLayoutConfig()
    header = layout.header

    return {
        "version": LAYOUT_SCHEMA_VERSION,
        "canvas": {"width": layout.width, "height": layout.height},
        "duration": layout.duration,
        "lanes": list(layout.lanes),
        "config": {
            "canvas_width": settings.canvas_width,
            "section_duration": settings.section_duration,
            "header_height": settings.header_height,
            "section_height": settings.section_height,
            "left_label_width": settings.left_label_width,
            "right_padding": settings.right_padding,
            "top_padding": settings.top_padding,
            "bottom_padding": settings.bottom_padding,
            "lane_height": settings.lane_height,
            "note_height": settings.note_height,
            "min_note_width": settings.min_note_width,
            "major_grid_step": settings.major_grid_step,
            "minor_grid_step": settings.minor_grid_step,
        },
        "header": {
            "title": header.title,
            "source": header.source,
            "profile": header.profile,
            "duration": header.duration,
            "transpose": header.transpose,
            "auto_transpose": header.auto_transpose,
            "range_mode": header.range_mode,
            "note_count": header.note_count,
            "playable_ratio": header.playable_ratio,
        },
        "sections": [
            {
                "index": section.index,
                "start_time": section.start_time,
                "end_time": section.end_time,
                "y": section.y,
                "height": section.height,
                "grid": [
                    {"x": line.x, "time": line.time, "kind": line.kind} for line in section.grid
                ],
                "notes": [
                    {
                        "id": fragment.note_id,
                        "source_note_id": fragment.source_note_id,
                        "x": fragment.x,
                        "y": fragment.y,
                        "width": fragment.width,
                        "height": fragment.height,
                        "label": fragment.label,
                        "lane": fragment.lane,
                        "start": fragment.start,
                        "duration": fragment.duration,
                        "fragment_start": fragment.fragment_start,
                        "fragment_duration": fragment.fragment_duration,
                        "is_continuation": fragment.is_continuation,
                        "confidence": fragment.confidence,
                    }
                    for fragment in section.notes
                ],
            }
            for section in layout.sections
        ],
    }


def layout_json(layout: LayoutScore, config: ScoreLayoutConfig | None = None) -> str:
    """Serialise the layout to a deterministic JSON string."""
    return json.dumps(build_layout_document(layout, config), indent=2, ensure_ascii=False) + "\n"


def write_layout_json(
    layout: LayoutScore,
    output_path: str | Path,
    config: ScoreLayoutConfig | None = None,
) -> Path:
    """Write the layout debug JSON and return the resolved path."""
    target = Path(output_path)
    try:
        parent = target.parent
        if parent and not parent.exists():
            parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.is_dir():
            raise IsADirectoryError(f"{target} is a directory")
        target.write_text(layout_json(layout, config), encoding="utf-8", newline="\n")
    except OSError as exc:
        raise OutputNotWritableError(f"cannot write layout JSON to {target}: {exc}") from exc

    return target


__all__ = ["LAYOUT_SCHEMA_VERSION", "build_layout_document", "layout_json", "write_layout_json"]
