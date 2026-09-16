"""Write a rendered score to an image file.

This is the only module that turns a :class:`LayoutScore` into a file on disk;
the drawing itself lives in :mod:`app.score.renderer_png`.
"""

from __future__ import annotations

from pathlib import Path

from app.core.errors import ScoreRenderError
from app.score.layout import LayoutScore, ScoreLayoutConfig
from app.score.renderer_png import render_score
from app.score.styles import DEFAULT_STYLE, ScoreStyle

PNG_FORMAT = "PNG"


def export_score_png(
    layout: LayoutScore,
    output_path: str | Path,
    *,
    style: ScoreStyle = DEFAULT_STYLE,
    config: ScoreLayoutConfig | None = None,
) -> Path:
    """Render *layout* and write it as a PNG file. Returns the resolved path."""
    target = Path(output_path)
    try:
        image = render_score(layout, style=style, config=config)
        parent = target.parent
        if parent and not parent.exists():
            parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.is_dir():
            raise IsADirectoryError(f"{target} is a directory")
        image.save(str(target), format=PNG_FORMAT)
    except ScoreRenderError:
        raise
    except (OSError, ValueError) as exc:
        raise ScoreRenderError(f"cannot write score PNG to {target}: {exc}") from exc

    return target


__all__ = ["PNG_FORMAT", "export_score_png"]
