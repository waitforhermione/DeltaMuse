"""Deterministic, cross-platform default output naming.

Rules:

- outputs live next to the input file;
- a cropped segment is encoded in the file name as ``00m48s-01m22s`` (no ``:``,
  no characters that are illegal on Windows);
- an existing target is never silently overwritten - ``unique_path`` appends
  ``_2``, ``_3``, ... which is deterministic and testable.
"""

from __future__ import annotations

from pathlib import Path


def format_segment_tag(start: float, end: float) -> str:
    """``48.0`` -> ``00m48s``; hours appear only when needed (``01h20m29s``)."""
    return f"{_clock_tag(start)}-{_clock_tag(end)}"


def _clock_tag(seconds: float) -> str:
    total = max(int(round(seconds)), 0)
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours:02d}h{minutes:02d}m{secs:02d}s"
    return f"{minutes:02d}m{secs:02d}s"


def default_output_path(
    input_path: str | Path,
    suffix: str,
    *,
    segment: tuple[float, float] | None = None,
    stem_suffix: str = "_score",
) -> Path:
    """Build the default output path next to the input file.

    ``song.mp3`` + ``.png``                 -> ``song_score.png``
    ``song.mp3`` + ``.png`` (48-82 s)       -> ``song_00m48s-01m22s_score.png``
    ``song.mp3`` + ``.score.json``          -> ``song.score.json``
    ``song.mp3`` + ``.score.json`` (48-82)  -> ``song_00m48s-01m22s.score.json``
    """
    source = Path(input_path)
    tag = "" if segment is None else f"_{format_segment_tag(*segment)}"
    if suffix == ".png":
        return source.with_name(f"{source.stem}{tag}{stem_suffix}{suffix}")
    return source.with_name(f"{source.stem}{tag}{suffix}")


def unique_path(path: Path) -> Path:
    """Return *path*, or the first free ``<stem>_2``, ``<stem>_3``, ... variant."""
    if not path.exists():
        return path
    for index in range(2, 10_000):
        candidate = path.with_name(f"{path.stem}_{index}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise FileExistsError(f"could not find a free output name for {path}")


__all__ = ["format_segment_tag", "default_output_path", "unique_path"]
