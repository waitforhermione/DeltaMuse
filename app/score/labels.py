"""Human readable note labels for the static score.

The label is produced once, here, and the renderer (Milestone 5) must never
re-interpret ``semitone`` / ``octave_offset`` on its own.

Order is fixed: ``key`` -> ``#`` -> ``octave``::

    semitone=False, octave=0  -> Z
    semitone=True,  octave=0  -> Z#
    octave=+1                 -> Z↑      octave=-1  -> Z↓
    octave=+2                 -> Z↑2     octave=-2  -> Z↓2
    semitone=True,  octave=+1 -> Z#↑
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from app.core.instrument_profile import InstrumentProfile
from app.core.models import PlayableNote

DEFAULT_SEMITONE_SYMBOL = "#"
DEFAULT_OCTAVE_UP_SYMBOL = "↑"
DEFAULT_OCTAVE_DOWN_SYMBOL = "↓"


@dataclass(frozen=True)
class NoteLabelSymbols:
    """The modifier glyphs used to build a note label."""

    semitone: str = DEFAULT_SEMITONE_SYMBOL
    octave_up: str = DEFAULT_OCTAVE_UP_SYMBOL
    octave_down: str = DEFAULT_OCTAVE_DOWN_SYMBOL

    def __post_init__(self) -> None:
        for name, symbol in (
            ("semitone", self.semitone),
            ("octave_up", self.octave_up),
            ("octave_down", self.octave_down),
        ):
            if not isinstance(symbol, str) or not symbol:
                raise ValueError(f"{name} symbol must be a non-empty string, got {symbol!r}")

    @classmethod
    def from_profile(cls, profile: InstrumentProfile) -> "NoteLabelSymbols":
        return cls(
            semitone=profile.semitone_symbol,
            octave_up=profile.octave_up_symbol,
            octave_down=profile.octave_down_symbol,
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "NoteLabelSymbols":
        """Build from the ``symbols`` object of a ``score.json`` document."""
        if not data:
            return cls()
        return cls(
            semitone=str(data.get("semitone", DEFAULT_SEMITONE_SYMBOL)),
            octave_up=str(data.get("octave_up", DEFAULT_OCTAVE_UP_SYMBOL)),
            octave_down=str(data.get("octave_down", DEFAULT_OCTAVE_DOWN_SYMBOL)),
        )


DEFAULT_SYMBOLS = NoteLabelSymbols()


def octave_suffix(octave_offset: int, symbols: NoteLabelSymbols = DEFAULT_SYMBOLS) -> str:
    """``0`` -> ``""``, ``+1`` -> ``↑``, ``-2`` -> ``↓2``."""
    if octave_offset == 0:
        return ""
    if octave_offset > 0:
        arrow = symbols.octave_up
        count = octave_offset
    else:
        arrow = symbols.octave_down
        count = -octave_offset
    return arrow if count == 1 else f"{arrow}{count}"


def format_note_label(
    note: PlayableNote,
    symbols: NoteLabelSymbols = DEFAULT_SYMBOLS,
) -> str:
    """Return the label a renderer should draw for *note*."""
    label = note.key_label
    if note.semitone:
        label += symbols.semitone
    return label + octave_suffix(note.octave_offset, symbols)


__all__ = [
    "DEFAULT_SEMITONE_SYMBOL",
    "DEFAULT_OCTAVE_UP_SYMBOL",
    "DEFAULT_OCTAVE_DOWN_SYMBOL",
    "NoteLabelSymbols",
    "DEFAULT_SYMBOLS",
    "octave_suffix",
    "format_note_label",
]
