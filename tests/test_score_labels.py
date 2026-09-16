"""Note label formatting (key -> # -> octave)."""

from __future__ import annotations

import pytest

from app.core.models import PlayableNote
from app.score.labels import (
    DEFAULT_SYMBOLS,
    NoteLabelSymbols,
    format_note_label,
    octave_suffix,
)

from conftest import make_profile


def note(
    key: str = "Z",
    *,
    semitone: bool = False,
    octave_offset: int = 0,
) -> PlayableNote:
    return PlayableNote(
        source_pitch=60,
        pitch=60,
        start=0.0,
        duration=0.5,
        lane=0,
        key_label=key,
        semitone=semitone,
        octave_offset=octave_offset,
    )


class TestOctaveSuffix:
    @pytest.mark.parametrize(
        ("offset", "expected"),
        [(0, ""), (1, "↑"), (-1, "↓"), (2, "↑2"), (-2, "↓2"), (3, "↑3"), (-3, "↓3")],
    )
    def test_suffixes(self, offset: int, expected: str) -> None:
        assert octave_suffix(offset, DEFAULT_SYMBOLS) == expected


class TestFormatNoteLabel:
    @pytest.mark.parametrize(
        ("semitone", "octave_offset", "expected"),
        [
            (False, 0, "Z"),
            (True, 0, "Z#"),
            (False, 1, "Z↑"),
            (False, -1, "Z↓"),
            (True, 1, "Z#↑"),
            (True, -1, "Z#↓"),
            (False, 2, "Z↑2"),
            (False, -2, "Z↓2"),
            (True, 2, "Z#↑2"),
            (True, -3, "Z#↓3"),
        ],
    )
    def test_documented_combinations(
        self, semitone: bool, octave_offset: int, expected: str
    ) -> None:
        assert format_note_label(note("Z", semitone=semitone, octave_offset=octave_offset)) == expected

    def test_uses_the_key_label_verbatim(self) -> None:
        # The last lane of the shipped profile is a comma.
        assert format_note_label(note(",")) == ","
        assert format_note_label(note(",", semitone=True, octave_offset=1)) == ",#↑"

    def test_multicharacter_keys_are_supported(self) -> None:
        assert format_note_label(note("F#")) == "F#"

    def test_custom_symbols(self) -> None:
        symbols = NoteLabelSymbols(semitone="s", octave_up="+", octave_down="-")
        assert format_note_label(note("Z", semitone=True, octave_offset=2), symbols) == "Zs+2"

    def test_symbols_from_the_profile(self, profile) -> None:
        symbols = NoteLabelSymbols.from_profile(profile)
        assert symbols.semitone == "#"
        assert symbols.octave_up == "↑"
        assert symbols.octave_down == "↓"
        assert format_note_label(note("Z", semitone=True), symbols) == "Z#"

    def test_symbols_from_a_score_document(self) -> None:
        symbols = NoteLabelSymbols.from_dict({"semitone": "x", "octave_up": "u", "octave_down": "d"})
        assert format_note_label(note("Z", semitone=True, octave_offset=-1), symbols) == "Zxd"

    def test_symbols_fall_back_to_defaults(self) -> None:
        assert NoteLabelSymbols.from_dict(None) == DEFAULT_SYMBOLS
        assert NoteLabelSymbols.from_dict({}) == DEFAULT_SYMBOLS

    def test_empty_symbols_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="semitone"):
            NoteLabelSymbols(semitone="", octave_up="↑", octave_down="↓")
