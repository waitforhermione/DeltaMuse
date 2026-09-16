"""Visual preset tests."""

from __future__ import annotations

import pytest
from PIL import Image

from app.score.layout import build_layout
from app.score.renderer_png import render_score
from app.score.styles import (
    COMPACT,
    DEFAULT_PRESET_NAME,
    PRACTICE,
    PRESETS,
    ScoreLayoutConfig,
    ScoreStyle,
    get_preset,
)

from conftest import LANES, playable_note as playable


class TestPresets:
    def test_both_presets_exist(self) -> None:
        assert set(PRESETS) == {"compact", "practice"}
        assert DEFAULT_PRESET_NAME == "compact"

    def test_presets_differ_meaningfully(self) -> None:
        assert PRACTICE.style.header_font_size > COMPACT.style.header_font_size
        assert PRACTICE.style.note_font_size > COMPACT.style.note_font_size
        assert PRACTICE.layout.section_height > COMPACT.layout.section_height
        assert PRACTICE.layout.lane_height > COMPACT.layout.lane_height
        assert PRACTICE.layout.note_height > COMPACT.layout.note_height
        assert PRACTICE.layout.min_note_width >= COMPACT.layout.min_note_width

    def test_unknown_preset_is_rejected(self) -> None:
        with pytest.raises(Exception, match="unknown visual preset"):
            get_preset("neon")

    def test_invalid_style_is_still_rejected(self) -> None:
        with pytest.raises(Exception):
            ScoreStyle(note_fill="not-a-colour")


class TestRenderBothPresets:
    def test_png_outputs_for_both_presets(self, tmp_path) -> None:
        notes = [playable(60, 0, 0.0, 1.0), playable(64, 2, 6.0, 1.0)]

        for preset in PRESETS.values():
            layout = build_layout(list(notes), lanes=LANES, config=preset.layout)
            image = render_score(layout, style=preset.style, config=preset.layout)
            assert image.size == (int(preset.layout.canvas_width), int(layout.height))
            path = tmp_path / f"{preset.name}.png"
            image.save(path)
            with Image.open(path) as reopened:
                assert reopened.size == image.size

    def test_practice_scores_are_taller_for_long_songs(self) -> None:
        notes = [playable(60 + (index % 8), index, float(index), 0.5) for index in range(8)]
        compact = build_layout(notes, lanes=LANES, config=COMPACT.layout)
        practice = build_layout(notes, lanes=LANES, config=PRACTICE.layout)

        assert practice.height > compact.height
        # Same music -> same number of sections in both presets.
        assert practice.section_count == compact.section_count

    def test_music_semantics_are_unchanged_across_presets(self) -> None:
        # pitch 64, lane 2, start 15.0 s, duration 1.5 s
        note = playable(64, 2, 15.0, 1.5)
        compact = build_layout([note], lanes=LANES, config=COMPACT.layout)
        practice = build_layout([note], lanes=LANES, config=PRACTICE.layout)

        for preset, layout in ((COMPACT, compact), (PRACTICE, practice)):
            fragment = layout.fragments()[0]
            config: ScoreLayoutConfig = preset.layout
            # 12 s -> section 2, x at 1/2 of the row, width 1/4 of the row.
            assert fragment.section_index == 2
            x_rel = (fragment.x - config.left_label_width) / config.usable_width()
            w_rel = fragment.width / config.usable_width()
            assert x_rel == pytest.approx(0.5, abs=1e-6)
            assert w_rel == pytest.approx(0.25, abs=1e-6)

    def test_labels_and_header_are_drawn_in_practice_mode(self) -> None:
        layout = build_layout(
            [playable(60, 0, 0.0, 1.0, key_label="Z")],
            lanes=LANES,
            config=PRACTICE.layout,
        )
        calls: list[tuple[str, object]] = []

        class Sink:
            def rectangle(self, xy, **kwargs) -> None:
                calls.append(("rect", xy))

            def rounded_rectangle(self, xy, **kwargs) -> None:
                calls.append(("rect", xy))

            def line(self, xy, **kwargs) -> None:
                calls.append(("line", xy))

            def text(self, xy, text, **kwargs) -> None:
                calls.append(("text", text))

            def text_length(self, text, font=None) -> float:
                return len(text) * 10.0

        render_score(layout, style=PRACTICE.style, config=PRACTICE.layout, sink_factory=lambda _img: Sink())
        texts = [value for kind, value in calls if kind == "text"]

        assert "Z" in texts  # lane label and note label
        assert any("00:00" in text for text in texts)  # section title
        assert any("Transpose:" in text for text in texts)  # header metadata
