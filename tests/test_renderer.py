"""PNG renderer tests (draw-call recording, no OCR)."""

from __future__ import annotations

import pytest
from PIL import Image

from app.core.errors import EmptyScoreError, ScoreRenderError
from app.export.image_exporter import export_score_png
from app.score.layout import LayoutHeader, LayoutScore, ScoreLayoutConfig, build_layout
from app.score.renderer_png import _glyph_available, clip_note, format_time, load_font, render_score
from app.score.styles import DEFAULT_STYLE, ScoreStyle

from conftest import LANES, playable_note as playable

CONFIG = ScoreLayoutConfig()


class RecordingSink:
    """Records every draw call so tests can assert on them."""

    def __init__(self, image: object) -> None:
        self.image = image
        self.calls: list[tuple[str, tuple[float, ...], dict[str, object]]] = []

    def rectangle(self, xy, fill=None, outline=None, width=1) -> None:
        self.calls.append(("rectangle", tuple(xy), {"fill": fill, "outline": outline, "width": width}))

    def rounded_rectangle(self, xy, radius=0, fill=None, outline=None, width=1) -> None:
        self.calls.append(
            ("rounded_rectangle", tuple(xy), {"fill": fill, "outline": outline, "width": width, "radius": radius})
        )

    def line(self, xy, fill=None, width=1) -> None:
        self.calls.append(("line", tuple(xy), {"fill": fill, "width": width}))

    def text(self, xy, text, fill=None, font=None) -> None:
        self.calls.append(("text", tuple(xy), {"text": text, "fill": fill}))

    def text_length(self, text, font=None) -> float:
        return len(text) * 8.0

    # -- helpers ----------------------------------------------------------- #
    def texts(self) -> list[str]:
        return [call[2]["text"] for call in self.calls if call[0] == "text"]

    def rects(self) -> list[tuple[str, tuple[float, ...]]]:
        return [
            (kind, xy) for kind, xy, _ in self.calls if kind in ("rectangle", "rounded_rectangle")
        ]

    def note_rects(self) -> list[tuple[float, float, float, float]]:
        return [
            xy
            for kind, xy, kwargs in self.calls
            if kind in ("rectangle", "rounded_rectangle")
            and kwargs["fill"] in (DEFAULT_STYLE.note_fill, DEFAULT_STYLE.continuation_fill)
        ]

    def note_labels(self) -> list[str]:
        return [
            call[2]["text"] for call in self.calls if call[0] == "text" and call[2]["fill"] == DEFAULT_STYLE.note_label
        ]


def render_layout(layout: LayoutScore) -> RecordingSink:
    sink = RecordingSink(None)
    render_score(layout, config=CONFIG, sink_factory=lambda _img: sink)
    return sink


def render_notes(*notes: object, **layout_kwargs: object) -> tuple[RecordingSink, LayoutScore]:
    layout = build_layout(list(notes), lanes=LANES, config=CONFIG, **layout_kwargs)  # type: ignore[arg-type]
    return render_layout(layout), layout


# --------------------------------------------------------------------------- #
# basic rendering
# --------------------------------------------------------------------------- #
class TestBasicRendering:
    def test_one_section(self) -> None:
        layout = build_layout([playable(60, 0, 0.0, 1.0)], lanes=LANES, config=CONFIG)
        image = render_score(layout, config=CONFIG)

        assert image.size == (int(layout.width), int(layout.height))

    def test_two_sections(self) -> None:
        layout = build_layout(
            [playable(60, 0, 0.0, 1.0), playable(62, 1, 6.0, 1.0)], lanes=LANES, config=CONFIG
        )
        image = render_score(layout, config=CONFIG)

        assert image.size[1] == int(CONFIG.total_height(2))

    def test_png_can_be_reopened(self, tmp_path) -> None:
        layout = build_layout([playable(60, 0, 0.0, 1.0)], lanes=LANES, config=CONFIG)
        path = export_score_png(layout, tmp_path / "score.png", config=CONFIG)

        with Image.open(path) as reopened:
            assert reopened.format == "PNG"
            assert reopened.size == (int(layout.width), int(layout.height))

    def test_empty_note_list_is_rejected_at_layout_time(self) -> None:
        with pytest.raises(EmptyScoreError):
            build_layout([], lanes=LANES)

    def test_render_rejects_a_sectionless_layout(self) -> None:
        empty = LayoutScore(
            width=2000.0,
            height=592.0,
            header=LayoutHeader(),
            duration=0.0,
            lanes=LANES,
            sections=(),
        )
        with pytest.raises(EmptyScoreError):
            render_score(empty, config=CONFIG)


# --------------------------------------------------------------------------- #
# lane labels & section titles
# --------------------------------------------------------------------------- #
class TestLaneLabels:
    def test_every_lane_label_is_drawn(self) -> None:
        sink, _layout = render_notes(playable(60, 0, 0.0, 1.0))
        texts = sink.texts()

        for lane in LANES:
            assert lane in texts, f"lane label {lane!r} was never drawn"

    def test_lane_labels_are_repeated_per_section(self) -> None:
        # Note keys that cannot be confused with lane labels.
        sink, layout = render_notes(
            playable(60, 0, 0.0, 1.0, key_label="k1"), playable(62, 1, 6.0, 1.0, key_label="k2")
        )
        repeated = [text for text in sink.texts() if text in LANES]

        assert len(repeated) == 16  # 8 lanes x 2 sections

    def test_section_titles_use_the_layout_times(self) -> None:
        sink, _layout = render_notes(playable(60, 0, 0.0, 1.0))
        assert "00:00–00:06" in sink.texts()

    def test_partial_last_section_keeps_the_uniform_scale(self) -> None:
        """The last row is still announced as a full 6 s section."""
        sink, _layout = render_notes(playable(60, 0, 8.5, 0.47))
        assert "00:06–00:12" in sink.texts()


# --------------------------------------------------------------------------- #
# note geometry
# --------------------------------------------------------------------------- #
class TestNoteGeometry:
    def test_notes_are_drawn_at_their_layout_geometry(self) -> None:
        layout = build_layout(
            [
                playable(60, 0, 0.0, 1.0),
                playable(66, 3, 2.0, 0.5, key_label="V", semitone=True),
            ],
            lanes=LANES,
            config=CONFIG,
        )
        sink = render_layout(layout)
        drawn = sink.note_rects()

        for fragment in layout.fragments():
            expected = (
                fragment.x,
                fragment.y,
                fragment.x + fragment.width,
                fragment.y + fragment.height,
            )
            assert any(all(abs(a - b) < 1e-6 for a, b in zip(xy, expected)) for xy in drawn), (
                f"fragment {fragment.note_id} was not drawn at {expected}"
            )

    def test_the_background_is_drawn_first(self) -> None:
        sink, layout = render_notes(playable(60, 0, 0.0, 1.0))
        first_kind, first_xy, first_kwargs = sink.calls[0]

        assert first_kind == "rectangle"
        assert first_kwargs["fill"] == DEFAULT_STYLE.background
        assert first_xy == (0.0, 0.0, layout.width, layout.height)


# --------------------------------------------------------------------------- #
# clipping
# --------------------------------------------------------------------------- #
class TestClipping:
    def test_note_beyond_the_section_edge_is_clipped(self) -> None:
        """A 5 ms fragment at the end of section 0 is widened by min_note_width
        in the layout; the renderer must clip it back to the drawing area."""
        layout = build_layout([playable(60, 0, 5.999, 0.001)], lanes=LANES, config=CONFIG)
        fragment = layout.fragments()[0]
        assert fragment.x + fragment.width > CONFIG.canvas_width - CONFIG.right_padding

        sink = render_layout(layout)
        right_edges = [xy[2] for xy in sink.note_rects()]

        assert right_edges
        assert max(right_edges) <= CONFIG.canvas_width - CONFIG.right_padding + 1e-6

    def test_clip_note_helper(self) -> None:
        layout = build_layout([playable(60, 0, 5.999, 0.001)], lanes=LANES, config=CONFIG)
        fragment = layout.fragments()[0]
        right = CONFIG.canvas_width - CONFIG.right_padding

        left, top, clipped, bottom = clip_note(fragment, right)
        assert left == fragment.x
        assert clipped == pytest.approx(right)
        assert (top, bottom) == (fragment.y, fragment.y + fragment.height)

    def test_zero_width_after_clipping_is_rejected(self) -> None:
        layout = build_layout([playable(60, 0, 0.0, 1.0)], lanes=LANES, config=CONFIG)
        fragment = layout.fragments()[0]
        with pytest.raises(ScoreRenderError, match="no visible width"):
            clip_note(fragment, right=fragment.x - 1.0)


# --------------------------------------------------------------------------- #
# continuation
# --------------------------------------------------------------------------- #
class TestContinuation:
    def test_first_fragment_labelled_continuation_not(self) -> None:
        sink, layout = render_notes(playable(60, 0, 5.5, 2.0, key_label="Z"))

        assert layout.fragment_count == 2
        assert sink.note_labels().count("Z") == 1  # only the first fragment shouts

    def test_continuation_fragment_is_still_drawn_as_a_bar(self) -> None:
        sink, layout = render_notes(playable(60, 0, 5.5, 2.0, key_label="Z"))
        continuation = layout.sections[1].notes[0]
        expected = (
            continuation.x,
            continuation.y,
            continuation.x + continuation.width,
            continuation.y + continuation.height,
        )

        assert any(all(abs(a - b) < 1e-6 for a, b in zip(xy, expected)) for xy in sink.note_rects())


# --------------------------------------------------------------------------- #
# modifier labels
# --------------------------------------------------------------------------- #
class TestModifierLabels:
    def test_modifier_labels_are_drawn(self) -> None:
        sink, _layout = render_notes(
            playable(60, 0, 0.0, 0.9, key_label="Z"),
            playable(63, 1, 1.0, 0.9, key_label="X", semitone=True),
            playable(76, 2, 2.0, 0.9, key_label="C", octave_offset=1),
            playable(53, 3, 3.0, 0.9, key_label="V", octave_offset=-1),
            playable(80, 4, 4.0, 0.9, key_label="B", semitone=True, octave_offset=1),
        )

        for expected in ("Z", "X#", "C↑", "V↓", "B#↑"):
            assert expected in sink.note_labels()

    def test_narrow_fragments_skip_the_label(self) -> None:
        """A fragment too small to hold its label must not draw spilling text."""
        config = ScoreLayoutConfig(min_note_width=10.0)
        layout = build_layout(
            [playable(60, 0, 5.999, 0.001, key_label="Z")], lanes=LANES, config=config
        )
        sink = RecordingSink(None)
        render_score(layout, config=config, sink_factory=lambda _img: sink)

        assert sink.note_labels() == []


# --------------------------------------------------------------------------- #
# header
# --------------------------------------------------------------------------- #
class TestHeader:
    def test_header_fields_are_formatted(self) -> None:
        sink, _layout = render_notes(
            playable(60, 0, 0.0, 1.0),
            header=LayoutHeader(
                title="My Song",
                source="song.mp3",
                profile="delta_harmonica",
                duration=34.0,
                transpose=-7,
                auto_transpose=True,
                range_mode="octave_fold",
                note_count=7,
                playable_ratio=0.9764,
            ),
        )
        texts = " | ".join(sink.texts())

        assert "My Song" in texts
        assert "Source: song.mp3" in texts
        assert "Profile: delta_harmonica" in texts
        assert "Duration: 34.0 s" in texts
        assert "Transpose: -7 (auto)" in texts
        assert "Notes: 7" in texts
        assert "Playable: 97.6%" in texts

    def test_segment_is_shown_with_the_original_times(self) -> None:
        sink, _layout = render_notes(
            playable(60, 0, 0.0, 1.0),
            header=LayoutHeader(segment=(48.0, 82.0)),
        )
        texts = " | ".join(sink.texts())

        assert "Segment: 00:48–01:22" in texts

    def test_no_segment_line_without_cropping(self) -> None:
        sink, _layout = render_notes(playable(60, 0, 0.0, 1.0))
        assert "Segment:" not in " | ".join(sink.texts())

    def test_positive_transpose_gets_a_plus_sign(self) -> None:
        sink, _layout = render_notes(
            playable(60, 0, 0.0, 1.0), header=LayoutHeader(transpose=3, auto_transpose=True)
        )
        assert "Transpose: +3 (auto)" in " | ".join(sink.texts())

    def test_source_is_shown_as_a_basename(self) -> None:
        sink, _layout = render_notes(
            playable(60, 0, 0.0, 1.0), header=LayoutHeader(source=r"C:\music\song.mp3")
        )
        texts = " | ".join(sink.texts())
        assert "Source: song.mp3" in texts
        assert "C:" not in texts

    def test_manual_transpose_has_no_auto_marker(self) -> None:
        sink, _layout = render_notes(
            playable(60, 0, 0.0, 1.0),
            header=LayoutHeader(transpose=-2, auto_transpose=False),
        )
        texts = " | ".join(sink.texts())

        assert "Transpose: -2" in texts
        assert "(auto)" not in texts

    def test_empty_fields_are_omitted(self) -> None:
        sink, _layout = render_notes(
            playable(60, 0, 0.0, 1.0),
            # title / source / profile empty, and note_count 0 -> all omitted.
            header=LayoutHeader(title="", source=None, profile="", duration=3.0, note_count=3),
        )
        texts = " | ".join(sink.texts())

        assert "Source:" not in texts
        assert "Profile:" not in texts
        # Duration / transpose / notes are always shown.
        assert "Duration:" in texts and "Transpose:" in texts and "Notes:" in texts


# --------------------------------------------------------------------------- #
# deterministic draw order
# --------------------------------------------------------------------------- #
class TestDrawOrder:
    def test_fixed_paint_order(self) -> None:
        """background -> lane rows -> lane lines -> grid -> notes -> labels."""
        layout = build_layout(
            [
                playable(60, 0, 0.0, 1.0),
                playable(66, 3, 6.0, 1.0, key_label="V", semitone=True),
            ],
            lanes=LANES,
            config=CONFIG,
        )
        sink = RecordingSink(None)
        render_score(layout, config=CONFIG, sink_factory=lambda _img: sink)

        order: dict[str, int] = {}
        for position, (kind, _xy, kwargs) in enumerate(sink.calls):
            fill = kwargs["fill"]
            if kind == "rectangle" and fill == DEFAULT_STYLE.background:
                order.setdefault("background", position)
            elif kind == "rectangle" and fill == DEFAULT_STYLE.lane_background:
                order.setdefault("lane_row", position)
            elif kind == "line" and fill == DEFAULT_STYLE.lane_line:
                order.setdefault("lane_line", position)
            elif kind == "line" and fill == DEFAULT_STYLE.grid_minor:
                order.setdefault("grid_minor", position)
            elif kind in ("rounded_rectangle", "rectangle") and fill == DEFAULT_STYLE.note_fill:
                order.setdefault("note", position)
            elif kind == "text" and fill == DEFAULT_STYLE.note_label:
                order.setdefault("note_label", position)

        assert list(order) == [
            "background",
            "lane_line",  # lane 0 has no alternating background
            "lane_row",  # lane 1 is shaded
            "grid_minor",
            "note",
            "note_label",
        ]

    def test_same_layout_paints_identically(self) -> None:
        notes = [
            playable(60, 0, 0.0, 1.0),
            playable(66, 3, 5.5, 2.0, key_label="V", semitone=True),
            playable(72, 7, 6.0, 1.0),
        ]
        first = RecordingSink(None)
        second = RecordingSink(None)
        render_score(build_layout(notes, lanes=LANES, config=CONFIG), config=CONFIG,
                     sink_factory=lambda _img: first)
        render_score(build_layout(list(reversed(notes)), lanes=LANES, config=CONFIG), config=CONFIG,
                     sink_factory=lambda _img: second)

        assert first.calls == second.calls


# --------------------------------------------------------------------------- #
# fonts & helpers
# --------------------------------------------------------------------------- #
class TestFonts:
    def test_chosen_font_renders_the_modifier_glyphs(self) -> None:
        font = load_font(16)
        for glyph in ("↑", "↓", "–", "#"):
            assert _glyph_available(font, glyph), f"the selected font cannot draw {glyph!r}"

    def test_font_is_cached(self) -> None:
        assert load_font(21) is load_font(21)

    def test_invalid_font_size_is_rejected(self) -> None:
        with pytest.raises(ScoreRenderError, match="font size"):
            load_font(0)


class TestFormatTime:
    @pytest.mark.parametrize(
        ("seconds", "expected"),
        [(0.0, "00:00"), (6.0, "00:06"), (8.97, "00:08.97"), (65.5, "01:05.50"), (61.0, "01:01")],
    )
    def test_formatting(self, seconds: float, expected: str) -> None:
        assert format_time(seconds) == expected

    def test_negative_time_is_rejected(self) -> None:
        with pytest.raises(ScoreRenderError, match="time"):
            format_time(-1.0)


class TestStyle:
    def test_default_style_is_light_and_plain(self) -> None:
        assert DEFAULT_STYLE.background == "#ffffff"
        assert DEFAULT_STYLE.note_fill != DEFAULT_STYLE.background

    def test_invalid_colour_is_rejected(self) -> None:
        with pytest.raises(ScoreRenderError, match="hex colour"):
            ScoreStyle(note_fill="blue")

    def test_non_positive_font_size_is_rejected(self) -> None:
        with pytest.raises(ScoreRenderError, match="header_font_size"):
            ScoreStyle(header_font_size=0)

    def test_negative_corner_radius_is_rejected(self) -> None:
        with pytest.raises(ScoreRenderError, match="corner_radius"):
            ScoreStyle(corner_radius=-1)


class TestExporter:
    def test_writes_and_creates_directories(self, tmp_path) -> None:
        layout = build_layout([playable(60, 0, 0.0, 1.0)], lanes=LANES, config=CONFIG)
        path = export_score_png(layout, tmp_path / "nested" / "score.png", config=CONFIG)
        assert path.is_file()

    def test_rejects_a_directory_target(self, tmp_path) -> None:
        layout = build_layout([playable(60, 0, 0.0, 1.0)], lanes=LANES, config=CONFIG)
        with pytest.raises(ScoreRenderError, match="cannot write score PNG"):
            export_score_png(layout, tmp_path, config=CONFIG)
