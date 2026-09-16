"""User settings tests (load / save / validate / corrupt fallback / priority)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.config.user_settings import (
    DEFAULTS,
    UserSettings,
    load_settings,
    parse_setting_value,
    save_settings,
    settings_path,
    with_updated_field,
)
from app.core.errors import PianoScoreError


class TestDefaults:
    def test_defaults(self) -> None:
        settings, warnings = load_settings()  # no file -> defaults, no warnings
        assert settings == UserSettings()
        assert warnings == []

    def test_builtin_default_style_is_practice(self) -> None:
        assert DEFAULTS["default_style"] == "practice"

    def test_settings_path_is_in_the_user_config_dir(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
        monkeypatch.delenv("APPDATA", raising=False)
        assert settings_path() == tmp_path / "cfg" / "piano_to_harmonica_score" / "settings.json"

    def test_settings_path_on_windows(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))
        assert settings_path() == tmp_path / "appdata" / "piano_to_harmonica_score" / "settings.json"


class TestLoadSave:
    def test_round_trip(self, tmp_path: Path) -> None:
        path = tmp_path / "settings.json"
        settings = with_updated_field(UserSettings(), "default_style", "compact")
        save_settings(settings, path)

        loaded, warnings = load_settings(path)
        assert loaded.default_style == "compact"
        assert warnings == []

    def test_partial_file_fills_the_rest_with_defaults(self, tmp_path: Path) -> None:
        path = tmp_path / "settings.json"
        path.write_text(json.dumps({"default_style": "compact"}), encoding="utf-8")

        loaded, _warnings = load_settings(path)
        assert loaded.default_style == "compact"
        assert loaded.segment_context_before == 2.0

    def test_invalid_field_falls_back_per_field(self, tmp_path: Path) -> None:
        path = tmp_path / "settings.json"
        path.write_text(
            json.dumps({"default_style": "neon", "segment_context_before": 1.5}),
            encoding="utf-8",
        )

        loaded, warnings = load_settings(path)
        assert loaded.default_style == "practice"  # fell back
        assert loaded.segment_context_before == 1.5  # kept
        assert any("default_style" in note for note in warnings)

    def test_corrupt_json_falls_back_with_a_warning(self, tmp_path: Path) -> None:
        path = tmp_path / "settings.json"
        path.write_text("{not json", encoding="utf-8")

        loaded, warnings = load_settings(path)
        assert loaded == UserSettings()
        assert warnings and "unreadable" in warnings[0]

    def test_non_object_json_falls_back(self, tmp_path: Path) -> None:
        path = tmp_path / "settings.json"
        path.write_text("[1, 2]", encoding="utf-8")

        loaded, warnings = load_settings(path)
        assert loaded == UserSettings()
        assert warnings

    def test_unknown_keys_are_ignored(self, tmp_path: Path) -> None:
        path = tmp_path / "settings.json"
        path.write_text(json.dumps({"secret": "x", "default_style": "compact"}), encoding="utf-8")

        loaded, warnings = load_settings(path)
        assert loaded.default_style == "compact"
        assert any("secret" in note for note in warnings)

    def test_missing_file_never_raises(self, tmp_path: Path) -> None:
        loaded, warnings = load_settings(tmp_path / "nope.json")
        assert loaded == UserSettings()
        assert warnings == []


class TestValidation:
    @pytest.mark.parametrize(
        ("key", "raw", "expected"),
        [
            ("default_style", "practice", "practice"),
            ("default_style", "COMPACT", "compact"),
            ("default_transpose", "auto", "auto"),
            ("default_transpose", "-7", -7),
            ("default_section_duration", "4.5", 4.5),
            ("segment_context_before", "1", 1.0),
        ],
    )
    def test_parse_setting_value(self, key: str, raw: str, expected: object) -> None:
        assert parse_setting_value(key, raw) == expected

    @pytest.mark.parametrize(
        ("key", "raw"),
        [
            ("default_style", "neon"),
            ("default_transpose", "13"),
            ("default_transpose", "loud"),
            ("default_section_duration", "0"),
            ("segment_context_after", "-1"),
            ("unknown_key", "x"),
        ],
    )
    def test_parse_setting_value_rejects(self, key: str, raw: str) -> None:
        with pytest.raises(PianoScoreError):
            parse_setting_value(key, raw)
