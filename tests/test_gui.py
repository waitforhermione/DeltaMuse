"""GUI tests with QT_QPA_PLATFORM=offscreen (no real TransKun, fake service)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6", reason="PySide6 is not installed")

from PySide6.QtCore import QMimeData  # noqa: E402
from PySide6.QtWidgets import QApplication, QMainWindow as _QMainWindow  # noqa: E402

from app.application.conversion_service import ConversionOutcome  # noqa: E402
from app.config.user_settings import load_settings, settings_path  # noqa: E402
from app.gui.main_window import MainWindow  # noqa: E402
from app.core.models import NoteEvent  # noqa: E402

QMainWindow_init = _QMainWindow.__init__


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def window(qapp):
    window = MainWindow()
    window.resize(700, 600)
    yield window
    window.close()


@pytest.fixture
def midi_file(tmp_path: Path):
    from app.midi.midi_writer import write_midi

    notes = [NoteEvent(pitch=60, start=i * 0.5, duration=0.4, velocity=80) for i in range(4)]
    return write_midi(notes, tmp_path / "song.mid")


class FakeService:
    """A Qt-free fake standing in for ConversionService."""

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.requests: list = []

    def inspect(self, path):
        from app.application.conversion_service import InputFileInfo

        path = Path(path)
        if not path.is_file():
            return InputFileInfo(path=path, kind="missing", error="Input file not found.")
        return InputFileInfo(path=path, kind="audio", duration=103.2)

    def run(self, request, on_progress=None):
        self.requests.append(request)
        if on_progress is not None:
            for stage in (1, 2, 3, 4, 5):
                on_progress(stage, f"stage {stage}")
        if self.fail:
            raise RuntimeError("No piano notes were detected in song.mp3.")
        output = Path(request.output_path or "fake_out.png")
        output.write_bytes(b"png")
        return ConversionOutcome(
            output=output, notes=163, transpose="-2 (auto)", playable=1.0, elapsed=17.8
        )


def _init_with(self, service):
    QMainWindow_init(self)
    self._service = service
    self._settings, self._setting_warnings = load_settings()
    self._thread = None
    self._worker = None
    self._outcome = None
    self._info = None
    self._build_ui()
    self._restore_geometry()
    self._update_generate_state()


def pump(window, limit: int = 8000) -> None:
    while window._thread is not None and limit > 0:
        QApplication.processEvents()
        limit -= 1
    QApplication.processEvents()


class TestMainWindow:
    def test_constructs(self, window) -> None:
        assert window.windowTitle().startswith("Delta Harmonica Score")
        assert not window.generate_button.isEnabled()

    def test_load_shows_file_info(self, window, midi_file: Path) -> None:
        window.set_input(str(midi_file))
        assert "song.mid" in window.info_label.text()
        assert window.generate_button.isEnabled()

    def test_drag_and_drop(self, window, midi_file: Path) -> None:
        from PySide6.QtCore import QUrl

        accepted: list = []

        class FakeEvent:
            def __init__(self, path: str) -> None:
                self.mime = QMimeData()
                self.mime.setUrls([QUrl.fromLocalFile(path)])

            def mimeData(self):
                return self.mime

            def acceptProposedAction(self):
                accepted.append(True)

        window.dragEnterEvent(FakeEvent(str(midi_file)))
        assert accepted
        window.dropEvent(FakeEvent(str(midi_file)))
        assert window.input_edit.text()
        assert window.generate_button.isEnabled()

    def test_unsupported_file_is_reported(self, window, tmp_path: Path) -> None:
        path = tmp_path / "notes.txt"
        path.write_text("x")
        window.set_input(str(path))
        assert "✗" in window.info_label.text()
        assert not window.generate_button.isEnabled()

    def test_segment_validation(self, window, midi_file: Path) -> None:
        window.set_input(str(midi_file))
        window.segment_radio.setChecked(True)
        window.start_edit.setText("00:00")
        assert not window.generate_button.isEnabled()  # end missing
        window.end_edit.setText("00:01")
        assert window.generate_button.isEnabled()
        window.end_edit.setText("ab:cd")
        assert not window.generate_button.isEnabled()
        window.end_edit.setText("00:00")
        assert not window.generate_button.isEnabled()  # end <= start

    def test_start_beyond_duration_caught_early(self, window, midi_file: Path) -> None:
        window.set_input(str(midi_file))
        window.segment_radio.setChecked(True)
        window.start_edit.setText("99:00")
        window.end_edit.setText("99:30")
        assert not window.generate_button.isEnabled()

    def test_style_selector_defaults_from_settings(self, qapp) -> None:
        path = settings_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"default_style": "compact"}), encoding="utf-8")
        fresh = MainWindow()
        assert fresh.style_combo.currentData() == "compact"
        fresh.close()

    def test_advanced_settings_collapsed(self, window) -> None:
        assert window.advanced.isCheckable()
        assert not window.advanced.isChecked()

    def test_entire_mode_needs_no_times(self, window, midi_file: Path) -> None:
        window.set_input(str(midi_file))
        window.entire_radio.setChecked(True)
        window.start_edit.setText("")
        window.end_edit.setText("")
        assert window.generate_button.isEnabled()


class TestWorkerFlow:
    def test_worker_success_flow(self, qapp, monkeypatch, midi_file: Path, tmp_path: Path) -> None:
        service = FakeService()
        monkeypatch.setattr(MainWindow, "__init__", lambda self: _init_with(self, service))
        window = MainWindow()
        window.set_input(str(midi_file))
        window.output_edit.setText(str(tmp_path / "out.png"))
        window._generate()
        pump(window)

        assert "Complete ✓" in window.result_label.text()
        assert "Notes: 163" in window.result_label.text()
        assert "Playable: 100.0%" in window.result_label.text()
        assert window.open_button.isEnabled()
        assert window.reveal_button.isEnabled()
        assert window.generate_button.isEnabled()  # a second run is possible
        assert service.requests[0].style == "practice"
        window.close()

    def test_worker_error_shows_message_box(
        self, qapp, monkeypatch, midi_file: Path, tmp_path: Path
    ) -> None:
        service = FakeService(fail=True)
        monkeypatch.setattr(MainWindow, "__init__", lambda self: _init_with(self, service))
        shown: list = []
        monkeypatch.setattr(
            "app.gui.main_window.QMessageBox.critical", lambda *a, **kw: shown.append(a[2])
        )
        window = MainWindow()
        window.set_input(str(midi_file))
        window.output_edit.setText(str(tmp_path / "out.png"))
        window._generate()
        pump(window)

        assert shown and "No piano notes" in shown[0]
        assert window.generate_button.isEnabled()
        window.close()

    def test_progress_stage_is_displayed(
        self, qapp, monkeypatch, midi_file: Path, tmp_path: Path
    ) -> None:
        service = FakeService()
        monkeypatch.setattr(MainWindow, "__init__", lambda self: _init_with(self, service))
        window = MainWindow()
        window.set_input(str(midi_file))
        window.output_edit.setText(str(tmp_path / "out.png"))
        window._generate()
        pump(window)

        assert "[5/5]" in window.stage_label.text()
        window.close()

    def test_generate_cannot_be_submitted_twice(
        self, qapp, monkeypatch, midi_file: Path, tmp_path: Path
    ) -> None:
        service = FakeService()
        monkeypatch.setattr(MainWindow, "__init__", lambda self: _init_with(self, service))
        window = MainWindow()
        window.set_input(str(midi_file))
        window.output_edit.setText(str(tmp_path / "out.png"))
        window._generate()
        assert not window.generate_button.isEnabled()
        window._generate()  # ignored while running
        pump(window)
        assert len(service.requests) == 1
        window.close()


class TestSettingsPersistence:
    def test_geometry_is_persisted(self, qapp, monkeypatch, midi_file: Path) -> None:
        window = MainWindow()
        window.resize(1234, 567)
        window.close()

        window2 = MainWindow()
        assert window2.size().width() >= 1234 - 40
        window2.close()

    def test_last_directories_are_persisted(self, qapp, monkeypatch) -> None:
        window = MainWindow()
        window._remember_dir("gui_last_input_dir", "C:/music")
        window.close()

        settings, _warnings = load_settings()
        assert settings.gui_last_input_dir == "C:/music"

