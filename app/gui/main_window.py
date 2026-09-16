"""Main window of the Delta Harmonica Score desktop app.

Presentation only: it collects user input, validates it early, drives a
:class:`~app.gui.worker.ConversionWorker` on a background thread, and displays
the :class:`~app.application.conversion_service.ConversionOutcome`. All
conversion logic lives in the Qt-free application service.
"""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from app.application.conversion_service import (
    ConversionRequest,
    ConversionService,
    SUPPORTED_INPUT_EXTENSIONS,
)
from app.config.user_settings import load_settings, save_settings, with_updated_field
from app.core.time_range import parse_time_value
from app.gui.worker import start_worker
from app.version import APP_NAME, VERSION

_FILE_FILTER = "Supported files (*.mp3 *.wav *.mid *.midi);;All files (*.*)"


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self._service = ConversionService()
        self._settings, self._setting_warnings = load_settings()
        self._thread = None
        self._worker = None
        self._outcome = None
        self._info = None

        self.setWindowTitle(f"{APP_NAME} {VERSION}")
        self.resize(640, 0)
        self._build_ui()
        self._restore_geometry()
        self._update_generate_state()

    # ------------------------------------------------------------------ #
    # UI construction
    # ------------------------------------------------------------------ #
    def _build_ui(self) -> None:
        central = QWidget(self)
        layout = QVBoxLayout(central)

        layout.addWidget(QLabel("Input file (.mp3 / .wav / .mid):"))
        input_row = QHBoxLayout()
        self.input_edit = QLineEdit()
        self.input_edit.setPlaceholderText("Choose or drop a piano recording…")
        self.input_edit.textChanged.connect(self._on_input_changed)
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse_input)
        input_row.addWidget(self.input_edit)
        input_row.addWidget(browse)
        layout.addLayout(input_row)
        self.setAcceptDrops(True)

        self.info_label = QLabel(" ")
        self.info_label.setStyleSheet("color: #666;")
        layout.addWidget(self.info_label)

        range_box = QGroupBox("Time range")
        range_layout = QVBoxLayout(range_box)
        self.entire_radio = QRadioButton("Entire song")
        self.entire_radio.setChecked(True)
        self.entire_radio.toggled.connect(self._on_range_mode_changed)
        self.segment_radio = QRadioButton("Segment")
        range_layout.addWidget(self.entire_radio)
        range_layout.addWidget(self.segment_radio)

        segment_row = QHBoxLayout()
        segment_row.addWidget(QLabel("Start"))
        self.start_edit = QLineEdit()
        self.start_edit.setPlaceholderText("00:48")
        self.start_edit.textChanged.connect(lambda _text: self._on_range_edited())
        segment_row.addWidget(self.start_edit)
        segment_row.addWidget(QLabel("End"))
        self.end_edit = QLineEdit()
        self.end_edit.setPlaceholderText("01:22")
        self.end_edit.textChanged.connect(lambda _text: self._on_range_edited())
        segment_row.addWidget(self.end_edit)
        range_layout.addLayout(segment_row)

        self.range_hint = QLabel(" ")
        self.range_hint.setStyleSheet("color: #b00;")
        range_layout.addWidget(self.range_hint)
        layout.addWidget(range_box)

        style_row = QHBoxLayout()
        style_row.addWidget(QLabel("Style:"))
        self.style_combo = QComboBox()
        self.style_combo.addItem("Practice", "practice")
        self.style_combo.addItem("Compact", "compact")
        default_style = getattr(self._settings, "default_style", "practice")
        index = self.style_combo.findData(default_style)
        self.style_combo.setCurrentIndex(max(index, 0))
        style_row.addWidget(self.style_combo)
        style_row.addStretch(1)
        layout.addLayout(style_row)

        layout.addWidget(QLabel("Output:"))
        output_row = QHBoxLayout()
        self.output_edit = QLineEdit()
        self.output_edit.setPlaceholderText("Auto: <song>_score.png")
        self.output_edit.textEdited.connect(self._update_generate_state)
        output_browse = QPushButton("Browse…")
        output_browse.clicked.connect(self._browse_output)
        output_row.addWidget(self.output_edit)
        output_row.addWidget(output_browse)
        layout.addLayout(output_row)

        self.advanced = QGroupBox("Advanced settings")
        self.advanced.setCheckable(True)
        self.advanced.setChecked(False)
        advanced_layout = QVBoxLayout(self.advanced)

        transpose_row = QHBoxLayout()
        transpose_row.addWidget(QLabel("Transpose:"))
        self.transpose_combo = QComboBox()
        self.transpose_combo.addItem("Auto", "auto")
        for semitones in range(-12, 13):
            self.transpose_combo.addItem(f"{semitones:+d}" if semitones else "0", semitones)
        transpose_row.addWidget(self.transpose_combo)
        transpose_row.addStretch(1)
        advanced_layout.addLayout(transpose_row)

        range_row = QHBoxLayout()
        range_row.addWidget(QLabel("Range mode:"))
        self.range_mode_combo = QComboBox()
        self.range_mode_combo.addItem("Octave fold", "octave_fold")
        self.range_mode_combo.addItem("Nearest", "nearest")
        self.range_mode_combo.addItem("Drop", "drop")
        range_row.addWidget(self.range_mode_combo)
        range_row.addStretch(1)
        advanced_layout.addLayout(range_row)

        section_row = QHBoxLayout()
        section_row.addWidget(QLabel("Section duration (s):"))
        self.section_spin = QSpinBox()
        self.section_spin.setRange(2, 30)
        self.section_spin.setValue(int(getattr(self._settings, "default_section_duration", 6)))
        section_row.addWidget(self.section_spin)
        section_row.addStretch(1)
        advanced_layout.addLayout(section_row)

        context_row = QHBoxLayout()
        context_row.addWidget(QLabel("Context before (s):"))
        self.context_before_spin = QSpinBox()
        self.context_before_spin.setRange(0, 10)
        self.context_before_spin.setValue(int(getattr(self._settings, "segment_context_before", 2)))
        context_row.addWidget(self.context_before_spin)
        context_row.addWidget(QLabel("after (s):"))
        self.context_after_spin = QSpinBox()
        self.context_after_spin.setRange(0, 10)
        self.context_after_spin.setValue(int(getattr(self._settings, "segment_context_after", 2)))
        context_row.addWidget(self.context_after_spin)
        context_row.addStretch(1)
        advanced_layout.addLayout(context_row)
        layout.addWidget(self.advanced)

        self.generate_button = QPushButton("Generate Score")
        self.generate_button.setDefault(True)
        self.generate_button.clicked.connect(self._generate)
        layout.addWidget(self.generate_button)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # indeterminate
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)
        self.stage_label = QLabel(" ")
        self.stage_label.hide()
        layout.addWidget(self.stage_label)

        self.result_label = QLabel(" ")
        self.result_label.hide()
        layout.addWidget(self.result_label)
        buttons_row = QHBoxLayout()
        self.open_button = QPushButton("Open Score")
        self.open_button.clicked.connect(self._open_score)
        self.open_button.setEnabled(False)
        self.reveal_button = QPushButton("Show in Folder")
        self.reveal_button.clicked.connect(self._show_in_folder)
        self.reveal_button.setEnabled(False)
        buttons_row.addWidget(self.open_button)
        buttons_row.addWidget(self.reveal_button)
        layout.addLayout(buttons_row)

        self.setCentralWidget(central)

    # ------------------------------------------------------------------ #
    # input / validation
    # ------------------------------------------------------------------ #
    def dragEnterEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        urls = event.mimeData().urls()
        if urls and urls[0].toLocalFile().lower().endswith(tuple(SUPPORTED_INPUT_EXTENSIONS)):
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        urls = event.mimeData().urls()
        if urls:
            self.set_input(urls[0].toLocalFile())

    def set_input(self, path: str) -> None:
        self.input_edit.setText(path)
        self._on_input_changed(path)

    def _browse_input(self) -> None:
        start_dir = self._settings.gui_last_input_dir or str(Path.home())
        chosen, _filter = QFileDialog.getOpenFileName(self, "Choose input file", start_dir, _FILE_FILTER)
        if chosen:
            self.set_input(chosen)
            self._remember_dir("gui_last_input_dir", str(Path(chosen).parent))

    def _browse_output(self) -> None:
        start_dir = self._settings.gui_last_output_dir or str(Path.home())
        chosen, _filter = QFileDialog.getSaveFileName(
            self, "Choose output file", start_dir, "Score image (*.png);;Score JSON (*.json)"
        )
        if chosen:
            self.output_edit.setText(chosen)
            self._remember_dir("gui_last_output_dir", str(Path(chosen).parent))

    def _on_input_changed(self, text: str) -> None:
        self._info = self._service.inspect(text) if text.strip() else None
        if self._info is None or not self._info.path.is_file():
            self.info_label.setText(" ")
            self._outcome = None
            self._refresh_result_buttons()
            self._update_generate_state()
            return
        if not self._info.supported:
            self.info_label.setText(f"✗ {self._info.error}")
        else:
            duration = f", {self._info.duration:.1f} s" if self._info.duration else ""
            self.info_label.setText(f"✓ {self._info.path.name} — {self._info.kind}{duration}")
        self._outcome = None
        self._refresh_result_buttons()
        self._update_generate_state()

    def _on_range_mode_changed(self) -> None:
        self._validate_range()
        self._update_generate_state()

    def _on_range_edited(self) -> None:
        self._validate_range()
        self._update_generate_state()

    def _validate_range(self) -> str | None:
        """Return an error message, or ``None`` when the range is valid."""
        if self.entire_radio.isChecked():
            self.range_hint.setText(" ")
            return None
        try:
            start = parse_time_value(self.start_edit.text().strip())
            end = parse_time_value(self.end_edit.text().strip())
        except Exception as exc:  # noqa: BLE001
            self.range_hint.setText(f"✗ {exc}")
            return str(exc)
        if start is None:
            self.range_hint.setText("✗ Start time is required in segment mode.")
            return "start required"
        if end is None:
            self.range_hint.setText("✗ End time is required in segment mode.")
            return "end required"
        if end <= start:
            self.range_hint.setText("✗ End must be after start.")
            return "end <= start"
        duration = self._info.duration if self._info else None
        if duration is not None and start >= duration:
            self.range_hint.setText(f"✗ Start is beyond the file duration ({duration:.0f} s).")
            return "start beyond duration"
        self.range_hint.setText(" ")
        return None

    def _update_generate_state(self) -> None:
        enabled = (
            self._info is not None
            and self._info.supported
            and self._validate_range() is None
            and self._thread is None
        )
        self.generate_button.setEnabled(enabled)

    # ------------------------------------------------------------------ #
    # generation
    # ------------------------------------------------------------------ #
    def _generate(self) -> None:
        if self._thread is not None or not self.generate_button.isEnabled():
            return  # no double submission
        request = ConversionRequest(
            input_path=Path(self.input_edit.text().strip()),
            output_path=Path(self.output_edit.text()) if self.output_edit.text().strip() else None,
            style=self.style_combo.currentData(),
            start=None if self.entire_radio.isChecked() else self.start_edit.text().strip(),
            end=None if self.entire_radio.isChecked() else self.end_edit.text().strip(),
            transpose=self.transpose_combo.currentData(),
            range_mode=self.range_mode_combo.currentData(),
            section_duration=float(self.section_spin.value()),
            context_before=float(self.context_before_spin.value()),
            context_after=float(self.context_after_spin.value()),
        )

        self.generate_button.setEnabled(False)
        self.progress_bar.show()
        self.stage_label.setText("Starting…")
        self.stage_label.show()
        self._outcome = None
        self._refresh_result_buttons()

        self._thread, self._worker = start_worker(self._service, request)
        self._worker.progress.connect(self._on_progress)
        self._worker.succeeded.connect(self._on_success)
        self._worker.failed.connect(self._on_failure)
        self._thread.finished.connect(self._on_thread_finished)
        self._thread.start()

    def _on_progress(self, stage: int, text: str) -> None:
        self.stage_label.setText(f"[{stage}/5] {text}")

    def _on_success(self, outcome) -> None:
        self._outcome = outcome
        self.result_label.setText(
            "Complete ✓\n"
            f"Notes: {outcome.notes}    Transpose: {outcome.transpose}    "
            f"Playable: {outcome.playable * 100:.1f}%    Elapsed: {outcome.elapsed:.1f} s\n"
            f"{outcome.output}"
        )
        self.result_label.show()

    def _on_failure(self, message: str) -> None:
        self.progress_bar.hide()
        self.stage_label.hide()
        QMessageBox.critical(self, APP_NAME, message or "The conversion failed.")

    def _on_thread_finished(self) -> None:
        self.progress_bar.hide()
        self.stage_label.hide()
        if self._thread is not None:
            self._thread.deleteLater()
            self._thread = None
            self._worker = None
        self._refresh_result_buttons()
        self._update_generate_state()

    def _refresh_result_buttons(self) -> None:
        has_output = self._outcome is not None and self._outcome.output.is_file()
        self.open_button.setEnabled(has_output)
        self.reveal_button.setEnabled(has_output)

    # ------------------------------------------------------------------ #
    # results
    # ------------------------------------------------------------------ #
    def _open_score(self) -> None:
        if self._outcome is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._outcome.output)))

    def _show_in_folder(self) -> None:
        if self._outcome is None:
            return
        target = self._outcome.output
        if sys.platform == "win32":
            import subprocess

            subprocess.Popen(["explorer", "/select,", str(target)])
        else:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(target.parent)))

    # ------------------------------------------------------------------ #
    # settings
    # ------------------------------------------------------------------ #
    def _remember_dir(self, key: str, value: str) -> None:
        self._settings = with_updated_field(self._settings, key, value)
        save_settings(self._settings)

    def _restore_geometry(self) -> None:
        geometry = self._settings.gui_window_geometry
        if geometry and "x" in geometry:
            width, height = (int(part) for part in geometry.split("x", 1))
            self.resize(max(width, 480), max(height, 320))

    def _persist_geometry(self) -> None:
        size = self.size()
        self._settings = with_updated_field(
            self._settings, "gui_window_geometry", f"{size.width()}x{size.height()}"
        )
        save_settings(self._settings)

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        self._persist_geometry()
        super().closeEvent(event)


def run_gui() -> int:
    """Create the Qt application and show the main window."""
    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


def run_smoke_test(argv: list[str]) -> int:
    """Headless service run used to verify a bundled EXE (no Qt event loop).

    ``DeltaHarmonicaScore.exe --smoke-test INPUT [--start T] [--end T] [--out FILE]``
    prints the conversion result as JSON and exits 0/1.
    """
    import json

    from app.application.conversion_service import ConversionOutcome

    input_path = Path(argv[0]) if argv else None
    start = end = out = None
    index = 1
    while index < len(argv):
        if argv[index] == "--start":
            start = argv[index + 1]
            index += 2
        elif argv[index] == "--end":
            end = argv[index + 1]
            index += 2
        elif argv[index] == "--out":
            out = Path(argv[index + 1])
            index += 2
        else:
            index += 1
    if input_path is None:
        print("usage: --smoke-test INPUT [--start T] [--end T] [--out FILE]")
        return 2

    service = ConversionService()
    try:
        outcome = service.run(
            ConversionRequest(
                input_path=input_path, output_path=out, style="practice", start=start, end=end
            )
        )
    except Exception as exc:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        Path("smoke_fail.log").write_text(traceback.format_exc(), encoding="utf-8")
        print(f"SMOKE FAIL: {exc}")
        return 1
    print(
        json.dumps(
            {
                "output": str(outcome.output),
                "notes": outcome.notes,
                "transpose": outcome.transpose,
                "playable": outcome.playable,
                "elapsed": round(outcome.elapsed, 2),
            }
        )
    )
    return 0

