"""Background conversion worker.

The model must never run on the Qt main thread: the worker owns a
:class:`ConversionService` call, runs inside a :class:`QThread`, and talks to
the window exclusively through signals. It never touches a QWidget.
"""

from __future__ import annotations

import traceback

from PySide6.QtCore import QObject, QThread, Signal

from app.application.conversion_service import ConversionOutcome, ConversionRequest, ConversionService


class ConversionWorker(QObject):
    progress = Signal(int, str)
    succeeded = Signal(object)  # ConversionOutcome
    failed = Signal(str)

    def __init__(self, service: ConversionService, request: ConversionRequest) -> None:
        super().__init__()
        self._service = service
        self._request = request

    def run(self) -> None:
        try:
            outcome = self._service.run(self._request, on_progress=self._on_progress)
            self.succeeded.emit(outcome)
        except Exception as exc:  # noqa: BLE001 - the GUI shows a message, log keeps details
            traceback.print_exc()
            self.failed.emit(str(exc))

    def _on_progress(self, stage: int, text: str) -> None:
        self.progress.emit(stage, text)


def start_worker(service: ConversionService, request: ConversionRequest) -> tuple[QThread, ConversionWorker]:
    """Create a thread + worker wired together (not started yet).

    The caller keeps references so Qt does not garbage-collect them mid-run and
    connects to the worker's signals before calling ``thread.start()``.
    """
    thread = QThread()
    worker = ConversionWorker(service, request)
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    # Without this the thread's event loop keeps running after run() returns
    # and ``finished`` would never fire (leaving the UI disabled forever).
    worker.succeeded.connect(thread.quit)
    worker.failed.connect(thread.quit)
    return thread, worker


__all__ = ["ConversionWorker", "start_worker", "QObject", "QThread"]
