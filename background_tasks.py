"""Cooperative Qt jobs. Workers never access widgets; stale results are discarded."""
import threading
import logging
from PyQt5.QtCore import QObject, QThread, pyqtSignal, pyqtSlot


class CalculationCancelled(Exception):
    pass


def check_cancelled(cancelled):
    if cancelled(): raise CalculationCancelled()


# Keep parentless QThreads alive even when a tab is replaced during calculation.
_RUNNING = set()


class CalculationThread(QThread):
    result = pyqtSignal(int, object)
    error = pyqtSignal(int, str)
    progress = pyqtSignal(int, str)

    def __init__(self, generation, function):
        super().__init__()
        self.generation = generation
        self.function = function
        self.cancel_event = threading.Event()

    def cancel(self): self.cancel_event.set()

    def run(self):
        try:
            def check(): check_cancelled(self.cancel_event.is_set)
            def progress(message):
                check(); self.progress.emit(self.generation, message)
            check()
            result = self.function(check, progress)
            check()
            self.result.emit(self.generation, result)
        except CalculationCancelled:
            pass
        except Exception as error:
            logging.getLogger(__name__).exception('Background calculation failed')
            self.error.emit(self.generation, f'{type(error).__name__}: {error}')


class TaskController(QObject):
    ready = pyqtSignal(object)
    failed = pyqtSignal(str)
    progress = pyqtSignal(str)
    busy_changed = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.generation = 0
        self.workers = set()
        self.pending = None

    @property
    def busy(self): return bool(self.workers or self.pending)

    def submit(self, function):
        # One active job per controller; newest request replaces the pending one.
        self.generation += 1
        for worker in self.workers: worker.cancel()
        self.pending = (self.generation, function)
        self.busy_changed.emit(True)
        if not self.workers: self._start_pending()

    def _start_pending(self):
        if self.pending is None: return
        generation, function = self.pending; self.pending = None
        worker = CalculationThread(generation, function)
        self.workers.add(worker); _RUNNING.add(worker)
        worker.result.connect(self._result)
        worker.error.connect(self._error)
        worker.progress.connect(self._progress)
        worker.finished.connect(self._finished)
        worker.finished.connect(lambda current=worker: _release_worker(current))
        self.destroyed.connect(worker.cancel)
        worker.start()

    def cancel(self):
        self.generation += 1; self.pending = None
        for worker in self.workers: worker.cancel()

    @pyqtSlot(int, object)
    def _result(self, generation, result):
        if generation == self.generation: self.ready.emit(result)

    @pyqtSlot(int, str)
    def _error(self, generation, error):
        if generation == self.generation: self.failed.emit(error)

    @pyqtSlot(int, str)
    def _progress(self, generation, message):
        if generation == self.generation: self.progress.emit(message)

    @pyqtSlot()
    def _finished(self):
        worker = self.sender()
        self.workers.discard(worker)
        self._start_pending()
        self.busy_changed.emit(self.busy)


def cancel_all():
    for worker in tuple(_RUNNING): worker.cancel()


def running_jobs(): return any(worker.isRunning() for worker in _RUNNING)


def _release_worker(worker):
    _RUNNING.discard(worker)
    worker.deleteLater()
