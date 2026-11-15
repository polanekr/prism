"""Wait for queued Qt job results without blocking the GUI event loop."""
import time
from PyQt5.QtWidgets import QApplication
from PyQt5.QtTest import QTest


def wait_for_tasks(controller, timeout=20):
    deadline = time.monotonic()+timeout
    while controller.busy and time.monotonic() < deadline:
        QApplication.processEvents(); QTest.qWait(5)
    QApplication.processEvents()
    if controller.busy:
        controller.cancel()
        raise AssertionError('Background calculation did not finish within timeout.')


def dispose_widget(widget):
    """Destroy Qt scenes deterministically before cyclic Python collection.

    processEvents alone does not deliver DeferredDelete events. Leaving many
    closed scenes alive can collect pyqtgraph items after their ViewBox dies.
    """
    from PyQt5.QtCore import QCoreApplication, QEvent
    widget.close()
    widget.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    QApplication.processEvents()
