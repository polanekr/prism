"""Run source-tree tests with the same bundled Qt paths on Windows and Linux."""
import os
from pathlib import Path
import subprocess
import sys

from prism_app import qt_environment


if __name__ == '__main__':
    environment, _ = qt_environment()
    environment['QT_QPA_PLATFORM'] = 'offscreen'
    root = Path(__file__).resolve().parent
    raise SystemExit(subprocess.call(
        [sys.executable, '-m', 'unittest', 'discover', '-s', str(root / 'tests'), '-v', *sys.argv[1:]],
        cwd=root, env=environment))
