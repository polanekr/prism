"""Cross-platform launcher and installation diagnostics; import Qt only after setup."""
import argparse
from importlib import metadata, util
import json
import multiprocessing
import os
from pathlib import Path
import platform
import shutil
import sys


CORE_PACKAGES = {
    'numpy': 'numpy', 'scipy': 'scipy', 'matplotlib': 'matplotlib',
    'PyQt5': 'PyQt5', 'pyqtgraph': 'pyqtgraph', 'scikit-image': 'skimage',
    'scikit-learn': 'sklearn', 'tifffile': 'tifffile', 'tqdm': 'tqdm',
}
OPTIONAL_PACKAGES = {
    'bayes': {'pymc': 'pymc', 'arviz': 'arviz', 'h5netcdf': 'h5netcdf', 'h5py': 'h5py'},
    'excel': {'openpyxl': 'openpyxl'},
}
_DLL_HANDLES = []


def qt_environment(environ=None, system=None, qt_root=None):
    """Prepare this interpreter's Qt paths without importing a Qt library."""
    environment = dict(os.environ if environ is None else environ)
    system = platform.system() if system is None else system
    if qt_root is None:
        spec = util.find_spec('PyQt5')
        if spec is None or spec.origin is None:
            raise RuntimeError('PyQt5 is not installed. Run uv sync --locked first.')
        qt_root = Path(spec.origin).parent / 'Qt5'
    qt_root = Path(qt_root)
    environment['PYQTGRAPH_QT_LIB'] = 'PyQt5'
    environment['QT_PLUGIN_PATH'] = str(qt_root / 'plugins')
    environment['QT_QPA_PLATFORM_PLUGIN_PATH'] = str(qt_root / 'plugins' / 'platforms')
    if system == 'Linux':
        library = str(qt_root / 'lib')
        other = [part for part in environment.get('LD_LIBRARY_PATH', '').split(os.pathsep)
                 if part and part != library]
        environment['LD_LIBRARY_PATH'] = os.pathsep.join([library, *other])
        # Match the existing X11 launcher on desktops; retain an explicit platform.
        environment.setdefault('QT_QPA_PLATFORM', 'xcb')
    return environment, qt_root


def prepare_runtime(arguments, check=False):
    environment, qt_root = qt_environment()
    if check:
        environment['QT_QPA_PLATFORM'] = 'offscreen'
    # Linux's dynamic loader reads LD_LIBRARY_PATH at process startup.
    if sys.platform.startswith('linux') and environment.get('LD_LIBRARY_PATH') != os.environ.get('LD_LIBRARY_PATH'):
        os.execvpe(sys.executable, [sys.executable, '-m', 'prism_app', *arguments], environment)
    os.environ.update(environment)
    if sys.platform == 'win32' and hasattr(os, 'add_dll_directory'):
        dll_directory = qt_root / 'bin'
        if dll_directory.is_dir():
            _DLL_HANDLES.append(os.add_dll_directory(str(dll_directory)))


def check_installation(require_bayes=False, require_excel=False):
    """Check imports and Qt plugin loading without showing a window or scanning."""
    report = {'python': platform.python_version(), 'platform': platform.platform(),
              'architecture': platform.machine(), 'application': str(Path(__file__).resolve()),
              'core': {}, 'optional': {}, 'errors': []}
    if not (3, 12) <= sys.version_info[:2] < (3, 15):
        report['errors'].append('Supported Python versions: 3.12, 3.13, 3.14.')
    for distribution, module in CORE_PACKAGES.items():
        try:
            __import__(module)
            report['core'][distribution] = metadata.version(distribution)
        except Exception as error:
            report['errors'].append(f'{distribution}: {error}')
    for extra, packages in OPTIONAL_PACKAGES.items():
        results = {}
        available = True
        for distribution, module in packages.items():
            try:
                __import__(module)
                results[distribution] = metadata.version(distribution)
            except Exception as error:
                available = False
                results[distribution] = f'unavailable: {error}'
        report['optional'][extra] = {'available': available, 'packages': results}
        if not available and ((extra == 'bayes' and require_bayes) or (extra == 'excel' and require_excel)):
            report['errors'].append(f'Required optional feature unavailable: {extra}')
    if not report['errors']:
        from PyQt5.QtCore import QT_VERSION_STR, qVersion
        from PyQt5.QtWidgets import QApplication
        app = QApplication.instance() or QApplication([])
        report['qt'] = {'version': qVersion(), 'bindings_built_against': QT_VERSION_STR,
                        'platform_plugin': app.platformName()}
        # Import the complete application to catch missing packaged modules too.
        import prism_gui  # noqa: F401
        import scan  # noqa: F401
    executable = 'naps2.console' if sys.platform == 'win32' else 'scanimage'
    report['scanner_executable'] = shutil.which(executable)
    report['scanner_supported'] = sys.platform.startswith('linux')
    report['ok'] = not report['errors']
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report['ok'] else 1


def main(argv=None):
    multiprocessing.freeze_support()
    parser = argparse.ArgumentParser(description='PRISM film dosimetry')
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument('--check', action='store_true', help='Check the installation without opening a window.')
    actions.add_argument('--replay-run', type=Path, help='Replay a saved processing run after verifying its input hashes.')
    parser.add_argument('--replay-output', type=Path, help='A NEW output folder for --replay-run.')
    actions.add_argument('--validate', action='store_true', help='Run numerical validation and save HTML/JSON reports.')
    parser.add_argument('--require-bayes', action='store_true', help='Require MCMC support during --check.')
    parser.add_argument('--require-excel', action='store_true', help='Require Excel export during --check.')
    actions.add_argument('--scanner', action='store_true', help='Open only the scanner dialog.')
    parser.add_argument('--target-directory', type=Path, help='Initial scanner output directory.')
    parser.add_argument('--validation-output', type=Path, help='A NEW directory for validation results.')
    parser.add_argument('--no-median-filter', action='store_true', help='Disable spatial median filters in validation conversions.')
    parser.add_argument('--quick', action='store_true', help='Small pilot validation; not a precise coverage estimate.')
    parser.add_argument('--seed', type=int, help='Validation random seed (default: 42).')
    parser.add_argument('--repetitions', type=int, help='Independent calibration/measurement experiments.')
    parser.add_argument('--posterior-samples', type=int, help='Parameter samples propagated per validation calibration.')
    parser.add_argument('--reference-manifest', type=Path, help='JSON manifest of independent reference measurements.')
    parser.add_argument('--mcmc-trace', type=Path, action='append', help='A NetCDF trace to diagnose; repeatable.')
    parser.add_argument('--no-synthetic', action='store_true', help='Evaluate only supplied references/calibration/traces.')
    parser.add_argument('--calibration-folder', type=Path, help='Existing calibration TIFF folder for dose-group holdout.')
    parser.add_argument('--calibration-roi-size', type=int, help='Central ROI side in pixels for supplied calibration (default: 50).')
    parser.add_argument('--calibration-doses', help='Comma-separated doses in calibration filename order.')
    arguments = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(arguments)
    if (args.require_bayes or args.require_excel) and not args.check:
        parser.error('--require-bayes and --require-excel are used with --check.')
    if not args.validate and any((args.validation_output, args.quick, args.no_median_filter, args.seed is not None,
                                 args.repetitions is not None, args.posterior_samples is not None,
                                 args.reference_manifest, args.mcmc_trace, args.no_synthetic,
                                 args.calibration_folder, args.calibration_doses, args.calibration_roi_size is not None)):
        parser.error('Validation options require --validate.')
    if bool(args.replay_run) != bool(args.replay_output):
        parser.error('--replay-run and --replay-output must be supplied together.')
    try:
        prepare_runtime(arguments, check=args.check or args.validate or bool(args.replay_run))
        if args.replay_run:
            from project_state import replay_run
            replay_run(args.replay_run, args.replay_output)
            print(f'Replay completed: {args.replay_output.resolve()}')
            return 0
        if args.check:
            return check_installation(args.require_bayes, args.require_excel)
        if args.validate:
            from datetime import datetime, timezone
            from validation import ValidationConfig, run_validation
            config = ValidationConfig(seed=args.seed if args.seed is not None else 42,
                repetitions=args.repetitions if args.repetitions is not None else (6 if args.quick else 60),
                image_size=12 if args.quick else 24, median_filtering=not args.no_median_filter,
                posterior_samples=args.posterior_samples if args.posterior_samples is not None else (4 if args.quick else 16))
            output = args.validation_output or Path('validation-results') / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
            doses = [float(value.strip()) for value in args.calibration_doses.split(',')] if args.calibration_doses else None
            report = run_validation(output, config, args.reference_manifest, args.mcmc_trace or (),
                synthetic=not args.no_synthetic, progress=lambda message: print(message, file=sys.stderr),
                calibration_folder=args.calibration_folder, calibration_doses=doses,
                calibration_roi_size=args.calibration_roi_size if args.calibration_roi_size is not None else 50)
            print(f"Validation {report['status']}: {output.resolve() / 'report.html'}")
            print('Physical validity is not established by this numerical report.')
            return 1 if report['errors'] else 0
        if args.scanner and sys.platform == 'win32':
            from scan import WINDOWS_SCAN_UNAVAILABLE
            raise RuntimeError(WINDOWS_SCAN_UNAVAILABLE)
        from PyQt5.QtWidgets import QApplication
        app = QApplication.instance() or QApplication([sys.argv[0]])
        app.setStyle('Fusion')
        if args.scanner:
            from scan import FilmScannerDialog
            window = FilmScannerDialog(target_directory=str(args.target_directory) if args.target_directory else None)
        else:
            from prism_gui import PrismDosimetryGUI
            window = PrismDosimetryGUI()
        window.show()
        return app.exec_()
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        print(f'PRISM could not start: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
