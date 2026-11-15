"""Versioned JSON state and processing provenance, without executable serialization."""
from datetime import datetime, timezone
from importlib import metadata
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import warnings
import subprocess
import tempfile
import uuid

import numpy as np


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    return value


def atomic_json(path, value):
    path = Path(path)
    payload = json.dumps(json_value(value), ensure_ascii=False, indent=2, allow_nan=False)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         suffix='.tmp', delete=False) as stream:
            temporary = stream.name
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def relative_path(path, base):
    path = Path(path).resolve()
    try:
        return Path(os.path.relpath(path, Path(base).resolve())).as_posix()
    except ValueError:
        # Windows inputs on another drive cannot have a relative path.
        return path.as_posix()


def file_record(path, base=None):
    path = Path(path).resolve()
    label = relative_path(path, base) if base is not None else str(path)
    if not path.is_file():
        return {'path': label, 'status': 'missing'}
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return {'path': label, 'status': 'present', 'sha256': digest.hexdigest(), 'bytes': path.stat().st_size}


def verify_files(records, base):
    for record in records:
        current = file_record(Path(base) / record['path'])
        if current['status'] != 'present' or current['sha256'] != record.get('sha256'):
            raise ValueError(f"Missing or changed project input: {record['path']}")


def calibration_snapshot(engine):
    result = {'fitted_params': engine.calib_params, 'calib_cov': engine.calib_cov,
              'calib_samples': engine.calib_samples, 'calib_sigma_od': engine.calib_sigma_od, 'random_seed': engine.random_seed}
    if engine.calib_doses is not None:
        result['doses_gy'] = engine.calib_doses
    if engine.calib_ods:
        result['measured_ods'] = engine.calib_ods
    return json_value(result)


@lru_cache(maxsize=1)
def _package_versions():
    """Installed dependency snapshot for this process; sources and inputs stay uncached."""
    packages = {}
    for name in ('prism', 'numpy', 'scipy', 'PyQt5', 'pyqtgraph', 'tifffile', 'scikit-learn', 'scikit-image', 'matplotlib', 'pymc', 'arviz'):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    return tuple(packages.items())


def runtime_record():
    root = Path(__file__).parent
    packages = dict(_package_versions())
    result = {'python': platform.python_version(), 'platform': platform.platform(), 'packages': packages,
              'sources': [file_record(root / name) for name in (
                  'dosimetry.py', 'uncertainty.py', 'scan_data.py', 'project_state.py',
                  'film_alignment.py', 'sample_statistics.py', 'prism_gui.py', 'bio_dosimetry.py',
                  'calibration.py', 'dose_solvers.py', 'film_processing.py', 'morphology.py',
                  'morphology_tab.py', 'bio_analysis.py', 'processing_settings.py', 'background_tasks.py', 'bayes_support.py', 'viewer_layout.py', 'beam_analysis.py', 'beam_analysis_panel.py')],
              'git': {'status': 'unavailable'}}
    if (root / '.git').exists():
        try:
            def git(*args):
                return subprocess.run(['git', '-C', str(root), *args], capture_output=True,
                                      text=True, check=True, timeout=5).stdout.strip()
            result['git'] = {'commit': git('rev-parse', 'HEAD'), 'working_tree': git('status', '--porcelain')}
        except (OSError, subprocess.SubprocessError):
            pass
    return result


def begin_run(engine, files, blank, settings):
    from scan_data import scan_paths
    root = Path(engine.output_folder).resolve()
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '-' + uuid.uuid4().hex[:12]
    folder = root / '.runs' / run_id
    folder.mkdir(parents=True)
    inputs = []
    for path in [*files, *([blank] if blank else [])]:
        inputs.append(file_record(path, root))
        inputs.extend(file_record(companion, root) for key, companion in scan_paths(path).items()
                      if key not in ('mean', 'lock') and companion.is_file())
    report = {'schema_version': 1, 'run_id': run_id, 'status': 'running',
              'started_at': datetime.now(timezone.utc).isoformat(), 'settings': settings,
              'inputs': inputs, 'calibration': calibration_snapshot(engine),
              'film_paths': [relative_path(path, root) for path in files],
              'blank_path': relative_path(blank, root) if blank else None,
              'batch_metadata': json_value(engine.batch_metadata), 'environment': runtime_record(), 'outputs': []}
    sources = folder / 'source'; sources.mkdir()
    source_root = Path(__file__).parent
    for path in [*(source_root / Path(record['path']).name for record in report['environment']['sources']),
                 source_root / 'pyproject.toml', source_root / 'uv.lock', source_root / '.python-version']:
        if path.is_file(): shutil.copy2(path, sources / path.name)
    report['source_archive'] = [file_record(path, folder) for path in sorted(sources.iterdir())]
    atomic_json(folder / 'run.json', report)
    return folder, report


def replay_run(manifest, output):
    """Replay verified inputs with the exact saved calibration into a new directory."""
    from dosimetry import GafchromicEngine
    manifest = Path(manifest).resolve()
    data = json.loads(manifest.read_text(encoding='utf-8'))
    if data.get('schema_version') != 1 or data.get('status') != 'completed':
        raise ValueError('Replay requires a completed schema-1 processing run.')
    base = manifest.parents[2]
    verify_files(data['inputs'], base)
    verify_files(data.get('source_archive', []), manifest.parent)
    current = runtime_record()
    saved_sources = {Path(row['path']).name: row.get('sha256') for row in data['environment']['sources']}
    current_sources = {Path(row['path']).name: row.get('sha256') for row in current['sources']}
    runtime_matches = all(current.get(key) == data['environment'].get(key) for key in ('python', 'packages', 'platform')) and saved_sources == current_sources
    if not runtime_matches:
        warnings.warn('Replay uses different source or runtime versions; inspect replay_source.json and the archived sources.')
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / 'replay_calibration.json', data['calibration'])
    engine = GafchromicEngine(str(output), random_seed=data['settings'].get('random_seed', 42))
    engine.load_calibration(str(output / 'replay_calibration.json'))
    engine.batch_metadata = data.get('batch_metadata') or {}
    from processing_settings import ProcessingSettings
    settings = ProcessingSettings.from_dict(data['settings'])
    engine.process_films([str(base / path) for path in data['film_paths']],
                         str(base / data['blank_path']) if data.get('blank_path') else '',
                         settings=settings)
    comparisons = [{'file': Path(row['path']).name,
                    'bitwise_match': file_record(output / Path(row['path']).name).get('sha256') == row.get('sha256')}
                   for row in data.get('outputs', []) if row['path'].endswith('.npy')]
    atomic_json(output / 'replay_source.json', {'source': file_record(manifest),
                'source_and_runtime_match': runtime_matches, 'array_comparisons': comparisons,
                'note': 'Inputs and calibration verified; numerical equivalence depends on source and runtime versions.'})


CONTROL_NAMES = ('combo_solver', 'chk_conv_clean', 'chk_median_filter', 'combo_calib_method',
                 'chk_calib_clean', 'input_doses', 'combo_cmap', 'combo_prof_mode', 'spin_band',
                 'slider_blend', 'chk_isodose', 'chk_show_roi', 'combo_mcmc_adaptation',
                 'spin_uncertainty_workers', 'combo_hist_normalization', 'combo_dah_band', 'combo_hist_range', 'chk_anchor_boundary')
MORPH_NAMES = ('combo_shot_mode', 'chk_snr_mask', 'slider_sigma')


def control_state(owner, names):
    from PyQt5.QtWidgets import QCheckBox, QComboBox, QLineEdit
    result = {}
    for name in names:
        widget = getattr(owner, name)
        if isinstance(widget, QCheckBox): value = widget.isChecked()
        elif isinstance(widget, QComboBox): value = widget.currentIndex()
        elif isinstance(widget, QLineEdit): value = widget.text()
        else: value = widget.value()
        result[name] = value
    return result


def apply_controls(owner, values):
    from PyQt5.QtWidgets import QCheckBox, QComboBox, QLineEdit
    for name, value in values.items():
        widget = getattr(owner, name)
        if isinstance(widget, QCheckBox): widget.setChecked(value)
        elif isinstance(widget, QComboBox): widget.setCurrentIndex(value)
        elif isinstance(widget, QLineEdit): widget.setText(value)
        else: widget.setValue(value)


def save_session(window):
    """Explicit project snapshot. Derived analyses are recomputed after restore."""
    root = Path(window.project_dir).resolve()
    records = []
    def track(path):
        if path is None: return None
        record = file_record(path, root)
        if record['status'] != 'present': raise ValueError(f'Missing input: {path}')
        records.append(record)
        return record['path']
    def track_map(path):
        result = track(path)
        if path is not None:
            path = Path(path)
            stem = path.stem.removesuffix('_dose')
            for suffix in ('_std.npy', '_meta.json'):
                companion = path.with_name(stem + suffix)
                if companion.is_file(): track(companion)
        return result
    bio = window.bio_tab
    state = {'schema_version': 1, 'saved_at': datetime.now(timezone.utc).isoformat(),
             'environment': runtime_record(), 'files': records,
             'calibration': calibration_snapshot(window.engine) if window.engine and window.engine.calib_params else None,
             'controls': control_state(window, CONTROL_NAMES),
             'morphology': control_state(window.morphology_tab, MORPH_NAMES),
             'beam_experiments': window.morphology_tab.measured.snapshot(),
             'blank': {'file': track(window.raw_dir / window.combo_blank_film.currentText())
                       if window.combo_blank_film.currentText() else None,
                       'rgb': window.global_blank_rgb, 'std': window.global_blank_std},
             'dose_map': track_map(getattr(window, 'current_dose_path', None)),
             'raw_image': track(getattr(window, 'current_raw_path', None)),
             'roi': window.roi.saveState(), 'line_roi': window.line_roi.saveState(),
             'film_boundary': window.film_boundary.saveState(),
             'crosshair': [window.v_line.value(), window.h_line.value()],
             'active_tab': window.tabs.currentIndex(),
             'viewer_layout': window.viewer_layout.snapshot(),
             'dose_levels': list(window._map_display_states.get('dose', {}).get(
                 'levels', window.colorbar.getLevels())),
             'bio': {'dvh_range': bio.combo_dvh_range.currentIndex(), 'front': track_map(bio.film_paths.get('front')), 'back': track_map(bio.film_paths.get('back')),
                     'geometry': {key: spin.value() for key, spin in bio.spins.items()},
                     'blend': bio.sl_blend.value(),
                     'alignment': [bio.sl_shift_x.value(), bio.sl_shift_y.value(), bio.sl_rot.value()],
                     'rois': [{'kind': getattr(roi, 'project_kind', 0), 'state': roi.saveState()} for roi in bio.rois]}}
    # Sessions reopen in dose mode: do not save the active STD palette as
    # the dose palette, or its window as dose levels.
    dose_preset = window._map_display_states.get('dose', {}).get('preset')
    if dose_preset:
        state['controls']['combo_cmap'] = window.combo_cmap.findText(dose_preset)
    atomic_json(root / 'project.json', state)
    return root / 'project.json'


def restore_session(window):
    """Validate files, calibration and a staged bio tab before replacing active state."""
    from dosimetry import GafchromicEngine
    from bio_dosimetry import BioDosimetryTab
    import pyqtgraph as pg
    root = Path(window.project_dir).resolve()
    state = json.loads((root / 'project.json').read_text(encoding='utf-8'))
    if state.get('schema_version') != 1:
        raise ValueError('Unsupported project state version; original file was preserved.')
    verify_files(state['files'], root)
    if 'viewer_layout' in state: window.viewer_layout.validate(state['viewer_layout'])
    if 'beam_experiments' in state: window.morphology_tab.measured.validate_state(state['beam_experiments'])
    # Validate serializable controls before touching the current UI.
    def validate_controls(owner, values, names):
        from PyQt5.QtWidgets import QCheckBox, QComboBox, QLineEdit
        if set(values) != set(names): raise ValueError('Incomplete project controls.')
        for name, value in values.items():
            widget = getattr(owner, name)
            if isinstance(widget, QCheckBox): valid = isinstance(value, bool)
            elif isinstance(widget, QLineEdit): valid = isinstance(value, str)
            elif isinstance(widget, QComboBox): valid = type(value) is int and 0 <= value < widget.count()
            else: valid = isinstance(value, (int, float)) and np.isfinite(value) and widget.minimum() <= value <= widget.maximum()
            if not valid: raise ValueError(f'Invalid saved control: {name}')
    # Schema-1 files predate these presentation/performance controls.
    defaults = {'combo_mcmc_adaptation': 0, 'spin_uncertainty_workers': 1,
                'combo_hist_normalization': 1, 'combo_dah_band': 1, 'combo_hist_range': 1, 'chk_anchor_boundary': False}
    state['controls'] = {**defaults, **state['controls']}
    validate_controls(window, state['controls'], CONTROL_NAMES)
    validate_controls(window.morphology_tab, state['morphology'], MORPH_NAMES)
    blank = state['blank']
    for key in ('rgb', 'std'):
        if blank[key] is not None:
            values = np.asarray(blank[key], float)
            if values.shape != (3,) or not np.all(np.isfinite(values)) or np.any(values < 0) or (key == 'rgb' and np.any(values <= 0)):
                raise ValueError('Invalid saved blank RGB/STD.')
    engine = GafchromicEngine(str(window.processed_dir))
    if state['calibration'] is not None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'calibration.json'
            atomic_json(path, state['calibration']); engine.load_calibration(str(path))
    # Stage ROI states as well as maps: malformed snapshots leave the active session intact.
    def validate_roi(data):
        for name in ('pos', 'size'):
            values = np.asarray(data[name], float)
            if values.shape != (2,) or not np.all(np.isfinite(values)):
                raise ValueError('Invalid ROI coordinates.')
        if np.any(np.asarray(data['size']) <= 0) or not np.isfinite(data['angle']):
            raise ValueError('Invalid ROI size or angle.')
        if 'points' in data:
            points = np.asarray(data['points'], float)
            if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2 or not np.all(np.isfinite(points)):
                raise ValueError('Invalid ROI vertices.')
    validate_roi(state['roi']); validate_roi(state['line_roi'])
    if 'film_boundary' in state:
        validate_roi(state['film_boundary'])
        if state['film_boundary']['angle'] != 0: raise ValueError('Film boundary must be axis-aligned.')
    if len(state['crosshair']) != 2: raise ValueError('Invalid crosshair.')
    roi = pg.EllipseROI([0, 0], [1, 1]); roi.setState(state['roi'])
    line = pg.LineSegmentROI([[0, 0], [1, 1]]); line.setState(state['line_roi'])
    for value in state['crosshair']:
        if not np.isfinite(value): raise ValueError('Invalid crosshair.')
    staged = BioDosimetryTab()
    try:
        bio_state = state['bio']
        dvh_range = bio_state.get('dvh_range', 0)
        if type(dvh_range) is not int or not 0 <= dvh_range < staged.combo_dvh_range.count():
            raise ValueError('Invalid dDVH dose range.')
        staged.combo_dvh_range.setCurrentIndex(dvh_range)
        if set(bio_state['geometry']) != set(staged.spins): raise ValueError('Incomplete geometry.')
        for key, value in bio_state['geometry'].items():
            if not np.isfinite(value) or not 0 <= value <= 50: raise ValueError('Invalid geometry.')
            staged.spins[key].setValue(value)
        if type(bio_state['blend']) is not int or not 0 <= bio_state['blend'] <= 100: raise ValueError('Invalid bio blend.')
        staged.sl_blend.setValue(bio_state['blend'])
        sliders = (staged.sl_shift_x, staged.sl_shift_y, staged.sl_rot)
        if len(bio_state['alignment']) != 3: raise ValueError('Invalid alignment.')
        for slider, value in zip(sliders, bio_state['alignment']):
            if type(value) is not int or not slider.minimum() <= value <= slider.maximum(): raise ValueError('Invalid alignment.')
            slider.setValue(value)
        for target in ('front', 'back'):
            if bio_state[target] and not staged.load_film(target, root / bio_state[target], report_errors=False):
                raise ValueError(f'Cannot restore {target} map.')
        for entry in bio_state['rois']:
            if type(entry['kind']) is not int or not 0 <= entry['kind'] < staged.combo_roi_type.count() or staged.back_dose is None:
                raise ValueError('Invalid sample ROI.')
            validate_roi(entry['state'])
            staged.combo_roi_type.setCurrentIndex(entry['kind']); staged.add_selected_roi()
            staged.rois[-1].setState(entry['state'])
        # Preflight displayed maps before mutation.
        for name in ('dose_map',):
            if state[name]:
                array = np.load(root / state[name], allow_pickle=False)
                if array.ndim != 2 or not array.size or not np.all(np.isfinite(array)) or np.any(array < 0):
                    raise ValueError('Invalid saved dose map.')
                stem = Path(state[name]).stem.removesuffix('_dose')
                std_path = (root / state[name]).with_name(stem + '_std.npy')
                if std_path.is_file():
                    sigma = np.load(std_path, allow_pickle=False)
                    if sigma.shape != array.shape or not np.all(np.isfinite(sigma)) or np.any(sigma < 0):
                        raise ValueError('Invalid saved STD map.')
                meta_path = (root / state[name]).with_name(stem + '_meta.json')
                if meta_path.is_file():
                    meta = json.loads(meta_path.read_text(encoding='utf-8'))
                    spacing = float(meta.get('pixel_spacing_mm', .0846))
                    if not np.isfinite(spacing) or spacing <= 0: raise ValueError('Invalid saved spacing.')
        if state['raw_image']:
            import tifffile
            tifffile.imread(root / state['raw_image'])
    except BaseException:
        staged.deleteLater()
        raise
    levels = np.asarray(state['dose_levels'], float)
    if levels.shape != (2,) or not np.all(np.isfinite(levels)) or levels[0] >= levels[1]:
        staged.deleteLater(); raise ValueError('Invalid dose color levels.')
    if type(state['active_tab']) is not int or not 0 <= state['active_tab'] < window.tabs.count():
        staged.deleteLater(); raise ValueError('Invalid active tab.')
    window.current_dose_map = None; window.current_std_map = None; window.current_meta = {}
    window.current_dose_path = None; window.current_raw_path = None; window.current_raw_img = None
    window.img_item_dose.clear(); window.img_item_raw.clear()
    window._clear_analysis_views()
    window.engine = engine
    window.calib_dir = None
    window.lbl_calib_status.setText('Loaded project calibration' if engine.calib_params else 'No calibration loaded')
    apply_controls(window, state['controls'])
    old_morph = window.morphology_tab
    morph_index = window.tabs.indexOf(old_morph)
    window.morphology_tab = type(old_morph)(window)
    apply_controls(window.morphology_tab, state['morphology'])
    window.tabs.removeTab(morph_index)
    window.tabs.insertTab(morph_index, window.morphology_tab, '🌌 Advanced Beam Morphology')
    old_morph.deleteLater()
    if blank['file']:
        window.combo_blank_film.setCurrentText(Path(blank['file']).name)
    window.global_blank_rgb = blank['rgb']; window.global_blank_std = blank['std']
    if state['raw_image']: window.load_and_display_raw(root / state['raw_image'])
    if state['dose_map']: window.load_and_display_dose(root / state['dose_map'])
    if 'film_boundary' in state: window.film_boundary.setState(state['film_boundary'])
    window.roi.setState(state['roi']); window.line_roi.setState(state['line_roi'])
    window.v_line.setValue(state['crosshair'][0]); window.h_line.setValue(state['crosshair'][1])
    old = window.bio_tab
    old.tasks.cancel()
    index = window.tabs.indexOf(old)
    window.tabs.removeTab(index); window.tabs.insertTab(index, staged, '🧬 Radiobiology & 3D Sample Dosimetry')
    window.bio_tab = staged; old.deleteLater()
    window.tabs.setCurrentIndex(state['active_tab'])
    if 'viewer_layout' in state: window.viewer_layout.restore(state['viewer_layout'])
    window.colorbar.setLevels(*levels)
    window.update_roi_stats(); window.update_profiles()
    window.morphology_tab.invalidate_analysis()
    if 'beam_experiments' in state: window.morphology_tab.measured.restore(state['beam_experiments'])
    return state
