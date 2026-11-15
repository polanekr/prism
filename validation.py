"""Reproducible numerical studies; none of these establishes physical validity."""
from collections import defaultdict
from contextlib import redirect_stdout
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib import metadata
import base64
import csv
import hashlib
import html
import json
import math
from pathlib import Path
import platform
import re
import shutil
import sys
import tempfile
import time
import traceback
import warnings

import numpy as np
from scipy.ndimage import median_filter
import tifffile

from dosimetry import GafchromicEngine
from scan_data import ScanAccumulator, scan_paths


CHANNELS = ('R', 'G', 'B')
METHODS = ('fast', 'mayer', 'admm')
# Independent, explicit forward-model fixture, not a call to the engine's OD curve.
FORWARD_PARAMETERS = {'R': [0., 1.2, 6., 1.], 'G': [0., .9, 10., 1.], 'B': [0., .6, 18., 1.]}
LRA_COEFFICIENTS = np.array([1.2e-5, .5e-5, .1e-5])
CALIBRATION_DOSES = (0., 1., 2., 4., 8., 12., 20.)
PROBE_DOSES = (3., 7., 16.)
SOURCE_MODULES = ('validation.py', 'dosimetry.py', 'uncertainty.py', 'scan_data.py', 'project_state.py', 'dose_solvers.py', 'calibration.py', 'film_processing.py', 'processing_settings.py')


@dataclass(frozen=True)
class ValidationConfig:
    seed: int = 42
    repetitions: int = 60
    image_size: int = 24
    posterior_samples: int = 16
    scan_repeats: int = 3
    sigma_od: float = .003
    median_filtering: bool = True

    def validate(self):
        if not isinstance(self.median_filtering, bool):
            raise ValueError('median_filtering must be boolean.')
        for name in ('seed', 'repetitions', 'image_size', 'posterior_samples', 'scan_repeats'):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f'{name} must be an integer.')
        if self.seed < 0 or self.repetitions < 3 or self.image_size < 12:
            raise ValueError('Require a nonnegative seed, at least 3 repetitions and image_size >= 12.')
        if self.posterior_samples < 2 or self.scan_repeats < 2:
            raise ValueError('At least two parameter samples and two scan repetitions are required.')
        if not np.isfinite(self.sigma_od) or self.sigma_od < 0:
            raise ValueError('sigma_od must be finite and nonnegative.')


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _rng(seed, study, index=0):
    return np.random.default_rng(np.random.SeedSequence([seed, study, index]))


def write_synthetic_film(path, dose, rng, config, thickness=1., position_mm=0., matched_lra=True):
    """Write 16-bit mean TIFF + SD/SEM sidecars from an explicit forward model.

    The matched LRA case uses the inverse of the engine's assumed quadratic
    correction. Recovery then tests model consistency, not actual scanner LRA.
    Noise is independent Gaussian OD noise before intensity quantization.
    """
    path = Path(path)
    dose = np.asarray(dose, float)
    if dose.ndim != 2 or not np.all(np.isfinite(dose)) or np.any(dose < 0):
        raise ValueError('Synthetic doses must be finite, nonnegative 2D maps.')
    spacing = 25.4 / 300
    left = 108 + position_mm - dose.shape[1] * spacing / 2
    channels = []
    for ch in CHANNELS:
        a, b, c, exponent = FORWARD_PARAMETERS[ch]
        power = np.power(dose, exponent)
        channels.append(a + b * power / (power + c))
    ideal_od = np.stack(channels, axis=-1) * np.asarray(thickness)[..., None]
    penalty = LRA_COEFFICIENTS[None, None, :] * (
        left + np.arange(dose.shape[1])[None, :, None] * spacing - 108)**2
    if matched_lra:
        discriminant = 1 - 4 * penalty * ideal_od
        if np.any(discriminant < 0):
            raise ValueError('Synthetic LRA inversion is outside its real-valued domain.')
        observed_od = 2 * ideal_od / (1 + np.sqrt(discriminant))
    else:
        observed_od = ideal_od
    accumulator = ScanAccumulator()
    clipped = 0
    for _ in range(config.scan_repeats):
        noise = rng.normal(0, config.sigma_od, observed_od.shape)
        intensity = np.array([60000., 56000., 52000.]) * 10**-(observed_od + noise)
        clipped += int(np.count_nonzero((intensity < 1) | (intensity > 65535)))
        accumulator.add(np.rint(np.clip(intensity, 1, 65535)).astype(np.uint16))
    mean, std, sem = accumulator.finish()
    paths = scan_paths(path)
    description = f'left_mm={left};scanner_center_mm=108.0;scan_count={config.scan_repeats}'
    for key, image in (('mean', mean), ('std', std), ('sem', sem)):
        tifffile.imwrite(paths[key], image, photometric='rgb', resolution=(300, 300),
                         resolutionunit='INCH', description=description)
    paths['meta'].write_text(json.dumps({'format_version': 1, 'synthetic': True,
        'scan_count': config.scan_repeats, 'noise_model': 'independent_repeat_sem'}, indent=2), encoding='utf-8')
    return {'filename': path.name, 'sha256': _sha256(path),
            'sidecar_hashes': {key: _sha256(paths[key]) for key in ('std', 'sem', 'meta')},
            'clipped_channel_values': clipped,
            'position_mm': position_mm, 'matched_lra': matched_lra}


def _make_calibration(folder, config, rng):
    folder.mkdir()
    return [write_synthetic_film(folder / f'{i:02d}_dose_{dose:g}.tif',
                np.full((config.image_size, config.image_size), dose), rng, config)
            for i, dose in enumerate(CALIBRATION_DOSES)]


def _fit_calibration(folder, output, config):
    engine = GafchromicEngine(str(output))
    engine.run_calibration(str(folder), list(CALIBRATION_DOSES), roi_size=max(2, config.image_size // 4),
                           n_posterior_samples=config.posterior_samples)
    return engine


def error_metrics(estimate, reference):
    estimate, reference = np.asarray(estimate, float), np.asarray(reference, float)
    if estimate.shape != reference.shape or estimate.size == 0:
        raise ValueError('Reference and estimate must have matching, nonempty shapes.')
    if not np.all(np.isfinite(estimate)) or not np.all(np.isfinite(reference)):
        raise ValueError('Reference and estimate must be finite.')
    error = estimate - reference
    return {'bias_gy': float(np.mean(error)), 'mae_gy': float(np.mean(np.abs(error))),
            'rmse_gy': float(np.sqrt(np.mean(error**2))),
            'max_abs_error_gy': float(np.max(np.abs(error))),
            'mean_estimate_gy': float(np.mean(estimate)), 'mean_reference_gy': float(np.mean(reference)),
            'peak_error_gy': float(np.max(estimate) - np.max(reference)), 'cell_count': int(estimate.size)}


def _process(engine, path, blank, method, output, median_filtering=True):
    engine.output_folder = str(output)
    output.mkdir(parents=True, exist_ok=True)
    engine.process_films([str(path)], str(blank), method=method, use_cleaning=False, median_filtering=median_filtering)
    stem = path.stem
    dose = np.load(output / f'{stem}_dose.npy', allow_pickle=False)
    sigma = np.load(output / f'{stem}_std.npy', allow_pickle=False)
    if dose.shape != sigma.shape or not np.all(np.isfinite(sigma)) or np.any(sigma < 0):
        raise ValueError('The processing pipeline returned an invalid STD map.')
    return dose, sigma


def _phantoms(size):
    y, x = np.indices((size, size))
    centre = (size - 1) / 2
    gaussian = 12 * np.exp(-((x - centre)**2 + (y - centre)**2) / (2 * (size / 5)**2))
    hotspot = np.full((size, size), 2.)
    hotspot[size // 2, size // 2] = 20.
    peak3 = np.full((size, size), 2.); peak5 = peak3.copy()
    peak3[size//2-1:size//2+2, size//2-1:size//2+2] = 20.
    peak5[size//2-2:size//2+3, size//2-2:size//2+3] = 20.
    ripple = 1 + .02 * np.sin(2 * np.pi * x / size)
    return [
        ('zero', np.zeros((size, size)), 1., 0., True),
        ('uniform_3', np.full((size, size), 3.), 1., 0., True),
        ('uniform_10', np.full((size, size), 10.), 1., 0., True),
        ('gaussian_peak', gaussian, 1., 0., True),
        ('single_pixel_hotspot', hotspot, 1., 0., True),
        ('peak_3px', peak3, 1., 0., True),
        ('peak_5px', peak5, 1., 0., True),
        ('thickness_ripple', np.full((size, size), 7.), ripple, 0., True),
        ('high_dose_80', np.full((size, size), 80.), 1., 0., True),
        ('left_matched_lra', np.full((size, size), 7.), 1., -50., True),
        ('right_matched_lra', np.full((size, size), 7.), 1., 50., True),
        ('lra_mismatch', np.full((size, size), 7.), 1., 50., False),
    ]


def _compare_solvers(root, config, engine, blank):
    rows, sources = [], []
    for index, (case, truth, thickness, position, matched) in enumerate(_phantoms(config.image_size)):
        path = root / f'{case}.tif'
        sources.append(write_synthetic_film(path, truth, _rng(config.seed, 2, index), config,
                                           thickness, position, matched))
        np.save(root / f'{case}_truth.npy', truth)
        for method in METHODS:
            start = time.perf_counter()
            row = {'case': case, 'method': method, 'status': 'computed',
                   'outside_calibration_range': bool(np.max(truth) > max(CALIBRATION_DOSES))}
            try:
                estimate, sigma = _process(engine, path, blank, method, root / 'processed' / method, config.median_filtering)
                row.update(error_metrics(estimate, truth))
                row['mean_sigma_gy'] = float(np.mean(sigma))
            except Exception as error:
                traceback.print_exc(file=sys.stdout)
                row.update(status='error', error=str(error))
            row['elapsed_seconds'] = time.perf_counter() - start
            rows.append(row)
    return rows, sources


def leave_one_dose_out(calibration_folder, doses, work_folder, roi_size=6, posterior_samples=4):
    """Hold out entire nonzero dose groups, refitting the production standard calibration.

    The zero-dose blank remains a required reference. Excluding the highest
    dose is explicitly labelled extrapolation. Original files are never modified.
    """
    files, doses = GafchromicEngine._calibration_inputs(calibration_folder, doses)
    results = []
    work_folder = Path(work_folder)
    work_folder.mkdir(parents=True, exist_ok=True)
    for held_dose in np.unique(doses[doses > 0]):
        retained = doses != held_dose
        if np.unique(doses[retained]).size < 4:
            results.append({'held_dose_gy': float(held_dose), 'status': 'skipped_insufficient_training_doses'})
            continue
        with tempfile.TemporaryDirectory(prefix='fold-', dir=work_folder) as temporary:
            temporary = Path(temporary)
            train = temporary / 'training'; train.mkdir()
            for path, keep in zip(files, retained):
                if keep:
                    for key, companion in scan_paths(path).items():
                        if key != 'lock' and companion.exists():
                            shutil.copy2(companion, train / companion.name)
            try:
                engine = GafchromicEngine(str(temporary / 'calibration'))
                engine.run_calibration(str(train), doses[retained].tolist(), roi_size=roi_size,
                                       n_posterior_samples=posterior_samples)
                blank, _ = engine._load_tiff_with_meta(files[0])
                blank_filtered = median_filter(blank, size=(5, 5, 1))
                od0 = engine._calculate_od_map(median_filter(blank, size=(3, 3, 1)), blank_filtered)
                zero_mean, _ = engine._get_roi_stats(od0, size=roi_size)
                for path in np.asarray(files, dtype=object)[~retained]:
                    raw, _ = engine._load_tiff_with_meta(path)
                    od = engine._calculate_od_map(median_filter(raw, size=(3, 3, 1)), blank_filtered)
                    measured, _ = engine._get_roi_stats(od, size=roi_size)
                    measured = np.maximum(measured - zero_mean, 0)
                    for ch_index, ch in enumerate(CHANNELS):
                        prediction = float(engine.inv_rational_func_dose(measured[ch_index], engine.calib_params[ch]))
                        if not np.isfinite(prediction):
                            raise ValueError('Nonfinite held-out dose prediction.')
                        results.append({'held_dose_gy': float(held_dose), 'filename': path.name,
                            'channel': ch, 'estimated_dose_gy': prediction,
                            'error_gy': prediction - float(held_dose), 'status': 'computed',
                            'extrapolation': bool(held_dose > np.max(doses[retained]))})
            except Exception as error:
                traceback.print_exc(file=sys.stdout)
                results.append({'held_dose_gy': float(held_dose), 'status': 'error', 'error': str(error)})
    return results


def wilson_interval(successes, count, z=1.959963984540054):
    """Binomial interval for independent experiments, not correlated image cells."""
    if count < 1 or not 0 <= successes <= count:
        raise ValueError('Require a positive count and 0 <= successes <= count.')
    rate = successes / count
    denominator = 1 + z*z / count
    centre = (rate + z*z / (2 * count)) / denominator
    half = z * math.sqrt(rate*(1-rate)/count + z*z/(4*count*count)) / denominator
    return [max(0., centre-half), min(1., centre+half)]


def _coverage_study(root, config, progress):
    observations = []
    for repeat in range(config.repetitions):
        if progress and (repeat % 5 == 0 or repeat == config.repetitions - 1):
            progress(f'Independent calibration/measurement experiment {repeat + 1}/{config.repetitions}')
        truth = PROBE_DOSES[repeat % len(PROBE_DOSES)]
        # Each experiment has new calibration films AND a new independent measurement.
        with tempfile.TemporaryDirectory(prefix='coverage-', dir=root) as temporary:
            temporary = Path(temporary)
            rng = _rng(config.seed, 3, repeat)
            try:
                _make_calibration(temporary / 'training', config, rng)
                engine = _fit_calibration(temporary / 'training', temporary / 'calibration', config)
                path = temporary / 'measurement.tif'
                write_synthetic_film(path, np.full((config.image_size, config.image_size), truth), rng, config)
                blank = sorted((temporary / 'training').glob('*_dose_0.tif'))[0]
            except Exception as error:
                traceback.print_exc(file=sys.stdout)
                for method in METHODS:
                    observations.append({'repeat': repeat, 'method': method, 'reference_dose_gy': truth,
                                         'status': 'error', 'error': str(error)})
                continue
            for method in METHODS:
                row = {'repeat': repeat, 'method': method, 'reference_dose_gy': truth, 'status': 'computed'}
                try:
                    dose, sigma = _process(engine, path, blank, method, temporary / method, config.median_filtering)
                    point = (config.image_size // 2, config.image_size // 2)
                    value, uncertainty = float(dose[point]), float(sigma[point])
                    error = value - truth
                    row.update(estimated_dose_gy=value, sigma_gy=uncertainty, error_gy=error,
                               within_1sigma=abs(error) <= uncertainty,
                               within_nominal_95=abs(error) <= 1.959963984540054 * uncertainty)
                except Exception as error:
                    traceback.print_exc(file=sys.stdout)
                    row.update(status='error', error=str(error))
                observations.append(row)
    groups = defaultdict(list)
    for row in observations:
        groups[(row['method'], row['reference_dose_gy'])].append(row)
    summary = []
    for (method, truth), rows in sorted(groups.items()):
        valid = [row for row in rows if row['status'] == 'computed']
        item = {'method': method, 'reference_dose_gy': truth, 'experiments': len(rows),
                'usable_experiments': len(valid), 'failed_experiments': len(rows) - len(valid)}
        if valid:
            successes = sum(row['within_nominal_95'] for row in valid)
            item.update(bias_gy=float(np.mean([row['error_gy'] for row in valid])),
                        empirical_sd_gy=float(np.std([row['estimated_dose_gy'] for row in valid], ddof=1)) if len(valid) > 1 else None,
                        mean_reported_sigma_gy=float(np.mean([row['sigma_gy'] for row in valid])),
                        coverage_1sigma=sum(row['within_1sigma'] for row in valid) / len(valid),
                        coverage_nominal_95=successes / len(valid),
                        coverage_95_wilson_interval=wilson_interval(successes, len(valid)))
        summary.append(item)
    return {'scope': 'Independent synthetic experiments; one central pixel per experiment, grouped by fixed dose.',
            'interval_assumption': 'Gaussian dose +/- 1.959964 sigma is a nominal band; coverage is measured, not assumed.',
            'failures': 'Failed experiments are listed and excluded from the rate; inspect failure counts before interpreting coverage.',
            'summary': summary, 'observations': observations}


def _load_map(path, name):
    array = np.asarray(np.load(path, allow_pickle=False), dtype=float)
    if array.ndim != 2 or array.size == 0 or not np.all(np.isfinite(array)) or np.any(array < 0):
        raise ValueError(f'{name} must be a finite, nonnegative, nonempty 2D NPY map.')
    return array


def evaluate_reference_manifest(manifest_path):
    """Compare supplied processed dose maps against independent reference data.

    A scalar reference is a reference to the ROI mean, not an invented uniform
    reference map. Independence and alignment are user declarations, not verified
    by this software. No clinical/physical pass threshold is invented.
    """
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('schema_version') != 1 or manifest.get('unit') != 'Gy':
        raise ValueError('Reference manifest requires schema_version=1 and unit="Gy".')
    cases = manifest.get('cases')
    if not isinstance(cases, list) or not cases:
        raise ValueError('Provide a nonempty cases list.')
    rows, sources, identifiers = [], [], set()
    def resolve(value):
        return (manifest_path.parent / value).resolve()
    def read(value, name):
        path = resolve(value)
        data = _load_map(path, name)
        sources.append({'path': str(path), 'sha256': _sha256(path)})
        return data
    for case in cases:
        identifier = case.get('id', '')
        if not isinstance(identifier, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,79}', identifier):
            raise ValueError('Reference case IDs must be simple, nonempty identifiers (maximum 80 characters).')
        if identifier in identifiers:
            raise ValueError(f'Duplicate reference case ID: {identifier}')
        identifiers.add(identifier)
        estimate = read(case['dose_map'], 'dose_map')
        mask = np.ones(estimate.shape, dtype=bool)
        if 'mask' in case:
            raw_mask = read(case['mask'], 'mask')
            if raw_mask.shape != estimate.shape or not np.all(np.isin(raw_mask, [0, 1])):
                raise ValueError('The mask must match the dose map and contain only 0 and 1.')
            mask = raw_mask.astype(bool)
        if not np.any(mask):
            raise ValueError(f'Empty reference ROI: {identifier}')
        sigma = read(case['sigma_map'], 'sigma_map') if 'sigma_map' in case else None
        if sigma is not None and sigma.shape != estimate.shape:
            raise ValueError('The sigma map must match the dose map.')
        has_map, has_mean = 'reference_map' in case, 'reference_mean_gy' in case
        if has_map == has_mean:
            raise ValueError('Provide either reference_map or reference_mean_gy, exclusively.')
        row = {'case': identifier, 'status': 'computed', 'acceptance': 'not_assessed',
               'reference_independence': 'declared_not_verified' if case.get('independent_reference') is True else 'not_declared'}
        if has_map:
            if case.get('alignment_confirmed') is not True:
                raise ValueError('Map comparisons require alignment_confirmed=true on a common physical grid.')
            reference = read(case['reference_map'], 'reference_map')
            if reference.shape != estimate.shape:
                raise ValueError('Reference and dose maps must use the same grid.')
            row.update(kind='map', **error_metrics(estimate[mask], reference[mask]))
            if 'reference_sigma_map' in case:
                reference_sigma = read(case['reference_sigma_map'], 'reference_sigma_map')
                if reference_sigma.shape != estimate.shape:
                    raise ValueError('Reference sigma must use the same grid.')
                if sigma is not None and case.get('uncertainties_independent') is True:
                    combined = np.hypot(sigma[mask], reference_sigma[mask])
                    row['spatial_fraction_inside_nominal_95_band'] = float(np.mean(
                        np.abs(estimate[mask] - reference[mask]) <= 1.959963984540054 * combined))
                    row['band_note'] = 'Spatial inclusion fraction, not independent-trial coverage; independent uncertainty sources assumed.'
        else:
            reference = float(case['reference_mean_gy'])
            if not np.isfinite(reference) or reference < 0:
                raise ValueError('Reference mean dose must be finite and nonnegative.')
            mean = float(np.mean(estimate[mask]))
            row.update(kind='roi_mean', estimated_mean_gy=mean, reference_mean_gy=reference,
                       mean_difference_gy=mean-reference, cell_count=int(mask.sum()))
            if sigma is not None:
                row['estimated_mean_sigma_upper_bound_gy'] = float(np.mean(sigma[mask]))
            if 'reference_sigma_gy' in case:
                reference_sigma = float(case['reference_sigma_gy'])
                if not np.isfinite(reference_sigma) or reference_sigma < 0:
                    raise ValueError('Reference sigma must be finite and nonnegative.')
                row['reference_sigma_gy'] = reference_sigma
                if sigma is not None and case.get('uncertainties_independent') is True:
                    row['combined_mean_sigma_upper_bound_gy'] = float(np.hypot(np.mean(sigma[mask]), reference_sigma))
        rows.append(row)
    return {'status': 'computed', 'acceptance': 'not_assessed',
            'manifest': str(manifest_path), 'manifest_sha256': _sha256(manifest_path),
            'notes': 'Reference origin, independence, alignment and acceptance criteria require domain review.',
            'cases': rows, 'sources': sources}


def mcmc_diagnostics(path):
    """Rank R-hat, bulk/tail ESS and divergences, without automatic convergence approval."""
    path = Path(path).resolve()
    result = {'path': str(path), 'convergence': 'not_established_by_software'}
    try:
        result['sha256'] = _sha256(path)
        import arviz as az
        trace = az.from_netcdf(path)
        try:
            posterior = trace.posterior
            chains, draws = posterior.sizes.get('chain', 0), posterior.sizes.get('draw', 0)
            result.update(chains=int(chains), draws_per_chain=int(draws))
            if hasattr(trace, 'sample_stats') and 'diverging' in trace.sample_stats:
                result['divergences'] = int(np.asarray(trace.sample_stats['diverging']).sum())
            else:
                result['divergences'] = None
            if chains < 2 or draws < 4:
                result.update(status='insufficient_chains_or_draws',
                              note='At least two chains and four retained draws per chain are needed to compute rank R-hat; this minimum does not establish convergence.')
                return result
            rhat = az.rhat(trace, method='rank')
            bulk = az.ess(trace, method='bulk')
            tail = az.ess(trace, method='tail')
            parameters = {}
            undefined = 0
            for name in posterior.data_vars:
                parameters[name] = {}
                for label, data, operation in (('rhat_max', rhat, np.max),
                                               ('ess_bulk_min', bulk, np.min), ('ess_tail_min', tail, np.min)):
                    values = np.asarray(data[name], dtype=float)
                    if np.all(np.isfinite(values)):
                        parameters[name][label] = float(operation(values))
                    else:
                        parameters[name][label] = None
                        undefined += int(np.count_nonzero(~np.isfinite(values)))
            result.update(status='computed' if not undefined else 'computed_with_undefined_values',
                          parameters=parameters, undefined_diagnostic_elements=undefined,
                          note='Inspect chain mixing, ESS, R-hat and divergences together. No acceptance thresholds were supplied.')
        finally:
            trace.close()
    except Exception as error:
        result.update(status='error', error=str(error))
    return result


def _csv(path, rows):
    if not rows:
        return
    names = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, names)
        writer.writeheader()
        for row in rows:
            safe = {}
            for key, value in row.items():
                if isinstance(value, (list, dict)):
                    value = json.dumps(value, allow_nan=False)
                # User-supplied diagnostic error text must remain text in spreadsheet viewers.
                if isinstance(value, str) and value.startswith(('=', '+', '-', '@')):
                    value = "'" + value
                safe[key] = value
            writer.writerow(safe)


def _table(rows):
    if not rows:
        return '<p>Nincs adat ebben a szakaszban.</p>'
    names = list(dict.fromkeys(key for row in rows for key in row))
    def cell(value):
        if isinstance(value, float):
            return f'{value:.6g}'
        if isinstance(value, (list, dict)):
            return json.dumps(value, ensure_ascii=False, allow_nan=False)
        return '' if value is None else str(value)
    return '<div class="table"><table><thead><tr>' + ''.join(
        f'<th>{html.escape(name)}</th>' for name in names) + '</tr></thead><tbody>' + ''.join(
        '<tr>' + ''.join(f'<td>{html.escape(cell(row.get(name)))}</td>' for name in names) + '</tr>'
        for row in rows) + '</tbody></table></div>'


def _figures(root, report):
    from matplotlib.figure import Figure
    paths = []
    rows = [row for row in report.get('solver_comparison', []) if row['status'] == 'computed']
    if rows:
        cases = list(dict.fromkeys(row['case'] for row in rows))
        figure = Figure(figsize=(12, 8), layout='constrained')
        axes = figure.subplots(1, 2)
        for method_index, method in enumerate(METHODS):
            lookup = {row['case']: row for row in rows if row['method'] == method}
            y = np.arange(len(cases)) + (method_index-1)*.24
            for ax, metric in zip(axes, ('bias_gy', 'rmse_gy')):
                ax.barh(y, [lookup.get(case, {}).get(metric, np.nan) for case in cases],
                        height=.22, label=method)
                ax.set_yticks(np.arange(len(cases)), cases)
                ax.invert_yaxis() if method_index == 0 else None
                ax.set_xlabel('Gy'); ax.set_title(metric.replace('_gy', '').upper())
                ax.axvline(0, color='black', linewidth=.6)
        axes[0].legend()
        figure.suptitle('Synthetic model recovery / stress tests — not physical validation')
        path = root / 'solver_comparison.png'; figure.savefig(path, dpi=130); paths.append(path)
    coverage = report.get('coverage', {}).get('summary', [])
    if coverage:
        figure = Figure(figsize=(9, 5), layout='constrained'); ax = figure.subplots()
        for index, method in enumerate(METHODS):
            selected = [row for row in coverage if row['method'] == method and row['usable_experiments']]
            if not selected:
                continue
            x = np.array([row['reference_dose_gy'] for row in selected]) + (index-1)*.2
            rates = np.array([row['coverage_nominal_95'] for row in selected])
            intervals = np.array([row['coverage_95_wilson_interval'] for row in selected])
            ax.errorbar(x, rates*100, yerr=np.maximum(0, np.vstack([rates-intervals[:, 0], intervals[:, 1]-rates]))*100,
                        fmt='o', capsize=4, label=method)
        ax.axhline(95, color='grey', linestyle='--', label='Nominal Gaussian band target')
        ax.set_ylim(-2, 102); ax.set_xlabel('Reference dose [Gy]'); ax.set_ylabel('Observed inclusion [%]')
        ax.set_title('Independent experiments; Wilson intervals for inclusion rates')
        ax.legend()
        path = root / 'coverage.png'; figure.savefig(path, dpi=130); paths.append(path)
    return paths


def _write_report(root, report):
    _csv(root / 'solver_metrics.csv', report.get('solver_comparison', []))
    _csv(root / 'cross_validation.csv', report.get('cross_validation', []))
    _csv(root / 'coverage.csv', report.get('coverage', {}).get('summary', []))
    _csv(root / 'coverage_observations.csv', report.get('coverage', {}).get('observations', []))
    _csv(root / 'reference_metrics.csv', report.get('references', {}).get('cases', []))
    _csv(root / 'reference_cross_validation.csv', report.get('reference_cross_validation', {}).get('results', []))
    pictures = _figures(root, report)
    image_markup = ''.join('<img alt="Numerical validation plot" src="data:image/png;base64,'
                           + base64.b64encode(path.read_bytes()).decode('ascii') + '">' for path in pictures)
    sections = [
        ('Solverek — szintetikus modell és érzékenységi esetek', _table(report.get('solver_comparison', []))),
        ('Kihagyott dóziscsoportok — standard kalibráció', _table(report.get('cross_validation', []))),
        ('Bizonytalansági sávok megfigyelt lefedettsége', _table(report.get('coverage', {}).get('summary', []))),
        ('Független referenciák', '<p>' + html.escape(report['references']['status']) + '</p>'
         + _table(report['references'].get('cases', []))),
        ('Megadott kalibráció keresztvalidációja',
         _table(report.get('reference_cross_validation', {}).get('results', []))),
        ('MCMC-diagnosztika', _table(report['mcmc_diagnostics'])),
    ]
    document = '''<!doctype html><html lang="hu"><meta charset="utf-8">
<title>PRISM numerikus validáció</title><style>
body{font:16px/1.55 system-ui,sans-serif;color:#172b3a;max-width:1400px;margin:36px auto;padding:0 24px}
h1,h2{line-height:1.2}.scope{background:#eef4f7;padding:18px;border-left:4px solid #356981}
.table{overflow:auto}table{border-collapse:collapse;font-size:13px;margin:20px 0;width:100%}
th,td{padding:9px;border:1px solid #d7e1e7;text-align:left;vertical-align:top}th{background:#eef4f7}
img{max-width:100%;height:auto}pre{white-space:pre-wrap;background:#f4f6f8;padding:15px}
</style><h1>PRISM — numerikus validáció</h1>'''
    document += '<p>' + html.escape(report['created_at']) + ' | ' + html.escape(report['status']) + '</p>'
    document += '<div class="scope"><strong>A fizikai megfelelőség nincs igazolva.</strong> '
    document += html.escape(report['physical_validation']['reason']) + '</div>'
    document += '<h2>Értelmezés és feltételezések</h2><ul>' + ''.join(
        '<li>' + html.escape(note) + '</li>' for note in report['limitations']) + '</ul>'
    document += '<h2>Futtatási beállítások</h2><pre>' + html.escape(json.dumps(report['configuration'], indent=2)) + '</pre>'
    document += image_markup
    document += ''.join('<h2>' + html.escape(title) + '</h2>' + content for title, content in sections)
    if report['errors']:
        document += '<h2>Végrehajtási hibák</h2><pre>' + html.escape(json.dumps(report['errors'], ensure_ascii=False, indent=2)) + '</pre>'
    document += '<h2>Reprodukálhatóság</h2><pre>' + html.escape(json.dumps(report['environment'], indent=2)) + '</pre>'
    document += '<p>A részletes gépi adatok: report.json; motor-kimenet: engine.log. '
    document += 'A JSON tartalmazza a bemenetek és a számítási modulok SHA-256 lenyomatait.</p></html>'
    (root / 'report.html').write_text(document, encoding='utf-8')
    (root / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def run_validation(output, config=None, reference_manifest=None, traces=(), synthetic=True, progress=None,
                   calibration_folder=None, calibration_doses=None, calibration_roi_size=50):
    """Create a new, self-contained report directory. Never overwrite an existing run."""
    config = config or ValidationConfig()
    config.validate()
    if not isinstance(calibration_roi_size, int) or isinstance(calibration_roi_size, bool) or calibration_roi_size < 2:
        raise ValueError('calibration_roi_size must be an integer >= 2.')
    if (calibration_folder is None) != (calibration_doses is None):
        raise ValueError('Provide both calibration_folder and calibration_doses.')
    if not synthetic and reference_manifest is None and not traces and calibration_folder is None:
        raise ValueError('Provide reference data or MCMC traces when skipping synthetic studies.')
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root / '.incomplete').write_text('Validation run not completed.\n', encoding='utf-8')
    packages = {}
    for package in ('numpy', 'scipy', 'matplotlib', 'tifffile', 'scikit-image', 'scikit-learn', 'pymc', 'arviz'):
        try:
            packages[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            packages[package] = None
    report = {
        'schema_version': 1, 'created_at': datetime.now(timezone.utc).isoformat(),
        'status': 'running', 'acceptance': 'not_assessed_no_thresholds_provided',
        'configuration': {**asdict(config), 'synthetic': synthetic, 'calibration_doses_gy': list(CALIBRATION_DOSES),
                          'probe_doses_gy': list(PROBE_DOSES), 'forward_parameters': FORWARD_PARAMETERS,
                          'reference_calibration_roi_size': calibration_roi_size, 'scanner_dpi': 300, 'blank_intensities': [60000, 56000, 52000],
                          'lra_coefficients': LRA_COEFFICIENTS.tolist()},
        'environment': {'python': platform.python_version(), 'platform': platform.platform(), 'packages': packages},
        'source_hashes': {name: _sha256(Path(__file__).parent / name) for name in SOURCE_MODULES},
        'physical_validation': {'status': 'not_established',
            'reason': 'A szimuláció nem igazolja a szkenner vagy a fizikai modell pontosságát. '
                      'A referenciák eredete, függetlensége és az elfogadási határok szakmai értékelést igényelnek.'},
        'limitations': [
            'A vizsgálat a standard kalibrációt illeszti. MCMC-konvergenciát csak külön megadott NetCDF-láncokból számol; automatikusan nem futtat Bayes-kalibrációt.',
            'A paraméterminták száma kovarianciaalapú hibaterjesztési beállítás; nem MCMC-lánchossz. A termelési kalibrátor rögzített 42-es magot használ e mintákhoz.',
            'A konverzió mediánszűrése a median_filtering beállítást követi. A kalibráció saját ROI-előszűrése és az ADMM belső regularizációja ettől független.',
            'A generátor a kalibrációval azonos racionális modellcsaládot használ. Ez modell-visszanyerési vizsgálat, nem független fizikai bizonyíték.',
            'A matched LRA esetek az előírt kvadratikus korrekcióval konzisztensek; a lra_mismatch eset szándékosan eltér tőle.',
            'A zaj független normális OD-zaj; a szkenner driftje, valódi térbeli korrelációja és anyagfüggése nincs igazolva.',
            'Minden Monte Carlo-ismétlésben új kalibráció és mérés készül. Dózisonként egy középső képpont számít kísérletnek; a képpontok száma nem növeli a független mintaszámot.',
            'A lefedettség a névleges Gauss-sáv tényleges befogadási aránya a szimulációban. A Wilson-intervallum ezt az arányt jellemzi; kevés ismétlésnél széles.',
            'A hibás kísérletek külön látszanak, a lefedettségi arányból kimaradnak. Hibák esetén az arány önmagában félrevezető lehet.',
            'A csúcsvesztés és a modellhibás/extrapolációs esetek eredményeit nem rejti el automatikus megfelelt minősítés.',
            'A keresztvalidáció teljes nemnulla dóziscsoportokat hagy ki, és a termelési standard kalibrációt illeszti újra; a blank megmarad referenciának.',
            'A hőmérséklet, az energiafüggés, a dózisteljesítmény, a FRONT/BACK geometria és a szóródási modell fizikai validálása nincs ebben a szintetikus vizsgálatban.',
        ],
        'solver_comparison': [], 'cross_validation': [], 'coverage': {'status': 'not_run'},
        'references': {'status': 'not_provided'}, 'mcmc_diagnostics': [], 'errors': [],
        'reference_cross_validation': {'status': 'not_provided'},
    }
    start = time.perf_counter()
    with (root / 'engine.log').open('w', encoding='utf-8') as log, redirect_stdout(log), warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        if synthetic:
            try:
                if progress: progress('Synthetic TIFF calibration and solver comparison')
                report['calibration_inputs'] = _make_calibration(root / 'calibration_films', config, _rng(config.seed, 1))
                engine = _fit_calibration(root / 'calibration_films', root / 'calibration', config)
                blank = sorted((root / 'calibration_films').glob('*_dose_0.tif'))[0]
                fixtures = root / 'phantoms'; fixtures.mkdir()
                report['solver_comparison'], report['synthetic_inputs'] = _compare_solvers(fixtures, config, engine, blank)
                if progress: progress('Leave-one-dose-group-out cross-validation')
                report['cross_validation'] = leave_one_dose_out(root / 'calibration_films', CALIBRATION_DOSES,
                    root / 'cross_validation_work', roi_size=max(2, config.image_size//4), posterior_samples=config.posterior_samples)
                report['coverage'] = _coverage_study(root, config, progress)
            except Exception as error:
                traceback.print_exc(file=sys.stdout)
                report['errors'].append({'stage': 'synthetic', 'error': str(error)})
        if reference_manifest is not None:
            try:
                report['references'] = evaluate_reference_manifest(reference_manifest)
            except Exception as error:
                report['references'] = {'status': 'error', 'error': str(error)}
        if calibration_folder is not None:
            try:
                files, doses = GafchromicEngine._calibration_inputs(calibration_folder, calibration_doses)
                inputs = [{'path': str(path.resolve()), 'sha256': _sha256(path),
                    'companions': {key: {'path': str(companion.resolve()), 'sha256': _sha256(companion)}
                        for key, companion in scan_paths(path).items()
                        if key not in ('lock', 'mean') and companion.exists()}} for path in files]
                report['reference_cross_validation'] = {
                    'status': 'computed', 'nominal_doses_gy': doses.tolist(), 'inputs': inputs,
                    'note': 'User-supplied nominal doses are treated as fixed; reference dose uncertainty is not propagated.',
                    'results': leave_one_dose_out(calibration_folder, calibration_doses,
                        root / 'reference_cv_work', roi_size=calibration_roi_size,
                        posterior_samples=config.posterior_samples)}
            except Exception as error:
                report['reference_cross_validation'] = {'status': 'error', 'error': str(error)}
        for path in traces:
            report['mcmc_diagnostics'].append(mcmc_diagnostics(path))
        report['warnings'] = list(dict.fromkeys(str(item.message) for item in captured))
    for stage, rows in (('solver_comparison', report['solver_comparison']),
                        ('cross_validation', report['cross_validation']),
                        ('coverage', report['coverage'].get('observations', [])),
                        ('references', [report['references']]), ('mcmc', report['mcmc_diagnostics'])):
        report['errors'].extend({'stage': stage, **row} for row in rows if row.get('status') == 'error')
    real_cv = report['reference_cross_validation']
    report['errors'].extend({'stage': 'reference_cross_validation', **row}
                           for row in [real_cv, *real_cv.get('results', [])] if row.get('status') == 'error')
    report['status'] = 'completed_with_errors' if report['errors'] else 'completed'
    report['elapsed_seconds'] = time.perf_counter() - start
    _write_report(root, report)
    (root / '.incomplete').unlink()
    return report
