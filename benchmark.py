"""Isolated, repeatable processing benchmarks; timing never gates CI."""
import argparse
import contextlib
import csv
import io
import json
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time
import tracemalloc


def measure_case(folder, size, method, samples, repeats, profile=False):
    import numpy as np
    from dosimetry import GafchromicEngine
    from project_state import atomic_json, runtime_record
    from validation import ValidationConfig, FORWARD_PARAMETERS, write_synthetic_film
    root = Path(folder)
    rng = np.random.default_rng(42)
    y, x = np.indices((size, size))
    truth = 2 + 8*np.exp(-((x-size/2)**2+(y-size/2)**2)/(2*(size/5)**2))
    with tempfile.TemporaryDirectory(dir=root) as temporary, contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        temporary = Path(temporary)
        config = ValidationConfig(image_size=max(12, size), posterior_samples=samples)
        film = temporary/'film.tif'; blank = temporary/'blank.tif'
        write_synthetic_film(film, truth, rng, config)
        write_synthetic_film(blank, np.zeros_like(truth), rng, config)
        engine = GafchromicEngine(str(temporary/'processed'))
        engine.calib_params = {ch: np.array(p) for ch, p in FORWARD_PARAMETERS.items()}
        engine.calib_sigma_od = {ch: .003 for ch in 'RGB'}
        for ch in 'RGB':
            draws = np.tile(engine.calib_params[ch], (samples, 1))
            draws[:, 1] *= 1 + rng.normal(0, .001, samples)
            engine.calib_samples[ch] = draws
        engine._prepare_ratio_lut()
        def process(): engine.process_films([str(film)], str(blank), method=method)
        process()  # warmup, imports and first allocations excluded
        times = []; peaks = []
        if profile:
            import cProfile
            profiler = cProfile.Profile(); profiler.enable()
        for _ in range(repeats):
            tracemalloc.start(); start = time.perf_counter()
            process()
            times.append(time.perf_counter()-start)
            peaks.append(tracemalloc.get_traced_memory()[1]); tracemalloc.stop()
        if profile:
            profiler.disable(); profiler.dump_stats(str(root/'profile.pstats'))
        dose = np.load(temporary/'processed'/'film_dose.npy')
        sigma = np.load(temporary/'processed'/'film_std.npy')
        np.savez_compressed(root/'maps.npz', dose=dose, sigma=sigma)
    row = {'size': size, 'method': method, 'parameter_samples': samples, 'repeats': repeats,
           'median_seconds': statistics.median(times), 'times_seconds': times,
           'peak_traced_bytes': max(peaks), 'peak_traced_bytes_per_repeat': peaks,
           'profiled': profile, 'rmse_gy': float(np.sqrt(np.mean((dose-truth)**2))), 'mean_sigma_gy': float(sigma.mean())}
    atomic_json(root/'case.json', row)
    atomic_json(root/'environment.json', runtime_record())


def compare_kernels(output, reference_source, sizes, samples, repeats):
    """Paired old/new numerical work, without TIFF IO or provenance overhead."""
    import hashlib
    import importlib.util
    import shutil
    import numpy as np
    from scipy.ndimage import median_filter, maximum_filter
    from dosimetry import GafchromicEngine
    from project_state import atomic_json, runtime_record
    from validation import FORWARD_PARAMETERS
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(reference_source, output/'reference_engine.py')
    spec = importlib.util.spec_from_file_location('prism_reference_engine', reference_source)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    rows = []
    with tempfile.TemporaryDirectory(dir=output) as temporary:
        for size in sizes:
            y, x = np.indices((size, size))
            truth = 2+8*np.exp(-((x-size/2)**2+(y-size/2)**2)/(2*(size/5)**2))
            od = np.stack([p[1]*truth/(truth+p[2]) for p in FORWARD_PARAMETERS.values()], axis=-1)
            noise = np.full_like(od, .003)
            for method in ('fast', 'mayer', 'admm'):
                print(f'Paired kernels {size}-{method}', flush=True)
                engines = [module.GafchromicEngine(temporary), GafchromicEngine(temporary)]
                for engine in engines:
                    rng = np.random.default_rng(42)
                    engine.calib_params = {ch: np.array(values) for ch, values in FORWARD_PARAMETERS.items()}
                    engine.calib_sigma_od = {ch: .003 for ch in 'RGB'}
                    engine.calib_samples = {}
                    for ch in 'RGB':
                        draws = np.tile(engine.calib_params[ch], (samples, 1))
                        draws[:, 1] *= 1+rng.normal(0, .001, samples)
                        engine.calib_samples[ch] = draws
                def compute(engine):
                    dose, fit = engine._solve_dose(od, method, noise)
                    calibration = engine._compute_calibration_std_map(od, dose, method, noise)
                    dose = median_filter(dose, size=3)
                    sigma = np.hypot(maximum_filter(fit, size=3), calibration)
                    return dose, np.maximum(sigma, np.where(dose > .01, dose*.005, 1e-4))
                for engine in engines: compute(engine)
                timings = [[], []]; cpu = [[], []]; results = [None, None]
                for repeat in range(repeats):
                    for index in ((0, 1) if repeat%2 == 0 else (1, 0)):
                        start = time.perf_counter(); start_cpu = time.process_time()
                        results[index] = compute(engines[index])
                        cpu[index].append(time.process_time()-start_cpu)
                        timings[index].append(time.perf_counter()-start)
                peaks = []
                for engine in engines:
                    tracemalloc.start(); compute(engine)
                    peaks.append(tracemalloc.get_traced_memory()[1]); tracemalloc.stop()
                rows.append({'size': size, 'method': method, 'samples': samples, 'repeats': repeats,
                    'reference_seconds': statistics.median(timings[0]), 'current_seconds': statistics.median(timings[1]),
                    'reference_cpu_seconds': statistics.median(cpu[0]), 'current_cpu_seconds': statistics.median(cpu[1]),
                    'speed_ratio_reference_over_current': statistics.median(timings[0])/statistics.median(timings[1]),
                    'reference_peak_traced_bytes': peaks[0], 'current_peak_traced_bytes': peaks[1],
                    'dose_max_abs_difference': float(np.max(np.abs(results[0][0]-results[1][0]))),
                    'sigma_max_abs_difference': float(np.max(np.abs(results[0][1]-results[1][1])))})
    atomic_json(output/'report.json', {'schema_version': 1, 'cases': rows, 'environment': runtime_record(),
        'reference_sha256': hashlib.sha256(reference_source.read_bytes()).hexdigest(),
        'scope': 'Paired numerical kernels; same process/data, alternating order, warmup excluded. Memory measured separately with tracemalloc; timing excludes tracing, TIFF IO and run provenance.'})
    with (output/'summary.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, list(rows[0])); writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sizes', default='64,128,256')
    parser.add_argument('--samples', type=int, default=16)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--profile', action='store_true', help='Save cProfile call statistics; adds timing overhead.')
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--reference-engine', type=Path, help='Explicit local reference engine source for paired kernel comparison.')
    parser.add_argument('--worker-method', choices=('fast', 'mayer', 'admm'), help=argparse.SUPPRESS)
    args = parser.parse_args()
    sizes = [int(value) for value in args.sizes.split(',')]
    if not sizes or min(sizes) < 12 or args.samples < 2 or args.repeats < 1:
        parser.error('Require sizes >=12, samples >=2, repeats >=1.')
    if args.reference_engine:
        compare_kernels(args.output, args.reference_engine, sizes, args.samples, args.repeats)
        return
    if args.worker_method:
        measure_case(args.output, sizes[0], args.worker_method, args.samples, args.repeats, args.profile)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for size in sizes:
        for method in ('fast', 'mayer', 'admm'):
            key = f'{size}-{method}'; folder = args.output/key; folder.mkdir()
            print(f'Benchmark {key}', flush=True)
            process = subprocess.run([sys.executable, '-m', 'benchmark', '--output', str(folder.resolve()),
                '--worker-method', method, '--sizes', str(size), '--samples', str(args.samples),
                '--repeats', str(args.repeats), *(['--profile'] if args.profile else [])], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            (folder/'engine.log').write_text(process.stdout, encoding='utf-8')
            if process.returncode:
                raise RuntimeError(f'Benchmark failed: see {folder / "engine.log"}')
            row = json.loads((folder/'case.json').read_text())
            if args.baseline:
                import numpy as np
                old = json.loads((args.baseline/key/'case.json').read_text())
                row['speed_ratio_baseline_over_current'] = old['median_seconds']/row['median_seconds']
                row['peak_traced_ratio_current_over_baseline'] = row['peak_traced_bytes']/old['peak_traced_bytes']
                with np.load(args.baseline/key/'maps.npz') as reference, np.load(folder/'maps.npz') as current:
                    for field in ('dose', 'sigma'):
                        row[field+'_max_abs_difference'] = float(np.max(np.abs(reference[field]-current[field])))
            rows.append(row)
    from project_state import atomic_json
    atomic_json(args.output/'report.json', {'schema_version': 1, 'cases': rows,
        'memory_measure': 'tracemalloc peak allocations during processing, including NumPy tracked allocations; not process RSS.',
        'timing_measure': 'Warm processing, TIFF IO and provenance included; fresh child interpreter per case. Tracemalloc enabled during timings.',
        'limits': 'Same seed and environment required for comparisons. Small synthetic maps do not predict full scanner performance.'})
    with (args.output/'summary.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


if __name__ == '__main__': main()
