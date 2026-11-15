"""Acquisition statistics, sidecars and exclusive publication of scan files."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import warnings
from functools import lru_cache

import numpy as np
import tifffile


AUXILIARY_SCAN_SUFFIXES = ('_scan_std', '_scan_sem')


def scan_paths(path):
    path = Path(path)
    return {
        'mean': path,
        'std': path.with_name(path.stem + '_scan_std' + path.suffix),
        'sem': path.with_name(path.stem + '_scan_sem' + path.suffix),
        'meta': path.with_name(path.stem + '_scan_meta.json'),
        'lock': path.with_name('.' + path.name + '.scan.lock'),
    }


def is_raw_film(path):
    path = Path(path)
    return (path.suffix.lower() in ('.tif', '.tiff')
            and not path.stem.lower().endswith(AUXILIARY_SCAN_SUFFIXES))


def _exists(path):
    return path.exists() or path.is_symlink()


def scan_target_available(path):
    return not any(_exists(p) for p in scan_paths(path).values())


def next_scan_path(directory, prefix, start, max_index=9999):
    """Find an unused set of scan/sidecar paths; publication also checks races."""
    if not prefix or any(c in prefix for c in '<>:"/\\|?*') or prefix in ('.', '..'):
        raise ValueError('Use a filename prefix without path separators or reserved characters.')
    if not 1 <= start <= max_index:
        raise ValueError('The starting index is outside the supported range.')
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError('The target directory does not exist.')
    for index in range(start, max_index + 1):
        candidate = directory / f'{prefix}_{index:03d}.tif'
        if scan_target_available(candidate):
            return candidate, index
    raise ValueError('No unused scan index remains. Choose another prefix or directory.')


@contextmanager
def reserve_scan(path):
    """Reserve a new filename without replacing any existing scan or sidecar."""
    if not is_raw_film(path):
        raise ValueError('The scan filename must be an RGB TIFF, not a noise sidecar.')
    paths = scan_paths(path)
    # An existing lock is never removed by this acquisition.
    lock = paths['lock'].open('x', encoding='utf-8')
    identity = os.fstat(lock.fileno())
    try:
        with lock:
            lock.write(f'pid={os.getpid()}\n')
        if any(_exists(p) for key, p in paths.items() if key != 'lock'):
            raise FileExistsError(f'Scan or sidecar already exists: {path}')
        yield paths
    finally:
        try:
            current = paths['lock'].lstat()
            if (current.st_dev, current.st_ino) == (identity.st_dev, identity.st_ino):
                paths['lock'].unlink()
        except FileNotFoundError:
            pass


def publish_scan_files(staged_files, check_cancel=lambda: None):
    """Exclusively create outputs; roll back only files created by this call.

    The caller holds reserve_scan until publication completes. Readers skip
    locked scans. An x-mode open prevents overwrites even by a racing writer.
    """
    owned = []
    try:
        for source, target in staged_files:
            check_cancel()
            target = Path(target)
            with target.open('xb') as output:
                identity = os.fstat(output.fileno())
                owned.append((target, identity.st_dev, identity.st_ino))
                with Path(source).open('rb') as incoming:
                    while chunk := incoming.read(1024 * 1024):
                        check_cancel()
                        output.write(chunk)
                output.flush()
                os.fsync(output.fileno())
        check_cancel()
    except BaseException:
        for target, device, inode in reversed(owned):
            try:
                current = target.lstat()
                if (current.st_dev, current.st_ino) == (device, inode):
                    target.unlink()
            except FileNotFoundError:
                pass
        raise


class ScanAccumulator:
    """Online mean and repeated-scan sample variance with bounded image storage."""
    def __init__(self):
        self.count = 0
        self.mean = None
        self.m2 = None

    def add(self, image):
        image = np.asarray(image)
        if image.dtype != np.uint16 or image.ndim != 3 or image.shape[2] != 3 or 0 in image.shape:
            raise ValueError('The scanner must return a nonempty 16-bit RGB image.')
        if self.mean is not None and image.shape != self.mean.shape:
            raise ValueError('Repeated scans must have identical image dimensions.')
        image = image.astype(np.float64)
        if self.count == 0:
            self.mean = image
            self.m2 = np.zeros(image.shape)
            self.count = 1
            return
        self.count += 1
        delta = image - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (image - self.mean)

    def finish(self):
        if self.count == 0:
            raise ValueError('No scan was acquired.')
        mean = np.rint(np.clip(self.mean, 0, 65535)).astype(np.uint16)
        if self.count == 1:
            return mean, None, None
        std = np.sqrt(np.maximum(self.m2 / (self.count - 1), 0))
        return mean, std.astype(np.float32), (std / np.sqrt(self.count)).astype(np.float32)


def ensure_scan_ready(path):
    if scan_paths(path)['lock'].exists():
        raise ValueError(f'Acquisition is still in progress or has a stale lock: {path}')


def load_scan_noise(path, expected_shape):
    """Return noise of the averaged image and its provenance, or None if unknown.

    New SEM is preferred. Known repeat count permits SD/sqrt(N). Legacy SD
    without a count is retained conservatively, never silently divided.
    """
    ensure_scan_ready(path)
    paths = scan_paths(path)
    meta = None
    if paths['meta'].exists():
        with paths['meta'].open(encoding='utf-8') as f:
            meta = json.load(f)
        count = meta.get('scan_count')
        if not isinstance(count, int) or isinstance(count, bool) or count < 1:
            raise ValueError(f'Invalid scan count in {paths["meta"]}')
        if count == 1:
            if paths['std'].exists() or paths['sem'].exists():
                raise ValueError('A single-scan acquisition must not have repeated-scan noise files.')
            return None, 'unavailable_single_scan'
    if paths['sem'].exists():
        source_path, source, divisor = paths['sem'], 'mean_sem', 1.0
    elif paths['std'].exists():
        source_path = paths['std']
        if meta is not None:
            source, divisor = 'repeat_std_to_sem', np.sqrt(meta['scan_count'])
        else:
            source, divisor = 'legacy_repeat_std_count_unknown', 1.0

    elif meta is not None:
        raise ValueError(f'Missing noise files for repeated acquisition: {path}')
    else:
        return None, 'unavailable'
    noise = tifffile.imread(source_path).astype(np.float64) / divisor
    if noise.shape != tuple(expected_shape) or not np.all(np.isfinite(noise)) or np.any(noise < 0):
        raise ValueError(f'Invalid scan noise shape or values: {source_path}')
    if source == 'legacy_repeat_std_count_unknown':
        stat = source_path.stat()
        _warn_legacy_noise(str(Path(path).resolve()), stat.st_mtime_ns, stat.st_size)
    return noise, source


class LegacyScanNoiseWarning(UserWarning):
    """An old SD sidecar cannot identify the noise of an averaged scan."""


@lru_cache(maxsize=4096)
def _warn_legacy_noise(path, modified_ns, size):
    # Explicit deduplication survives warnings.catch_warnings in sampler libraries.
    # Only the notice is cached; noise arrays and metadata are always read afresh.
    warnings.warn(f'Repetition count unavailable for {path}; using legacy repeat SD without sqrt(N) reduction. '
                  'A verified scan_count in the scan metadata resolves this warning.',
                  LegacyScanNoiseWarning, stacklevel=3)
