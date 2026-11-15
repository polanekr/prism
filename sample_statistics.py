"""Volume-weighted sample statistics, independent of the Qt interface."""

import numpy as np


def axisymmetric_volume_weights(mask):
    """Approximate chord lengths of the solid of revolution of a tube mask.

    Each occupied raster cell is sampled at its centre. The same rasterized
    contour therefore defines both the sample boundary and its volume weights.
    Lengths are in pixels; a common physical scale cancels in relative volumes.
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError('A 2D tube mask is required.')
    weights = np.zeros(mask.shape, dtype=float)
    for row, occupied in enumerate(mask):
        columns = np.flatnonzero(occupied)
        if columns.size == 0:
            continue
        centre = (columns[0] + columns[-1]) / 2.0
        radius = (columns[-1] - columns[0] + 1) / 2.0
        weights[row, columns] = 2 * np.sqrt(np.maximum(
            radius**2 - (columns - centre)**2, 0.0))
    return weights


def calculate_sample_statistics(doses, sigmas, weights, bins=50):
    """Summarize valid sample cells, including genuinely zero-dose cells.

    Spatial spread and mean pixel sigma describe different quantities. The
    latter is not the uncertainty of the sample mean: that requires a model
    of covariance between cells. Relative quantities are undefined at zero
    mean dose. Empty histogram bins have no mean pixel sigma (NaN).
    """
    doses, sigmas, weights = (np.asarray(a, dtype=float) for a in (doses, sigmas, weights))
    if (doses.ndim != 1 or doses.shape != sigmas.shape or doses.shape != weights.shape
            or doses.size == 0):
        raise ValueError('Dose, sigma and weight must be matching, nonempty 1D arrays.')
    if any(not np.all(np.isfinite(a)) or np.any(a < 0) for a in (doses, sigmas, weights)):
        raise ValueError('Dose, sigma and weight must be finite and nonnegative.')
    if not isinstance(bins, (int, np.integer)) or bins < 1:
        raise ValueError('The histogram bin count must be a positive integer.')
    positive = weights > 0
    doses, sigmas, weights = doses[positive], sigmas[positive], weights[positive]
    total_weight = np.sum(weights)
    if not np.isfinite(total_weight) or total_weight <= 0:
        raise ValueError('The sample must have a finite, positive total volume weight.')
    relative_weights = weights / total_weight
    mean = float(np.sum(doses * relative_weights))
    spread = float(np.sqrt(np.sum(relative_weights * (doses - mean)**2)))
    mean_sigma = float(np.sum(sigmas * relative_weights))
    maximum = float(np.max(doses))
    minimum = float(np.min(doses))
    padding = max(.01, maximum*.01) if maximum-minimum <= np.spacing(maximum)*bins*2 else 0.
    limits = (max(0., minimum-padding), maximum+padding)
    volume, edges = np.histogram(doses, bins=bins, range=limits,
                                weights=relative_weights)
    sigma_sums, _ = np.histogram(doses, bins=edges, weights=relative_weights * sigmas)
    bin_sigmas = np.divide(sigma_sums, volume, out=np.full(volume.shape, np.nan), where=volume > 0)
    centres = (edges[:-1] + edges[1:]) / 2
    return {
        'hist_x': centres,
        'hist_edges': edges,
        'hist_y': volume * 100.0,
        'dose_stats': np.vstack([centres, bin_sigmas]),
        'weights': weights,
        'raw_voxels': doses,
        'raw_sigmas': sigmas,
        'mean_dose': mean,
        'spread_dose': spread,
        # Keep the existing export key, with an explicit definition in the UI/export.
        'uncert_dose': mean_sigma,
        'spread_pct': spread / mean * 100.0 if mean > 0 else None,
        'uncert_pct': mean_sigma / mean * 100.0 if mean > 0 else None,
        'min_dose': float(np.min(doses)),
        'max_dose': maximum,
    }
