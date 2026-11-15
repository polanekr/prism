"""Local covariance approximations and explicitly bounded display envelopes."""

import numpy as np


def coupled_dose_variance(h_dd, h_dt, h_tt):
    """Marginal dose variance of a local 2x2 dose/thickness information matrix.

    An unidentifiable dose has infinite variance, never an arbitrary small
    replacement. If thickness has no influence (both h_tt and h_dt are zero),
    the remaining scalar dose problem is used.
    """
    h_dd, h_dt, h_tt = np.broadcast_arrays(
        np.asarray(h_dd, float), np.asarray(h_dt, float), np.asarray(h_tt, float))
    coupling = np.divide(h_dt**2, h_tt, out=np.zeros(h_dd.shape), where=h_tt > 0)
    information = h_dd - coupling
    valid = (np.isfinite(h_dd) & np.isfinite(h_dt) & np.isfinite(h_tt)
             & np.isfinite(information) & (h_tt >= 0)
             & ((h_tt > 0) | (h_dt == 0))
             & (information > 64 * np.finfo(float).eps * np.maximum(h_dd, 1e-30)))
    return np.divide(1.0, information, out=np.full(h_dd.shape, np.inf), where=valid)


def mean_sigma_bound(sigmas, axis):
    """Upper bound on a mean's sigma when spatial covariance is unknown.

    Positive correlation can prevent any sqrt(N) reduction. For equal positive
    weights the arithmetic mean of marginal sigmas bounds all covariance choices.
    """
    sigmas = np.asarray(sigmas, float)
    if not np.all(np.isfinite(sigmas)) or np.any(sigmas < 0):
        raise ValueError('Standard deviations must be finite and nonnegative.')
    return np.mean(sigmas, axis=axis)


def cumulative_area(doses, thresholds):
    """Percentage of all cells at or above each threshold, without bin truncation."""
    doses = np.sort(np.asarray(doses, float).ravel())
    thresholds = np.asarray(thresholds, float)
    if doses.size == 0 or not np.all(np.isfinite(doses)) or not np.all(np.isfinite(thresholds)):
        raise ValueError('Finite doses and thresholds are required.')
    return (doses.size - np.searchsorted(doses, thresholds, side='left')) / doses.size * 100


def dose_distribution_bounds(doses, sigmas, thresholds):
    """D90 and DAH envelopes under the pixel-wise dose +/- sigma box.

    These are sensitivity envelopes, not probabilistic confidence intervals.
    No random/systematic decomposition is inferred from marginal sigma alone.
    """
    doses, sigmas = np.asarray(doses, float), np.asarray(sigmas, float)
    if (doses.shape != sigmas.shape or doses.size == 0
            or not np.all(np.isfinite(doses)) or np.any(doses < 0)
            or not np.all(np.isfinite(sigmas)) or np.any(sigmas < 0)):
        raise ValueError('Matching finite, nonnegative doses and sigmas are required.')
    low, high = np.maximum(0, doses - sigmas), doses + sigmas
    return {
        'd90_low': float(np.percentile(low, 10)),
        'd90_high': float(np.percentile(high, 10)),
        'dah_low': cumulative_area(low, thresholds),
        'dah_high': cumulative_area(high, thresholds),
    }


def beam_dose_distribution(doses, bins=50, normalization='peak', max_knots=1024):
    """Histogram scaling is independent of the empirical area-above-dose curve.

    The distribution can be peak-normalized for shape inspection or normalized
    by total area. DAH always uses the full ROI denominator, including true zero
    doses. At most max_knots observed doses are used for a responsive plot.
    """
    values = np.sort(np.asarray(doses, float).ravel())
    if not values.size or not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError('Finite nonnegative ROI doses are required.')
    if normalization not in ('peak', 'area'):
        raise ValueError('Histogram normalization must be peak or area.')
    if type(bins) is not int or bins < 1 or type(max_knots) is not int or max_knots < 2:
        raise ValueError('Invalid histogram bin count or DAH knot count.')
    maximum = float(values[-1])
    minimum = float(values[0])
    padding = max(.01, maximum*.01) if maximum-minimum <= np.spacing(maximum)*bins*2 else 0.
    limits = (max(0., minimum-padding), maximum+padding)
    counts, edges = np.histogram(values, bins=bins, range=limits)
    denominator = counts.max() if normalization == 'peak' else values.size
    heights = counts / denominator * 100.
    positions = np.unique(np.linspace(0, values.size-1, min(max_knots, values.size), dtype=int))
    observed = values[positions]
    thresholds = np.unique(np.r_[0., observed, np.nextafter(observed, np.inf)])
    area = (values.size-np.searchsorted(values, thresholds, side='left')) / values.size * 100.
    return {'edges': edges, 'histogram': heights, 'dah_dose': thresholds, 'dah_area': area}
