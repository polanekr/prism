"""Widget-free sample dosimetry and ROI evaluation on the BACK reference grid."""
from dataclasses import dataclass
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from film_alignment import resample_to_reference
from processing_settings import BioSettings
from sample_statistics import calculate_sample_statistics


@dataclass(frozen=True)
class SampleRegion:
    identifier: int
    shape: tuple
    vectors: tuple
    origin: tuple
    mask: np.ndarray
    weights: np.ndarray


def sample_coordinates(region):
    grid = np.indices(tuple(int(np.ceil(value)) for value in region.shape), dtype=float)
    return np.einsum('ij,ihw->jhw', np.asarray(region.vectors), grid) + np.asarray(region.origin)[:, None, None]


def calculate_sample_maps(front, back, front_std, back_std, transform, settings, check=lambda: None):
    check()
    front_aligned, coverage = resample_to_reference(front, transform, back.shape)
    check()
    front_sigma = (resample_to_reference(front_std, transform, back.shape)[0]
                   if front_std is not None else front_aligned*.02)
    back_sigma = back_std if back_std is not None else back*.02
    weight = settings.weight_back
    d_front_safe = np.clip(front_aligned, 1e-4, None); d_back_safe = np.clip(back, 1e-4, None)
    dose = front_aligned**(1-weight) * back**weight
    derivative_dose = d_front_safe**(1-weight) * d_back_safe**weight
    sigma = np.sqrt(((1-weight)*derivative_dose/d_front_safe*front_sigma)**2) + np.sqrt((weight*derivative_dose/d_back_safe*back_sigma)**2)
    dose = np.where(coverage, dose, 0.); sigma = np.where(coverage, sigma, 0.)
    check()
    if settings.scatter_sigma_mm > 0:
        width = settings.scatter_sigma_mm/settings.spacing_mm
        support = gaussian_filter(coverage.astype(float), sigma=width)
        dose = gaussian_filter(dose, sigma=width)/np.maximum(support, 1e-12)
        sigma = gaussian_filter(sigma, sigma=width)/np.maximum(support, 1e-12)
    check()
    return dose, sigma, coverage


def analyze_samples(front, back, front_std, back_std, transform, settings, regions,
                    check=lambda: None, progress=lambda message: None):
    progress('Aligning films and propagating uncertainty')
    dose, sigma, coverage = calculate_sample_maps(front, back, front_std, back_std, transform, settings, check)
    results = {}; skipped = []
    for region in regions:
        check(); progress(f'Sample #{region.identifier}')
        mask = region.mask
        if not mask.size or not np.any(mask & (region.weights > 0)):
            skipped.append(region.identifier); continue
        coordinates = sample_coordinates(region)
        options = dict(order=1, mode='constant', cval=0., prefilter=False)
        roi_coverage = map_coordinates(coverage.astype(float), coordinates, **options)
        if np.any(mask & (roi_coverage < .999999)):
            raise ValueError(f'Sample #{region.identifier} extends outside the common film area. Adjust alignment or ROI.')
        values = map_coordinates(dose, coordinates, **options)
        errors = map_coordinates(sigma, coordinates, **options)
        result = calculate_sample_statistics(values[mask], errors[mask], region.weights[mask])
        result['z_weight_back'] = settings.weight_back
        results[region.identifier] = result
    check()
    if not results: raise ValueError('No ROI contains a sample with positive volume. Adjust ROIs and recalculate.')
    return {'results': results, 'skipped': skipped}
