"""Measured film-plane footprint and explicitly conditional exposure metrics."""
from dataclasses import asdict, dataclass
import math
import numpy as np
from scipy.ndimage import binary_erosion


@dataclass(frozen=True)
class Experiment:
    exposure_id: str = ''
    shots: int | None = None
    count_basis: str = 'reported'
    duration_s: float | None = None
    start_s: float | None = None
    end_s: float | None = None
    frequency_hz: float | None = None
    particle_pulse_fs: float | None = None
    laser_pulse_fs: float | None = None
    pulse_plane: str = 'unknown'
    pulse_shape: str = 'unknown'
    pulse_structure: str = 'unknown'
    pulse_duration_basis: str = 'assumed'
    background_gy: float = 0.
    core_radius_mm: float | None = None

    def __post_init__(self):
        if not isinstance(self.exposure_id, str): raise ValueError('Exposure ID must be text.')
        if self.shots is not None and (type(self.shots) is not int or self.shots < 1):
            raise ValueError('Shot count must be a positive integer; blank means unknown.')
        for key in ('duration_s', 'frequency_hz', 'particle_pulse_fs', 'laser_pulse_fs', 'core_radius_mm'):
            value = getattr(self, key)
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError(f'{key} must be positive or blank.')
        for value in (self.start_s, self.end_s, self.background_gy):
            if value is not None and not math.isfinite(value): raise ValueError('Nonfinite exposure value.')
        if self.background_gy < 0: raise ValueError('Background must be nonnegative.')
        if (self.start_s is None) != (self.end_s is None): raise ValueError('Supply both time-window endpoints or neither.')
        if self.start_s is not None:
            if self.end_s <= self.start_s: raise ValueError('Window end must be after start.')
            if self.duration_s is not None and not math.isclose(self.duration_s, self.end_s-self.start_s, rel_tol=1e-6, abs_tol=1e-12):
                raise ValueError('Duration and time-window endpoints disagree.')
        if self.count_basis not in ('reported', 'estimated'): raise ValueError('Invalid shot-count basis.')
        if self.pulse_plane not in ('unknown', 'film', 'source'): raise ValueError('Invalid particle-pulse plane.')
        if self.pulse_structure not in ('unknown', 'one_per_event'): raise ValueError('Invalid pulse structure.')
        if self.pulse_duration_basis not in ('reported', 'assumed'): raise ValueError('Invalid duration provenance.')
        if self.pulse_shape not in ('unknown', 'rectangular', 'gaussian_fwhm'): raise ValueError('Invalid pulse shape.')

    @property
    def exposure_seconds(self):
        return self.end_s-self.start_s if self.start_s is not None else self.duration_s

    def to_dict(self): return asdict(self)


def temporal_metrics(mean_dose, experiment):
    e = experiment; duration = e.exposure_seconds
    pulse = mean_dose/e.shots if e.shots is not None else None
    average = mean_dose/duration if duration is not None else None
    characteristic = peak = None
    reason = 'Requires shot count and particle pulse duration at the film plane.'
    if pulse is not None and e.particle_pulse_fs is not None and e.pulse_plane == 'film':
        if e.pulse_structure != 'one_per_event':
            reason = 'Requires a declared single particle pulse per counted irradiation event.'
        elif e.pulse_shape == 'unknown':
            reason = 'Specify rectangular duration or Gaussian FWHM; these have different meanings.'
        else:
            characteristic = pulse/(e.particle_pulse_fs*1e-15)
            peak = characteristic*(math.sqrt(4*math.log(2)/math.pi) if e.pulse_shape == 'gaussian_fwhm' else 1.)
            reason = 'Conditional model value, not time-resolved measurement. Assumes a common pulse width/shape.'
            if e.frequency_hz is not None and e.particle_pulse_fs*1e-15*e.frequency_hz > 1:
                raise ValueError('Pulse duration exceeds the repetition period.')
    for value in (pulse, average, characteristic, peak):
        if value is not None and not math.isfinite(value): raise ValueError('Derived rate overflows; check time units.')
    return {'dose_per_shot_gy': pulse, 'average_rate_gy_s': average,
            'pulse_characteristic_rate_gy_s': characteristic, 'conditional_peak_rate_gy_s': peak,
            'pulse_rate_note': reason, 'shot_count_basis': e.count_basis,
            'exposure_seconds': duration, 'count_over_window_hz': e.shots/duration if e.shots and duration else None,
            'uncertainty_note': 'Dose uncertainty is not divided by sqrt(shots); timing/count/model uncertainties are not quantified.'}


def analyze_measured_footprint(dose, sigma, spacing, region, film_bounds, experiment, check=lambda: None, ellipse=True):
    """Use native film pixels inside the ellipse AND film boundary, without smoothing.

    region vectors/origin use source-array (row, column) coordinates. Morphology
    weights are max(D-background, 0); raw dose averages retain genuine zeros.
    """
    dose = np.asarray(dose)
    if dose.ndim != 2 or not np.all(np.isfinite(dose)) or np.any(dose < 0): raise ValueError('Invalid film dose map.')
    if not math.isfinite(spacing) or spacing <= 0: raise ValueError('Invalid pixel spacing.')
    if sigma is not None and (sigma.shape != dose.shape or not np.all(np.isfinite(sigma)) or np.any(sigma < 0)):
        raise ValueError('Invalid film STD map.')
    check()
    shape = np.asarray(region.shape, float); basis = np.asarray(region.vectors, float).T
    origin = np.asarray(region.origin, float)
    if shape.shape != (2,) or np.any(shape <= 0) or abs(np.linalg.det(basis)) < 1e-12: raise ValueError('Invalid ROI transform.')
    center = origin+basis@(shape/2-.5)
    extent = np.sqrt(np.sum((basis*(shape/2)[None, :])**2, axis=1))
    x1, x2, y1, y2 = film_bounds
    y0, ystop = max(0,y1,int(np.floor(center[0]-extent[0]))), min(dose.shape[0],y2,int(np.ceil(center[0]+extent[0]))+1)
    x0, xstop = max(0,x1,int(np.floor(center[1]-extent[1]))), min(dose.shape[1],x2,int(np.ceil(center[1]+extent[1]))+1)
    if ystop <= y0 or xstop <= x0: raise ValueError('ROI does not overlap the selected film area.')
    yy, xx = np.mgrid[y0:ystop,x0:xstop]
    local = np.einsum('ij,jhw->ihw',np.linalg.inv(basis),np.array([yy,xx])-origin[:,None,None])
    mask = np.sum(((local+.5)/(shape[:,None,None]/2)-1)**2,axis=0) < 1 if ellipse else np.ones(yy.shape, bool)
    if not mask.any(): raise ValueError('No native film pixels inside the ROI.')
    raw = dose[y0:ystop,x0:xstop].astype(float,copy=True)
    signal = np.where(mask,np.maximum(raw-experiment.background_gy,0.),0.)
    values = raw[mask]; total = float(signal.sum()); warnings = []
    if center[0]-extent[0] < y1-.5 or center[0]+extent[0] > y2-.5 or center[1]-extent[1] < x1-.5 or center[1]+extent[1] > x2-.5:
        warnings.append('ROI clipped by film boundary; footprint metrics describe the retained area only.')
    if experiment.shots is not None and experiment.exposure_seconds is not None and experiment.frequency_hz is not None:
        if abs(experiment.shots-experiment.exposure_seconds*experiment.frequency_hz) > 1.000001:
            warnings.append('Count differs from nominal frequency × exposure window; check gating or missed events. No count was inferred.')
    mean_sigma = float(np.mean(sigma[y0:ystop,x0:xstop][mask])) if sigma is not None else None
    metrics = {'definition':'Measured film-plane, dose-weighted footprint in scanner X/right, Y/down coordinates.',
        'roi_pixels': int(mask.sum()), 'roi_area_mm2': float(mask.sum()*spacing**2),
        'mean_dose_gy':float(values.mean()),'min_dose_gy':float(values.min()),'max_dose_gy':float(values.max()),
        'spatial_std_gy':float(values.std()),'mean_pixel_sigma_bound_gy':mean_sigma,
        'dose_integral_gy_mm2':float(values.sum()*spacing**2),'signal_integral_gy_mm2':total*spacing**2,
        'centroid_mm':None,'covariance_mm2':None,'sigma_major_mm':None,'sigma_minor_mm':None,
        'ellipticity_major_minor':None,'orientation_deg_y_down':None,'radii_mm':None,'edge_signal_fraction':None,'halo_signal_fraction':None,'halo_core_ratio':None,
        'shape_uncertainty_note':'Width/radius uncertainty is not estimated; ROI, background and correlated dose errors remain relevant.',
        'warnings':warnings,'experiment':experiment.to_dict(),'temporal':temporal_metrics(float(values.mean()),experiment)}
    metrics['temporal']['dose_only_uncertainty_bounds'] = {
        'mean_dose_per_event_gy': mean_sigma/experiment.shots if mean_sigma is not None and experiment.shots else None,
        'exposure_average_gy_s': mean_sigma/experiment.exposure_seconds if mean_sigma is not None and experiment.exposure_seconds else None}
    radii = fractions = np.array([])
    if total > 0:
        points = np.stack([xx[mask],yy[mask]],axis=1)*spacing
        weights = signal[mask]/total
        centroid = np.sum(points*weights[:,None],axis=0); delta=points-centroid
        covariance=(delta*weights[:,None]).T@delta
        eig, axes=np.linalg.eigh(covariance); eig=np.maximum(eig,0.)
        major,minor=np.sqrt(eig[::-1]); vector=axes[:,-1]
        radius=np.linalg.norm(delta,axis=1); order=np.argsort(radius)
        radii=radius[order];fractions=np.cumsum(weights[order]);fractions[-1]=1.
        enclosed={str(p):float(radii[np.searchsorted(fractions,p/100)]) for p in (50,80,90)}
        if experiment.core_radius_mm is not None:
            core_fraction=float(weights[radius<=experiment.core_radius_mm].sum())
            metrics['halo_signal_fraction']=max(0.,1.-core_fraction)
            metrics['halo_core_ratio']=(1.-core_fraction)/core_fraction if core_fraction>0 else None
        edge=mask & ~binary_erosion(mask)
        edge_fraction=float(signal[edge].sum()/total)
        if np.any(signal[edge] > .1*signal.max()): warnings.append('Signal reaches ROI edge (>10% of peak); full-beam widths/halo may be truncated.')
        metrics.update(centroid_mm=centroid.tolist(),covariance_mm2=covariance.tolist(),sigma_major_mm=float(major),sigma_minor_mm=float(minor),
            ellipticity_major_minor=float(major/minor) if minor > 0 else None,
            orientation_deg_y_down=float(np.degrees(np.arctan2(vector[1],vector[0]))%180) if eig[1]-eig[0]>1e-6*max(eig[1],1e-30) else None,
            radii_mm=enclosed,edge_signal_fraction=edge_fraction)
    else: warnings.append('No positive signal above background: centroid, widths and enclosed radii are undefined.')
    check()
    return {'metrics':metrics,'dose_crop':raw,'mask':mask,'origin_mm':(x0*spacing,y0*spacing),
            'sigma_crop':np.array(sigma[y0:ystop,x0:xstop],copy=True) if sigma is not None else None,
            'spacing_mm':spacing,'radii':radii,'fractions':fractions}
