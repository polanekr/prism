"""Numerical morphology primitives; no GUI or calibration imports."""
import numpy as np
from scipy.fft import fft2, fftshift, ifft2
from scipy.stats import skew, kurtosis
from skimage.transform import radon

class BeamMorphology:
    """Advanced morphological analysis for LWFA beam footprints."""
    
    @staticmethod
    def compute_fft_and_autocorr(dose_roi, spacing_mm):
        """2D Power Spectral Density és Autokorreláció."""
        # Átlag levonása a tiszta frekvenciaképhez
        data_zeromean = dose_roi - np.mean(dose_roi)
        
        # 2D FFT és Power Spectrum (PSD)
        f = fft2(data_zeromean)
        fshift = fftshift(f)
        psd = np.abs(fshift)**2
        
        # Logaritmikus PSD a vizualizációhoz
        psd_log = np.log10(np.maximum(psd, 1e-10))
        
        # Autokorreláció (Wiener-Khinchin tétel alapján)
        autocorr = fftshift(np.real(ifft2(np.abs(f)**2)))
        peak = np.max(autocorr)
        autocorr = autocorr / peak if peak > 0 else np.zeros_like(autocorr)
        
        # Radiális autokorreláció kinyerése (struktúra mérete)
        h, w = autocorr.shape
        cy, cx = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        r = np.hypot(x - cx, y - cy) * spacing_mm
        
        # Binezés a radiális profilhoz
        r_flat = r.flatten()
        a_flat = autocorr.flatten()
        bins = np.linspace(0, min(h, w) * spacing_mm / 2.0, 50)
        bin_centers = (bins[1:] + bins[:-1]) / 2.0
        radial_prof, _ = np.histogram(r_flat, bins=bins, weights=a_flat)
        counts, _ = np.histogram(r_flat, bins=bins)
        radial_prof = radial_prof / np.maximum(counts, 1)
        
        return psd_log, autocorr, bin_centers, radial_prof

    @staticmethod
    def compute_radon_sinogram(dose_roi):
        """Irányított sávosodás kimutatása Radon-transzformációval."""
        import warnings
        from skimage.transform import radon
        
        h, w = dose_roi.shape
        
        # Pontosan a scikit-image belső logikájának megfelelő maszkolás (egész osztással)
        radius = min(h, w) // 2
        y, x = np.ogrid[0:h, 0:w]
        outside_circle = (y - h // 2)**2 + (x - w // 2)**2 > radius**2
        
        masked_roi = np.copy(dose_roi)
        masked_roi[outside_circle] = 0.0
        
        theta = np.linspace(0., 180., max(masked_roi.shape), endpoint=False)
        
        # Figyelmeztetések blokkolása a futás idejére
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            sinogram = radon(masked_roi, theta=theta, circle=True)
            
        return sinogram, theta

    @staticmethod
    def compute_zernike_modes(dose_roi):
        """
        Zernike-polinom illesztés a nyalábtorzulások (Aberration) kvantifikálására.
        Főbb módusok: Defocus, Astigmatism X/Y, Coma X/Y, Spherical.
        """
        h, w = dose_roi.shape
        cy, cx = h / 2.0, w / 2.0
        y, x = np.ogrid[:h, :w]
        
        # Normálisan skálázott polárkoordináták a [-1, 1] egységkörön
        R_max = min(h, w) / 2.0
        rho = np.hypot(x - cx, y - cy) / R_max
        phi = np.arctan2(y - cy, x - cx)
        
        mask = rho <= 1.0
        rho_m = rho[mask]
        phi_m = phi[mask]
        data_m = dose_roi[mask]
        
        # Első 9 ANSI standard Zernike mód (Piston és Tilt nélkül)
        # Z_2^0 (Defocus), Z_2^-2, Z_2^2 (Astig), Z_3^-1, Z_3^1 (Coma), Z_4^0 (Spherical)
        Z = np.zeros((len(rho_m), 6))
        Z[:, 0] = np.sqrt(3) * (2*rho_m**2 - 1)                          # Defocus
        Z[:, 1] = np.sqrt(6) * (rho_m**2) * np.sin(2*phi_m)              # Astigmatism Y
        Z[:, 2] = np.sqrt(6) * (rho_m**2) * np.cos(2*phi_m)              # Astigmatism X
        Z[:, 3] = np.sqrt(8) * (3*rho_m**3 - 2*rho_m) * np.sin(phi_m)    # Coma Y
        Z[:, 4] = np.sqrt(8) * (3*rho_m**3 - 2*rho_m) * np.cos(phi_m)    # Coma X
        Z[:, 5] = np.sqrt(5) * (6*rho_m**4 - 6*rho_m**2 + 1)             # Spherical
        
        # Legkisebb négyzetek módszere a koefficiens-illesztéshez
        coeffs, _, _, _ = np.linalg.lstsq(Z, data_m, rcond=None)
        
        # Rekonstruált nyaláb a Zernike módusokból
        reconstructed = np.zeros_like(dose_roi)
        reconstructed[mask] = Z @ coeffs
        
        labels = ["Defocus", "Astig Y", "Astig X", "Coma Y", "Coma X", "Spherical"]
        return coeffs, labels, reconstructed

    @staticmethod
    def compute_moments(dose_roi):
        """Kurtosis (csúcsosság) és Skewness (ferdeség) a mátrixra."""
        val = dose_roi.flatten()
        return skew(val), kurtosis(val)

    @staticmethod
    def deconvolve_scattering(dose_roi, sigma_mm, spacing_mm, iterations=15):
        """
        Richardson-Lucy dekonvolúció a PMMA/levegő szóródás eltávolítására.
        Visszaállítja a nyaláb eredeti (vákuum/kilépő ablak) élességét.
        """
        from skimage.restoration import richardson_lucy
        
        # PSF (Point Spread Function) generálása a Moliere-szórás alapján
        sigma_px = sigma_mm / spacing_mm
        if sigma_px < 0.5:
            return dose_roi # Nincs érdemi szóródás
            
        kernel_size = int(np.ceil(sigma_px * 6)) // 2 * 2 + 1
        y, x = np.mgrid[-kernel_size//2 + 1:kernel_size//2 + 1, 
                        -kernel_size//2 + 1:kernel_size//2 + 1]
        psf = np.exp(-(x**2 + y**2) / (2.0 * sigma_px**2))
        psf /= psf.sum()
        
        # Hátteret (nullákat) megemeljük picit a nullával osztás elkerülésére
        dose_safe = np.clip(dose_roi, 1e-4, None)
        
        # Dekonvolúció (élesítés)
        with np.errstate(divide='ignore', invalid='ignore'):
            vacuum_profile = richardson_lucy(dose_safe, psf, num_iter=iterations, clip=False)
            
        return np.nan_to_num(vacuum_profile, nan=0.0)

    @staticmethod
    def compute_radiobiology_metrics(dose_roi, spacing_mm):
        """
        Radiobiológiai és térben frakcionált (GRID/Microbeam) metrikák kiszámítása.
        """
        from scipy.ndimage import gaussian_filter, maximum_filter, minimum_filter, uniform_filter
        
        metrics = {}
        
        # 1. Local Dose Gradient (LDG) - Lokális Dózisgradiens
        gy, gx = np.gradient(dose_roi, spacing_mm)
        gradient_mag = np.hypot(gx, gy)
        metrics['mean_ldg'] = float(np.mean(gradient_mag))
        metrics['max_ldg'] = float(np.max(gradient_mag))
        
        # 2. Fractional Hotspot Volume (FHV)
        # Hotspot definíció: a lokális átlagdózis 1.5-szeresét meghaladó régiók
        mean_dose = np.mean(dose_roi)
        hotspot_mask = dose_roi > (1.5 * mean_dose)
        metrics['fhv_pct'] = float(np.sum(hotspot_mask) / dose_roi.size * 100.0)
        
        # 3. Peak-to-Valley Dose Ratio (PVDR)
        # Enyhe simítás a zaj csökkentésére a csúcsok és völgyek keresése előtt
        smooth_dose = gaussian_filter(dose_roi, sigma=0.5/spacing_mm)
        
        # Lokális maximumok és minimumok keresése (pl. 3x3 pixel ablakban)
        local_max = maximum_filter(smooth_dose, size=3) == smooth_dose
        local_min = minimum_filter(smooth_dose, size=3) == smooth_dose
        
        peaks = smooth_dose[local_max]
        valleys = smooth_dose[local_min]
        
        # Kiszűrjük a háttérzajt: csak a releváns nyalábrész völgyeit és jelentős csúcsait nézzük
        max_d = np.max(smooth_dose)
        valleys_active = valleys[valleys > 0.1 * max_d]
        peaks_active = peaks[peaks > 0.4 * max_d]
        
        if len(peaks_active) > 0 and len(valleys_active) > 0:
            metrics['pvdr'] = float(np.mean(peaks_active) / np.mean(valleys_active))
        else:
            metrics['pvdr'] = 1.0
            
        # 4. Lakunaritás (Hézagosság) dobozszámláló módszerrel
        # Binarizálás az átlag feletti "anyag" (pamacsok) és az alatti "hézagok" elválasztására
        binary_map = dose_roi > mean_dose
        box_size = max(3, int(1.0 / spacing_mm)) # kb. 1 mm-es vizsgálóablak
        
        # A dobozba eső "tömeg" kiszámítása
        box_mass = uniform_filter(binary_map.astype(float), size=box_size) * (box_size**2)
        
        mean_mass = np.mean(box_mass)
        var_mass = np.var(box_mass)
        
        if mean_mass > 0:
            metrics['lacunarity'] = float(var_mass / (mean_mass**2))
        else:
            metrics['lacunarity'] = 0.0
            
        return metrics

    @staticmethod
    def compute_advanced_segmentation_and_entropy(dose_roi, spacing_mm):
        """
        Watershed-szegmentáció a pamacsok (filamentumok/jitter-szigetek) izolálására
        és térbeli Shannon-entrópia számítása.
        """
        from skimage.segmentation import watershed
        from skimage.feature import peak_local_max
        from scipy.stats import entropy
        
        # 1. Térbeli Shannon-entrópia (Dóziseloszlás rendezetlensége)
        d_flat = dose_roi.flatten()
        d_positive = d_flat[d_flat > 0.001]
        if d_positive.size > 0:
            hist, _ = np.histogram(d_positive, bins=64, density=True)
            hist = hist[hist > 0]
            shannon_entropy = float(entropy(hist, base=2))
        else:
            shannon_entropy = 0.0

        # 2. Watershed szegmentáció (Pamacs-szigetek száma és mérete)
        mean_d = np.mean(dose_roi)
        thresh_mask = dose_roi > mean_d
        
        # Lokális koordináták keresése a dóziscsúcsokon (~0.5 mm távolság)
        min_dist_px = max(2, int(0.5 / spacing_mm))
        coordinates = peak_local_max(dose_roi, min_distance=min_dist_px, labels=thresh_mask)
        
        num_islands = len(coordinates)
        mean_island_area_mm2 = 0.0
        island_dose_fraction = 0.0
        segmented_map = np.zeros_like(dose_roi, dtype=int)
        
        if num_islands > 0:
            markers = np.zeros(dose_roi.shape, dtype=int)
            markers[coordinates[:, 0], coordinates[:, 1]] = np.arange(1, num_islands + 1)
                
            segmented_map = watershed(-dose_roi, markers, mask=thresh_mask)
            
            pixel_area_mm2 = spacing_mm ** 2
            island_sizes = np.bincount(segmented_map.ravel(), minlength=num_islands + 1)[1:num_islands + 1] * pixel_area_mm2
            mean_island_area_mm2 = float(np.mean(island_sizes)) if island_sizes.size else 0.0
            
            total_dose_roi = np.sum(dose_roi)
            if total_dose_roi > 0:
                island_dose = np.sum(dose_roi[segmented_map > 0])
                island_dose_fraction = float(island_dose / total_dose_roi * 100.0)

        return {
            "shannon_entropy": shannon_entropy,
            "num_islands": int(num_islands),
            "mean_island_area_mm2": mean_island_area_mm2,
            "island_dose_fraction_pct": island_dose_fraction,
            "segmented_map": segmented_map
        }


def analyze_morphology(dose, sigma, spacing, settings, check=lambda: None, progress=lambda message: None):
    """Compute a complete morphology result without retaining any widget reference."""
    from scipy.ndimage import zoom
    data = np.asarray(dose, dtype=float)
    if data.ndim != 2 or min(data.shape) < 2 or not np.all(np.isfinite(data)) or np.any(data < 0):
        raise ValueError('Morphology requires a finite nonnegative 2D map, at least 2×2.')
    if not np.isfinite(spacing) or spacing <= 0: raise ValueError('Invalid pixel spacing.')
    check()
    if settings.snr_mask and sigma is not None:
        if sigma.shape != data.shape or not np.all(np.isfinite(sigma)) or np.any(sigma < 0):
            raise ValueError('STD must match morphology input.')
        data = np.where(data / np.maximum(sigma, 1e-6) > 3., data, 0.)
    original_shape = data.shape
    if max(data.shape) > settings.max_size:
        scale = settings.max_size/max(data.shape)
        data = zoom(data, scale, order=1); spacing /= scale
    check(); progress('Scattering correction')
    if settings.scatter_sigma_mm > 0:
        data = BeamMorphology.deconvolve_scattering(data, settings.scatter_sigma_mm, spacing, iterations=15)
    check(); progress('FFT and autocorrelation')
    psd, autocorr, radii, radial = BeamMorphology.compute_fft_and_autocorr(data, spacing)
    half = np.flatnonzero(radial < .5)
    correlation = float(radii[half[0]]) if half.size and np.ptp(data) > 0 else None
    check(); progress('Radon transform')
    sinogram, theta = BeamMorphology.compute_radon_sinogram(data)
    check(); progress('Zernike modes')
    coefficients, labels, reconstruction = BeamMorphology.compute_zernike_modes(data)
    check(); progress('Moments and segmentation')
    skewness, kurt = BeamMorphology.compute_moments(data) if np.ptp(data) > 0 else (None, None)
    bio = BeamMorphology.compute_radiobiology_metrics(data, spacing)
    segmentation = BeamMorphology.compute_advanced_segmentation_and_entropy(data, spacing)
    metrics = {'shot_mode': settings.shot_mode, 'skewness': float(skewness) if skewness is not None and np.isfinite(skewness) else None,
        'kurtosis': float(kurt) if kurt is not None and np.isfinite(kurt) else None, 'shannon_entropy': segmentation['shannon_entropy'],
        'pvdr': bio['pvdr'], 'fhv_pct': bio['fhv_pct'], 'mean_ldg_gy_mm': bio['mean_ldg'],
        'lacunarity': bio['lacunarity'], 'num_islands': segmentation['num_islands'],
        'mean_island_area_mm2': segmentation['mean_island_area_mm2'],
        'island_dose_fraction_pct': segmentation['island_dose_fraction_pct'],
        'scatter_sigma_mm': settings.scatter_sigma_mm, 'correlation_length_mm': correlation,
        'zernike_coeffs': {label: float(value) for label, value in zip(labels, coefficients)},
        'input_shape': list(original_shape), 'working_shape': list(data.shape), 'working_spacing_mm': float(spacing)}
    check()
    return {'psd': psd, 'autocorr': autocorr, 'radii': radii, 'radial': radial,
        'sinogram': sinogram, 'reconstruction': reconstruction, 'coefficients': coefficients,
        'labels': labels, 'segmented_map': segmentation['segmented_map'], 'metrics': metrics}


def analyze_morphology_region(dose, sigma, region, spacing, settings,
                              check=lambda: None, progress=lambda message: None):
    """Extract the viewer's elliptical ROI off-thread using its numeric transform."""
    from scipy.ndimage import map_coordinates
    from bio_analysis import sample_coordinates
    check(); progress('Extracting analysis ROI')
    coordinates = sample_coordinates(region)
    shape = coordinates.shape[1:]
    if min(shape) < 2: raise ValueError('Analysis ROI is too small.')
    y, x = np.indices(shape)
    mask = np.hypot((y+.5)/(shape[0]/2)-1, (x+.5)/(shape[1]/2)-1) < 1
    options = dict(order=1, mode='constant', cval=0., prefilter=False)
    data = map_coordinates(np.asarray(dose, float), coordinates, **options)*mask
    errors = map_coordinates(np.asarray(sigma, float), coordinates, **options)*mask if sigma is not None else None
    check()
    return analyze_morphology(data, errors, spacing, settings, check, progress)
