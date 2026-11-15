"""TIFF conversion and saved processing provenance; independent of widgets."""
import json
import os
from pathlib import Path
import numpy as np
import tifffile
from scipy.ndimage import median_filter, maximum_filter, zoom
from tqdm.auto import tqdm
from scan_data import is_raw_film, ensure_scan_ready, load_scan_noise
from processing_settings import ProcessingSettings
from dose_solvers import calibration_worker_count

class NumpyEncoder(json.JSONEncoder):
    """Helper to serialise numpy arrays to JSON."""
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


class ProcessingMixin:
    def _load_tiff_with_meta(self, path: str):
        """Loads TIFF and extracts DPI/Resolution and absolute scanner position metadata."""
        ensure_scan_ready(path)
        img = tifffile.imread(path).astype(np.float64)
        if img.ndim == 3 and img.shape[2] == 4:
            img = img[:, :, :3]
        if (img.ndim != 3 or img.shape[2] != 3 or 0 in img.shape
                or not np.all(np.isfinite(img)) or np.any(img < 0)):
            raise ValueError(f'Expected a finite, nonnegative RGB TIFF: {path}')

        # Alapértelmezett értékek (fallback)
        meta = {
            'pixel_spacing_mm': 0.0846, 
            'x_offset_mm': None, 
            'scanner_center_mm': 108.0
        } 
        try:
            with tifffile.TiffFile(path) as tif:
                tags = tif.pages[0].tags
                
                # Felbontás (DPI) kiolvasása
                if 'XResolution' in tags:
                    x_res = tags['XResolution'].value
                    dpi = x_res[0] / x_res[1] if isinstance(x_res, tuple) else float(x_res)
                    if dpi < 1: dpi = 72.0
                    unit = tags['ResolutionUnit'].value if 'ResolutionUnit' in tags else 2
                    if unit == 2: meta['pixel_spacing_mm'] = 25.4 / dpi
                    elif unit == 3: meta['pixel_spacing_mm'] = 10.0 / dpi
                    
                # Pozíció (LRA korrekcióhoz) kiolvasása az ImageDescription tag-ből
                if 'ImageDescription' in tags:
                    desc = tags['ImageDescription'].value
                    if isinstance(desc, str):
                        for part in desc.split(';'):
                            if '=' in part:
                                key, val = part.split('=')
                                if key == 'left_mm': meta['x_offset_mm'] = float(val)
                                if key == 'scanner_center_mm': meta['scanner_center_mm'] = float(val)
        except Exception: pass
        return img, meta


    def process_films(self, file_list: list, blank_path: str, method: str = 'hybrid',
                      use_cleaning: bool = False, blank_rgb: tuple = None, median_filtering: bool = True, settings: ProcessingSettings = None):
        """Process with an immutable run identity and a saved calibration/settings snapshot."""
        from project_state import begin_run, atomic_json, file_record, json_value
        from datetime import datetime, timezone
        specification = settings or ProcessingSettings(method=method, cleaning=use_cleaning,
            median_filtering=median_filtering, blank_rgb=blank_rgb, random_seed=self.random_seed)
        if not isinstance(specification, ProcessingSettings):
            raise ValueError('settings must be ProcessingSettings.')
        self.random_seed = specification.random_seed
        files = list(file_list)
        method = specification.method; use_cleaning = specification.cleaning
        median_filtering = specification.median_filtering; blank_rgb = specification.blank_rgb
        settings = json_value(specification.to_dict())
        folder, report = begin_run(self, files, blank_path if blank_rgb is None else None, settings)
        try:
            self._process_films(files, blank_path, method, use_cleaning, blank_rgb, median_filtering,
                                report['run_id'], specification.uncertainty_workers, specification.anchor_bounds)
            report['status'] = 'completed'
        except BaseException as error:
            report.update(status='failed', error=f'{type(error).__name__}: {error}')
            raise
        finally:
            for path in files:
                stem = Path(path).stem
                meta_path = Path(self.output_folder) / f'{stem}_meta.json'
                if meta_path.exists():
                    try:
                        data = json.loads(meta_path.read_text(encoding='utf-8'))
                        if data.get('run_id') == report['run_id']:
                            report['outputs'].extend(file_record(Path(self.output_folder) / f'{stem}{suffix}', self.output_folder)
                                for suffix in ('_dose.npy', '_std.npy', '_std_fit.npy', '_std_calib.npy', '_meta.json'))
                    except (OSError, ValueError):
                        pass
            report['finished_at'] = datetime.now(timezone.utc).isoformat()
            atomic_json(folder / 'run.json', report)


    def _process_films(self, file_list, blank_path, method, use_cleaning, blank_rgb,
                       median_filtering, run_id, uncertainty_workers, anchor_bounds=None):
        """
        Batch processes a list of film scans into dose maps (*_dose.npy) AND 
        pixel-wise uncertainty maps (*_std.npy).
        Incorporates exact scanner noise propagation from _scan_std.tif if available.
        """
        if not self.calib_params: raise ValueError("No calibration loaded!")
        method = self._canonical_method(method)
        if any(not is_raw_film(path) for path in file_list):
            raise ValueError('Only raw film TIFFs can be converted; scanner SD/SEM sidecars are not films.')
        for path in file_list:
            ensure_scan_ready(path)

        print(f"\n--- Processing {len(file_list)} films (Method={method}, Cleaning={use_cleaning}) ---")
        
        # --- BLANK FILM ÉS ZAJÁNAK BETÖLTÉSE ---
        if blank_rgb is None:
            blank_img_raw, _ = self._load_tiff_with_meta(blank_path)
            blank_img = median_filter(blank_img_raw, size=(5, 5, 1)) if median_filtering else blank_img_raw
            
            blank_std_img, blank_noise_source = load_scan_noise(blank_path, blank_img_raw.shape)
            if blank_std_img is None:
                blank_std_img = np.zeros_like(blank_img_raw)
        else:
            blank_noise_source = 'assumed_uniform_10_intensity_units'
        
        for idx, fpath in tqdm(enumerate(file_list), total=len(file_list)):
            fname = os.path.basename(fpath)
            base_name = os.path.splitext(fname)[0]
            
            img_raw, meta = self._load_tiff_with_meta(fpath)
            img = median_filter(img_raw, size=(3, 3, 1)) if median_filtering else img_raw
            
            # --- BLANK FILM KÉP ÉS ZAJ SKÁLÁZÁSA (Ha eltér a méret) ---
            if blank_rgb is not None:
                current_blank = np.ones_like(img) * np.array(blank_rgb)
                current_blank_std = np.ones_like(img) * 10.0
            else:
                if img.shape != blank_img.shape:
                    scale_h = img.shape[0] / blank_img.shape[0]
                    scale_w = img.shape[1] / blank_img.shape[1]
                    current_blank = zoom(blank_img, (scale_h, scale_w, 1.0), order=1)
                    current_blank_std = zoom(blank_std_img, (scale_h, scale_w, 1.0), order=1)
                else:
                    current_blank = blank_img
                    current_blank_std = blank_std_img
            
            spacing = meta.get("pixel_spacing_mm", 0.0846)
            x_offset = meta.get("x_offset_mm", None)
            scanner_center = meta.get("scanner_center_mm", 108.0)
            
            # 1. Alap OD térkép kiszámítása
            od_map = self._calculate_od_map(img, current_blank, spacing, x_offset, scanner_center)
            
            if use_cleaning:
                od_map = self._clean_od_map_biharmonic(od_map)
                
            # --- ZAJ (sigma_OD) KISZÁMÍTÁSA ---
            img_std, scan_noise_source = load_scan_noise(fpath, img_raw.shape)
            if img_std is not None:
                # Noise of the averaged intensity, not the SD of individual repetitions.
                
                # Hibaterjedés: sigma_OD = (1 / (ln(10) * I)) * sigma_I
                term_img = (img_std / (np.log(10) * np.maximum(img_raw, 1.0))) ** 2
                term_blank = (current_blank_std / (np.log(10) * np.maximum(current_blank, 1.0))) ** 2
                
                od_noise_map = np.sqrt(term_img + term_blank)
                if median_filtering:
                    od_noise_map = median_filter(od_noise_map, size=(3, 3, 1))
            else:
                # Fallback: zajbecslés a szűretlen és szűrt OD különbségéből
                raw_od_map = self._calculate_od_map(img_raw, current_blank, spacing, x_offset, scanner_center)
                od_noise_map = self._estimate_od_noise_map(raw_od_map, od_map)

            if (od_noise_map.shape != od_map.shape or not np.all(np.isfinite(od_noise_map))
                    or np.any(od_noise_map < 0)):
                raise ValueError(f'Invalid OD uncertainty for {fname}: expected finite, nonnegative RGB noise.')
            
            # 2. Dózis és illesztési/mérési szórás (std_fit) kiszámítása
            anchor_mask = self._anchor_valid_mask(od_map, anchor_bounds) if method != 'fast' else None
            solver_diagnostics = {}
            dose_map, std_fit_map = self._solve_dose(od_map, method, od_noise_map,
                anchor_mask=anchor_mask, diagnostics=solver_diagnostics)
            if solver_diagnostics:
                print(f"  -> Anchor: {solver_diagnostics['anchor_status']}, "
                      f"{solver_diagnostics['anchor_pixels']} core pixels; "
                      f"thickness at limit: {100*solver_diagnostics['thickness_at_limit_fraction']:.2f}%.")

            
            # 3. Kalibrációs görbe bizonytalanságának (std_calib) kiszámítása
            source = self._calibration_uncertainty_source()
            samples = min(len(self.calib_samples[ch]) for ch in 'RGB') if source == 'parameter_samples' else 1
            effective_workers = calibration_worker_count(dose_map.shape, uncertainty_workers, samples)
            print(f'  -> Calibration uncertainty: {samples} realizations, {effective_workers} worker(s).')
            std_calib_map = self._compute_calibration_std_map(od_map, dose_map, method, od_noise_map,
                median_filtering=median_filtering, workers=effective_workers, anchor_mask=anchor_mask)
            
            # 4. Teljes pixelenkénti szórás kombinálása: sigma_tot = sqrt(sigma_fit^2 + sigma_calib^2)
            # Parameter realizations already include this final dose filter.
            # Spatial noise covariance is unknown: do not median-filter sigmas
            # as if they were doses, or assume independent neighbouring cells.
            if median_filtering:
                dose_map = median_filter(dose_map, size=3)
                std_fit_map = maximum_filter(std_fit_map, size=3)
            std_total_map = np.sqrt(std_fit_map**2 + std_calib_map**2)
            invalid_components = []
            for component, values in (('dose', dose_map), ('measurement_STD', std_fit_map),
                                      ('calibration_STD', std_calib_map), ('total_STD', std_total_map)):
                nonfinite = int(np.count_nonzero(~np.isfinite(values)))
                negative = int(np.count_nonzero(np.isfinite(values) & (values < 0)))
                if nonfinite or negative:
                    invalid_components.append(f'{component}: {nonfinite} nonfinite, {negative} negative pixels')
            if invalid_components:
                raise ValueError(f'Invalid output for {fname} (solver={method}): '
                    + '; '.join(invalid_components)
                    + '. No output was saved for this film. '
                    + 'Nonfinite uncertainty can indicate unidentifiable dose/thickness; '
                    + 'negative dose indicates a numerical constraint failure.')

            # Fizikai korlátok biztosítása (ne legyen 0 vagy NaN szórás ott, ahol van dózis)
            min_std = np.where(dose_map > 0.01, dose_map * 0.005, 1e-4)
            std_total_map = np.maximum(std_total_map, min_std)
            
            # MENTÉS: *_dose.npy ÉS *_std.npy
            dose_out_path = os.path.join(self.output_folder, f"{base_name}_dose.npy")
            std_out_path = os.path.join(self.output_folder, f"{base_name}_std.npy")
            np.save(dose_out_path, dose_map)
            np.save(std_out_path, std_total_map)
            np.save(os.path.join(self.output_folder, f'{base_name}_std_fit.npy'), std_fit_map)
            np.save(os.path.join(self.output_folder, f'{base_name}_std_calib.npy'), std_calib_map)
            
            active_mask = dose_map > (0.1 * np.max(dose_map))
            mean_rel_std = float(np.mean(std_total_map[active_mask] / dose_map[active_mask]) * 100.0) if np.any(active_mask) else 0.0
            
            meta_out = {
                "original_filename": fname,
                "run_id": run_id,
                "provenance": f".runs/{run_id}/run.json",
                "median_filtering": median_filtering,
                "std_filename": f"{base_name}_std.npy",
                "std_fit_filename": f"{base_name}_std_fit.npy",
                "std_calib_filename": f"{base_name}_std_calib.npy",
                "uncertainty_model_version": 3,
                "anchor_bounds_normalized": anchor_bounds,
                "solver_diagnostics": solver_diagnostics,
                "calibration_uncertainty_workers": effective_workers,
                "calibration_uncertainty_samples": samples,
                "calibration_uncertainty_source": self._calibration_uncertainty_source(),
                "calibration_uncertainty_available": self._calibration_uncertainty_source() != 'unavailable',
                "parameter_channel_correlation": "independent_channel_blocks",
                "spatial_covariance": "not_stored; use conservative envelopes for averages",
                "fit_uncertainty_after_median": "maximum_of_local_3x3_sigmas" if median_filtering else "not_applied",
                "uncertainty_floor": {"relative_above_0.01_gy": 0.005, "otherwise_gy": 1e-4},
                "pixel_spacing_mm": meta.get("pixel_spacing_mm", 0.0846),
                "max_dose": float(np.max(dose_map)),
                "mean_relative_uncertainty_pct": round(mean_rel_std, 2),
                "unit": "Gy",
                "method": method,
                "cleaning_applied": use_cleaning,
                "noise_model": "measured_from_scanner" if img_std is not None else "estimated_from_image",
                "scan_noise_source": scan_noise_source,
                "blank_noise_source": blank_noise_source
            }
            
            if self.batch_metadata:
                for key, value in self.batch_metadata.items():
                    if key in meta_out:
                        continue
                    if isinstance(value, (list, tuple, np.ndarray)):
                        if idx < len(value):
                            val_to_write = value[idx]
                            if hasattr(val_to_write, 'item'): val_to_write = val_to_write.item()
                            meta_out[key] = val_to_write
                    else:
                        meta_out[key] = value

            with open(os.path.join(self.output_folder, f"{base_name}_meta.json"), 'w') as f:
                json.dump(meta_out, f, cls=NumpyEncoder, indent=4)

        print(f"Processing complete. Dose (*_dose.npy) and Uncertainty (*_std.npy) saved to: {self.output_folder}")


