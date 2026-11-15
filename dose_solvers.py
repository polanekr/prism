"""Numerical dose estimators and uncertainty propagation, with no Qt dependency."""
import warnings
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from scipy.ndimage import median_filter, uniform_filter
from skimage.restoration import denoise_tv_chambolle
from uncertainty import coupled_dose_variance

class DoseSolverMixin:
    def _calculate_od_map(self, img: np.ndarray, blank_img: np.ndarray, 
                          pixel_spacing_mm: float = 0.0846, 
                          x_offset_mm: float = None, 
                          scanner_center_mm: float = 108.0) -> np.ndarray:
        """
        Calculates the Optical Density (OD) map with absolute position-aware,
        OD- and channel-dependent Lateral Response Artifact (LRA) correction.
        """
        img_safe = np.where(img <= 0, 1e-6, img)
        blank_safe = np.where(blank_img <= 0, 1e-6, blank_img)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            od_map = np.log10(blank_safe / img_safe)
            
        od_map = np.nan_to_num(od_map, nan=0.0, posinf=0.0, neginf=0.0)
        od_map[od_map < 0] = 0.0

        # --- FIZIKAI LRA KORREKCIÓ ---
        h, w = od_map.shape[:2]
        x_px = np.arange(w)
        
        if x_offset_mm is not None:
            # Abszolút pozíció a szkenner üveglapján (ha a scan.py-ból jött a fájl)
            absolute_x_mm = x_offset_mm + (x_px * pixel_spacing_mm)
            dist_from_center_mm = absolute_x_mm - scanner_center_mm
        else:
            # Fallback: feltételezzük, hogy a film a szkenner középvonalára lett helyezve
            dist_from_center_mm = (x_px - w / 2.0) * pixel_spacing_mm
            
        # Csatornafüggő LRA együtthatók Epson szkennerhez (EBT3 film)
        k_coeffs = np.array([1.2e-5, 0.5e-5, 0.1e-5]) 
        
        lra_penalty = k_coeffs[None, :] * (dist_from_center_mm[:, None] ** 2)
        od_map_corr = od_map * (1.0 - lra_penalty[None, :, :] * od_map)
        
        return np.maximum(od_map_corr, 0.0)


    def _estimate_od_noise_map(self, raw_od_map: np.ndarray, filtered_od_map: np.ndarray) -> np.ndarray:
        """
        Becsli a lokális pixelenkénti OD mérési zajt a szűretlen és a mediánszűrt OD különbségéből,
        kiegészítve a kalibrációs alapzajjal (baseline scanner noise floor).
        """
        diff = np.abs(raw_od_map - filtered_od_map)
        # Levágjuk a meredek élekből/műtermékekből adódó kiugró különbségeket (max 0.015 OD zaj)
        diff_clipped = np.minimum(diff, 0.015)
        local_var = uniform_filter(diff_clipped ** 2, size=(5, 5, 1))
        
        noise_floor = np.array([
            min(self.calib_sigma_od.get('R', 0.003), 0.005),
            min(self.calib_sigma_od.get('G', 0.003), 0.005),
            min(self.calib_sigma_od.get('B', 0.004), 0.006)
        ])[None, None, :]
        
        return np.sqrt(np.maximum(local_var, noise_floor ** 2))


    @staticmethod
    def rational_func_od(d, a: float, b: float, c: float, e: float = 1.0):
        """
        Általánosított racionális (Hill/Mendez-Lewis) modell:
        OD(D) = a + (b * D^e) / (D^e + c^e)  (vagy c_eff = c^e kompatibilitással)
        """
        d_safe = np.maximum(d, 0.0)
        if e == 1.0:
            return a + (b * d_safe) / (d_safe + c)
        de = np.power(d_safe, e)
        return a + (b * de) / (de + c)


    @staticmethod
    def rational_func_od_3p(d, a: float, b: float, c: float):
        """Kifejezetten a scipy.optimize.curve_fit számára (csak 3 szabad paraméter)."""
        d_safe = np.maximum(d, 0.0)
        return a + (b * d_safe) / (d_safe + c)


    @staticmethod
    def rational_func_dod_dd(d, a: float, b: float, c: float, e: float = 1.0):
        """
        Analitikus parciális derivált a dózis szerint:
        d(OD)/dD = (b * c * e * D^(e-1)) / (D^e + c)^2
        """
        d_safe = np.maximum(d, 1e-4)
        if e == 1.0:
            return (b * c) / np.maximum((d_safe + c) ** 2, 1e-12)
        de = np.power(d_safe, e)
        num = b * c * e * np.power(d_safe, e - 1.0)
        den = (de + c) ** 2
        return num / np.maximum(den, 1e-12)


    @staticmethod
    def inv_rational_func_dose(od, params):
        """Inverz racionális függvény: OD -> Dózis [Gy] aszimptota-védelemmel."""
        a, b, c = params[:3]
        e = params[3] if len(params) > 3 else 1.0
        
        max_valid_od = a + b * 0.995
        od_safe = np.clip(od, a, max_valid_od)
        
        num = c * (od_safe - a)
        den = (a + b) - od_safe
        
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.maximum(num / np.maximum(den, 1e-12), 0.0)
            val = ratio if e == 1.0 else np.power(ratio, 1.0 / e)
        return np.nan_to_num(val, nan=0.0)


    @staticmethod
    def _canonical_method(method):
        if method == 'hybrid':  # Historical API alias for the vectorized solver.
            return 'fast'
        if method not in ('fast', 'admm', 'mayer'):
            raise ValueError(f'Unknown dose solver: {method}')
        return method


    def _solve_dose(self, od_map, method, od_noise_map=None, params_dict=None, dose_only=False, anchor_mask=None, diagnostics=None):
        solver = getattr(self, '_solver_' + self._canonical_method(method))
        kwargs = {'od_noise_map': od_noise_map}
        if self._canonical_method(method) != 'fast':
            kwargs.update(anchor_mask=anchor_mask, diagnostics=diagnostics)
        if dose_only: kwargs['dose_only'] = True
        if params_dict is not None:
            kwargs['params_dict'] = params_dict
        return solver(od_map, **kwargs)


    def _anchor_valid_mask(self, od_map, bounds=None, params_dict=None):
        """Select calibration-supported signal, independently of posterior draws.

        A mask controls thickness estimation/anchoring only, never clips doses.
        Bounds are normalized (left, top, right, bottom). Without bounds the
        whole scan is eligible, but nonphysical/saturated responses are not.
        Five percent OD headroom matches the existing thickness correction limit;
        it is a numerical eligibility rule, not an accuracy guarantee.
        """
        valid = np.all(np.isfinite(od_map) & (od_map >= 0), axis=-1)
        valid &= od_map[..., 0] > .015
        maximum = float(np.max(self.calib_doses)) if self.calib_doses is not None else None
        for i, ch in enumerate('RGB'):
            params = (self.calib_params or params_dict)[ch]
            ceiling = params[0] + .995*params[1]
            if maximum is not None:
                ceiling = min(ceiling, 1.05*float(self.rational_func_od(maximum, *params)))
            valid &= od_map[..., i] < ceiling
        if bounds is not None:
            bounds = np.asarray(bounds, dtype=float)
            if (bounds.shape != (4,) or not np.all(np.isfinite(bounds))
                    or np.any(bounds < 0) or np.any(bounds > 1)
                    or bounds[0] >= bounds[2] or bounds[1] >= bounds[3]):
                raise ValueError('Anchor bounds must be normalized left, top, right, bottom.')
            h, w = valid.shape
            x, y = np.arange(w)/w, np.arange(h)/h
            valid &= ((y >= bounds[1]) & (y < bounds[3]))[:, None]
            valid &= ((x >= bounds[0]) & (x < bounds[2]))[None, :]
        return valid

    def _anchor_mask(self, od_map, anchor_mask, params_dict=None):
        valid = self._anchor_valid_mask(od_map, params_dict=params_dict)
        if anchor_mask is not None:
            supplied = np.asarray(anchor_mask)
            if supplied.shape != valid.shape or supplied.dtype != bool:
                raise ValueError('Anchor mask must be a boolean array matching the dose map.')
            valid &= supplied
        return valid

    def _calibration_uncertainty_source(self):
        if all(ch in self.calib_samples and len(self.calib_samples[ch]) >= 2 for ch in 'RGB'):
            return 'parameter_samples'
        if all(ch in self.calib_cov for ch in 'RGB'):
            return 'parameter_covariance'
        return 'unavailable'


    def _compute_calibration_std_map(self, od_map, dose_map, method='fast', od_noise_map=None, median_filtering=True, workers=1, anchor_mask=None):
        """Propagate parameters through the selected solver AND final median filter.

        Parameter samples are global per channel, shared across all pixels.
        Online sample variance uses O(H*W) memory, not a stack of whole images.
        Without samples, use the within-channel parameter covariance and a
        finite-difference Jacobian of the same estimator. Cross-channel parameter
        covariance is unavailable in the current calibration format.
        """
        source = self._calibration_uncertainty_source()
        if source == 'unavailable':
            warnings.warn('Calibration uncertainty unavailable: provide parameter samples or covariance. '
                          'The output STD omits this component.')
            return np.zeros(dose_map.shape)

        if self._canonical_method(method) != 'fast':
            anchor_mask = self._anchor_mask(od_map, anchor_mask)

        def evaluate(params):
            candidate, _ = self._solve_dose(od_map, method, od_noise_map, params, dose_only=True,
                                            anchor_mask=anchor_mask)
            if median_filtering:
                candidate = median_filter(candidate, size=3)
            if candidate.shape != dose_map.shape or not np.all(np.isfinite(candidate)):
                raise ValueError('Nonfinite dose during calibration uncertainty propagation.')
            return candidate

        if source == 'parameter_samples':
            count = min(len(self.calib_samples[ch]) for ch in 'RGB')
            mean = np.zeros(dose_map.shape)
            m2 = np.zeros(dose_map.shape)
            worker_count = calibration_worker_count(dose_map.shape, workers, count)
            parameters = ({ch: self.calib_samples[ch][i] for ch in 'RGB'} for i in range(count))
            for i, candidate in enumerate(ordered_realizations(evaluate, parameters, worker_count)):
                delta = candidate - mean
                mean += delta / (i + 1)
                m2 += delta * (candidate - mean)
            return np.sqrt(np.maximum(m2 / (count - 1), 0))

        variance = np.zeros(dose_map.shape)
        for ch in 'RGB':
            parameters = self.calib_params[ch]
            covariance = self.calib_cov[ch]
            derivatives = []
            for j, value in enumerate(parameters):
                if not np.any(covariance[j]):
                    derivatives.append(np.zeros(dose_map.shape))
                    continue
                step = 1e-5 * max(abs(value), 1.0)
                if j > 0:
                    step = min(step, value / 4)
                plus = {key: np.array(params, dtype=float, copy=True) for key, params in self.calib_params.items()}
                minus = {key: np.array(params, dtype=float, copy=True) for key, params in self.calib_params.items()}
                plus[ch][j] += step
                minus[ch][j] -= step
                derivatives.append((evaluate(plus) - evaluate(minus)) / (2 * step))
            jacobian = np.stack(derivatives)
            variance += np.einsum('ihw,ij,jhw->hw', jacobian, covariance, jacobian)
        return np.sqrt(np.maximum(variance, 0))


    def _solver_fast(self, od_map: np.ndarray, od_noise_map: np.ndarray = None, params_dict: dict = None, dose_only=False):
        """
        Fast vectorized solver returning BOTH (dose_map, std_fit_map).
        """
        if params_dict is None:
            params_dict = self.calib_params
        dr = self.inv_rational_func_dose(od_map[:, :, 0], params_dict['R'])
        dg = self.inv_rational_func_dose(od_map[:, :, 1], params_dict['G'])
        
        weight_r = np.clip((10.0 - dr) / 4.0, 0.0, 1.0)
        weight_g = 1.0 - weight_r
        dose_map = np.nan_to_num(weight_r * dr + weight_g * dg, nan=0.0)
        if dose_only: return dose_map, None
        
        if od_noise_map is None:
            od_noise_map = np.full_like(od_map, 0.003)
            
        # Analitikus hibaterjedés: sigma_D = |dD/dOD| * sigma_OD + csatorna-eltérés
        dod_dd_r = self.rational_func_dod_dd(dr, *params_dict['R'])
        dod_dd_g = self.rational_func_dod_dd(dg, *params_dict['G'])
        
        std_r = od_noise_map[:, :, 0] / np.maximum(dod_dd_r, 1e-4)
        std_g = od_noise_map[:, :, 1] / np.maximum(dod_dd_g, 1e-4)
        
        # Ha a két csatorna (R és G) eltérő dózist mond az átmeneti zónában, az növeli a bizonytalanságot
        channel_discrepancy = weight_r * weight_g * np.abs(dr - dg)
        # Channel weights depend on the red dose; include their derivative.
        dw_r = np.where((dr > 6.0) & (dr < 10.0), -0.25, 0.0)
        jac_r = weight_r + (dr - dg) * dw_r
        std_fit_map = np.sqrt((jac_r * std_r)**2 + (weight_g * std_g)**2 + channel_discrepancy**2)
        
        return dose_map, np.nan_to_num(std_fit_map, nan=0.0)


    def _solver_admm(self, od_map: np.ndarray, od_noise_map: np.ndarray = None, params_dict: dict = None, max_iter: int = 6, dose_only=False, anchor_mask=None, diagnostics=None):
        """
        Plug-and-Play ADMM (Alternating Direction Method of Multipliers) Solver.
        Globálisan, iteratívan választja szét a Dózist és a Vastagságot:
        - Dózis (D): L1 Total Variation (TV) regularizáció (megőrzi az éles LWFA profilokat és csúcsokat).
        - Vastagság (t): L2 Makroszkopikus simítás (fizikai folytonosság).
        Runtime depends on map size, iteration count and propagated parameter samples.
        """
        if params_dict is None: params_dict = self.calib_params
        if od_noise_map is None: od_noise_map = np.full_like(od_map, 0.003)
        if max_iter < 1:
            raise ValueError('ADMM requires at least one iteration.')

        pR, pG, pB = params_dict['R'], params_dict['G'], params_dict['B']

        # Warm start a fast solver alapjaival
        dR = self.inv_rational_func_dose(od_map[:, :, 0], pR)
        dG = self.inv_rational_func_dose(od_map[:, :, 1], pG)
        dB = self.inv_rational_func_dose(od_map[:, :, 2], pB)

        wR = np.clip((10.0 - dR) / 4.0, 0.0, 1.0)
        wG = 1.0 - wR
        max_calib_d = float(np.max(self.calib_doses)) if self.calib_doses is not None else 15.0
        wB = np.clip((dG - 30.0) / 20.0, 0.0, 0.25) if max_calib_d >= 30.0 else np.zeros_like(dG)
        wG = np.maximum(0.0, wG - wB)

        D = np.nan_to_num(wR * dR + wG * dG + wB * dB, nan=0.0)
        t = np.ones_like(D)
        valid_anchor = self._anchor_mask(od_map, anchor_mask, params_dict)
        core_weights = thickness_anchor_weights(D, valid_anchor)

        # ADMM Dual & Segédváltozók
        Z_D = np.copy(D)
        U_D = np.zeros_like(D)
        rho = 20.0  # ADMM büntetőparaméter a TV csatoláshoz
        tv_weight = 0.05  # Erős, de éleket megtartó simítás

        signal_mask = od_map[:, :, 0] > 0.015

        noise_variance = np.maximum(od_noise_map ** 2, 1e-6)
        for _ in range(max_iter):
            # 1. Gradiens Lépés (Adathűség - Data Fidelity)
            dod_dt = np.stack([
                self.rational_func_od(D, *pR),
                self.rational_func_od(D, *pG),
                self.rational_func_od(D, *pB)
            ], axis=-1)

            od_pred = dod_dt * t[..., None]

            dod_dD = np.stack([
                self.rational_func_dod_dd(D, *pR) * t,
                self.rational_func_dod_dd(D, *pG) * t,
                self.rational_func_dod_dd(D, *pB) * t
            ], axis=-1)
            
            ch_reliability = np.stack([
                np.where(D < 8.0, 1.0, np.exp(-(D - 8.0) / 3.0)),
                np.where(D > 1.5, 1.0, 0.5),
                np.where(D > 2.0, 0.8, 0.3)
            ], axis=-1)
            W = ch_reliability / noise_variance

            res = od_map - od_pred
            H_DD = np.sum(W * (dod_dD ** 2), axis=-1)
            H_tt = np.sum(W * (dod_dt ** 2), axis=-1)
            
            g_D = np.sum(W * res * dod_dD, axis=-1)
            g_t = np.sum(W * res * dod_dt, axis=-1)

            # 2. Dózis (D) frissítése ADMM momentummal
            D_unreg = D + g_D / np.maximum(H_DD, 1e-6)
            D = (H_DD * D_unreg + rho * (Z_D - U_D)) / (H_DD + rho)
            D = np.where(signal_mask, np.maximum(D, 0.0), 0.0)

            # 3. Vastagság (t) frissítése Proximal L2 operátorral
            t_unreg = t + g_t / np.maximum(H_tt, 1e-6)
            t_smooth = masked_uniform_mean(t_unreg, valid_anchor, size=15, fill=1.)
            w_sum = np.sum(core_weights)
            mean_t_core = np.sum(t_smooth * core_weights) / w_sum if w_sum > 0 else 1.0
            t = np.where(valid_anchor, np.clip(t_smooth - mean_t_core + 1.0, 0.95, 1.05), 1.0)

            # 4. Z-update (Proximal Total Variation operátor a nyaláb profiljára)
            # The dual offset can make the unconstrained TV result negative,
            # even when the preceding D update was nonnegative. Enforce the
            # physical dose constraint on this state too, before the dual update.
            Z_D = np.maximum(denoise_tv_chambolle(D + U_D, weight=tv_weight), 0.0)

            # 5. U-update (Dual Multiplier)
            U_D = U_D + D - Z_D

        record_anchor_diagnostics(diagnostics, valid_anchor, core_weights, t, .05)
        if diagnostics is not None: diagnostics['iterations'] = max_iter
        if dose_only: return Z_D, None

        # Re-evaluate the local information at the returned state, including
        # dose/thickness coupling. This is a local approximation, not the full
        # covariance of the spatially regularized nonlinear solver.
        dod_dD = np.stack([self.rational_func_dod_dd(Z_D, *p) * t for p in (pR, pG, pB)], axis=-1)
        dod_dt = np.stack([self.rational_func_od(Z_D, *p) for p in (pR, pG, pB)], axis=-1)
        reliability = np.stack([
            np.exp(-np.maximum(Z_D - 8.0, 0) / 3),
            np.where(Z_D > 1.5, 1.0, 0.5),
            np.where(Z_D > 2, 0.8, 0.3),
        ], axis=-1)
        W = reliability / np.maximum(od_noise_map**2, 1e-6)
        H_DD = np.sum(W * dod_dD**2, axis=-1)
        H_Dt = np.sum(W * dod_dD * dod_dt, axis=-1)
        H_tt = np.sum(W * dod_dt**2, axis=-1)
        std_noise = np.sqrt(coupled_dose_variance(H_DD, H_Dt, H_tt))
        
        od_corr = od_map / t[:, :, None]
        dR_c = self.inv_rational_func_dose(od_corr[:, :, 0], pR)
        dG_c = self.inv_rational_func_dose(od_corr[:, :, 1], pG)
        dB_c = self.inv_rational_func_dose(od_corr[:, :, 2], pB)
        ch_disc = wR * wG * np.abs(dR_c - dG_c) + wG * wB * np.abs(dG_c - dB_c)
        
        std_fit_map = np.sqrt(std_noise ** 2 + ch_disc ** 2)
        return Z_D, std_fit_map


    def _solver_mayer(self, od_map: np.ndarray, od_noise_map: np.ndarray = None, params_dict: dict = None, dose_only=False, anchor_mask=None, diagnostics=None):
        """
        Locally linearized, prior-regularized multichannel thickness estimator.
        The final dose uses the existing dose-dependent channel blend; absolute
        accuracy must be validated against independent reference films.
        - A B csatornát elsődlegesen a rétegvastagság (delta_t) szeparálására használja,
          és csak a G csatorna valódi telítődése (>30 Gy) felett engedi közvetlenül a dózisba.
        - Dózis-súlyozott nullapont-horgonyzást alkalmaz a vastagságtérképre a nyalábmagban.
        """
        if params_dict is None:
            params_dict = self.calib_params
        if od_noise_map is None:
            od_noise_map = np.full_like(od_map, 0.003)

        pR, pG, pB = params_dict['R'], params_dict['G'], params_dict['B']

        # 1. Csatornánkénti direkt inverzió
        dR = self.inv_rational_func_dose(od_map[:, :, 0], pR)
        dG = self.inv_rational_func_dose(od_map[:, :, 1], pG)
        dB = self.inv_rational_func_dose(od_map[:, :, 2], pB)

        # Dinamikus csatornasúlyok az alapdózishoz (konzisztens a fast és hybrid solverekkel!):
        # - 0..8 Gy: R dominál, 6..10 Gy között sima átmenet G-be
        wR = np.clip((10.0 - dR) / 4.0, 0.0, 1.0)
        wG = 1.0 - wR
        
        # A B csatornát CSAK extrém magas dózisnál (>30 Gy, ahol már a G is telítődik)
        # és csak kis súllyal keverjük a bázisba, hogy ne húzza le a 10-30 Gy-es tartományt!
        max_calib_d = float(np.max(self.calib_doses)) if self.calib_doses is not None else 15.0
        if max_calib_d >= 30.0:
            wB = np.clip((dG - 30.0) / 20.0, 0.0, 0.25)
            wG = np.maximum(0.0, wG - wB)
        else:
            wB = np.zeros_like(dG)

        # Torzításmentes bázis dózistérkép (pontosan egyezik a fast solver skálájával <30 Gy-ig)
        D_base = np.nan_to_num(wR * dR + wG * dG + wB * dB, nan=0.0)

        valid_anchor = self._anchor_mask(od_map, anchor_mask, params_dict)
        core_weights = thickness_anchor_weights(D_base, valid_anchor)

        # 2. Fizikai modell predikció és Jacobi-deriváltak a D_base munkapontban
        od_pred = np.stack([
            self.rational_func_od(D_base, *pR),
            self.rational_func_od(D_base, *pG),
            self.rational_func_od(D_base, *pB)
        ], axis=-1)
        
        dod_dD = np.stack([
            self.rational_func_dod_dd(D_base, *pR),
            self.rational_func_dod_dd(D_base, *pG),
            self.rational_func_dod_dd(D_base, *pB)
        ], axis=-1)

        # 3. Egzakt Bayes-i súlyozás az OD-térben (1 / sigma_OD^2)
        # A B csatorna itt kap kulcsszerepet: mivel alacsony az érzékenysége a dózisra (kicsi dod_dD),
        # de érzékeny a rétegvastagságra (od_pred), kiválóan megköti a delta_t-t!
        ch_reliability = np.stack([
            np.where(D_base < 8.0, 1.0, np.exp(-(D_base - 8.0) / 3.0)), # R telítődik >10 Gy felett
            np.where(D_base > 1.5, 1.0, 0.5),                          # G a fő munkacsatorna
            np.where(D_base > 2.0, 0.8, 0.3)                           # B stabilizálja a vastagságot
        ], axis=-1)
        
        W_ch = ch_reliability / np.maximum(od_noise_map ** 2, 1e-6)
        res = od_map - od_pred

        # 2x2-es Gauss-Newton Hesse-mátrix (H) és gradiens (g) a (delta_D, delta_t) rendszerre:
        # Modell: OD_meas ≈ OD_pred(D_base) + dod_dD * delta_D + OD_pred(D_base) * delta_t
        H_DD = np.sum(W_ch * (dod_dD ** 2), axis=-1)
        H_Dt = np.sum(W_ch * dod_dD * od_pred, axis=-1)
        
        # JAVÍTVA: A W_ch már tartalmazza az 1/sigma_OD^2 szorzót, így a vastagság priorja
        # egyszerűen 1 / sigma_t^2 (ahol sigma_t = 0.015, azaz 1.5% gyári EBT3 vastagságtűrés).
        sigma_t_prior = 0.015
        H_tt = np.sum(W_ch * (od_pred ** 2), axis=-1) + (1.0 / (sigma_t_prior ** 2))

        g_D = np.sum(W_ch * res * dod_dD, axis=-1)
        g_t = np.sum(W_ch * res * od_pred, axis=-1)

        # Cramer-szabály a csatolt (delta_D, delta_t) rendszerre
        det = np.maximum(H_DD * H_tt - H_Dt ** 2, 1e-6)
        delta_t_raw = (H_DD * g_t - H_Dt * g_D) / det
        
        # Csak ott becsülünk vastagságot, ahol van érdemi jel (OD > 0.015)
        signal_mask = od_map[:, :, 0] > 0.015
        delta_t_raw = np.where(signal_mask, np.clip(delta_t_raw, -0.035, 0.035), 0.0)

        # 4. MULTI-SCALE VASTAGSÁG-KORREKCIÓ (Finom homogenitás + Csúcsdózis-védelem)
        # a) Enyhe (3x3 px) szűrés a nyers delta_t-n: megtartja a valódi (0.2-1.0 mm-es) 
        #    filmvastagság-ingadozásokat, így a térbeli STD ~1.1%-ról ~0.5-0.7%-ra csökken!
        delta_t_local = masked_uniform_mean(delta_t_raw, valid_anchor, size=3)
        
        # b) Makroszkopikus (35x35 px ~ 3 mm) trend: ezzel biztosítjuk, hogy a korrekció
        #    ne tudja "ellapítani" egy szűk LWFA nyaláb valódi dóziscsúcsát!
        delta_t_macro = masked_uniform_mean(delta_t_raw, valid_anchor, size=35)
        
        # A nyalábmag súlyozott átlagát levonjuk a makroszkopikus trendből (nullapont-horgonyzás)
        beam_core_weights = core_weights
        weight_sum = np.sum(beam_core_weights)
        if weight_sum > 0:
            mean_dt_core = np.sum(delta_t_macro * beam_core_weights) / weight_sum
        else:
            mean_dt_core = 0.0
            
        # Kombináció: a nagyfrekvenciás film-inhomogenitást (local - macro) 100%-ban korrigáljuk,
        # a makroszkopikus vastagságtrendet pedig óvatosan (50%-os súllyal) a horgonyzott nullapont körül:
        delta_t_smooth = (delta_t_local - delta_t_macro) + 0.5 * (delta_t_macro - mean_dt_core)
        delta_t_smooth = np.where(valid_anchor, np.clip(delta_t_smooth, -0.03, 0.03), 0.0)
        record_anchor_diagnostics(diagnostics, valid_anchor, core_weights, 1.+delta_t_smooth, .03)

        # 5. Korrigált dózis kiszámítása:
        # Miután a simított delta_t_smooth ismert, az optimális dózis-frissítés analitikusan:
        # delta_D = (g_D - H_Dt * delta_t_smooth) / H_DD, vagy ekvivalensen a korrigált OD-ból:
        od_corr = od_map / (1.0 + delta_t_smooth[:, :, None])
        dR_c = self.inv_rational_func_dose(od_corr[:, :, 0], pR)
        dG_c = self.inv_rational_func_dose(od_corr[:, :, 1], pG)
        dB_c = self.inv_rational_func_dose(od_corr[:, :, 2], pB)
        
        dose_map = np.nan_to_num(wR * dR_c + wG * dG_c + wB * dB_c, nan=0.0)
        if dose_only: return dose_map, None

        # 6. Local curvature approximation, not an exact spatial posterior.
        # A 2x2-es inverz Hesse-mátrix (kovariancia) [0,0] eleme adja a vastagság-bizonytalansággal
        # növelt dózis-varianciát: Var(D) = H_tt / det
        var_D_cramer = coupled_dose_variance(H_DD, H_Dt, H_tt)
        std_noise = np.sqrt(np.maximum(var_D_cramer, 0.0))
        
        # Csatornák közötti maradvány-eltérés a vastagság-korrekció után
        ch_disc = wR * wG * np.abs(dR_c - dG_c) + wG * wB * np.abs(dG_c - dB_c)
        std_fit_map = np.sqrt(std_noise ** 2 + ch_disc ** 2)
        return dose_map, std_fit_map




def thickness_anchor_weights(dose, valid):
    """Fixed beam-core weights; no single pixel/global maximum sets the scale."""
    candidates = np.asarray(dose)[valid]
    if not candidates.size: return np.zeros_like(dose, dtype=float)
    peak = float(np.percentile(candidates, 99))
    if peak <= 0: return np.zeros_like(dose, dtype=float)
    bounded = np.clip(dose/peak, 0., 1.)
    return np.where(valid & (dose >= .5*peak), bounded**2, 0.)


def masked_uniform_mean(values, valid, size, fill=0.):
    """Normalized convolution: excluded pixels cannot enter thickness trends."""
    numerator = uniform_filter(np.where(valid, values, 0.), size=size)
    denominator = uniform_filter(valid.astype(float), size=size)
    return np.divide(numerator, denominator, out=np.full_like(numerator, fill), where=denominator > 1e-12)


def record_anchor_diagnostics(target, valid, weights, thickness, limit):
    if target is None: return
    values = thickness[valid]
    target.update(anchor_model_version=1, anchor_status='available' if np.any(weights) else 'unavailable',
                  valid_pixels=int(valid.sum()), excluded_pixels=int(valid.size-valid.sum()),
                  anchor_pixels=int(np.count_nonzero(weights)),
                  thickness_min=float(values.min()) if values.size else None,
                  thickness_max=float(values.max()) if values.size else None,
                  thickness_at_limit_fraction=float(np.mean(np.abs(values-1.) >= limit-1e-9)) if values.size else 0.)


def calibration_worker_count(shape, requested, samples):
    """Bound independent ADMM work by a conservative 256 MiB scratch budget."""
    if type(requested) is not int or not 1 <= requested <= 8:
        raise ValueError('Uncertainty workers must be in 1..8.')
    bytes_per_worker = max(1, int(np.prod(shape))*8*64)
    return max(1, min(requested, samples, (256*1024**2)//bytes_per_worker))


def ordered_realizations(evaluate, parameters, workers):
    """Bound outstanding maps and retain serial accumulation order exactly."""
    if workers == 1:
        yield from map(evaluate, parameters)
        return
    iterator = iter(parameters)
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix='prism-calibration') as pool:
        pending = deque()
        for _ in range(workers):
            item = next(iterator, None)
            if item is None: break
            pending.append(pool.submit(evaluate, item))
        try:
            while pending:
                candidate = pending.popleft().result()
                yield candidate
                item = next(iterator, None)
                if item is not None: pending.append(pool.submit(evaluate, item))
        finally:
            for future in pending: future.cancel()
