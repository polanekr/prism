import numpy as np
import matplotlib.pyplot as plt
import pymc as pm
import pandas as pd
from scipy.interpolate import UnivariateSpline
import warnings

warnings.filterwarnings("ignore")

class FullPDDBayesianFitter:
    def __init__(self, z_meas, dose_meas, mode='electron'):
        """
        mode: 'electron' (Sigmoid) vagy 'photon' (Exp + Buildup)
        """
        z_arr = np.array(z_meas, dtype=float)
        d_arr = np.array(dose_meas, dtype=float)
        sort_idx = np.argsort(z_arr)
        
        self.z_meas = z_arr[sort_idx]
        self.dose_meas = d_arr[sort_idx]
        self.mode = mode
        
        self.model = None
        self.trace = None
        self.priors = {}
        self.ssd = None

    @classmethod
    def create_from_physical_params(cls, z_meas, d_meas, E0_MeV, E0_std_MeV=None, SSD_mm=None, normalize=True, mode_override=None):
        if mode_override:
            mode = mode_override
        else:
            mode = 'photon' if E0_MeV < 5.0 else 'electron'
            
        print(f"\n=== FIZIKAI MODELL: {mode.upper()} (E={E0_MeV:.2f} +/- {E0_std_MeV if E0_std_MeV else 0:.1f} MeV) ===")
        
        if z_meas is None or len(z_meas) == 0: raise ValueError("Üres adat!")

        # SKÁLÁZÁS
        max_d = np.max(d_meas) if np.max(d_meas) > 0 else 1.0
        d_meas_scaled = d_meas if max_d < 5.0 else (d_meas / max_d) * 100.0

        fitter = cls(z_meas, d_meas_scaled, mode=mode)
        fitter.ssd = SSD_mm 

        # --- PRIOROK (DINAMIKUS SZÓRÁSSAL) ---
        if mode == 'electron':
            # R50 becslése az energiából (E = 2.33 * R50)
            # R50 [mm] = (E [MeV] / 2.33) * 10
            R50_prior_mu = (E0_MeV / 2.33) * 10.0
            
            # --- AZ ÚJ LOGIKA ITT VAN: ---
            if E0_std_MeV is not None and E0_std_MeV > 0:
                # Hibaterjedés: Ha E bizonytalan, akkor R50 is az.
                # sigma_R50 = (sigma_E / 2.33) * 10.0
                R50_sigma = (E0_std_MeV / 2.33) * 10.0
                # Biztonsági minimum: ne legyen túl szűk a prior
                R50_sigma = max(R50_sigma, 5.0)
                print(f"  -> Prior lazítása: R50 sigma = {R50_sigma:.1f} mm (Energia bizonytalanság miatt)")
            else:
                # Alapértelmezett, ha nincs megadva szórás
                R50_sigma = 10.0

            fitter.priors = {
                'A': 100.0, 
                'R50': R50_prior_mu, 
                'R50_sigma': R50_sigma, # Itt használjuk a kiszámolt szórást
                'k': 0.5,
                'surf_dose': 0.8, 'buildup_rate': 0.1, 'tail': 1.0,
                'k_sigma': 0.5
            }
        else:
            # --- PHOTON PRIOROK ---
            fitter.priors = {
                'A': 100.0,           
                'mu': 0.03,           
                'surf_dose': 0.98,
                'buildup_rate': 10.0
            }

        return fitter

    @staticmethod
    def _model_func_numpy(z, params, mode, ssd=None):
        if mode == 'electron':
            A = params['A']; R50 = params['R50']; k = params['k']
            surf = params['surf_dose']; b_rate = params['buildup_rate']; tail = params['tail_bg']
            
            arg = np.clip(k * (z - R50), -50, 50)
            sigmoid = 1.0 / (1.0 + np.exp(arg))
            buildup = 1.0 - (1.0 - surf) * np.exp(-abs(b_rate) * z)
            dose = A * sigmoid * buildup + tail
            
        else: # PHOTON
            A = params['A']
            mu = params['mu']
            surf = params.get('surf_dose', 1.0)
            b_rate = params.get('buildup_rate', 10.0)
            
            # Buildup és Attenuáció
            buildup_term = 1.0 - (1.0 - surf) * np.exp(-abs(b_rate) * z)
            attenuation = np.exp(-mu * z)
            dose = A * buildup_term * attenuation

        if ssd is not None and ssd > 0:
            isl_factor = (ssd / (ssd + z))**2
            return dose * isl_factor
        else:
            return dose

    def build_model(self):
        print(f"PyMC Modell építése ({self.mode})...")
        
        with pm.Model() as self.model:
            sigma_noise = pm.HalfNormal('sigma_noise', sigma=5.0)

            if self.mode == 'electron':
                A = pm.Normal('A', mu=self.priors['A'], sigma=5.0)
                R50 = pm.Normal('R50', mu=self.priors['R50'], sigma=self.priors['R50_sigma'])
                k = pm.TruncatedNormal('k', mu=self.priors['k'], sigma=self.priors['k_sigma'], lower=0.001)
                surf = pm.Normal('surf_dose', mu=self.priors['surf_dose'], sigma=0.2)
                buildup = pm.TruncatedNormal('buildup_rate', mu=self.priors['buildup_rate'], sigma=0.1, lower=0)
                tail = pm.TruncatedNormal('tail_bg', mu=self.priors['tail'], sigma=5.0, lower=0)
                
                sigmoid = 1.0 / (1.0 + pm.math.exp(k * (self.z_meas - R50)))
                buildup_term = 1.0 - (1.0 - surf) * pm.math.exp(-buildup * self.z_meas)
                mu_dose_raw = A * sigmoid * buildup_term + tail

            else:
                # --- PHOTON MODELL ---
                A = pm.Normal('A', mu=self.priors['A'], sigma=10.0)
                mu = pm.TruncatedNormal('mu', mu=self.priors['mu'], sigma=0.05, lower=0.001)
                
                # JAVÍTÁS: upper=1.2 engedi, hogy a felszín > 100% legyen a fitben (matematikailag),
                # amit a normalizálás majd helyretesz.
                surf = pm.TruncatedNormal('surf_dose', mu=self.priors['surf_dose'], sigma=0.1, lower=0.5, upper=1.2)
                b_rate = pm.TruncatedNormal('buildup_rate', mu=self.priors['buildup_rate'], sigma=3.0, lower=0.5)
                
                buildup_term = 1.0 - (1.0 - surf) * pm.math.exp(-b_rate * self.z_meas)
                attenuation = pm.math.exp(-mu * self.z_meas)
                
                mu_dose_raw = A * buildup_term * attenuation

            if self.ssd is not None and self.ssd > 0:
                isl = (self.ssd / (self.ssd + self.z_meas))**2
                mu_dose = mu_dose_raw * isl
            else:
                mu_dose = mu_dose_raw
            
            pm.StudentT('Y_obs', nu=3, mu=mu_dose, sigma=sigma_noise, observed=self.dose_meas)

    def run_fit(self, draws=4000, chains=4, tune=1000):
        if self.model is None: raise ValueError("Hívd meg a build_model()-t!")
        with self.model:
            self.trace = pm.sample(draws=draws, tune=tune, chains=chains, target_accept=0.99, progressbar=True)
            
    def predict(self, z_depths):
        """
        Bayes-i predikció tetszőleges mélységekre a posterior eloszlás alapján.
        
        Args:
            z_depths (np.array): Mélység értékek [mm]
            
        Returns:
            mean_curve (np.array): Az illesztett görbék átlaga
            std_curve (np.array): A bizonytalanság (szórás)
        """
        if self.trace is None:
            raise ValueError("Nincs illesztés (trace). Futtasd a run_fit()-et előbb!")
            
        # 1. Adatok előkészítése Broadcastinghoz
        # A bemeneti z-t sorvektorrá alakítjuk: (1, M)
        z_grid = np.array(z_depths).reshape(1, -1)
        
        # 2. Posterior minták kinyerése
        post = self.trace.posterior
        # Összefűzzük a láncokat (chains) és mintákat (draws) egy dimenzióba -> (N_samples)
        stacked = post.stack(sample=("chain", "draw"))
        
        # 3. Paraméterek kinyerése szótárba
        # Minden paramétert (N_samples, 1) alakra hozunk, hogy a numpy broadcasting
        # automatikusan elvégezze a mátrixszorzást a (1, M) alakú z_grid-del.
        params_map = {}
        
        if self.mode == 'electron':
            param_names = ['A', 'R50', 'k', 'surf_dose', 'buildup_rate', 'tail_bg']
        else:
            param_names = ['A', 'mu', 'surf_dose', 'buildup_rate']
            
        for name in param_names:
            # Értékek kinyerése és formázása (N, 1)
            values = stacked[name].values  # Shape: (N_samples,)
            params_map[name] = values[:, np.newaxis]
            
        # 4. Dózis számítása minden mintára (Vectorized)
        # Eredmény shape: (N_samples, M_depths)
        # A már meglévő _model_func_numpy függvényt használjuk újra!
        # Mivel a params_map vektorokat tartalmaz, a numpy automatikusan mátrixot számol.
        d_pred_matrix = self._model_func_numpy(z_grid, params_map, mode=self.mode, ssd=self.ssd)
        
        # 5. Átlag és Szórás számítása oszloponként (minden mélységpontra)
        mean_curve = np.mean(d_pred_matrix, axis=0)
        std_curve = np.std(d_pred_matrix, axis=0)
        
        return mean_curve, std_curve

    def calculate_dosimetric_params(self):
        if self.trace is None: return None
        post = self.trace.posterior
        
        if self.mode == 'electron':
            vars = ['A', 'R50', 'k', 'surf_dose', 'buildup_rate', 'tail_bg']
        else:
            vars = ['A', 'mu', 'surf_dose', 'buildup_rate']
            
        p = {var: float(post[var].mean()) for var in vars}
        
        z_dense = np.linspace(0, np.max(self.z_meas) * 1.5, 500)
        d_dense = self._model_func_numpy(z_dense, p, self.mode, ssd=self.ssd)
        
        # JAVÍTÁS: A normalizálási alap a MÉRT maximum, nem a FIT maximum
        # Ez biztosítja, hogy a ploton a legmagasabb pontod 100%-on legyen.
        norm_base = np.max(self.dose_meas) 
        d_norm = (d_dense / norm_base) * 100.0
        
        return {'Params': p, 'Norm_Base': norm_base}, (z_dense, d_norm)

    def plot_results(self):
        if self.trace is None: return
        res, (z_fit, d_fit) = self.calculate_dosimetric_params()
        
        # JAVÍTÁS: Konzisztens normalizálás
        norm_base = res['Norm_Base']
        d_meas_rescaled = (self.dose_meas / norm_base) * 100.0
        
        plt.figure(figsize=(10, 6))
        
        # Fit görbe
        plt.plot(z_fit, d_fit, 'b-', linewidth=2, label=f"Bayes Fit ({self.mode.upper()})")
        
        # Mért pontok (Most már a max = 100%)
        plt.errorbar(self.z_meas, d_meas_rescaled, yerr=2.0, fmt='ro', label='Mért Adatok', zorder=5, alpha=0.7)
        
        plt.title(f"Dózis Illesztés ({self.mode})")
        plt.xlabel("Mélység [mm]")
        plt.ylabel("Relatív Dózis [%]")
        
        # 100% vonal
        plt.axhline(100, color='gray', linestyle='--', alpha=0.5)
        
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()