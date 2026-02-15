# PRISM Keretrendszer - Részletes Felhasználói Kézikönyv

**Verzió:** 1.0.0
**Dátum:** 2024
**Szerzők:** PRISM Development Team

---

## Bevezetés

A **PRISM (Probabilistic Reconstruction of Inhomogeneous Systems Methodology)** egy Python-alapú keretrendszer, amelyet kifejezetten a komplex, térbelileg inhomogén sugárterek (pl. lézeres gyorsítók, FLASH radioterápia, GRID terápia) dozimetriájára és sugárbiológiai elemzésére fejlesztettünk.

A hagyományos módszerekkel ellentétben a PRISM nem egyetlen "átlagdózissal" számol, hanem:
1.  Rekonstruálja a teljes **3D dóziseloszlást** ritkás mérésekből.
2.  Képes kezelni a **spektrális változásokat** (Beam Hardening) és a **szórt sugárzást**.
3.  A biológiai modellezés során figyelembe veszi a dozimetriai bizonytalanságot (**Error-in-Variables** megközelítés), így pontosabb RBE és túlélési görbe becslést ad.

---

## Tartalomjegyzék

1.  [Telepítés és Környezet](#1-telepítés-és-környezet)
2.  [Fázis I: Film Dozimetria](#2-fázis-i-film-dozimetria)
    * [Kalibráció](#a-kalibráció)
    * [Mérések Feldolgozása](#b-mérések-feldolgozása)
3.  [Fázis II: 3D Dózisrekonstrukció](#3-fázis-ii-3d-dózisrekonstrukció)
    * [Térfogat Építése](#a-térfogat-építése)
    * [Illesztés és Skálázás](#b-illesztés-és-skálázás)
4.  [Fázis III: Fizikai Analízis és Vizualizáció](#4-fázis-iii-fizikai-analízis-és-vizualizáció)
    * [3D Megjelenítés](#a-3d-megjelenítés)
    * [PDD és Profil Analízis](#b-pdd-és-profil-analízis)
    * [Gamma Analízis](#c-gamma-analízis)
5.  [Fázis IV: Biológiai Modellezés](#5-fázis-iv-biológiai-modellezés)
    * [Sejttúlélés (LQ Modell)](#a-sejttúlélés-lq-modell)
    * [Hal/Embrió Túlélés (Weibull)](#b-halembrió-túlélés-weibull)
6.  [Fázis V: Statisztikai Validáció](#6-fázis-v-statisztikai-validáció)

---

## 1. Telepítés és Környezet

A PRISM futtatásához Python 3.9 vagy újabb verzió szükséges.

### Függőségek telepítése

A projekt gyökérmappájában található `requirements.txt` fájl tartalmazza az összes szükséges csomagot.

```bash
pip install -r requirements.txt

Főbb csomagok:

    numpy, scipy, pandas: Matematikai alapok.

    pymc, arviz: Bayes-i statisztika és MCMC mintavételezés.

    tifffile, opencv-python: Képfeldolgozás (Gafchromic filmekhez).

    plotly, matplotlib: Vizualizáció.

    scikit-learn: Gaussian Process regresszió (a 3D rekonstrukcióhoz).

2. Fázis I: Film Dozimetria

Ez a modul (prism.dosimetry) felelős a szkennelt Gafchromic EBT3 filmek (.tif) konvertálásáért fizikai dózistérképekké (.npy).
A. Kalibráció

A pontos méréshez elengedhetetlen a filmek kalibrálása ismert dózispontokon (pl. 0, 0.5, 2, 5, 10 Gy). A szkennelést RGB módban, legalább 300 DPI felbontással kell végezni.
Python

from prism.dosimetry import GafchromicEngine

# 1. Motor inicializálása (ide menti az eredményeket)
engine = GafchromicEngine(output_folder="data/processed")

# 2. Kalibráció futtatása
# A mappában lévő .tif fájlokat sorban olvassa be (névsorrendben!)
# Fontos: A fájlok száma egyezzen meg a doses_gy lista hosszával.
engine.run_calibration(
    calibration_folder="data/raw/calibration_scans",
    doses_gy=[0, 0.5, 1, 2, 5, 8, 10], 
    roi_size=50  # A ROI mérete pixelben az átlagoláshoz
)

# 3. Ellenőrzés (Validáció)
# Visszaszámolja a dózist az OD görbéből, hogy lássuk a hibát (RMSE).
engine.validate_calibration_curve(channel='red')

B. Mérések Feldolgozása

A kísérleti filmek konvertálása dózistérképekké.
Python

import glob

# Fájlok listázása
film_files = sorted(glob.glob("data/raw/measurements/*.tif"))
blank_film = "data/raw/calibration_scans/calib_0Gy.tif" # Referencia (Blank)

# Kötegelt feldolgozás
engine.process_films(
    file_list=film_files, 
    blank_path=blank_film,
    method='hybrid',    # 'hybrid': Precíz, iteratív solver (ajánlott)
                        # 'fast': Gyorsabb, vektorizált közelítés
    use_cleaning=True   # bFDR tisztítás bekapcsolása (karcok, porszemek eltávolítása)
)

Kimenet: A data/processed mappában létrejönnek a *_dose.npy (dózistérkép) és *_meta.json (metadata) fájlok.
3. Fázis II: 3D Dózisrekonstrukció

A prism.reconstruction modul építi fel a teljes 3D térfogatot a ritkás 2D mérésekből.
A. Térfogat Építése
Python

from prism.reconstruction import Bayesian3DVolumeReconstructor

# 1. Inicializálás
# pdd_folder: Ahol a feldolgozott .npy fájlok vannak
recon = Bayesian3DVolumeReconstructor(
    pdd_folder="data/processed", 
    ssd_mm=300.0,      # Forrás-Felszín távolság
    energy_mev=0.35    # Névleges energia (kezdeti becsléshez, opcionális)
)

# 2. Rekonstrukció (Gaussian Process Regression)
recon.build_volume(
    roi_size_mm=40.0,           # Mekkora területet rekonstruáljunk oldalirányban
    z_res_mm=0.2,               # Z (mélység) felbontás
    divergence_correction=True, # 1/r^2 korrekció alkalmazása
    perform_alignment=True      # A filmek automatikus középpontra igazítása
)

B. Illesztés és Skálázás

Ha a biológiai kísérletnél használtunk egy "Front Filmet" (a plate tetején) és egy "Back Filmet" (a plate alján), akkor a rekonstruált modellt ehhez kell igazítani.
Python

# Front film betöltése referenciának
front_film_map, meta, _ = recon.load_dose_map("data/processed/exp_front_film_dose.npy")

# Regisztráció és Skálázás
# Ez a lépés "ráhúzza" a modellt a valós mérésre (XY eltolás + Intenzitás skálázás)
recon.register_and_scale(
    front_film_map=front_film_map, 
    front_film_depth_mm=0.0  # A film pozíciója a fantomban (Z=0)
)

# Opcionális: Spektrális korrekció (Ha van Back Film is)
# Ez korrigálja az ék okozta sugárkeményedést (Beam Hardening)
back_film_map, _, _ = recon.load_dose_map("data/processed/exp_back_film_dose.npy")

# Maszk hozzáadása a korrekcióhoz (pl. a 96-well plate területe)
recon.add_monolayer_well_mask("Plate_Area", center_coords=(0,0,15), radius_mm=40)

recon.correct_spectrum_with_cavity(
    rear_film_map=back_film_map,
    rear_film_depth_mm=15.0, # A plate vastagsága
    mask_name="Plate_Area",
    gap_thickness_mm=10.0    # A folyadék/levegő réteg vastagsága
)

# Mentés későbbi használatra
recon.save_volume("results/reconstructed_volume.npz")

4. Fázis III: Fizikai Analízis és Vizualizáció

A rekonstruált térfogat fizikai validációja és megjelenítése (prism.analytics és prism.viz).
A. 3D Megjelenítés
Python

from prism.viz import plot_3d_interactive, plot_dvh, plot_gamma_map

# Interaktív 3D Isodose felület (Plotly alapú)
# Megnyit egy böngésző ablakot vagy notebook widgetet
fig = plot_3d_interactive(recon.volume, recon.coords, level=50) # 50%-os izodózis
fig.show()

# DVH (Dózis-Térfogat Hisztogram)
plot_dvh(recon.volume)

B. PDD és Profil Analízis
Python

from prism.analytics import DoseAnalyst

analyst = DoseAnalyst()

# PDD (Mélydózis görbe) kinyerése a középső tengely mentén
pdd_z = recon.coords['z']
pdd_dose = recon.volume[:, 100, 100] # [Z, Y, X] indexelés

# Paraméterek számítása (R50, Rp, E0)
metrics, curve = analyst.analyze_pdd(pdd_z, pdd_dose)

print(f"R50 (félérték réteg): {metrics['R50_mm']:.2f} mm")
print(f"Becsült Energia (E0): {metrics['E0_MeV']:.2f} MeV")

C. Gamma Analízis

Összehasonlítás egy referencia (pl. Monte Carlo) szimulációval vagy másik méréssel.
Python

# Gamma Index számítása (3% / 3mm kritérium)
gamma_vals, pass_rate = analyst.calculate_gamma_index(
    ref_z=pdd_z, ref_d=pdd_dose, 
    eval_z=sim_z, eval_d=sim_dose, 
    dose_tol=3.0, dist_tol=3.0
)
print(f"Gamma Pass Rate: {pass_rate:.1f}%")

5. Fázis IV: Biológiai Modellezés

A rekonstruált 3D dózistérfogat összekapcsolása a biológiai adatokkal (prism.biology).
A. Sejttúlélés (LQ Modell)

Ez a modul a Linear-Quadratic modellt használja, de Error-in-Variables kiegészítéssel, ami figyelembe veszi, hogy a dózis nem egy pontszerű érték, hanem eloszlás.
Python

from prism.biology import CellSurvivalLQModel

# 1. Modell létrehozása
model = CellSurvivalLQModel(cell_line="U251") # Opciók: 'U251', 'HaCaT', 'Generic'

# 2. Adatok hozzáadása (Kísérletenként)
# Kinyerjük a dózisstatisztikát (hisztogramot) egy adott lyukra (pl. A1) a térfogatból
dvh_stats_A1 = recon.get_dvh_statistics(mask_name="Well_A1")

model.add_experiment_from_histogram(
    hist_data=dvh_stats_A1, 
    colony_count=42, 
    n_seeded=200
)

# ... További kísérletek hozzáadása ...

# 3. Bayes-i Mintavétel (MCMC)
idata = model.sample(draws=2000, chains=4)

# 4. Eredmények
model.plot_survival_curves()
ld50, hdi = model.calculate_ld50()
print(f"Számított LD50: {ld50:.2f} Gy")

B. Hal/Embrió Túlélés (Weibull)

Gradiens kísérletekhez, ahol a minták (pl. Zebradánió embriók) térben elosztva helyezkednek el egy 96-os lemezen.
Python

from prism.biology import FishSurvivalModel

fish_model = FishSurvivalModel()

# Adatok kinyerése a rekonstrukcióból (pl. egy teljes sor)
voxels_row_A, _ = recon.get_voxel_data("Well_Row_A")

# Kísérlet hozzáadása
# A napi halálozási adatokat kumulatív módon vagy napi bontásban is megadhatjuk
fish_model.add_experiment(
    dose_distribution=voxels_row_A,
    daily_deaths=[0, 1, 5, 2], # Halálok száma az 1., 2., 3., 4. napon
    n_start=20,                # Kezdő létszám
    observation_days=[1, 2, 3, 4]
)

# Mintavétel és analízis
fish_model.sample()

# Dózis-hatás görbe a 4. napon
fish_model.plot_dose_response(day=4) 

6. Fázis V: Statisztikai Validáció

A publikációhoz szükséges statisztikai tesztek.
Python

import arviz as az

# 1. Konvergencia Diagnosztika
# Ellenőrzi, hogy a Markov-láncok konvergáltak-e (R-hat < 1.01).
model.diagnose()

# 2. Modell Összehasonlítás (LOO - Leave-One-Out)
# Példa: Szignifikánsan jobb-e az LQ modell, mint egy egyszerű Lineáris modell?
# (Feltételezve, hogy futtattál egy 'linear_model'-t is)
comp = az.compare(
    {"Linear": linear_model.idata, "LQ": model.idata}, 
    ic="loo", 
    scale="deviance"
)
az.plot_compare(comp)
print(comp)

# 3. Posterior Predictive Check (PPC)
# Visszaellenőrzi, hogy a modell képes-e reprodukálni a mért adatokat.
model.check_posterior_predictive()

További segítség:
Ha kérdésed van, nyiss egy Issue-t a GitHub repóban, vagy vedd fel a kapcsolatot a fejlesztőkkel.