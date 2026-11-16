# PRISM

Radiochromic film dosimetry and measured beam-footprint analysis with a PyQt5 desktop interface.

- **Documentation:** [polanek.hu/en/software/prism/](https://polanek.hu/en/software/prism/)
- **Source code:** [github.com/polanekr/prism](https://github.com/polanekr/prism)
- **Current application:** desktop version **0.1.0**, Python 3.12–3.14, Windows x64 / Linux x86-64.
- **Scope:** calibrated 2D film dose maps, beam descriptors and the documented FRONT–BACK intermediate-plane sample-dose estimate. Full 3D reconstruction, proton-spectrum unfolding and biological response modelling are not implemented in this desktop version.
- **Earlier framework:** preserved on [`legacy/framework-2026-11-16`](https://github.com/polanekr/prism/tree/legacy/framework-2026-11-16); see [MIGRATION.md](MIGRATION.md). Its README and examples describe a different architecture.

## Obtain the current desktop application

```bash
git clone https://github.com/polanekr/prism.git
cd prism
uv sync --locked --all-extras
uv run --no-sync prism --check --require-bayes --require-excel
uv run --no-sync prism
```

For the reviewed release, use `git checkout v0.1.0` after cloning. Source installation is documented below and in the [English installation guide](https://polanek.hu/en/software/prism/installation/). No standalone Windows installer is supplied by this source release.

Gafchromic-film dozimetriai és nyalábanalízis alkalmazás PyQt5 felülettel.

## Telepítés Windowsra és Linuxra

Támogatott cél: **Windows 10/11 x64** és **Linux x86-64**, normál CPython
3.12–3.14 verzióval. Az ajánlott verzió **Python 3.13**, ezt a `.python-version`
is rögzíti. A Windows ARM64 és a szabad szálkezelésű Python nem része az
ellenőrzött telepítési célnak.

A projekt kizárólag **PyQt5**-öt használ. A Qt futtatókörnyezetet a PyQt5
függősége telepíti; PyQt6-ot vagy külön, kézzel kiválasztott Qt-verziót nem kell
hozzáadni. A `uv.lock` tartalmazza a Linux- és Windows-specifikus feloldást is.

### Ajánlott telepítés: uv

Telepíts legalább 0.12.1-es [uv-t a hivatalos útmutató szerint](https://docs.astral.sh/uv/getting-started/installation/),
majd nyiss új terminált, és lépj a projekt könyvtárába. Az uv szükség esetén
letölti a megfelelő Pythont is. A `.venv` operációs rendszerhez kötött:
Windows alatt új környezetet kell létrehozni, nem a Linuxos mappát átmásolni.

Windows PowerShellben, például a `C:\Kutatas\prism` könyvtárban:

```powershell
uv sync --locked --all-extras
.\start.bat --check --require-bayes --require-excel
.\start.bat
```

Linuxon, a projekt könyvtárában:

```bash
uv sync --locked --all-extras
./start.sh --check --require-bayes --require-excel
./start.sh
```

Mindkét indító a saját könyvtárából és annak `.venv` környezetéből indul,
akkor is, ha másik munkakönyvtárból hívod. Az indítók nem módosítják automatikusan
az egyszer már telepített környezetet. Telepítés után a `uv run --no-sync prism`
parancs is használható mindkét rendszeren.

### Választható funkciók

| Telepítés | Parancs |
| --- | --- |
| Alap dozimetria, GUI, CSV-export | `uv sync --locked` |
| Alap + Excel-export | `uv sync --locked --extra excel` |
| Alap + Bayes/MCMC-kalibráció és NetCDF trace-export | `uv sync --locked --extra bayes` |
| Minden funkció | `uv sync --locked --all-extras` |

A `bayes` kiegészítés a PyMC, ArviZ, h5netcdf és h5py csomagokat deklarálja.
Az `excel` kiegészítés az openpyxl-t használja, külön pandas-függőség nélkül.
Alaptelepítésnél az MCMC menüpont inaktív; az Excel-export hiányzó kiegészítésnél
telepítési útmutatót ad. A CSV-exporthoz egyik kiegészítés sem kell.
Az `uv sync` csak a megadott kiegészítéseket tartja meg, ezért frissítésnél is
add meg a használni kívánt `--extra` vagy `--all-extras` kapcsolókat.

### Rendszeroldali összetevők

- Windows alatt az alkalmazás és a numerikus függőségek x64-es wheeljei
  települnek. `DLL load failed` hiba esetén ellenőrizd, hogy x64-es Pythont
  használsz, és elérhető-e a [Microsoft Visual C++ x64 futtatókörnyezet](https://learn.microsoft.com/cpp/windows/latest-supported-vc-redist).
- Linuxon a GUI-hoz grafikus asztal és X11/XWayland szükséges. Debian/Ubuntu
  rendszeren a Qt rendszerkönyvtárai például a `libgl1`, `libegl1`,
  `libxkbcommon-x11-0`, `libxcb-icccm4`, `libxcb-image0`, `libxcb-keysyms1`,
  `libxcb-render-util0`, `libxcb-xinerama0` csomagokból érhetők el.
- A beépített szkennelés jelenleg Linuxon, SANE-nel használható (`scanimage`,
  Debian/Ubuntu alatt `sane-utils`), a megfelelő szkennermeghajtóval.
- **Windows alatt a beépített szkennelés nem támogatott.** A
  [NAPS2 dokumentált CLI-jének](https://www.naps2.com/doc/command-line) színes módja
  24 bites RGB (8 bit/csatorna), a PRISM pedig 48 bites RGB-t
  (16 bit/csatorna) követel. Windows alatt a gyártói szkennerszoftverrel készített,
  valóban 16 bit/csatornás RGB TIFF-eket másold a projekt `raw` könyvtárába,
  majd frissítsd a fájllistát. A kalibráció, konverzió és elemzés ezeken használható.
  Az integrált Windows-felvételhez külön 48 bites backend és hardveres validálás kell.

### Telepítésellenőrzés és tesztek

A `--check` kapcsoló ablak megnyitása nélkül ellenőrzi az importokat és a Qt
platformplugin betöltését, és JSON-jelentést ad. A `--require-bayes` és
`--require-excel` kapcsolókkal a megfelelő kiegészítés hiánya is hibát jelent.
A `scanner_executable: null` önmagában nem hibás alkalmazástelepítés.
Windows alatt a `scanner_supported: false` az integrált felvétel fent leírt
korlátját jelzi; Linuxon a külön szkennerprogramnak is elérhetőnek kell lennie.

Forráskódból a regressziós tesztek egységesen:

```text
uv run --no-sync python run_tests.py
```

A tesztindító mindkét platformon a telepített PyQt5 saját könyvtárait használja,
képernyőablak nélkül. Az opcionális funkciók tesztjei alaptelepítésnél kimaradnak.
A GitHub Actions munkafolyamat Windows/Linux és Python 3.12/3.13/3.14
alaptelepítést, továbbá Python 3.13-on teljes telepítést és külön wheel-telepítést
ellenőriz. A rövid MCMC-próba telepítési teszt, nem tudományos konvergenciavizsgálat.

### Telepítés pip-pel és csomagépítés

Ha uv helyett már meglévő Python 3.13-at használsz, Windowson:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install ".[bayes,excel]"
.\start.bat --check --require-bayes --require-excel
.\start.bat
```

Linuxon ugyanez a `python3.13 -m venv .venv` és a
`.venv/bin/python -m pip install ".[bayes,excel]"` parancsokkal működik.
A pip a verziókorlátokat használja, nem az uv lockfájlját; reprodukálható
környezethez az uv-s telepítés az ajánlott.

A `uv build` a `dist/` könyvtárba forrásarchívumot és
`prism-0.1.0-py3-none-any.whl` csomagot készít. A wheel a célgép virtuális
környezetébe pip-pel telepíthető; az opcionális funkciókat a wheel elérési útja
után írt `[bayes,excel]` választja ki. Telepítés után a `prism` parancs indítja
az alkalmazást; Linuxon a `prism --scanner --target-directory <könyvtár>`
csak a szkennert nyitja meg.
A célgéphez való numerikus és Qt-csomagokat a telepítő külön választja ki.

## Kalibráció

- A program a TIFF-fájlokat fájlnév szerint rendezi; a dózisokat ebben a sorrendben
  kell megadni. Használj nullával feltöltött sorszámokat: `00_blank.tif`,
  `01_film.tif`, … . A `_scan_std` és `_scan_sem` végű zajképek nem
  kalibrációs filmek, és a batch-konverzió film- és blanklistájából is kimaradnak.
- Az első film a 0 Gy referencia. A dózislista legyen nemcsökkenő, véges és
  nemnegatív; legalább négy különböző dózis szükséges, a nullát is beleértve.
- Sikertelen illesztéskor a program hibát jelez, és megtartja az előző aktív
  kalibrációt. Nem készít helyettesítő paraméterekből sikeresnek jelzett kalibrációt.
- Új JSON betöltése teljesen lecseréli a kalibrációs állapotot. A hiányzó opcionális
  minták és kovarianciák nem öröklődnek az előző kalibrációból.

Az RGB-szűrés javítása miatt a korábbi verzióval készített kalibrációkat és
dózistérképeket a nyers TIFF-ekből újra kell számítani és ellenőrizni.

## Filmillesztés

A FRONT film a képen látható transzformációval kerül a BACK film rácsára.
A két kép mérete eltérhet, a pixelméretüknek azonban egyeznie kell.
A filmen kívüli terület nem kap kiterjesztett peremdózist. A közös lefedettségen
kívülre nyúló mintára a számítás figyelmeztetést ad.

## Mintastatisztikák és dDVH

- A dDVH oszlopai az egyes dózisintervallumok térfogathányadát mutatják,
  összegük a lebegőpontos kerekítés pontosságán belül 100%. A 0 Gy és az
  alacsony dózisú mintarészek is szerepelnek a statisztikában; a ROI alakján
  kívüli képpontok nem.
- Az átlag és a térbeli szórás térfogatsúlyozott. Az Eppendorf-súlyok a megrajzolt
  csőkontúr raszterizált, forgásszimmetrikus közelítéséből származnak. Kör és
  poligon esetén egyenletes mintavastagságot feltételezünk.
- A `Mean pixel uncertainty [%]` a térfogatsúlyozott átlagos képponti
  bizonytalansági korlát és az átlagdózis aránya. Ez nem a mintaátlag
  bizonytalansága. Utóbbihoz a pixelek
  közötti korrelációt is modellezni kell. Nulla átlagdózisnál a százalékos
  értékek `N/A` jelölést kapnak.
- A geometria vagy a ROI-k módosítása érvényteleníti a korábbi eredményeket;
  exportálás előtt újra kell számolni őket. Kihagyott üres ROI esetén a többi
  minta megőrzi a saját azonosítóját.
- A CSV a pontos számított értékeket menti a képernyőn kerekített értékek helyett;
  a dDVH-export a binhatárokat is tartalmazza.
- A 2-es verziójú NPZ megtartja a korábbi mintánkénti tömböket, és hozzáadja a
  `hist_edges`, `hist_volume_pct`, `z_weight_back` adatokat és a mezők definícióit.
  A `dose_stats` második sora az adott bin térfogatsúlyozott átlagos pixelszórása;
  üres bin esetén `NaN`. Az export súlyozott mintaadat, nem teljes 3D dózisrács.

## Bizonytalansági modell (2-es verzió)

- A kalibráció paramétermintái ugyanazon a `fast`, `mayer` vagy `admm` solveren
  és végső mediánszűrőn haladnak át, mint a mentett dózis. A kalibrációs
  komponens ezek mintaszórása (`ddof=1`), nem a korábbi 16–84 percentilis
  félszélessége. A számítás nem tárol minden teljes képmintát a memóriában.
- Ha nincs legalább két paraméterminta mindhárom csatornához, a program a
  csatornán belüli paraméter-kovarianciát terjeszti numerikus Jacobi-mátrixszal.
  Ez elsőrendű közelítés; különösen telítésnél és kapcsolási határokon korlátozott.
- A `fast` módszer mérési hibája tartalmazza a változó csatornasúlyok
  deriváltját is. Az ADMM helyi kovarianciabecslése a visszaadott állapotban
  figyelembe veszi a dózis és vastagság kereszttagját. Nem azonosítható
  dózis/vastagság esetén a feldolgozás hibát jelez a véges szórás kitalálása helyett.
- A `*_std_fit.npy` az illesztési/mérési közelítés, a `*_std_calib.npy` a
  kalibrációs komponens. A `*_std.npy` a két komponens négyzetes összege,
  az explicit minimumkorláttal: 0,01 Gy felett a dózis 0,5%-a, egyébként
  0,0001 Gy. A fit-komponens a végső mediánszűrő környezetében a legnagyobb
  helyi szórást veszi át, mert a térbeli kovariancia nem ismert.
- A meta-JSON tárolja a komponensfájlokat, a kalibrációs hibabecslés forrását és
  a közelítéseket. Ha sem minták, sem teljes paraméter-kovarianciák nem érhetők el,
  a program figyelmeztet, és `calibration_uncertainty_available: false` értéket
  ment. Ekkor a kalibrációs komponenstömb nullája **hiányzó becslést** jelent;
  a teljes STD nem tartalmazza a kalibráció bizonytalanságát. A felület ezt jelzi.
- A profilátlagok sávja nem csökken automatikusan √N-nel: a marginális szórások
  átlagát használja konzervatív felső korlátként. A bio-modul az ismeretlen
  FRONT/BACK korreláció miatt az abszolút hibajárulékokat összeadja; az igazítás
  és a simítás ugyancsak pozitív súlyú bizonytalansági korlátot terjeszt.
- A D90/DAH sáv a `dózis ± szórás` képpontonkénti tartomány érzékenységi
  burkolója. Nem 95%-os konfidenciaintervallum. A felület nem talál ki
  véletlen/szisztematikus felbontást egyetlen STD-térképből, és nem simítja el
  a dóziseloszlást a sáv előállítása előtt.

A fit és a kalibráció komponensét a kombinálás függetlennek kezeli; a külön
kalibrált RGB-paraméterblokkok között sincs becsült kereszttag. A Mayer/ADMM
illesztési hibája helyi görbületi közelítés, nem a teljes térbeli posterior.
A bio-modul nulladózis körüli hibaterjedése továbbra is regularizált elsőrendű
közelítés. A szkenner/LRA és a fizikai szóródási modell teljes validálása nem
következik a numerikus tesztekből.

A módszertani háttér:
[NIST: kovarianciát is tartalmazó elsőrendű hibaterjedés](https://www.nist.gov/pml/nist-technical-note-1297/nist-tn-1297-appendix-law-propagation-uncertainty)
és [JCGM 101: eloszlások terjesztése a mérési modellen keresztül](https://www.bipm.org/en/web/guest/publications/guides).

A korábbi STD-térképeket és a belőlük készített profil- és mintaeredményeket
újra kell számítani. A kiválasztott solver ismételt futtatása időigényesebb
lehet a régi csatornakeverésnél.

## Szkennelés és fájlvédelem

- A `Next Idx` mező adja a kezdő sorszámot. A program átugorja a már létező
  vagy foglalt nevet, akkor is, ha csak egy hozzá tartozó zajkép vagy metaadat
  maradt a könyvtárban. A számláló sikeres mentés után lép tovább; hibánál
  ugyanaz a szabad sorszám újrapróbálható. A `Reset Index` sem enged felülírást.
- Egy felvételhez a `film.tif` átlagolt 16 bites RGB-kép és a
  `film_scan_meta.json` tartozik. Legalább két ismétlésnél a `film_scan_std.tif`
  az egyedi felvételek mintaszórása (`ddof=1`), a `film_scan_sem.tif` pedig
  az átlag standard hibája, SD/√N. Ez független ismétléseket feltételez;
  a lámpadrift és az ismétlések korrelációja külön validálást igényel.
- Egyetlen felvételből nem becsülhető ismétlési szórás: ilyenkor egyik zaj-TIFF
  sem készül el, a metaadat ezt hiányzó becslésként jelöli. A dózismotor ekkor
  a meglévő képalapú zajbecslésre tér át.
- A kalibráció és a konverzió a SEM-et használja. Ha csak SD és ismert
  ismétlésszám érhető el, abból számolja a SEM-et. Régi, ismétlésszám nélküli
  SD-térképet figyelmeztetéssel, √N-es csökkentés nélkül használ; a forrást az
  eredmény metaadataiban rögzíti. A GUI egyenletes blank RGB-referenciájának
  10 intenzitásegységes zajközelítése továbbra is külön jelölt feltevés.
- A szkennelés a név lefoglalása után ideiglenes fájlokkal dolgozik. A végleges
  fájlok kizárólag újként hozhatók létre, így a mentés közben keletkező ütközés
  sem ír felül mérést. Hiba vagy megszakítás esetén a művelet a saját részleges
  fájljait eltávolítja; másik folyamat fájljait megőrzi.
- A rejtett `.film.tif.scan.lock` a folyamatban lévő mentést jelzi. A program
  ezeket a felvételeket nem listázza és nem dolgozza fel. Áramszünet vagy
  kényszerített folyamatleállítás után maradhat zárolás és részleges fájl;
  ezeket nem törli automatikusan. Kézi ellenőrzés előtt állítsd le az adott
  felvételt végző folyamatot, és őrizd meg az esetleg helyreállítható adatokat.
- Az expozíciónkénti időkorlát alapértéke 300 másodperc, a felületen állítható;
  az eszközkeresés legfeljebb 15 másodpercet vár. A `Cancel current task`, az
  ablakbezárás és az Escape ugyanazon megszakítási útvonalat használja. Az ablak
  a munkaszál kilépése után zárul be. A három bemelegítő expozíció befejezése
  önmagában nem bizonyítja a lámpa hőmérsékleti stabilitását.

A tesztek szimulált szkennerképekkel és valódi helyi tesztfolyamatokkal futnak.
Valós SANE-hardveren külön integrációs próba szükséges; a Windowsos beépített
felvétel a 48 bites backend elkészültéig le van tiltva.
A folyamatkezelés a [Python subprocess](https://docs.python.org/3/library/subprocess.html)
és Windows alatt a [Microsoft taskkill](https://learn.microsoft.com/windows-server/administration/windows-commands/taskkill)
leírt működésére épül.

## Ellenőrzési korlátok

A numerikus tesztek szintetikus adatokat használnak. A teljes kalibráció,
a szkenner viselkedése és a fizikai modell referenciamérésekkel történő
validálása külön ellenőrzést igényel. A Windowsra feloldott wheel-készlet
ellenőrzése nem helyettesíti a Windowsos munkafolyamat és a célgép próbáját.

## Numerikus validáció és menthető jelentés

A validáció ugyanazt a TIFF-betöltést, standard kalibrációt és dózisfeldolgozást
futtatja, mint az alkalmazás. Az alaptelepítés elegendő a szintetikus
vizsgálatokhoz; csak a NetCDF MCMC-diagnosztika igényli a `bayes` extrát.
Windows alatt is ugyanezek a parancsok használhatók, szkenner nélkül:

```sh
uv run --no-sync prism --validate --quick
uv run --no-sync prism --validate --validation-output validation-results/reszletes
```

A gyors futás 6 független kísérletet használ, dózisonként mindössze kettőt.
Ez működési próba. Az alapértelmezett részletes futás 60 kísérlet, három
próbadózisra elosztva, 24×24 képpontos képekkel és kalibrációnként 16
kovarianciaalapú paramétermintával. Az ismétlésszám és a mintaszám növelhető:

```sh
uv run --no-sync prism --validate --seed 42 --repetitions 600 --posterior-samples 100
```

Az eredmény mindig **új könyvtárba** kerül; meglévő futást nem ír felül.
Megszakított futásnál `.incomplete` jelző marad. A végrehajtási hibák bekerülnek
a jelentésbe, a parancs hibakódja ilyenkor 1. A sikeres végrehajtás önmagában
nem jelent tudományos vagy fizikai megfelelőséget.

A könyvtár tartalma:

- `report.html`: önállóan megnyitható jelentés, beágyazott ábrákkal;
- `report.json`: teljes pontosságú eredmények, beállítások, verziók és a
  forrásmodulok/bemenetek SHA-256 lenyomatai;
- `solver_metrics.csv`, `cross_validation.csv`, `coverage.csv`,
  `coverage_observations.csv`: feldolgozható táblázatok;
- szintetikus 16 bites TIFF-ek, SD/SEM kísérőképek, kalibráció, referencia-
  és kiszámított dózistérképek, külön PNG-ábrák, illetve `engine.log`.

A solver-összehasonlítás a `fast`, `mayer` és `admm` módszert méri nulladózison,
egyenletes és Gauss-eloszlású térképeken, egyképpontos csúcson, vastagsághibán,
80 Gy-es extrapoláción és eltérő LRA-helyzeteken. Torzítást, MAE/RMSE-t,
maximális hibát és csúcsvesztést közöl. A nagy dózis a racionális válasz
aszimptotikus tartományát vizsgálja; a TIFF-intenzitások levágását a bemeneti
metaadat külön számlálja.

A keresztvalidáció minden nemnulla dóziscsoportot teljes egészében kihagy:
az azonos dózisú ismétlések sem szivárognak vissza a tanítóhalmazba.
A blank referencia megmarad, a legmagasabb dózis kihagyása extrapolációként
jelölt. Ez csatornánkénti kalibrációellenőrzés, nem egy további solver-teszt.

A lefedettségi vizsgálat minden ismétlésben új kalibrációs képeket és új mérést
generál. Egy középső képpont ad egy megfigyelést; a korrelált képpontokat nem
számolja független ismétlésnek. A ±1σ és ±1,959964σ sáv befogadási arányát,
Wilson-intervallumát, a tényleges szórást és a jelentett σ-t közli.
A hibás kísérletek kimaradnak az arány nevezőjéből, de számuk és részleteik
láthatók. Azonos maggal a bemenetek reprodukálhatók; eltérő platformok és
numerikus könyvtárak között apró lebegőpontos eltérés lehetséges.
A paraméterminták a standard illesztés kovarianciájából származnak;
a `--posterior-samples` nem MCMC-lánchossz.

### Valódi kalibráció és független referenciák

Meglévő kalibrációs TIFF-ekből is készíthető dóziscsoportonkénti kihagyásos
vizsgálat. A dózisok a nyers TIFF-ek névsorának sorrendjében követik egymást,
a blank az első. A fájlok és kísérőadataik változatlanok maradnak:

```sh
uv run --no-sync prism --validate --no-synthetic --calibration-folder meres/kalibracio --calibration-doses 0,1,2,4,8,12,20 --calibration-roi-size 50
```

A megadott dózisok rögzített névleges értékek; a referencia besugárzás
bizonytalanságát ez az illesztés nem terjeszti tovább.

Külön referenciamérésekhez JSON-leíró adható meg. A fájlútvonalak a leíró
könyvtárához képest értendők. Példa kétféle összehasonlításra:

```json
{
  "schema_version": 1,
  "unit": "Gy",
  "cases": [
    {
      "id": "egyenletes_7Gy",
      "dose_map": "meres_dose.npy",
      "mask": "roi.npy",
      "sigma_map": "meres_std.npy",
      "reference_mean_gy": 7.0,
      "reference_sigma_gy": 0.07,
      "independent_reference": true,
      "uncertainties_independent": true
    },
    {
      "id": "profil_terkep",
      "dose_map": "profil_dose.npy",
      "reference_map": "referencia.npy",
      "alignment_confirmed": true,
      "independent_reference": true
    }
  ]
}
```

```sh
uv run --no-sync prism --validate --no-synthetic --reference-manifest meres/referenciak.json
uv run --no-sync prism --validate --no-synthetic --mcmc-trace kalibracio/trace.nc
```

A térképek véges, nemnegatív, kétdimenziós NPY-tömbök Gy egységben; a maszk
0/1 értékű. Térképi összevetéshez azonos fizikai rács és a regisztráció
kifejezett megerősítése szükséges. Skalár referencia csak a ROI átlagához
hasonlítható; ebből nem készül mesterséges egyenletes referenciatérkép.
Az átlag σ-jának felső korlátja a képponti σ-k átlaga; a program nem feltételez
független képpontokat. A két bizonytalanságot csak kifejezett függetlenségi
nyilatkozat esetén kombinálja. Térképi referenciaszóráshoz a
`reference_sigma_map` mező használható a `sigma_map` mellett.

Az opcionális MCMC-diagnosztika láncszámot, megtartott mintaszámot,
divergenciákat, rank R-hatot és bulk/tail ESS-t közöl. Egyetlen láncból nem
állít konvergenciát, az elégtelen vagy nem definiált diagnosztikát megjelöli.
A szintetikus alapfutás nem indít automatikusan Bayes-kalibrációt.
A módszertanhoz lásd az
[ArviZ diagnosztikai dokumentációját](https://python.arviz.org/projects/stats/en/latest/api/generated/arviz_stats.diagnose.html)
és a [JCGM Monte Carlo útmutatót](https://www.bipm.org/en/web/guest/publications/guides).

A generátor a kalibrációval azonos modellcsaládot használ; a zaj és az LRA
feltevései explicit szimulációs feltételek. Ez számszerű modell-visszanyerési
és érzékenységi vizsgálat. Valós szkenner-, geometriai és anyagfüggő
validációhoz független referenciamérés, igazolt referencia-bizonytalanság
és előre rögzített elfogadási kritériumok szükségesek. A program ezek nélkül
nem ad automatikus „megfelelt” minősítést.

## Kapcsolható mediánszűrés

A konverziós panelen az **Apply spatial median filters** kapcsoló vezérli a
blank 5×5-ös, a nyers RGB-kép 3×3-as és a kész dózistérkép 3×3-as
mediánszűrését. Alapértelmezésben be van kapcsolva, így a korábbi eredmények
viselkedése megmarad. Kikapcsolva ezek a lépések elmaradnak, a kalibrációs
paraméterminták hibaterjesztése is a szűretlen dózisbecslőn történik.
A szűrt mérési bizonytalanság konzervatív 3×3-as felső burkolója ilyenkor
szintén elmarad. A beállítás bekerül a film metaadataiba és a futásnaplóba.

A **bFDR artifact removal** külön művelet, saját kapcsolóval. Az ADMM belső
TV-regularizációját és a kalibráció ROI-előszűrését a konverziós kapcsoló
nem módosítja. Keskeny csúcsokhoz mindkét képtisztítási beállítást és a solver
választását figyelembe kell venni; a szűrés kikapcsolása önmagában nem
igazolja, hogy egy elszigetelt kiugrás valódi dózisjel.

API és validációs példa:

```python
engine.process_films(files, blank_path, method="fast", median_filtering=False)
```

```sh
uv run --no-sync prism --validate --no-median-filter
```

A validáció egy-, három- és ötképpontos szélességű négyzetes csúcsokat is
vizsgál. A kapcsolóval két külön futás készíthető; a jelentésben szerepel az
alkalmazott szűrés. Az egyképpontos csúcs szűretlen megőrzését külön
regressziós teszt ellenőrzi a `fast` és `mayer` módszernél.

## Reprodukálhatóság és projektállapot

A **Save Project State** a projekt gyökerébe menti a verziózott `project.json`
fájlt. A projekt megnyitása után a **Restore Project State** állítja vissza:

- az aktív kalibráció paramétereit, kovarianciáját, tárolt paramétermintáit,
  OD-zaját és véletlenmagját;
- a blank fájlt, a kiválasztott ROI-ból származó RGB-átlagot és szórást;
- a solver és a két külön képtisztítás beállítását;
- a megjelenített nyers és dózisképet, az elemzési ellipszist, ferde
  profilvonalat, keresztpozíciót, profilsávot és megjelenítési beállításokat;
- a FRONT/BACK térképeket, eltolást, forgatást, fantomgeometriát és az összes
  minta-ROI típusát, méretét, pozícióját, forgatását és poligonpontjait;
- a morfológiai elemzés vezérlőinek beállításait.

A mentés explicit: a munkamenet végén használd a mentés gombot.
A származtatott minta- és morfológiai eredményeket visszaállítás után újra kell
számítani. Projektváltáskor a korábbi kalibráció, blank, betöltött képek és
származtatott nézetek törlődnek az aktív munkamenetből.
A projektfájl cseréje ideiglenes fájlon keresztül történik; hibás JSON/adat nem
írja felül az előző mentést. Ismeretlen sémaverziót nem tölt be.

A hivatkozott képek és a hozzájuk tartozó STD/metaadat fájlok SHA-256
ellenőrzésen mennek át visszaállítás előtt. Hiányzó vagy módosult fájlnál a
visszaállítás hibával megáll. A kalibráció külön pillanatképként szerepel,
így egy később felülírt `calibration_master.json` nem cseréli le észrevétlenül.
A fájlútvonalak relatívak: a teljes projekt áthelyezhető, ha a relatív
könyvtárszerkezet megmarad. Külső fájlokra is lehet hivatkozni, de azokat a
mentés nem másolja be; hordozhatósághoz tartsd őket a projekten belül.
A `project.json` nem teljes adatbiztonsági mentés.

### Feldolgozási futásnapló és újrafuttatás

Minden konverzió egyedi azonosítót kap. A `processed/.runs/<azonosító>/run.json`
rögzíti a beállításokat, a tényleges kalibrációs pillanatképet, a bemenetek
és zajkísérők lenyomatát, a Python/csomagverziókat, a források lenyomatát,
és ahol elérhető, a Git commitot és a munkakönyvtár állapotát.
A `source/` mappa megőrzi a releváns forrásmodulokat, valamint forrásból
futtatásnál a csomagleírót és a lockfájlt is. A kész NPY-k lenyomata szintén
bekerül a naplóba; a film metaadata a futásazonosítóra hivatkozik.

A napló állapota `running`, `completed` vagy `failed`. Hiba részleges
kimeneteket hagyhat; kényszerített leállítás után `running` maradhat. Ezek nem
tekinthetők befejezett futásnak. A futásnapló nem archiválja a teljes mérési
adatállományt és a korábbi dózis-NPY-ket: új konverzió a szokásos nevű
kimeneteket továbbra is felülírhatja. Az eredeti bemeneteket és a `.runs`
könyvtárat őrizd meg együtt.

Egy befejezett futás a bemenetek és az archivált források lenyomatának
ellenőrzése után új, még nem létező könyvtárba ismételhető:

```sh
uv run --no-sync prism --replay-run projekt/processed/.runs/AZONOSITO/run.json --replay-output projekt/ujrafuttatas
```

Az újrafuttatás a jelenlegi telepített kódot használja. Eltérő forrás vagy
környezet esetén figyelmeztet; az archivált kódot nem hajtja végre
automatikusan. A `replay_source.json` jelzi a környezet egyezését és a
kiszámított NPY-k bitenkénti egyezését az eredeti napló lenyomataival.
Eltérő platform/BLAS/verzió esetén nem ígér bitazonos eredményt.

Az engine helyi véletlengenerátorokat használ, alapértékben `random_seed=42`.
Ez a standard kalibráció paramétermintáit, a bFDR mintavételezését, a
kalibrációs ábrák részmintáit és az MCMC csatornánként rögzített magját is
vezérli. Az API-ban például `GafchromicEngine(..., random_seed=123)` adható meg.
A mag a kalibrációval együtt mentődik és töltődik; régi kalibrációnál 42 az
alapérték. A determinisztikus mag nem helyettesíti az MCMC konvergenciájának
ellenőrzését.

## Karbantarthatóság és számítási feladatok

A `GafchromicEngine` továbbra is a publikus belépési pont. A megvalósítás
feladat szerint külön modulokba került:

| Modul | Felelősség |
| --- | --- |
| `calibration.py` | Kalibráció illesztése, betöltése és mentése |
| `dose_solvers.py` | OD-modell, fast/Mayer/ADMM solver, bizonytalanságterjesztés |
| `film_processing.py` | TIFF-betöltés, konverzió, kimenetek és futásnapló |
| `morphology.py` | Qt-től független morfológiai számítások |
| `morphology_tab.py` | Morfológiai vezérlők és eredmények megjelenítése |
| `bio_analysis.py` | FRONT/BACK interpoláció, minta-ROI-k és dózisstatisztikák |
| `processing_settings.py` | Változtathatatlan, ellenőrzött feldolgozási beállítások |
| `background_tasks.py` | Háttérfeladatok, megszakítás és elavult eredmények eldobása |

A GUI és a futásnapló ugyanazt a `ProcessingSettings` szerződést használja.
A régi `process_films(..., method=..., median_filtering=...)` hívás működik;
az új forma például:

```python
from processing_settings import ProcessingSettings

settings = ProcessingSettings(method="fast", median_filtering=False, random_seed=42)
engine.process_films(files, blank_path, settings=settings)
```

A `settings` argumentum megadása esetén az abban lévő értékek érvényesek.
A kalibrációs bizonytalanság számításánál minden paramétermintához továbbra is
ugyanaz a dózisbecslés készül, de az egyes realizációk később eldobott
illesztési STD-jét már nem számolja újra. A végleges mérési STD változatlanul
kiszámításra kerül. A mintaszám és a tudományos modell nem változott.
Az ADMM egy iteráción belül újrahasználja az azonos OD-modellértékeket;
a szegmentált szigetek területe egyetlen címkeszámlálással készül.

A bio- és morfológiai elemzés háttérszálon fut. A feladat egy saját bemeneti
pillanatképet kap; a munkaszálak nem olvasnak vagy módosítanak Qt-widgetet.
A **Cancel Calculation / Cancel Analysis** megszakítást kér. A már futó
NumPy/SciPy művelet végéig várni kell, majd a következő ellenőrzési pontnál
áll le; a szálat nem állítja le erőszakosan. Az ablak bezárása megvárja a
futó feladatok befejeződését vagy együttműködő megszakítását.

Bemenet-, geometria- vagy ROI-változás érvényteleníti a régi eredményt.
Egy korábban indult feladat eredménye ilyenkor nem jelenhet meg később.
A morfológiai vezérlők gyors egymásutánban érkező változásait 200 ms-os
késleltetés vonja össze. Egy vezérlőhöz legfeljebb egy futó feladat tartozik;
ha újabb kérés érkezik, a várakozó kérésekből csak a legutolsó marad meg.

## Teljesítménymérés

A mérőprogram új könyvtárba ment JSON/CSV összesítést, környezeti adatokat
és ellenőrzésre alkalmas dózis/STD-térképeket. Például:

```sh
uv run --no-sync python -m benchmark --output benchmark-results/alap --sizes 64,128,256 --samples 16 --repeats 3
uv run --no-sync python -m benchmark --output benchmark-results/uj --sizes 64,128,256 --samples 16 --repeats 3 --baseline benchmark-results/alap
```

Minden méret/solver külön Python-folyamatban fut, bemelegítés után.
Ez a mérés a TIFF-műveleteket és a futásnaplót is tartalmazza.
A `tracemalloc` bekapcsolva marad a mérés közben: a mért idő ezért mérési
többletköltséget is tartalmaz. A memóriaadat a nyomon követett allokációk
csúcsa, beleértve a NumPy által jelentett foglalásokat; nem a teljes folyamat
RSS-értéke. A `--baseline` a sebesség mellett a dózis és az STD maximális
abszolút eltérését is közli.

Egy kifejezetten megadott, helyi referencia-engine forrásával párosított,
csak numerikus összehasonlítás is készíthető:

```sh
uv run --no-sync python -m benchmark --output benchmark-results/solverek --sizes 128,256 --samples 16 --repeats 5 --reference-engine regi/dosimetry.py
```

Ez a parancs betölti a megadott Python-forrást. Azonos bemeneteken,
váltakozó sorrendben méri a régi és új számítást, TIFF és naplózás nélkül.
A memóriafoglalást külön menetben méri, így a futásidőt nem terheli a
`tracemalloc`. A referencia forrása és lenyomata megmarad a jelentés mellett.
A futásidők tájékoztató adatok, CI-ben nincs gépfüggő sebességküszöb.
A kis szintetikus térképek eredménye nem vetíthető ki automatikusan teljes
szkennerképekre. A gyorsulás csak azonos számítási beállítások és az eredmények
egyezésének ellenőrzése mellett értelmezhető.

### Mért eredmény — 2026. november 13.
Linuxon, Python 3.14 alatt, 16 paramétermintával és öt párosított ismétléssel,
256×256 képponton a következő numerikus futásidők adódtak:

| Solver | Korábbi idő | Új idő | Futásidő csökkenése |
| --- | ---: | ---: | ---: |
| fast | 0.238 s | 0.181 s | 24.2% |
| mayer | 0.747 s | 0.697 s | 6.6% |
| admm | 4.112 s | 3.453 s | 16.0% |

Mind a hat párosított esetben nulla volt a dózis és az STD maximális
abszolút eltérése. A `fast` numerikus részének követett csúcsmemóriája
körülbelül 29%-kal csökkent. Ezek a számok a numerikus részre vonatkoznak;
a TIFF-kezelés és a kibővített forrásarchiválás költségét nem tartalmazzák.
A kezdeti teljesfolyamat-mérés vegyes futásidőket mutatott, ezért ebből
nem következik minden feladatra általános gyorsulás.

A csomagverziók változatlan listája folyamatonként egyszer készül el;
a bemeneti és forrásfájlok lenyomata, valamint a Git-állapot minden futáskor
friss marad. Függőségtelepítés után indítsd újra az alkalmazást.
Részletes hívásprofilhoz a teljesfolyamat-méréshez add a `--profile` kapcsolót.
Az egyes esetek könyvtárában létrejövő `profile.pstats` például a
`python -m pstats profile.pstats` paranccsal vizsgálható. A profilozás további
futásidő-többletet jelent, ezért a profilozott és a profilozatlan idők külön
kezelendők.

### ADMM dóziskorlát és kimeneti ellenőrzés

Az ADMM nemnegatív dóziskorlátja minden TV-frissítés után is érvényesül,
a duális változó frissítése előtt. A duális eltolás miatt a korábbi,
korlátozatlan TV-eredmény alacsony jelű képpontokban negatív lehetett akkor
is, ha az előző dózislépés már nemnegatív volt. Ez egy teljes film mentését
megakadályozhatta. A paraméterminták ugyanazt a korlátozott solvert használják.
A valóban nem becsülhető bizonytalanságot a program továbbra sem helyettesíti
véges számmal: a hibaüzenet megnevezi a filmet, a solvert, az érintett dózis/
STD-komponenst és a hibás képpontok számát.

## Futás közbeni regressziók felülvizsgálata (2026. november 13.)

A főnézet **Beam Dose Distribution & Cumulative DAH** ábráján a kék
hisztogram alapértelmezése újra **peak = 100%**. Ez az eloszlás alakját mutatja;
a külön választható **area per bin** módnál a binértékek összege 100%.
A két normalizálás a piros DAH-t nem módosítja: az mindig a teljes ROI
küszöbdózist elérő területaránya, a valódi 0 Gy-es pontokkal együtt.
A görbe 0 Gy-nél 100%-ról indul, a legnagyobb dózis felett 0%-ra érkezik.
A megjelenítés legfeljebb 1024 megfigyelt dóziscsomópontot mintáz; a terület-
értékek minden megjelenített küszöbnél a teljes ROI-ból készülnek.

A dózistengely alapból a ROI tényleges minimum–maximum tartományát mutatja,
lineáris skálán; a **Dose axis: full, from 0 Gy** visszaadja a teljes tengelyt.
A nagyítás a DAH százalékait és a számításba vett képpontokat nem változtatja.
A **Relative height / area [%]** közös tengely: a kék görbe peak módban a
leggyakoribb dózisbinhez viszonyított magasság, a piros görbe mindig területarány.

A dózis- és az STD-térkép külön ablakolást és színskálát őriz a nézetek közötti
váltáskor. Az automatikus ablak a piros filmhatáron belüli értékek 1. és 99,5.
percentilise között nyílik; az **Auto window** ezt az aktuális nézetben újraszámolja.
A határ módosítása mindkét nézet automatikus ablakát frissíti. A színskála
hisztogramja is ezt a filmterületet mutatja. Az ablakolás csak a kontrasztot
módosítja, a mentett dózist/STD-t és a statisztikát nem. A profilpanelek nagyobb
helyet kapnak; a pozíciótengelyük a filmhatárt, a dózistengelyük a profilértékeket
követi, a térkép kézi nagyításától függetlenül. STD-nézetben is dózisprofilok
láthatók, a hozzájuk tartozó bizonytalansági sávval.

A főnézet DAH-sávja alapból kikapcsolt. A választható **±σ envelope** egy
érzékenységi burkoló, amit a grafikon címe is megjelöl; nem konfidenciaintervallum.
Képpontonkénti marginális STD-ből a térbeli kovariancia ismerete nélkül nem
állítható elő igazolt DAH-konfidenciaintervallum. A sáv elrejtése nem módosítja
a mentett bizonytalanságot vagy a dózist. A valóban nagy STD-t nem csonkoljuk
kisebbre a megjelenítés kedvéért. A bio-modul térfogatarányos dDVH-ja továbbra
is külön számítás, ott a binértékek összege 100%.

Az ADMM lassulásának egyik korábbi oka a 100 kalibrációs paraméterminta
solverrel való teljes újraértékelése: ez több munka, mint a régi, az ADMM-től
eltérő csatornakeverés. Az új **Calibration uncertainty workers** vezérlő
1–8 szálat kérhet; a GUI alapértéke legfeljebb 4. A párhuzamos realizációk
számát a képméret és egy konzervatív, 256 MiB-es becsült munkamemória-keret
is korlátozza. Ez a párhuzamos munkára vonatkozó becslés, nem teljesfolyamat-
RSS-korlát; egyetlen nagy kép is túllépheti. A minták száma és az összegzés
sorrendje változatlan, a függőben tartott térképek száma korlátos.
Az API-ban `ProcessingSettings(uncertainty_workers=4)` használható;
a régi hívások alapértéke továbbra is 1 szál.

Az MCMC-hez választható a **Full covariance (experimental)** adaptáció.
A modell, a priorok, a 4 lánc, a lánconkénti 3000 megtartott minta és az
1000 hangolási lépés változatlan. Ez a mintavételező előkondicionálását
változtatja, nem a posterior céleloszlását. A PyMC kísérletinek jelöli;
a diagonális adaptáció továbbra is választható és az API alapértelmezése.
A csatornánkénti futásidők és a választott adaptáció a kalibrációs JSON-ba
kerülnek. Az API-opció `run_mcmc_calibration(..., adaptation='adapt_full')`.
A gyorsabb futás mellett ugyanúgy ellenőrizni kell az R-hatot, az ESS-t és
a divergenciákat.

A régi, SD-kísérőképes felvételeknél az ismétlésszám hiányáról szóló
figyelmeztetés fájlonként deduplikált. A zajképet és a metaadatot minden
betöltés újraolvassa; a figyelmeztetés deduplikálása nem rejt el adatfrissítést.
Megbízhatóan ismert N esetén a `film_scan_meta.json` `scan_count` mezője
rögzíti az átlagolt képek számát. Így az egyedi képek SD-jéből az átlag
SEM-je `SD/√N` lesz, független ismétlések feltevésével. Ismeretlen N esetén
a konzervatív SD marad; a régi GUI alapértékéből nem találunk ki ismétlésszámot.
Az ismétlésszám utólagos helyesbítése után a kalibrációt és a konverziót
újra kell számítani. A meglévő kalibráció és eredmények nem frissülnek maguktól.

## Vastagsághorgonyzás és solverválasztás

A Mayer és az ADMM vastagsághorgonyzása közös érvényességi maszkot használ.
Az alapértelmezett automatikus maszk a teljes képen keresi a pozitív jelet,
de kizárja a nem véges/negatív OD-t, az inverzió telítési védelmébe eső
válaszokat és a kalibráció felső dózisához tartozó válasznál több mint 5%-kal
nagyobb OD-t bármely csatornán. Az 5% a meglévő vastagságkorlátból választott
numerikus ráhagyás; nem elfogadott fizikai mérési hibatűrés. Kalibrációs
dózissor nélkül csak a jel- és aszimptotakorlát használható. Az automatikus
maszk válaszérvényességet vizsgál, nem garantálja a film geometriai felismerését.

A **Use displayed film boundary for anchoring** kapcsolóval a megjelenített
film piros kerete is korlátozhatja a horgonyzást. Állítsd a keretet a tényleges
filmterületre, jelöld ki ugyanennek a filmnek a nyers TIFF-jét, majd konvertáld
újra. Ez nem azonos a fehér elemzési ROI-val. A kapcsoló másik filmre vagy több
filmre nem alkalmazza automatikusan a megjelenített keretet. Batch konverzióhoz
kapcsold ki: ott az automatikus válaszmaszk érvényes. API-ból a
`ProcessingSettings(anchor_bounds=(left, top, right, bottom))` normalizált,
0–1 közötti téglalapot fogad; egy futás minden képén ugyanazt a relatív területet
jelenti. Az értékek és a maszkból képzett diagnosztika bekerülnek a futásnaplóba
és a film metaadataiba; a GUI a projektállapottal a piros keretet is menti.

A magot az érvényes kezdeti dózisok 99. percentilisének felét elérő képpontok
adják. A dózisnégyzetes súlyok a percentilisnél telítődnek, ezért egyetlen
kiugró maximum nem választhatja ki vagy uralhatja a magot. Az ADMM ezt a magot
az iterációk alatt rögzítve tartja. A vastagság simítása maszkkal normalizált:
a kizárt pontok értékei nem kerülnek bele a helyi/makroszkopikus átlagokba.
A fődózis és az összes kalibrációs realizáció ugyanazt a névleges kalibrációból
képzett érvényességi maszkot kapja; a realizációk dózisfüggő súlyai a saját
paramétereikből készülnek. A párhuzamos feldolgozás nem módosít közös solverállapotot.

A maszk a vastagságkorrekciót korlátozza: a kívül eső dózisokat nem állítja
nullára és nem csonkolja a kalibráció maximumára. Érvényes mag hiányában a
vastagságtényező 1 marad, és `anchor_status: unavailable` kerül a diagnosztikába;
ez nem teszi fizikailag érvényessé az extrapolált dózist. A
`solver_diagnostics` mezőben szerepel az érvényes/kizárt és magképpontok száma,
a vastagságtényező tartománya és a korlátot elérő érvényes képpontok aránya
(`thickness_at_limit_fraction`, 0–1). A `uncertainty_model_version: 3` a
horgonyzás megváltozott dózis- és bizonytalanságbecslését jelöli. A régi
konverziók megnyitáskor nem számolódnak újra.

A következő eljárást javasoljuk a solver kiválasztásához; ez saját projekt-QA
workflow, nem univerzális klinikai elfogadási protokoll:

1. **Rögzítsd az alkalmazási tartományt és a döntési szabályt.** Filmfajta és
   gyártási tétel, besugárzási körülmények, várható dózis- és térbeli tartomány,
   szkenner, tájolás, DPI, időzítés, blank/LRA és szűrés legyen dokumentálva.
   Előre határozd meg a megengedett ROI-átlaghibát, profil-/csúcshibát és
   ismételhetőséget a mérés célja és a referencia bizonytalansága alapján.
   Ezeket ne az összehasonlítás eredményéhez igazítsd.
2. **Különítsd el a kalibrációt és a referenciákat.** A körülbelül 11 Gy-es
   munkatartomány pilotjához például független 8, 11 és 15 Gy-es referenciamezők,
   egy meredek szél és a legkisebb releváns nyalábfolt használható, ha mind a
   kalibrált/validált tartományon belül vannak. A dózist független, az adott
   sugárzási térhez megfelelő referenciamérés adja. Azonos film három
   szkennelése a szkennelési zajt méri; az ismételhetőségi vizsgálat külön
   besugárzott filmeket is igényel. A kalibrációra használt filmek önmagukban
   nem független solver-validációk.
3. **Ugyanazokat az adatokat számold mindhárom solverrel.** Ugyanaz a kalibráció,
   blank, filmhatár, elemzési maszk és szűrés legyen érvényes. Ments külön
   `comparison/fast`, `comparison/mayer`, `comparison/admm` könyvtárakba, hogy
   mindhárom eredmény és futásnapló megmaradjon. Szűk csúcsoknál külön vizsgáld
   a mediánszűrés hatását; az ADMM saját TV-regularizációja a mediánkapcsolótól
   függetlenül aktív.
4. **Számolj hibát és vizsgáld a hibaképet.** ROI-nként dózisátlag és relatív
   eltérés a referenciától; több független filmre bias, RMSE és ismételhetőség.
   Profiloknál csúcs, FWHM, penumbra, pozíció és megfelelő regisztráció mellett
   térképhiba is kell. Ellenőrizd a horgonyzás elérhetőségét és a vastagságkorlát
   elérését; ezek hibajelzők, nem önmagukban pontossági rangsorok. A kisebb
   képponti STD vagy a simább kép önmagában nem jobb mérési eredmény.
5. **Válassz a megfelelt módszerek közül.** Ha több solver teljesíti az előre
   rögzített feltételeket, az egyszerűbb/gyorsabb, stabilabb módszer előnyös.
   A választást egy újabb, a rangsorolásba be nem vont referenciamérésen is
   ellenőrizd. Ha egyik sem felel meg, először a kalibrációt, csatornaeltérést,
   geometriai/szkennelési feltételeket és a modell érvényességét vizsgáld;
   ne válassz azért solvert, mert éppen a várt dózist adja.
6. **Rögzítsd és ellenőrizd a protokollt.** A döntés az ellenőrzött film–szkenner–
   dózis–geometria tartományra érvényes. Új tétel, szkennelési feltétel,
   kalibráció vagy algoritmusváltozás után ellenőrizd újra. Mérésenként a blank,
   a referenciaellenőrzés és a mentett solverdiagnosztika alapján dönts arról,
   hogy a kiválasztott protokoll továbbra is alkalmazható-e.

| Solver | Ésszerű kiinduló szerep | Mit kell külön bizonyítani? |
|---|---|---|
| `fast` | Gyors előnézet, csatornaalapú kontroll, jól viselkedő homogén film | A vastagságkorrekció nélküli dózis pontossága és a csatornaváltás viselkedése |
| `mayer` | Vastagságkorrekcióval készülő első összehasonlítási alap | Csatornaeltérés/prior/horgonyzás okozta bias a releváns dózistartományban |
| `admm` | Térbeli regularizáció kipróbálása zajos eloszláson | Abszolút dózis mellett a szűk csúcsok, szélek és profilszélességek megőrzése |

A jelenlegi eszközökkel az egyes solverek független referenciáit ugyanabba a
referencialeíróba lehet felsorolni (külön `id`, közös elemzési maszk). A
[független referenciák leírása](#valódi-kalibráció-és-független-referenciák)
szerinti mezők használhatók. A kiértékelés nem választ automatikusan győztest:

```sh
uv run --no-sync prism --validate --no-synthetic --reference-manifest comparison/references.json --validation-output comparison/reference-review
```

A bizonytalanságok kombinálásánál a közös kalibrációtól származó korrelációt
nem szabad függetlenségnek tekinteni. A solverek eltérése sem független
referenciamérés. A fenti mérési feltételekhez és ellenőrzésekhez támpont az
[AAPM TG-235](https://aapm.onlinelibrary.wiley.com/doi/10.1002/mp.14497), a közös
szkennelésben végzett referenciaellenőrzéshez pedig
[Lewis és munkatársai protokollja](https://pmc.ncbi.nlm.nih.gov/articles/PMC9381144/).


## Laptopon is használható térképnézet

A főnézet **Layout: Automatic** beállítása a munkaterület logikai méretéhez
igazodik, így a kijelző DPI-nagyítását is figyelembe veszi. Kis ablakban
**Compact**, nagy munkaterületen **Detailed** elrendezésre vált; a köztes
mérettartományban megtartja az előző módot, hogy az elrendezés ne villogjon
átméretezéskor. A **Layout: Compact / Detailed** kézzel is rögzíthető.
A program induló ablaka a rendelkezésre álló képernyőterülethez igazodik.

- A bal oldali és az alsó profil külön húzható elválasztóval méretezhető.
  Kompakt nézetben kisebb alapméretet kapnak, így a dózistérkép marad a fő
  munkafelület. A pozíciótengely továbbra is a filmhatárt követi. A vízszintes
  profil rajzterülete a térképen látható filmhatár alá igazodik, annak
  szélességével; a tengelyfeliratok ezen kívül kapnak helyet. Átméretezéskor
  és térképnagyításkor az igazítás automatikusan követi a filmet.
- A **Raw film** és a **Statistics / DAH** kapcsolóval a nyers kép és az alsó
  elemzési rész külön visszakapcsolható. Kompakt módban alapból rejtettek;
  részletes módban alapból látszanak. A statisztikai rész szükség esetén
  görgethető, ezért nem kényszeríti a teljes ablakot képernyőnél szélesebbre.
- Az **Enlarge map** ideiglenesen elrejti a profilokat, nyers képet, alsó
  elemzést és az oldalsó vezérlőpanelt. ROI-t kijelölni, módosítani, illetve a
  térképet nagyítani és mozgatni ebben a nézetben is lehet. A **Restore view**
  visszahozza a korábbi paneleket és arányokat; a fókusznézetben végzett ROI-
  és térképnagyítás-módosítások megmaradnak.
- A két elrendezés külön őrzi a kézi panelarányokat és láthatósági választásokat.
  Ezek a **Save State** projektmentéssel megőrizhetők. A nagyított munkanézet
  ideiglenes: projekt-visszaállításkor a normál nézet nyílik meg.

Az elrendezés nem számolja újra a dózist, az STD-t vagy a ROI-t, és nem állítja
vissza az ablakolást, színskálát vagy a térkép nagyítását. A **Save figure**
a pillanatnyilag látható elrendezést exportálja. Teljes, nyers képet és
statisztikát is tartalmazó kompozithoz előbb állítsd vissza a normál nézetet,
és kapcsold be a kívánt paneleket.


A **Radiobiology 3D Sample Dosimetry** fül dDVH-ábráján a **Dose axis: sample
range** alapbeállítás a kiválasztott minta nem üres dózisbinjeire nagyít.
A **Dose axis: full, from 0 Gy** visszaadja a teljes dózistengelyt. A kapcsoló
kizárólag a megjelenített tartományt módosítja: a térfogatarányok, statisztikák,
binek és exportált adatok változatlanok. A nem üres szélső bineket teljesen
mutatja, egyenletes és valódi 0 Gy-es mintánál is; a valódi nulla dózisú
mintarészeket nem hagyja el. A választás a projektállapottal menthető.


Mindkét fő hisztogram alapból 50 egyenlő szélességű binre osztja a vizsgált
ROI/minta tényleges minimum–maximum dózistartományát. Így például 10–12 Gy
között 0,04 Gy a binszélesség, nem 12/50 = 0,24 Gy. Valódi 0 Gy-es pontok
esetén a minimum továbbra is nulla. Egyenletes dózisnál kis, nemnegatív
segédtartomány biztosít véges binszélességet; ez nem mérési szórás.
A dózisok, a minta statisztikái és az empirikus DAH nem változnak a binningtől.
Az újonnan kiszámolt/exportált dDVH binhatárai és binenkénti térfogatarányai
az új felbontást követik. A tengelynézet kapcsolója továbbra is csak nagyítás,
a binninget nem kapcsolja át. A minták korábban kiszámolt dDVH-ját a
**Calculate** művelettel kell újraszámolni; filmkonverzió nem szükséges.

## Kísérleti adatlap és mért nyalábjellemzők

Az **Advanced Beam Morphology → Measured beam & exposure** a filmen mérhető
alapjellemzők új alapnézete. A **Calculate** gomb a natív képpontokon számol,
nem használja a kísérleti dekonvolúciót, SNR-vágást vagy 300 pixeles előnézetet.
Alapból a piros filmkeret teljes téglalapja szerepel; a **Beam Analysis ellipse**
opció a központi ellipszist választja. A filmkereten kívüli és az ellipszisen
kívüli pontok nem mesterséges nulla dózisú mérések: nem szerepelnek a nevezőkben.
A maszkon belüli valódi nulla dózisok viszont megmaradnak.

A mérési adatok filmenként külön tárolódnak a **Save State** projektállapotban.
Új filmre váltáskor nem öröklődik automatikusan az előző film lövésszáma vagy
időablaka. Az üres mező ismeretlen; a program nem állít elő lövésszámot az
ismétlési frekvencia és az expozíciós idő szorzatából. A lézerimpulzus FWHM-je
kizárólag leíró adat, nem helyettesíti a filmen érvényes részecskeimpulzus-hosszt.
Az időmezők másodpercben, az impulzushosszak fs-ban értendők; tudományos
számformátum használható (pl. 1 ns = 1e6 fs). Az időtartam és a kezdő/végpontok
alternatív megadási módok; ha mindkettő szerepel, egyezniük kell. Az időablak
felhasználó által megadott expozíciós/kapuzási intervallum, nem N/f közelítés.

A ROI nyers dózisátlagából D/N és D/T számolható, ha a szükséges bemenetek
ismertek. Az N a megadott, besugárzáshoz tartozó események száma; az átlag akkor
is érvényes, ha az események dózisa különböző. Ebből nem ismerjük meg az egyes
lövések eloszlását. A feltételes csúcs-dózisteljesítményhez külön ki kell
választani, hogy egy részecskeimpulzus tartozik egy számlált eseményhez, és az
impulzushossz a **film síkjára** érvényes. Téglalap-időprofilnál D/(N τ), Gauss-
FWHM-nél sqrt(4 ln(2)/π) D/(N τ) az alkalmazott csúcsérték. Ezek feltételezett
időalakból származó értékek, nem időfelbontott mérések vagy FLASH-minősítések.
Hiányzó időalak, impulzusstruktúra vagy filmsíkbeli időtartam esetén nem készül
csúcsérték. A N/T és a névleges frekvencia eltérése ellenőrzési jelzés lehet,
nem bizonyítja a kiesett impulzusok számát.

A morfológia max(D − háttér, 0) dózisjellel súlyozott súlypontot, teljes 2×2-es
kovarianciamátrixot, fő-/melléktengely szerinti RMS-szélességet, ellipticitást,
főtengelyszöget és R50/R80/R90 befoglalósugarakat ad. A tengelyszög a film X-jobbra,
Y-lefelé koordinátarendszerében értendő, körszimmetriánál nem meghatározott.
A sugarak a súlypont körül a megtartott jelintegrál 50/80/90%-át foglalják be,
nem töltés- vagy energiasugarak. A háttérlevonás a morfológiai súlyokat módosítja;
a nyers dózisátlagot, D/N-t és D/T-t nem. Jel nélküli képen a méretek és a
súlypont nem meghatározottak. ROI-szélen jelentős jel vagy levágott ROI esetén
figyelmeztetés jelzi, hogy az eredmény csak a megtartott foltrészre vonatkozik.

A nézetek: natív filmkép súlyponttal és befoglalókörökkel; a befoglalt jelintegrál
görbéje; valamint olvasható eredménytábla és feltételezések. Az ismert bemenetek
alapján külön dózis/lövés-, átlagdózisteljesítmény- és feltételes csúcsértéktérkép
választható. Ezek a film dózistérképének globális osztókkal/szorzókkal képzett
nézetei, nem rekonstruált lövésenkénti vagy térben időfelbontott mérések.

Az átlagos képponti STD az átlag bizonytalanságának korrelációt nem kizáró felső
korlátjaként szerepel; nincs sqrt(N) csökkentés. A D/N és D/T dózisoldali
bizonytalanságkorlátja megjelenik az exportban. Az időzítés, lövésszám és alak-
modell bizonytalansága, illetve a morfológiai paraméterek konfidenciaintervalluma
egyelőre nincs számszerűsítve. A JSON export tartalmazza a beállításokat, bemeneti
geometriát, feldolgozási metaadatokat és dózis-/STD-tömbazonosítókat. A társított
NPZ-ben dózis, maszk, rendelkezésre álló STD, fizikai képskála és befoglalási
görbe szerepel, objektumszerializálás nélkül.

A korábbi eszközök az **Exploratory morphology** nézetben maradtak; a módválasztó
nem állít többé rögzített 500 Hz-et. Ezek nem módosítják a mért alapmutatókat.
A forrásrekonstrukció, anyagrétegenkénti transzport és lövésenkénti stabilitás
nem része ennek az első megvalósítási ütemnek.


A mag–halo bontáshoz opcionális **Core radius for halo [mm]** adható meg:
a mag a súlypont körüli rögzített sugarú kör, a halo a körön kívüli,
de az elemzési maszkon belüli háttérlevont jel. A halo-jel hányada és a
halo/mag arány ezzel a deklarált definícióval számolódik. Rögzített magsugár
nélkül nincs automatikusan gyártott haloarány. A térképen a fehér kör a
megadott maghatár, a sárga/cián/magenta kör az R50/R80/R90, a piros szakaszok
a főtengelyek ±2 RMS-szélességét jelölik; ez utóbbiak nem konfidenciasávok.
