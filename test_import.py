# test_import.py
import sys
import os

# Hozzáadjuk az aktuális könyvtárat az útvonalhoz
sys.path.append(os.getcwd())

print("--- PRISM Rendszerellenőrzés ---")

try:
    import prism
    print(f"✅ Csomag betöltve: PRISM v{prism.__version__}")
    
    from prism.dosimetry import GafchromicEngine
    print("✅ Dosimetry modul: OK")
    
    from prism.reconstruction import Bayesian3DVolumeReconstructor
    print("✅ Reconstruction modul: OK")
    
    from prism.biology import CellSurvivalLQModel
    print("✅ Biology modul: OK")

    from prism.analytics import DoseAnalyst
    print("✅ Analytics modul: OK")
    
    print("\nSIKER! A rendszer készen áll a használatra.")
    
except ImportError as e:
    print(f"\n❌ HIBA: {e}")
    print("Ellenőrizd, hogy minden fájl a 'prism' mappában van-e, és létezik-e az __init__.py!")
except Exception as e:
    print(f"\n❌ VÁRATLAN HIBA: {e}")