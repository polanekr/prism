"""
PRISM: Probabilistic Reconstruction of Inhomogeneous Systems Methodology

Main package initialization.
This exposes the most important classes directly under the 'prism' namespace.
"""

# Verzió
__version__ = "1.0.0"

# Importok a modulokból, hogy könnyebb legyen elérni őket
# Pl. 'from prism import CellSurvivalLQModel' a 'from prism.biology import ...' helyett

from .reconstruction import Bayesian3DVolumeReconstructor
from .dosimetry import GafchromicEngine
from .biology import (
    CellSurvivalLQModel, 
    FishSurvivalModel, 
    EUD_SurvivalModel,
    BioModelComparator
)
from .analytics import DoseAnalyst
from .viz import plot_3d_interactive, plot_dvh, plot_gamma_map

# Mit exportáljon, ha valaki azt írja: 'from prism import *'
__all__ = [
    "Bayesian3DVolumeReconstructor",
    "GafchromicEngine",
    "CellSurvivalLQModel",
    "FishSurvivalModel",
    "EUD_SurvivalModel",
    "BioModelComparator",
    "DoseAnalyst",
    "plot_3d_interactive", 
    "plot_dvh", 
    "plot_gamma_map"
]