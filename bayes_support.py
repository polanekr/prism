"""Optional Bayesian backend shared by calibration and the launcher."""
# Bayesian inference (optional)
BAYES_IMPORT_ERROR = None
pm = az = None
try:
    import pymc as pm
    import arviz as az
    BAYES_AVAILABLE = True
except Exception as error:
    BAYES_AVAILABLE = False
    BAYES_IMPORT_ERROR = str(error)

