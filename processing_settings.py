"""Immutable processing contracts shared by UI, engine and saved runs."""
from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class ProcessingSettings:
    method: str = 'fast'
    cleaning: bool = False
    median_filtering: bool = True
    blank_rgb: tuple[float, float, float] | None = None
    random_seed: int = 42
    uncertainty_workers: int = 1
    anchor_bounds: tuple[float, float, float, float] | None = None

    def __post_init__(self):
        if self.method == 'hybrid': object.__setattr__(self, 'method', 'fast')
        if self.method not in ('fast', 'mayer', 'admm'):
            raise ValueError(f'Unknown dose solver: {self.method}')
        if type(self.cleaning) is not bool or type(self.median_filtering) is not bool:
            raise ValueError('Cleaning and median filtering must be boolean.')
        if type(self.random_seed) is not int or not 0 <= self.random_seed < 2**32 - 3:
            raise ValueError('Invalid processing random seed.')
        if type(self.uncertainty_workers) is not int or not 1 <= self.uncertainty_workers <= 8:
            raise ValueError('Uncertainty workers must be an integer from 1 to 8.')
        if self.anchor_bounds is not None:
            bounds = tuple(float(value) for value in self.anchor_bounds)
            if (len(bounds) != 4 or any(not math.isfinite(value) or not 0 <= value <= 1 for value in bounds)
                    or bounds[0] >= bounds[2] or bounds[1] >= bounds[3]):
                raise ValueError('Anchor bounds must be normalized left, top, right, bottom.')
            object.__setattr__(self, 'anchor_bounds', bounds)
        if self.blank_rgb is not None:
            rgb = tuple(float(value) for value in self.blank_rgb)
            if len(rgb) != 3 or any(not math.isfinite(value) or value <= 0 for value in rgb):
                raise ValueError('Blank RGB must contain three finite positive intensities.')
            object.__setattr__(self, 'blank_rgb', rgb)

    def to_dict(self): return asdict(self)

    @classmethod
    def from_dict(cls, data): return cls(**data)


@dataclass(frozen=True)
class MorphologySettings:
    scatter_sigma_mm: float = 0.
    max_size: int = 300
    snr_mask: bool = False
    shot_mode: str = 'Multi-shot (500 Hz Accumulation)'

    def __post_init__(self):
        if not math.isfinite(self.scatter_sigma_mm) or self.scatter_sigma_mm < 0:
            raise ValueError('Scattering sigma must be finite and nonnegative.')
        if type(self.max_size) is not int or self.max_size < 3:
            raise ValueError('Morphology maximum side must be >=3.')


@dataclass(frozen=True)
class BioSettings:
    weight_back: float
    scatter_sigma_mm: float
    spacing_mm: float

    def __post_init__(self):
        if not all(math.isfinite(value) for value in (self.weight_back, self.scatter_sigma_mm, self.spacing_mm)):
            raise ValueError('Bio settings must be finite.')
        if not 0 <= self.weight_back <= 1 or self.scatter_sigma_mm < 0 or self.spacing_mm <= 0:
            raise ValueError('Invalid bio geometry or pixel spacing.')
