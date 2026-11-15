"""Resample film maps using the same forward XY transform as the viewer."""

import numpy as np
from scipy.ndimage import affine_transform


def resample_to_reference(data, source_to_reference_xy, output_shape):
    """Return a map and coverage mask on the reference grid.

    The homogeneous matrix maps source (x, y) to reference (x, y).
    SciPy requires the inverse mapping in array (row, column) order.
    Pixels outside the source are invalid, rather than extended edge doses.
    """
    data = np.asarray(data)
    transform = np.asarray(source_to_reference_xy, dtype=float)
    if data.ndim != 2 or 0 in data.shape or not np.all(np.isfinite(data)):
        raise ValueError('A finite, nonempty 2D film map is required.')
    if (transform.shape != (3, 3) or not np.all(np.isfinite(transform))
            or not np.allclose(transform[2], [0, 0, 1])):
        raise ValueError('A finite affine 3x3 XY transform is required.')
    inverse = np.linalg.inv(transform)
    kwargs = dict(matrix=inverse[:2, :2][::-1, ::-1],
                  offset=inverse[:2, 2][::-1], output_shape=output_shape,
                  order=1, mode='constant', cval=0.0, prefilter=False)
    aligned = affine_transform(data.astype(float), **kwargs)
    coverage = affine_transform(np.ones(data.shape), **kwargs) > 0.999999
    return aligned, coverage
