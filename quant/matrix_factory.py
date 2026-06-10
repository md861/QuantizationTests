"""Matrix generation utilities for quantization experiments.

The functions in this module intentionally keep the data-generation rules
simple and explicit. They are used to create controlled matrices for studying
how low-bit quantization behaves under different distribution shapes.
"""

from __future__ import annotations

from enum import Enum
from typing import TypeAlias

import numpy as np


Shape: TypeAlias = tuple[int, int]
DTypeLike: TypeAlias = np.dtype | type[np.floating]


class MatrixKind(str, Enum):
    """Supported synthetic matrix families."""

    GAUSSIAN = "gaussian"
    HEAVY_TAILED = "heavy_tailed"
    OUTLIER = "outlier"


def gaussian_matrix(
    shape: Shape,
    *,
    mean: float = 0.0,
    std: float = 1.0,
    seed: int | None = None,
    dtype: DTypeLike = np.float32,
) -> np.ndarray:
    """Generate a matrix with independent Gaussian entries."""

    _validate_shape(shape)
    if std <= 0:
        raise ValueError("std must be positive")

    rng = np.random.default_rng(seed)
    matrix = rng.normal(loc=mean, scale=std, size=shape)
    return matrix.astype(dtype, copy=False)


def heavy_tailed_matrix(
    shape: Shape,
    *,
    df: float = 2.0,
    scale: float = 1.0,
    seed: int | None = None,
    dtype: DTypeLike = np.float32,
) -> np.ndarray:
    """Generate a matrix from a scaled Student-t distribution.

    Smaller degrees of freedom produce heavier tails. The default `df=2.0`
    is intentionally outlier-prone while still easy to reason about.
    """

    _validate_shape(shape)
    if df <= 0:
        raise ValueError("df must be positive")
    if scale <= 0:
        raise ValueError("scale must be positive")

    rng = np.random.default_rng(seed)
    matrix = rng.standard_t(df=df, size=shape) * scale
    return matrix.astype(dtype, copy=False)


def outlier_matrix(
    shape: Shape,
    *,
    base_mean: float = 0.0,
    base_std: float = 1.0,
    outlier_fraction: float = 0.01,
    outlier_scale: float = 10.0,
    seed: int | None = None,
    dtype: DTypeLike = np.float32,
) -> np.ndarray:
    """Generate a Gaussian matrix with a controlled number of large outliers.

    Outliers are injected by selecting random entries and replacing them with
    signed values whose magnitudes are roughly `outlier_scale * base_std`.
    """

    _validate_shape(shape)
    if base_std <= 0:
        raise ValueError("base_std must be positive")
    if not 0 <= outlier_fraction <= 1:
        raise ValueError("outlier_fraction must be between 0 and 1")
    if outlier_scale <= 0:
        raise ValueError("outlier_scale must be positive")

    rng = np.random.default_rng(seed)
    matrix = rng.normal(loc=base_mean, scale=base_std, size=shape)

    total_values = matrix.size
    outlier_count = int(round(total_values * outlier_fraction))
    if outlier_count == 0:
        return matrix.astype(dtype, copy=False)

    flat_indices = rng.choice(total_values, size=outlier_count, replace=False)
    signs = rng.choice(np.array([-1.0, 1.0]), size=outlier_count)
    magnitudes = rng.normal(
        loc=outlier_scale * base_std,
        scale=base_std,
        size=outlier_count,
    )
    matrix.flat[flat_indices] = base_mean + signs * np.abs(magnitudes)
    return matrix.astype(dtype, copy=False)


def make_matrix(
    kind: MatrixKind | str,
    shape: Shape,
    *,
    seed: int | None = None,
    dtype: DTypeLike = np.float32,
    **kwargs: float,
) -> np.ndarray:
    """Generate a matrix by kind.

    Extra keyword arguments are forwarded to the selected generator.
    """

    matrix_kind = MatrixKind(kind)

    if matrix_kind is MatrixKind.GAUSSIAN:
        return gaussian_matrix(shape, seed=seed, dtype=dtype, **kwargs)
    if matrix_kind is MatrixKind.HEAVY_TAILED:
        return heavy_tailed_matrix(shape, seed=seed, dtype=dtype, **kwargs)
    if matrix_kind is MatrixKind.OUTLIER:
        return outlier_matrix(shape, seed=seed, dtype=dtype, **kwargs)

    raise ValueError(f"Unsupported matrix kind: {kind}")


def _validate_shape(shape: Shape) -> None:
    if len(shape) != 2:
        raise ValueError("shape must be a 2D tuple: (rows, columns)")
    rows, cols = shape
    if rows <= 0 or cols <= 0:
        raise ValueError("shape dimensions must be positive")
