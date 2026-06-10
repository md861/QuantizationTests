"""Per-channel scaling utilities for ParoQuant experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ChannelScaling:
    """Stores one reversible scaling transform across matrix columns."""

    factors: np.ndarray
    target_max_abs: float


def column_max_abs(matrix: np.ndarray) -> np.ndarray:
    """Return the maximum absolute value in each matrix column."""

    _validate_matrix(matrix)
    return np.max(np.abs(matrix.astype(np.float64, copy=False)), axis=0)


def compute_channel_scaling(
    matrix: np.ndarray,
    *,
    target_max_abs: float | None = None,
) -> ChannelScaling:
    """Compute per-column factors that balance column max-abs magnitudes.

    For each nonzero column j, the factor is:

        factor_j = target_max_abs / max(abs(W[:, j]))

    Zero columns receive factor 1.0 because they are already invariant under
    scaling and cannot be normalised by division.
    """

    max_abs = column_max_abs(matrix)
    target = _target_max_abs(max_abs, target_max_abs)
    factors = np.ones_like(max_abs, dtype=np.float64)
    nonzero = max_abs > 0.0
    factors[nonzero] = target / max_abs[nonzero]
    return ChannelScaling(factors=factors, target_max_abs=target)


def apply_channel_scaling(
    matrix: np.ndarray,
    scaling: ChannelScaling,
) -> np.ndarray:
    """Apply stored per-channel scaling factors to matrix columns."""

    _validate_matrix(matrix)
    factors = _validate_scaling(scaling, matrix.shape[1])
    result = matrix.astype(np.float64, copy=False) * factors
    return result.astype(matrix.dtype, copy=False)


def invert_channel_scaling(
    matrix: np.ndarray,
    scaling: ChannelScaling,
) -> np.ndarray:
    """Undo a previously applied per-channel scaling transform."""

    _validate_matrix(matrix)
    factors = _validate_scaling(scaling, matrix.shape[1])
    result = matrix.astype(np.float64, copy=False) / factors
    return result.astype(matrix.dtype, copy=False)


def balance_channel_max_abs(
    matrix: np.ndarray,
    *,
    target_max_abs: float | None = None,
) -> tuple[np.ndarray, ChannelScaling]:
    """Scale columns toward a shared max-abs target and return metadata."""

    scaling = compute_channel_scaling(matrix, target_max_abs=target_max_abs)
    return apply_channel_scaling(matrix, scaling), scaling


def _target_max_abs(max_abs: np.ndarray, target_max_abs: float | None) -> float:
    if target_max_abs is not None:
        if not np.isfinite(target_max_abs) or target_max_abs <= 0.0:
            raise ValueError("target_max_abs must be a positive finite value")
        return float(target_max_abs)

    nonzero = max_abs[max_abs > 0.0]
    if nonzero.size == 0:
        return 1.0
    return float(np.mean(nonzero))


def _validate_matrix(matrix: np.ndarray) -> None:
    if matrix.ndim != 2:
        raise ValueError("matrix must be a 2D array")
    if not np.issubdtype(matrix.dtype, np.floating):
        raise TypeError("matrix must contain floating-point values")


def _validate_scaling(scaling: ChannelScaling, n_cols: int) -> np.ndarray:
    factors = np.asarray(scaling.factors, dtype=np.float64)
    if factors.ndim != 1:
        raise ValueError("scaling factors must be a 1D array")
    if factors.shape[0] != n_cols:
        raise ValueError("scaling factor count must match matrix columns")
    if not np.all(np.isfinite(factors)):
        raise ValueError("scaling factors must be finite")
    if np.any(factors <= 0.0):
        raise ValueError("scaling factors must be positive")
    if not np.isfinite(scaling.target_max_abs) or scaling.target_max_abs <= 0.0:
        raise ValueError("target_max_abs must be a positive finite value")
    return factors
