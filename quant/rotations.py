"""Pairwise rotation utilities for ParoQuant experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GivensRotation:
    """Records a single pairwise Givens rotation."""

    i: int
    j: int
    theta: float


def rotation_matrix(n: int, i: int, j: int, theta: float) -> np.ndarray:
    """Return an n×n Givens rotation matrix for channel pair (i, j).

    The matrix is an identity with the (i, j) subblock replaced:

        R[i, i] =  cos(theta)    R[i, j] = -sin(theta)
        R[j, i] =  sin(theta)    R[j, j] =  cos(theta)

    Right-multiplying a matrix W by R rotates columns i and j of W:

        W' = W @ R
    """
    _validate_channel_pair(n, i, j)
    R = np.eye(n, dtype=np.float64)
    c, s = np.cos(theta), np.sin(theta)
    R[i, i] = c
    R[j, j] = c
    R[i, j] = -s
    R[j, i] = s
    return R


def apply_rotation(
    matrix: np.ndarray,
    i: int,
    j: int,
    theta: float,
) -> np.ndarray:
    """Apply a Givens rotation to columns i and j of matrix.

    The columns are updated in-place on a copy:

        col_i' =  cos(theta) * col_i + sin(theta) * col_j
        col_j' = -sin(theta) * col_i + cos(theta) * col_j

    The returned array has the same shape and dtype as the input.
    """
    _validate_matrix(matrix)
    _validate_channel_pair(matrix.shape[1], i, j)

    col_i = matrix[:, i].astype(np.float64)
    col_j = matrix[:, j].astype(np.float64)
    c, s = np.cos(theta), np.sin(theta)

    result = matrix.copy()
    result[:, i] = (c * col_i + s * col_j).astype(matrix.dtype)
    result[:, j] = (-s * col_i + c * col_j).astype(matrix.dtype)
    return result


def optimal_angle(
    matrix: np.ndarray,
    i: int,
    j: int,
    *,
    n_search: int = 360,
) -> float:
    """Find theta in [0, pi) minimising max-abs across columns i and j.

    Uses a uniform grid search.  The objective is pi-periodic: rotating by
    theta and theta+pi negates both columns, leaving max-abs unchanged.
    """
    _validate_matrix(matrix)
    _validate_channel_pair(matrix.shape[1], i, j)
    if n_search < 1:
        raise ValueError("n_search must be at least 1")

    col_i = matrix[:, i].astype(np.float64)
    col_j = matrix[:, j].astype(np.float64)

    best_theta = 0.0
    best_cost = float("inf")

    for theta in np.linspace(0.0, np.pi, n_search, endpoint=False):
        c, s = np.cos(theta), np.sin(theta)
        cost = max(
            float(np.max(np.abs(c * col_i + s * col_j))),
            float(np.max(np.abs(-s * col_i + c * col_j))),
        )
        if cost < best_cost:
            best_cost = cost
            best_theta = float(theta)

    return best_theta


def rotate_channel_pair(
    matrix: np.ndarray,
    i: int,
    j: int,
    *,
    n_search: int = 360,
) -> tuple[np.ndarray, float]:
    """Rotate columns i and j by the optimal angle.

    Returns ``(rotated_matrix, theta)`` so callers can record the angle used.
    """
    theta = optimal_angle(matrix, i, j, n_search=n_search)
    return apply_rotation(matrix, i, j, theta), theta


def apply_sequential_rotations(
    matrix: np.ndarray,
    rotations: list[GivensRotation],
) -> np.ndarray:
    """Apply a sequence of GivensRotations in order."""
    _validate_matrix(matrix)
    result = matrix
    for r in rotations:
        result = apply_rotation(result, r.i, r.j, r.theta)
    return result


def _validate_matrix(matrix: np.ndarray) -> None:
    if matrix.ndim != 2:
        raise ValueError("matrix must be a 2D array")
    if not np.issubdtype(matrix.dtype, np.floating):
        raise TypeError("matrix must contain floating-point values")


def _validate_channel_pair(n_cols: int, i: int, j: int) -> None:
    if not (0 <= i < n_cols):
        raise ValueError(f"channel index i={i} is out of range for n_cols={n_cols}")
    if not (0 <= j < n_cols):
        raise ValueError(f"channel index j={j} is out of range for n_cols={n_cols}")
    if i == j:
        raise ValueError("channel indices i and j must be distinct")
