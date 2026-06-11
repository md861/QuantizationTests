"""Pairwise rotation utilities for ParoQuant experiments."""

from __future__ import annotations

import heapq
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


def channel_widths(matrix: np.ndarray) -> np.ndarray:
    """Return the max-abs width of each channel/column."""
    _validate_matrix(matrix)
    return np.max(np.abs(matrix), axis=0).astype(np.float64)


def top_width_channel_pairs(
    matrix: np.ndarray,
    *,
    top_fraction: float = 0.10,
    independent: bool = True,
) -> list[tuple[int, int]]:
    """Select channel pairs with the largest max-abs width differences.

    ``top_fraction`` is applied to all possible unordered channel pairs.  When
    ``independent`` is true, pairs are greedily filtered so each channel appears
    in at most one returned pair, matching ParoQuant's independent-rotation
    constraint.
    """
    _validate_matrix(matrix)
    _validate_top_fraction(top_fraction)

    n_cols = matrix.shape[1]
    if n_cols < 2:
        return []

    total_pairs = n_cols * (n_cols - 1) // 2
    n_candidates = max(1, int(np.ceil(top_fraction * total_pairs)))
    candidates = _top_width_pair_candidates(
        channel_widths(matrix),
        n_candidates=n_candidates,
    )

    if not independent:
        return [(i, j) for _, i, j in candidates]

    selected: list[tuple[int, int]] = []
    used: set[int] = set()
    for _, i, j in candidates:
        if i in used or j in used:
            continue
        selected.append((i, j))
        used.update((i, j))
    return selected


def _top_width_pair_candidates(
    widths: np.ndarray,
    *,
    n_candidates: int,
) -> list[tuple[float, int, int]]:
    """Return candidate pairs with largest width differences.

    Width differences are maximized by pairing low-width and high-width
    channels. A heap over the sorted channel widths gives the same ordering as
    sorting all unordered pairs by ``(-difference, i, j)`` without materializing
    the full O(n²) pair list for very wide layers.
    """
    n_cols = len(widths)
    if n_cols < 2 or n_candidates < 1:
        return []

    sorted_widths = sorted((float(width), idx) for idx, width in enumerate(widths))
    heap: list[tuple[float, int, int, int, int]] = []
    for lo in range(n_cols - 1):
        hi = n_cols - 1
        item = _heap_pair_item(sorted_widths, lo, hi)
        heapq.heappush(heap, item)

    candidates: list[tuple[float, int, int]] = []
    while heap and len(candidates) < n_candidates:
        neg_diff, i, j, lo, hi = heapq.heappop(heap)
        candidates.append((-neg_diff, i, j))
        next_hi = hi - 1
        if next_hi > lo:
            heapq.heappush(heap, _heap_pair_item(sorted_widths, lo, next_hi))

    return candidates


def _heap_pair_item(
    sorted_widths: list[tuple[float, int]],
    lo: int,
    hi: int,
) -> tuple[float, int, int, int, int]:
    low_width, low_idx = sorted_widths[lo]
    high_width, high_idx = sorted_widths[hi]
    i, j = sorted((low_idx, high_idx))
    return (-(high_width - low_width), i, j, lo, hi)


def rotate_top_width_pairs(
    matrix: np.ndarray,
    *,
    top_fraction: float = 0.10,
    independent: bool = True,
    n_search: int = 360,
) -> tuple[np.ndarray, list[GivensRotation]]:
    """Rotate selected top-width-difference channel pairs in sequence."""
    pairs = top_width_channel_pairs(
        matrix,
        top_fraction=top_fraction,
        independent=independent,
    )
    result = matrix
    rotations: list[GivensRotation] = []
    for i, j in pairs:
        theta = optimal_angle(result, i, j, n_search=n_search)
        result = apply_rotation(result, i, j, theta)
        rotations.append(GivensRotation(i=i, j=j, theta=theta))
    return result, rotations


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


def _validate_top_fraction(top_fraction: float) -> None:
    if not np.isfinite(top_fraction):
        raise ValueError("top_fraction must be finite")
    if not (0.0 < top_fraction <= 1.0):
        raise ValueError("top_fraction must be in the interval (0, 1]")
