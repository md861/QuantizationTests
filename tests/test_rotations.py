"""Tests for pairwise Givens rotation utilities."""

from __future__ import annotations

import numpy as np
import pytest

from quant.rotations import (
    GivensRotation,
    apply_rotation,
    apply_sequential_rotations,
    channel_widths,
    optimal_angle,
    rotate_channel_pair,
    rotate_top_width_pairs,
    rotation_matrix,
    top_width_channel_pairs,
)


# ── rotation_matrix ───────────────────────────────────────────────────────────

def test_rotation_matrix_is_identity_at_zero_angle() -> None:
    R = rotation_matrix(5, 1, 3, 0.0)
    np.testing.assert_allclose(R, np.eye(5), atol=1e-12)


def test_rotation_matrix_is_orthogonal() -> None:
    R = rotation_matrix(6, 0, 4, 0.7)
    np.testing.assert_allclose(R.T @ R, np.eye(6), atol=1e-12)


def test_rotation_matrix_leaves_other_rows_and_cols_as_identity() -> None:
    n = 5
    R = rotation_matrix(n, 1, 3, 0.5)
    for k in range(n):
        if k not in {1, 3}:
            assert R[k, k] == pytest.approx(1.0)
            for m in range(n):
                if m != k:
                    assert R[k, m] == pytest.approx(0.0)


def test_rotation_matrix_subblock_values() -> None:
    theta = np.pi / 4
    R = rotation_matrix(4, 0, 2, theta)
    c, s = np.cos(theta), np.sin(theta)
    assert R[0, 0] == pytest.approx(c)
    assert R[2, 2] == pytest.approx(c)
    assert R[0, 2] == pytest.approx(-s)
    assert R[2, 0] == pytest.approx(s)


def test_rotation_matrix_returns_float64() -> None:
    R = rotation_matrix(4, 0, 1, 0.3)
    assert R.dtype == np.float64


# ── apply_rotation ────────────────────────────────────────────────────────────

def test_apply_rotation_at_zero_is_identity() -> None:
    rng = np.random.default_rng(7)
    matrix = rng.standard_normal((6, 6)).astype(np.float32)
    rotated = apply_rotation(matrix, 0, 3, 0.0)
    np.testing.assert_allclose(rotated, matrix, atol=1e-6)


def test_apply_rotation_preserves_frobenius_norm() -> None:
    rng = np.random.default_rng(42)
    matrix = rng.standard_normal((8, 8)).astype(np.float32)
    rotated = apply_rotation(matrix, 2, 5, 0.9)
    np.testing.assert_allclose(
        np.linalg.norm(matrix), np.linalg.norm(rotated), rtol=1e-5
    )


def test_apply_rotation_is_invertible() -> None:
    rng = np.random.default_rng(99)
    matrix = rng.standard_normal((5, 5)).astype(np.float64)
    theta = 1.2
    rotated = apply_rotation(matrix, 1, 4, theta)
    recovered = apply_rotation(rotated, 1, 4, -theta)
    np.testing.assert_allclose(recovered, matrix, atol=1e-10)


def test_apply_rotation_only_modifies_target_columns() -> None:
    rng = np.random.default_rng(13)
    matrix = rng.standard_normal((5, 6)).astype(np.float64)
    rotated = apply_rotation(matrix, 1, 4, 0.6)
    for k in range(6):
        if k not in {1, 4}:
            np.testing.assert_array_equal(rotated[:, k], matrix[:, k])


def test_apply_rotation_preserves_dtype_float32() -> None:
    rng = np.random.default_rng(0)
    matrix = rng.standard_normal((4, 4)).astype(np.float32)
    rotated = apply_rotation(matrix, 0, 2, 0.5)
    assert rotated.dtype == np.float32


def test_apply_rotation_preserves_dtype_float64() -> None:
    rng = np.random.default_rng(0)
    matrix = rng.standard_normal((4, 4)).astype(np.float64)
    rotated = apply_rotation(matrix, 0, 2, 0.5)
    assert rotated.dtype == np.float64


def test_apply_rotation_does_not_mutate_input() -> None:
    rng = np.random.default_rng(5)
    matrix = rng.standard_normal((4, 4)).astype(np.float64)
    original_copy = matrix.copy()
    apply_rotation(matrix, 0, 1, 0.7)
    np.testing.assert_array_equal(matrix, original_copy)


def test_apply_rotation_agrees_with_rotation_matrix() -> None:
    rng = np.random.default_rng(21)
    matrix = rng.standard_normal((5, 5)).astype(np.float64)
    theta = 0.6
    i, j = 1, 3
    via_function = apply_rotation(matrix, i, j, theta)
    via_matrix = matrix @ rotation_matrix(5, i, j, theta)
    np.testing.assert_allclose(via_function, via_matrix, atol=1e-12)


# ── optimal_angle ─────────────────────────────────────────────────────────────

def test_optimal_angle_does_not_increase_max_abs() -> None:
    rng = np.random.default_rng(42)
    matrix = rng.standard_normal((16, 16)).astype(np.float64)
    matrix[3, 0] = 50.0
    i, j = 0, 1
    before = max(
        float(np.max(np.abs(matrix[:, i]))),
        float(np.max(np.abs(matrix[:, j]))),
    )
    theta = optimal_angle(matrix, i, j)
    rotated = apply_rotation(matrix, i, j, theta)
    after = max(
        float(np.max(np.abs(rotated[:, i]))),
        float(np.max(np.abs(rotated[:, j]))),
    )
    assert after <= before + 1e-6


def test_optimal_angle_improves_single_outlier_column() -> None:
    matrix = np.ones((4, 4), dtype=np.float64)
    matrix[0, 0] = 100.0
    theta = optimal_angle(matrix, 0, 1)
    rotated = apply_rotation(matrix, 0, 1, theta)
    after = max(
        float(np.max(np.abs(rotated[:, 0]))),
        float(np.max(np.abs(rotated[:, 1]))),
    )
    assert after < 100.0 - 1e-6


def test_optimal_angle_returns_float() -> None:
    matrix = np.eye(4, dtype=np.float64)
    result = optimal_angle(matrix, 0, 1)
    assert isinstance(result, float)


def test_optimal_angle_is_in_range() -> None:
    rng = np.random.default_rng(7)
    matrix = rng.standard_normal((8, 8)).astype(np.float64)
    theta = optimal_angle(matrix, 0, 3)
    assert 0.0 <= theta < np.pi


# ── rotate_channel_pair ───────────────────────────────────────────────────────

def test_rotate_channel_pair_result_matches_apply_rotation() -> None:
    rng = np.random.default_rng(55)
    matrix = rng.standard_normal((8, 8)).astype(np.float64)
    matrix[2, 3] = 30.0
    rotated, theta = rotate_channel_pair(matrix, 2, 3)
    expected = apply_rotation(matrix, 2, 3, theta)
    np.testing.assert_allclose(rotated, expected, atol=1e-10)


def test_rotate_channel_pair_returns_angle_from_valid_range() -> None:
    rng = np.random.default_rng(9)
    matrix = rng.standard_normal((6, 6)).astype(np.float64)
    _, theta = rotate_channel_pair(matrix, 0, 5)
    assert 0.0 <= theta < np.pi


# ── top-width pair selection ─────────────────────────────────────────────────

def test_channel_widths_returns_column_max_abs() -> None:
    matrix = np.array(
        [
            [1.0, -5.0, 2.0],
            [-3.0, 4.0, -7.0],
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(channel_widths(matrix), np.array([3.0, 5.0, 7.0]))


def test_top_width_channel_pairs_selects_largest_width_differences() -> None:
    matrix = np.array(
        [
            [1.0, 10.0, 4.0, 7.0],
            [1.0, 10.0, 4.0, 7.0],
        ],
        dtype=np.float64,
    )
    pairs = top_width_channel_pairs(matrix, top_fraction=0.34, independent=False)
    assert pairs == [(0, 1), (0, 3), (1, 2)]


def test_top_width_channel_pairs_can_enforce_independence() -> None:
    matrix = np.array(
        [
            [1.0, 10.0, 4.0, 7.0],
            [1.0, 10.0, 4.0, 7.0],
        ],
        dtype=np.float64,
    )
    pairs = top_width_channel_pairs(matrix, top_fraction=1.0, independent=True)
    used: set[int] = set()
    for i, j in pairs:
        assert i not in used
        assert j not in used
        used.update((i, j))
    assert pairs[0] == (0, 1)


def test_rotate_top_width_pairs_preserves_norm_and_records_angles() -> None:
    rng = np.random.default_rng(123)
    matrix = rng.standard_normal((8, 8)).astype(np.float64)
    matrix[0, 0] = 40.0
    rotated, rotations = rotate_top_width_pairs(
        matrix,
        top_fraction=0.25,
        independent=True,
        n_search=72,
    )
    assert rotations
    assert all(0.0 <= r.theta < np.pi for r in rotations)
    np.testing.assert_allclose(
        np.linalg.norm(rotated), np.linalg.norm(matrix), rtol=1e-10
    )


def test_top_width_channel_pairs_rejects_invalid_fraction() -> None:
    matrix = np.ones((4, 4), dtype=np.float64)
    with pytest.raises(ValueError, match="top_fraction"):
        top_width_channel_pairs(matrix, top_fraction=0.0)
    with pytest.raises(ValueError, match="top_fraction"):
        top_width_channel_pairs(matrix, top_fraction=1.5)


# ── apply_sequential_rotations ────────────────────────────────────────────────

def test_apply_sequential_rotations_empty_list_is_identity() -> None:
    rng = np.random.default_rng(3)
    matrix = rng.standard_normal((5, 5)).astype(np.float64)
    result = apply_sequential_rotations(matrix, [])
    np.testing.assert_array_equal(result, matrix)


def test_apply_sequential_rotations_chains_correctly() -> None:
    rng = np.random.default_rng(17)
    matrix = rng.standard_normal((6, 6)).astype(np.float64)
    rotations = [
        GivensRotation(i=0, j=1, theta=0.5),
        GivensRotation(i=2, j=3, theta=1.1),
        GivensRotation(i=4, j=5, theta=0.3),
    ]
    result = apply_sequential_rotations(matrix, rotations)
    expected = matrix
    for r in rotations:
        expected = apply_rotation(expected, r.i, r.j, r.theta)
    np.testing.assert_allclose(result, expected, atol=1e-10)


def test_apply_sequential_rotations_preserves_frobenius_norm() -> None:
    rng = np.random.default_rng(88)
    matrix = rng.standard_normal((6, 6)).astype(np.float64)
    rotations = [
        GivensRotation(i=0, j=2, theta=0.4),
        GivensRotation(i=1, j=3, theta=0.9),
    ]
    result = apply_sequential_rotations(matrix, rotations)
    np.testing.assert_allclose(
        np.linalg.norm(matrix), np.linalg.norm(result), rtol=1e-10
    )


# ── validation errors ─────────────────────────────────────────────────────────

def test_rotation_matrix_raises_for_equal_indices() -> None:
    with pytest.raises(ValueError, match="distinct"):
        rotation_matrix(5, 2, 2, 0.5)


def test_rotation_matrix_raises_for_out_of_range_index() -> None:
    with pytest.raises(ValueError):
        rotation_matrix(5, -1, 2, 0.5)
    with pytest.raises(ValueError):
        rotation_matrix(5, 0, 10, 0.5)


def test_apply_rotation_raises_for_non_2d() -> None:
    with pytest.raises(ValueError, match="2D"):
        apply_rotation(np.ones((3, 4, 5), dtype=np.float32), 0, 1, 0.5)


def test_apply_rotation_raises_for_integer_dtype() -> None:
    with pytest.raises(TypeError, match="floating"):
        apply_rotation(np.ones((4, 4), dtype=np.int32), 0, 1, 0.5)


def test_apply_rotation_raises_for_equal_channel_indices() -> None:
    with pytest.raises(ValueError, match="distinct"):
        apply_rotation(np.ones((4, 4), dtype=np.float32), 1, 1, 0.5)


def test_optimal_angle_raises_for_invalid_n_search() -> None:
    matrix = np.ones((4, 4), dtype=np.float64)
    with pytest.raises(ValueError, match="n_search"):
        optimal_angle(matrix, 0, 1, n_search=0)


# ── Verification: entry-zeroing, Givens QR, column orthogonalisation ─────────

def _givens_qr(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """QR via cascaded Givens row-rotations built on apply_rotation.

    Row rotation on R uses the transpose trick:
        apply_rotation(R.T, i-1, i, theta).T  ≡  G_step @ R

    Q is accumulated via right-multiplication by G_step^T at each step:
        apply_rotation(Q, i-1, i, theta)  ≡  Q @ G_step^T

    After all steps: Q @ R == A  (complete form, Q is m×m).
    """
    m, n = A.shape
    R = A.astype(np.float64).copy()
    Q = np.eye(m, dtype=np.float64)
    for j in range(n):
        for i in range(m - 1, j, -1):
            a, b = R[i - 1, j], R[i, j]
            if abs(b) < 1e-14:
                continue
            theta = float(np.arctan2(b, a))
            R = np.ascontiguousarray(apply_rotation(R.T, i - 1, i, theta).T)
            Q = apply_rotation(Q, i - 1, i, theta)
    return Q, R


def test_exact_angle_zeros_target_entry() -> None:
    """arctan2(b, a) applied to cols (i, j) must zero entry M[k, j] exactly."""
    rng = np.random.default_rng(77)
    matrix = rng.standard_normal((6, 6)).astype(np.float64)
    k, i, j = 2, 0, 3
    a, b = float(matrix[k, i]), float(matrix[k, j])
    theta = float(np.arctan2(b, a))
    rotated = apply_rotation(matrix, i, j, theta)
    assert abs(rotated[k, j]) < 1e-10
    np.testing.assert_allclose(abs(rotated[k, i]), np.sqrt(a**2 + b**2), atol=1e-10)


def test_givens_qr_via_cascaded_rotations() -> None:
    """QR built from apply_rotation must satisfy Q@R=A, Q^T Q=I, R upper-triangular."""
    rng = np.random.default_rng(42)
    A = rng.standard_normal((6, 4)).astype(np.float64)
    Q, R = _givens_qr(A)
    np.testing.assert_allclose(Q @ R, A, atol=1e-10)
    np.testing.assert_allclose(Q.T @ Q, np.eye(Q.shape[0]), atol=1e-10)
    m, n = A.shape
    for row in range(1, m):
        for col in range(min(row, n)):
            assert abs(R[row, col]) < 1e-10, f"R[{row},{col}] = {R[row,col]:.2e} not zero"


def test_givens_qr_diagonal_magnitudes_match_numpy() -> None:
    """Absolute diagonal of our Givens-QR R must match numpy.linalg.qr (sign-free)."""
    rng = np.random.default_rng(99)
    A = rng.standard_normal((5, 5)).astype(np.float64)
    _, R_ours = _givens_qr(A)
    _, R_numpy = np.linalg.qr(A)
    np.testing.assert_allclose(
        np.abs(np.diag(R_ours)),
        np.abs(np.diag(R_numpy)),
        atol=1e-8,
    )


def test_exact_angle_orthogonalises_column_pair() -> None:
    """Jacobi angle 0.5*arctan2(2*dot, ||ci||²-||cj||²) must make columns orthogonal."""
    rng = np.random.default_rng(11)
    matrix = rng.standard_normal((8, 8)).astype(np.float64)
    i, j = 0, 3
    ci, cj = matrix[:, i].copy(), matrix[:, j].copy()
    dot_ij = float(np.dot(ci, cj))
    diff_norms = float(np.dot(ci, ci) - np.dot(cj, cj))
    theta = 0.5 * float(np.arctan2(2.0 * dot_ij, diff_norms))
    rotated = apply_rotation(matrix, i, j, theta)
    assert abs(np.dot(rotated[:, i], rotated[:, j])) < 1e-10
