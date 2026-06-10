"""Tests for pairwise Givens rotation utilities."""

from __future__ import annotations

import numpy as np
import pytest

from quant.rotations import (
    GivensRotation,
    apply_rotation,
    apply_sequential_rotations,
    optimal_angle,
    rotate_channel_pair,
    rotation_matrix,
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
