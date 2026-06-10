"""Tests for per-channel scaling utilities."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from quant.scaling import (
    ChannelScaling,
    apply_channel_scaling,
    balance_channel_max_abs,
    column_max_abs,
    compute_channel_scaling,
    invert_channel_scaling,
)
from quant.visualize import plot_channel_scaling_quantization_dashboard


def test_column_max_abs_returns_per_column_values() -> None:
    matrix = np.array(
        [
            [1.0, -5.0, 0.0],
            [-3.0, 2.0, 0.0],
        ],
        dtype=np.float32,
    )

    result = column_max_abs(matrix)

    np.testing.assert_allclose(result, np.array([3.0, 5.0, 0.0]))
    assert result.dtype == np.float64


def test_compute_channel_scaling_balances_nonzero_column_max_abs() -> None:
    matrix = np.array(
        [
            [1.0, -10.0, 0.0],
            [-2.0, 5.0, 0.0],
            [4.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )

    scaling = compute_channel_scaling(matrix)
    scaled = apply_channel_scaling(matrix, scaling)

    assert scaling.target_max_abs == pytest.approx(7.0)
    np.testing.assert_allclose(column_max_abs(scaled), np.array([7.0, 7.0, 0.0]))
    np.testing.assert_allclose(scaling.factors, np.array([7.0 / 4.0, 7.0 / 10.0, 1.0]))


def test_custom_target_max_abs_is_respected() -> None:
    matrix = np.array([[1.0, 10.0], [-2.0, -5.0]], dtype=np.float64)

    scaled, scaling = balance_channel_max_abs(matrix, target_max_abs=3.0)

    assert scaling.target_max_abs == pytest.approx(3.0)
    np.testing.assert_allclose(column_max_abs(scaled), np.array([3.0, 3.0]))


def test_scaling_then_inverse_recovers_original_matrix() -> None:
    rng = np.random.default_rng(19)
    matrix = rng.standard_normal((8, 5)).astype(np.float64)
    matrix[:, 0] *= 50.0

    scaled, scaling = balance_channel_max_abs(matrix)
    recovered = invert_channel_scaling(scaled, scaling)

    np.testing.assert_allclose(recovered, matrix, atol=1e-12)


def test_scaling_preserves_input_dtype() -> None:
    rng = np.random.default_rng(23)
    matrix = rng.standard_normal((6, 4)).astype(np.float32)

    scaled, scaling = balance_channel_max_abs(matrix)
    recovered = invert_channel_scaling(scaled, scaling)

    assert scaled.dtype == np.float32
    assert recovered.dtype == np.float32
    np.testing.assert_allclose(recovered, matrix, atol=1e-6)


def test_zero_matrix_uses_identity_factors() -> None:
    matrix = np.zeros((3, 4), dtype=np.float64)

    scaled, scaling = balance_channel_max_abs(matrix)

    assert scaling.target_max_abs == pytest.approx(1.0)
    np.testing.assert_array_equal(scaling.factors, np.ones(4))
    np.testing.assert_array_equal(scaled, matrix)


def test_apply_channel_scaling_does_not_mutate_input() -> None:
    matrix = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    original = matrix.copy()
    scaling = ChannelScaling(factors=np.array([2.0, 0.5]), target_max_abs=1.0)

    apply_channel_scaling(matrix, scaling)

    np.testing.assert_array_equal(matrix, original)


def test_scaling_rejects_non_2d_input() -> None:
    with pytest.raises(ValueError, match="2D"):
        compute_channel_scaling(np.ones((2, 2, 2), dtype=np.float32))


def test_scaling_rejects_integer_input() -> None:
    with pytest.raises(TypeError, match="floating"):
        compute_channel_scaling(np.ones((2, 2), dtype=np.int32))


def test_scaling_rejects_invalid_target() -> None:
    matrix = np.ones((2, 2), dtype=np.float64)

    with pytest.raises(ValueError, match="target_max_abs"):
        compute_channel_scaling(matrix, target_max_abs=0.0)


def test_apply_scaling_rejects_factor_count_mismatch() -> None:
    matrix = np.ones((2, 3), dtype=np.float64)
    scaling = ChannelScaling(factors=np.ones(2), target_max_abs=1.0)

    with pytest.raises(ValueError, match="columns"):
        apply_channel_scaling(matrix, scaling)


def test_apply_scaling_rejects_nonpositive_factors() -> None:
    matrix = np.ones((2, 2), dtype=np.float64)
    scaling = ChannelScaling(factors=np.array([1.0, 0.0]), target_max_abs=1.0)

    with pytest.raises(ValueError, match="positive"):
        apply_channel_scaling(matrix, scaling)


def test_channel_scaling_quantization_dashboard_saves_png() -> None:
    rng = np.random.default_rng(101)
    matrix = rng.standard_normal((24, 16)).astype(np.float32)
    matrix[:, 0] *= 50.0
    matrix[:, 1] *= 20.0
    matrix[:, 2] *= 10.0
    output_path = Path("plots/channel_scaling_dashboard.png")

    fig = plot_channel_scaling_quantization_dashboard(matrix, output_path=output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0
    assert len(fig.axes) >= 9
    plt.close(fig)
