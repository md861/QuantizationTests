"""Tests for symmetric quantization."""

import numpy as np
import pytest

from quant.matrix_factory import gaussian_matrix
from quant.quantizer import (
    grouped_symmetric_quantize,
    quantize_int4,
    quantize_int4_grouped,
    quantize_int8,
    quantize_int8_grouped,
    symmetric_quantize,
)


def test_int8_quantization_uses_expected_range() -> None:
    matrix = np.array([[-1.0, 0.0, 1.0]], dtype=np.float32)

    result = quantize_int8(matrix)

    assert result.bitwidth == 8
    assert result.qmin == -127
    assert result.qmax == 127
    assert result.quantized.min() >= -127
    assert result.quantized.max() <= 127
    assert result.quantized.dtype == np.int8


def test_int4_quantization_uses_expected_range() -> None:
    matrix = np.array([[-1.0, 0.0, 1.0]], dtype=np.float32)

    result = quantize_int4(matrix)

    assert result.bitwidth == 4
    assert result.qmin == -7
    assert result.qmax == 7
    assert result.quantized.min() >= -7
    assert result.quantized.max() <= 7
    assert result.quantized.dtype == np.int8


def test_symmetric_quantization_scale_formula() -> None:
    matrix = np.array([[-2.0, 0.0, 2.0]], dtype=np.float32)

    result = quantize_int8(matrix)

    assert result.scale == pytest.approx(2.0 / 127.0)
    np.testing.assert_array_equal(result.quantized, np.array([[-127, 0, 127]], dtype=np.int8))
    np.testing.assert_allclose(result.dequantized, matrix)


def test_zero_matrix_quantizes_without_division_by_zero() -> None:
    matrix = np.zeros((3, 4), dtype=np.float32)

    result = quantize_int4(matrix)

    assert result.scale == pytest.approx(1.0)
    np.testing.assert_array_equal(result.quantized, np.zeros((3, 4), dtype=np.int8))
    np.testing.assert_array_equal(result.dequantized, matrix)


def test_dequantized_matrix_preserves_shape_and_dtype() -> None:
    matrix = gaussian_matrix((5, 20), seed=42, dtype=np.float32)

    result = quantize_int8(matrix)

    assert result.quantized.shape == matrix.shape
    assert result.dequantized.shape == matrix.shape
    assert result.dequantized.dtype == matrix.dtype


def test_int8_has_lower_or_equal_reconstruction_error_than_int4() -> None:
    matrix = gaussian_matrix((20, 20), seed=9)

    int8 = quantize_int8(matrix)
    int4 = quantize_int4(matrix)
    int8_error = np.mean((matrix - int8.dequantized) ** 2)
    int4_error = np.mean((matrix - int4.dequantized) ** 2)

    assert int8_error <= int4_error


def test_symmetric_quantize_rejects_unsupported_bitwidth() -> None:
    matrix = np.ones((2, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="bitwidth"):
        symmetric_quantize(matrix, bitwidth=3)


def test_symmetric_quantize_rejects_non_2d_input() -> None:
    matrix = np.ones((2, 2, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="2D"):
        symmetric_quantize(matrix, bitwidth=8)


def test_symmetric_quantize_rejects_integer_input() -> None:
    matrix = np.ones((2, 2), dtype=np.int32)

    with pytest.raises(TypeError, match="floating-point"):
        symmetric_quantize(matrix, bitwidth=8)


def test_grouped_int4_quantization_uses_expected_range_and_metadata() -> None:
    matrix = np.array([[-2.0, 0.0, 4.0, 8.0]], dtype=np.float32)

    result = quantize_int4_grouped(matrix, group_size=2)

    assert result.bitwidth == 4
    assert result.qmin == -7
    assert result.qmax == 7
    assert result.quantized.min() >= -7
    assert result.quantized.max() <= 7
    assert result.quantized.dtype == np.int8
    assert result.group_size == 2
    np.testing.assert_allclose(result.scales, np.array([2.0 / 7.0, 8.0 / 7.0]))


def test_grouped_int8_quantization_wrapper_sets_bitwidth() -> None:
    matrix = np.ones((2, 3), dtype=np.float32)

    result = quantize_int8_grouped(matrix, group_size=1)

    assert result.bitwidth == 8
    assert result.qmax == 127
    assert result.scales is not None
    assert result.scales.shape == (3,)


def test_grouped_quantization_handles_last_partial_group() -> None:
    matrix = np.array([[1.0, 2.0, 3.0, 4.0, 20.0]], dtype=np.float32)

    result = quantize_int4_grouped(matrix, group_size=2)

    assert result.scales is not None
    assert result.scales.shape == (3,)
    np.testing.assert_allclose(
        result.scales,
        np.array([2.0 / 7.0, 4.0 / 7.0, 20.0 / 7.0]),
    )


def test_grouped_quantization_zero_group_uses_unit_scale() -> None:
    matrix = np.array([[0.0, 0.0, 3.0, -3.0]], dtype=np.float32)

    result = quantize_int4_grouped(matrix, group_size=2)

    np.testing.assert_allclose(result.scales, np.array([1.0, 3.0 / 7.0]))
    np.testing.assert_array_equal(result.dequantized[:, :2], np.zeros((1, 2), dtype=np.float32))


def test_group_size_covering_all_columns_matches_global_quantization() -> None:
    matrix = gaussian_matrix((8, 8), seed=41)

    global_result = quantize_int4(matrix)
    grouped_result = quantize_int4_grouped(matrix, group_size=matrix.shape[1])

    assert grouped_result.scale == pytest.approx(global_result.scale)
    np.testing.assert_array_equal(grouped_result.quantized, global_result.quantized)
    np.testing.assert_allclose(grouped_result.dequantized, global_result.dequantized)


def test_grouped_quantization_can_reduce_error_when_outlier_group_is_isolated() -> None:
    matrix = np.ones((4, 4), dtype=np.float32)
    matrix[0, 0] = 100.0

    global_result = quantize_int4(matrix)
    grouped_result = quantize_int4_grouped(matrix, group_size=1)
    global_error = np.mean((matrix - global_result.dequantized) ** 2)
    grouped_error = np.mean((matrix - grouped_result.dequantized) ** 2)

    assert grouped_error < global_error


def test_grouped_quantization_rejects_invalid_group_size() -> None:
    matrix = np.ones((2, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="group_size"):
        grouped_symmetric_quantize(matrix, bitwidth=4, group_size=0)
