"""Tests for quantization error metrics."""

import math

import numpy as np
import pytest

from quant.matrix_factory import gaussian_matrix
from quant.metrics import (
    compute_quantization_metrics,
    cosine_similarity,
    max_absolute_error,
    mean_absolute_error,
    mean_squared_error,
    relative_frobenius_error,
    signal_to_noise_ratio_db,
)
from quant.quantizer import quantize_int4


def test_basic_error_metrics_match_known_values() -> None:
    original = np.array([[1.0, -2.0], [3.0, -4.0]], dtype=np.float32)
    reconstructed = np.array([[1.5, -1.0], [2.0, -5.0]], dtype=np.float32)

    assert mean_squared_error(original, reconstructed) == pytest.approx(0.8125)
    assert mean_absolute_error(original, reconstructed) == pytest.approx(0.875)
    assert max_absolute_error(original, reconstructed) == pytest.approx(1.0)


def test_relative_frobenius_error_matches_known_value() -> None:
    original = np.array([[3.0, 4.0]], dtype=np.float32)
    reconstructed = np.array([[0.0, 0.0]], dtype=np.float32)

    assert relative_frobenius_error(original, reconstructed) == pytest.approx(1.0)


def test_cosine_similarity_matches_known_value() -> None:
    original = np.array([[1.0, 0.0]], dtype=np.float32)
    reconstructed = np.array([[0.0, 1.0]], dtype=np.float32)

    assert cosine_similarity(original, reconstructed) == pytest.approx(0.0)


def test_snr_is_infinite_for_exact_reconstruction() -> None:
    original = np.eye(3, dtype=np.float32)

    assert signal_to_noise_ratio_db(original, original) == float("inf")


def test_zero_matrices_have_exact_reconstruction_conventions() -> None:
    original = np.zeros((2, 2), dtype=np.float32)
    reconstructed = np.zeros((2, 2), dtype=np.float32)

    assert relative_frobenius_error(original, reconstructed) == pytest.approx(0.0)
    assert cosine_similarity(original, reconstructed) == pytest.approx(1.0)
    assert signal_to_noise_ratio_db(original, reconstructed) == float("inf")


def test_compute_quantization_metrics_includes_optional_integer_diagnostics() -> None:
    original = gaussian_matrix((5, 20), seed=42)
    result = quantize_int4(original)

    metrics = compute_quantization_metrics(
        original,
        result.dequantized,
        quantized=result.quantized,
        qmin=result.qmin,
        qmax=result.qmax,
    )

    assert metrics.mse >= 0.0
    assert metrics.mae >= 0.0
    assert 0.0 <= metrics.zero_fraction <= 1.0
    assert 0.0 <= metrics.saturation_fraction <= 1.0
    assert metrics.relative_spectrum_l2_error >= 0.0
    assert math.isfinite(metrics.stable_rank_change)


def test_compute_quantization_metrics_omits_integer_diagnostics_when_not_provided() -> None:
    original = np.eye(3, dtype=np.float32)

    metrics = compute_quantization_metrics(original, original.copy())

    assert metrics.zero_fraction is None
    assert metrics.saturation_fraction is None


def test_compute_quantization_metrics_requires_matching_shapes() -> None:
    original = np.zeros((2, 2), dtype=np.float32)
    reconstructed = np.zeros((2, 3), dtype=np.float32)

    with pytest.raises(ValueError, match="same shape"):
        compute_quantization_metrics(original, reconstructed)


def test_compute_quantization_metrics_requires_qmin_and_qmax_together() -> None:
    original = np.eye(2, dtype=np.float32)
    quantized = np.eye(2, dtype=np.int8)

    with pytest.raises(ValueError, match="together"):
        compute_quantization_metrics(original, original.copy(), quantized=quantized, qmin=-7)


def test_compute_quantization_metrics_rejects_non_integer_quantized_input() -> None:
    original = np.eye(2, dtype=np.float32)
    quantized = np.eye(2, dtype=np.float32)

    with pytest.raises(TypeError, match="integer"):
        compute_quantization_metrics(original, original.copy(), quantized=quantized)
