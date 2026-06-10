"""Tests for singular-value spectrum analysis."""

import math

import numpy as np
import pytest

from quant.spectrum import analyze_spectrum, compare_spectra, explained_energy, singular_values


def test_singular_values_match_numpy_svd() -> None:
    matrix = np.array([[3.0, 0.0], [0.0, 4.0]], dtype=np.float32)

    values = singular_values(matrix)

    np.testing.assert_allclose(values, np.array([4.0, 3.0]))


def test_analyze_spectrum_reports_summary_stats() -> None:
    matrix = np.diag(np.array([4.0, 2.0, 0.0], dtype=np.float32))

    stats = analyze_spectrum(matrix)

    np.testing.assert_allclose(stats.singular_values, np.array([4.0, 2.0, 0.0]))
    assert stats.rank == 2
    assert stats.spectral_norm == pytest.approx(4.0)
    assert stats.nuclear_norm == pytest.approx(6.0)
    assert stats.frobenius_norm == pytest.approx(math.sqrt(20.0))
    assert stats.condition_number == pytest.approx(2.0)
    assert stats.stable_rank == pytest.approx(20.0 / 16.0)
    np.testing.assert_allclose(stats.explained_energy, np.array([0.8, 1.0, 1.0]))


def test_explained_energy_handles_zero_spectrum() -> None:
    energy = explained_energy(np.zeros(3, dtype=np.float32))

    np.testing.assert_array_equal(energy, np.zeros(3))


def test_compare_spectra_reports_zero_error_for_matching_matrices() -> None:
    matrix = np.eye(3, dtype=np.float32)

    comparison = compare_spectra(matrix, matrix)

    assert comparison["spectrum_l2_error"] == pytest.approx(0.0)
    assert comparison["relative_spectrum_l2_error"] == pytest.approx(0.0)
    assert comparison["reference_rank"] == pytest.approx(3.0)
    assert comparison["candidate_rank"] == pytest.approx(3.0)


def test_spectrum_rejects_non_2d_matrix() -> None:
    with pytest.raises(ValueError, match="2D"):
        singular_values(np.zeros((2, 2, 2), dtype=np.float32))


def test_explained_energy_rejects_negative_values() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        explained_energy(np.array([1.0, -1.0]))
