"""Singular-value spectrum analysis utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SpectrumStats:
    """Summary statistics for a matrix singular-value spectrum."""

    singular_values: np.ndarray
    rank: int
    spectral_norm: float
    nuclear_norm: float
    frobenius_norm: float
    condition_number: float
    stable_rank: float
    explained_energy: np.ndarray


def singular_values(matrix: np.ndarray) -> np.ndarray:
    """Return singular values sorted from largest to smallest."""

    _validate_matrix(matrix)
    return np.linalg.svd(matrix, compute_uv=False)


def explained_energy(spectrum: np.ndarray) -> np.ndarray:
    """Return cumulative squared singular-value energy fractions."""

    _validate_spectrum(spectrum)
    energy = spectrum**2
    total_energy = float(np.sum(energy))
    if total_energy == 0.0:
        return np.zeros_like(spectrum, dtype=np.float64)
    return np.cumsum(energy, dtype=np.float64) / total_energy


def analyze_spectrum(matrix: np.ndarray, *, rank_tol: float = 1e-7) -> SpectrumStats:
    """Compute singular values and useful spectrum-level summary statistics."""

    if rank_tol < 0:
        raise ValueError("rank_tol must be non-negative")

    values = singular_values(matrix).astype(np.float64, copy=False)
    rank = int(np.sum(values > rank_tol))
    spectral_norm = float(values[0]) if values.size else 0.0
    nuclear_norm = float(np.sum(values))
    frobenius_norm = float(np.linalg.norm(values))

    nonzero_values = values[values > rank_tol]
    if nonzero_values.size == 0:
        condition_number = float("inf")
    else:
        condition_number = float(nonzero_values[0] / nonzero_values[-1])

    if spectral_norm == 0.0:
        stable_rank = 0.0
    else:
        stable_rank = float((frobenius_norm**2) / (spectral_norm**2))

    return SpectrumStats(
        singular_values=values,
        rank=rank,
        spectral_norm=spectral_norm,
        nuclear_norm=nuclear_norm,
        frobenius_norm=frobenius_norm,
        condition_number=condition_number,
        stable_rank=stable_rank,
        explained_energy=explained_energy(values),
    )


def compare_spectra(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    rank_tol: float = 1e-7,
) -> dict[str, float]:
    """Compare two matrices by their singular-value spectra."""

    reference_stats = analyze_spectrum(reference, rank_tol=rank_tol)
    candidate_stats = analyze_spectrum(candidate, rank_tol=rank_tol)

    reference_values = reference_stats.singular_values
    candidate_values = candidate_stats.singular_values
    length = min(reference_values.size, candidate_values.size)

    if length == 0:
        spectrum_l2_error = 0.0
        relative_spectrum_l2_error = 0.0
    else:
        difference = reference_values[:length] - candidate_values[:length]
        spectrum_l2_error = float(np.linalg.norm(difference))
        reference_norm = float(np.linalg.norm(reference_values[:length]))
        relative_spectrum_l2_error = (
            0.0 if reference_norm == 0.0 else spectrum_l2_error / reference_norm
        )

    return {
        "reference_rank": float(reference_stats.rank),
        "candidate_rank": float(candidate_stats.rank),
        "reference_stable_rank": reference_stats.stable_rank,
        "candidate_stable_rank": candidate_stats.stable_rank,
        "spectrum_l2_error": spectrum_l2_error,
        "relative_spectrum_l2_error": relative_spectrum_l2_error,
    }


def _validate_matrix(matrix: np.ndarray) -> None:
    if matrix.ndim != 2:
        raise ValueError("matrix must be a 2D array")


def _validate_spectrum(spectrum: np.ndarray) -> None:
    if spectrum.ndim != 1:
        raise ValueError("spectrum must be a 1D array")
    if np.any(spectrum < 0):
        raise ValueError("spectrum values must be non-negative")
