"""Error and similarity metrics for quantization experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quant.spectrum import compare_spectra


@dataclass(frozen=True)
class QuantizationMetrics:
    """Error, similarity, and diagnostic metrics for a quantized matrix."""

    mse: float
    mae: float
    relative_frobenius_error: float
    cosine_similarity: float
    snr_db: float
    max_abs_error: float
    mean_error: float
    error_std: float
    spectrum_l2_error: float
    relative_spectrum_l2_error: float
    reference_rank: float
    candidate_rank: float
    reference_stable_rank: float
    candidate_stable_rank: float
    stable_rank_change: float
    saturation_fraction: float | None = None
    zero_fraction: float | None = None


def mean_squared_error(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Return the mean squared reconstruction error."""

    original, reconstructed = _validate_pair(original, reconstructed)
    residual = reconstructed - original
    return float(np.mean(residual**2))


def mean_absolute_error(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Return the mean absolute reconstruction error."""

    original, reconstructed = _validate_pair(original, reconstructed)
    residual = reconstructed - original
    return float(np.mean(np.abs(residual)))


def relative_frobenius_error(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Return ||original - reconstructed||_F / ||original||_F."""

    original, reconstructed = _validate_pair(original, reconstructed)
    residual_norm = float(np.linalg.norm(original - reconstructed))
    original_norm = float(np.linalg.norm(original))
    if original_norm == 0.0:
        return 0.0 if residual_norm == 0.0 else float("inf")
    return residual_norm / original_norm


def cosine_similarity(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Return cosine similarity between flattened matrices."""

    original, reconstructed = _validate_pair(original, reconstructed)
    original_flat = original.ravel()
    reconstructed_flat = reconstructed.ravel()
    original_norm = float(np.linalg.norm(original_flat))
    reconstructed_norm = float(np.linalg.norm(reconstructed_flat))

    if original_norm == 0.0 and reconstructed_norm == 0.0:
        return 1.0
    if original_norm == 0.0 or reconstructed_norm == 0.0:
        return 0.0

    return float(np.dot(original_flat, reconstructed_flat) / (original_norm * reconstructed_norm))


def signal_to_noise_ratio_db(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Return signal-to-noise ratio in decibels."""

    original, reconstructed = _validate_pair(original, reconstructed)
    signal_power = float(np.sum(original**2))
    noise_power = float(np.sum((original - reconstructed) ** 2))

    if noise_power == 0.0:
        return float("inf")
    if signal_power == 0.0:
        return float("-inf")

    return float(10.0 * np.log10(signal_power / noise_power))


def max_absolute_error(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Return the largest absolute entrywise reconstruction error."""

    original, reconstructed = _validate_pair(original, reconstructed)
    return float(np.max(np.abs(reconstructed - original)))


def compute_quantization_metrics(
    original: np.ndarray,
    reconstructed: np.ndarray,
    *,
    quantized: np.ndarray | None = None,
    qmin: int | None = None,
    qmax: int | None = None,
) -> QuantizationMetrics:
    """Compute reconstruction, spectrum, and optional integer diagnostics."""

    original, reconstructed = _validate_pair(original, reconstructed)
    residual = reconstructed - original
    spectrum_metrics = compare_spectra(original, reconstructed)

    saturation_fraction = None
    zero_fraction = None
    if quantized is not None:
        _validate_quantized(quantized, original.shape)
        zero_fraction = float(np.mean(quantized == 0))

        if qmin is not None and qmax is not None:
            saturation_fraction = float(np.mean((quantized == qmin) | (quantized == qmax)))
        elif qmin is not None or qmax is not None:
            raise ValueError("qmin and qmax must be provided together")

    reference_stable_rank = spectrum_metrics["reference_stable_rank"]
    candidate_stable_rank = spectrum_metrics["candidate_stable_rank"]

    return QuantizationMetrics(
        mse=float(np.mean(residual**2)),
        mae=float(np.mean(np.abs(residual))),
        relative_frobenius_error=relative_frobenius_error(original, reconstructed),
        cosine_similarity=cosine_similarity(original, reconstructed),
        snr_db=signal_to_noise_ratio_db(original, reconstructed),
        max_abs_error=float(np.max(np.abs(residual))),
        mean_error=float(np.mean(residual)),
        error_std=float(np.std(residual)),
        spectrum_l2_error=spectrum_metrics["spectrum_l2_error"],
        relative_spectrum_l2_error=spectrum_metrics["relative_spectrum_l2_error"],
        reference_rank=spectrum_metrics["reference_rank"],
        candidate_rank=spectrum_metrics["candidate_rank"],
        reference_stable_rank=reference_stable_rank,
        candidate_stable_rank=candidate_stable_rank,
        stable_rank_change=candidate_stable_rank - reference_stable_rank,
        saturation_fraction=saturation_fraction,
        zero_fraction=zero_fraction,
    )


def _validate_pair(original: np.ndarray, reconstructed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if original.shape != reconstructed.shape:
        raise ValueError("original and reconstructed must have the same shape")
    if original.ndim != 2:
        raise ValueError("matrices must be 2D arrays")
    if not np.issubdtype(original.dtype, np.floating):
        raise TypeError("original must contain floating-point values")
    if not np.issubdtype(reconstructed.dtype, np.floating):
        raise TypeError("reconstructed must contain floating-point values")
    return original.astype(np.float64, copy=False), reconstructed.astype(np.float64, copy=False)


def _validate_quantized(quantized: np.ndarray, expected_shape: tuple[int, ...]) -> None:
    if quantized.shape != expected_shape:
        raise ValueError("quantized must have the same shape as original")
    if not np.issubdtype(quantized.dtype, np.integer):
        raise TypeError("quantized must contain integer values")
