"""Symmetric low-bit quantization utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class QuantizationResult:
    """Container for quantized values and reconstruction metadata."""

    quantized: np.ndarray
    dequantized: np.ndarray
    scale: float
    bitwidth: int
    qmin: int
    qmax: int
    scales: np.ndarray | None = None
    group_size: int | None = None


def symmetric_quantize(matrix: np.ndarray, *, bitwidth: int) -> QuantizationResult:
    """Quantize and dequantize a matrix with symmetric signed quantization.

    Uses the baseline formula:

        scale = max(abs(matrix)) / (2 ** (bitwidth - 1) - 1)
        q = round(matrix / scale)
        matrix_hat = scale * q

    The zero matrix is handled explicitly with `scale=1.0` to avoid division
    by zero while still reconstructing exactly to all zeros.
    """

    _validate_matrix(matrix)
    qmin, qmax = _symmetric_range(bitwidth)
    output_dtype = _integer_dtype(bitwidth)

    max_abs = float(np.max(np.abs(matrix)))
    if max_abs == 0.0:
        quantized = np.zeros_like(matrix, dtype=output_dtype)
        dequantized = np.zeros_like(matrix, dtype=matrix.dtype)
        return QuantizationResult(
            quantized=quantized,
            dequantized=dequantized,
            scale=1.0,
            bitwidth=bitwidth,
            qmin=qmin,
            qmax=qmax,
        )

    scale = max_abs / qmax
    quantized = np.round(matrix / scale)
    quantized = np.clip(quantized, qmin, qmax).astype(output_dtype)
    dequantized = (quantized.astype(np.float64) * scale).astype(matrix.dtype, copy=False)

    return QuantizationResult(
        quantized=quantized,
        dequantized=dequantized,
        scale=scale,
        bitwidth=bitwidth,
        qmin=qmin,
        qmax=qmax,
    )


def grouped_symmetric_quantize(
    matrix: np.ndarray,
    *,
    bitwidth: int,
    group_size: int,
) -> QuantizationResult:
    """Quantize contiguous column groups with one symmetric scale per group."""

    _validate_matrix(matrix)
    if group_size <= 0:
        raise ValueError("group_size must be positive")

    qmin, qmax = _symmetric_range(bitwidth)
    output_dtype = _integer_dtype(bitwidth)
    n_cols = matrix.shape[1]

    quantized = np.zeros_like(matrix, dtype=output_dtype)
    dequantized = np.zeros_like(matrix, dtype=matrix.dtype)
    scales: list[float] = []

    for start in range(0, n_cols, group_size):
        end = min(start + group_size, n_cols)
        group = matrix[:, start:end]
        scale = _scale_for_values(group, qmax)
        scales.append(scale)

        group_quantized = np.round(group / scale)
        group_quantized = np.clip(group_quantized, qmin, qmax).astype(output_dtype)
        quantized[:, start:end] = group_quantized
        dequantized[:, start:end] = (
            group_quantized.astype(np.float64) * scale
        ).astype(matrix.dtype, copy=False)

    scale_array = np.array(scales, dtype=np.float64)
    return QuantizationResult(
        quantized=quantized,
        dequantized=dequantized,
        scale=float(np.mean(scale_array)),
        bitwidth=bitwidth,
        qmin=qmin,
        qmax=qmax,
        scales=scale_array,
        group_size=group_size,
    )


def quantize_int8(matrix: np.ndarray) -> QuantizationResult:
    """Apply symmetric INT8 quantization."""

    return symmetric_quantize(matrix, bitwidth=8)


def quantize_int4(matrix: np.ndarray) -> QuantizationResult:
    """Apply symmetric INT4 quantization."""

    return symmetric_quantize(matrix, bitwidth=4)


def quantize_int8_grouped(matrix: np.ndarray, *, group_size: int) -> QuantizationResult:
    """Apply grouped symmetric INT8 quantization over column groups."""

    return grouped_symmetric_quantize(matrix, bitwidth=8, group_size=group_size)


def quantize_int4_grouped(matrix: np.ndarray, *, group_size: int) -> QuantizationResult:
    """Apply grouped symmetric INT4 quantization over column groups."""

    return grouped_symmetric_quantize(matrix, bitwidth=4, group_size=group_size)


def _scale_for_values(values: np.ndarray, qmax: int) -> float:
    max_abs = float(np.max(np.abs(values)))
    if max_abs == 0.0:
        return 1.0
    return max_abs / qmax


def _symmetric_range(bitwidth: int) -> tuple[int, int]:
    if bitwidth not in {4, 8}:
        raise ValueError("bitwidth must be 4 or 8")

    qmax = (2 ** (bitwidth - 1)) - 1
    qmin = -qmax
    return qmin, qmax


def _integer_dtype(bitwidth: int) -> type[np.signedinteger]:
    if bitwidth == 8:
        return np.int8
    if bitwidth == 4:
        # NumPy has no int4 dtype, so INT4 values are stored in int8.
        return np.int8
    raise ValueError("bitwidth must be 4 or 8")


def _validate_matrix(matrix: np.ndarray) -> None:
    if matrix.ndim != 2:
        raise ValueError("matrix must be a 2D array")
    if not np.issubdtype(matrix.dtype, np.floating):
        raise TypeError("matrix must contain floating-point values")
