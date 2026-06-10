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


def quantize_int8(matrix: np.ndarray) -> QuantizationResult:
    """Apply symmetric INT8 quantization."""

    return symmetric_quantize(matrix, bitwidth=8)


def quantize_int4(matrix: np.ndarray) -> QuantizationResult:
    """Apply symmetric INT4 quantization."""

    return symmetric_quantize(matrix, bitwidth=4)


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
