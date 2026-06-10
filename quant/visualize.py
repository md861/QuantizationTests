"""Visualization helpers for matrices, residuals, and quantization effects."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from math import isfinite
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from quant.metrics import QuantizationMetrics, compute_quantization_metrics
from quant.quantizer import QuantizationResult
from quant.spectrum import analyze_spectrum, singular_values


def plot_matrix_heatmap(
    matrix: np.ndarray,
    *,
    title: str = "Matrix",
    ax: Axes | None = None,
    cmap: str = "viridis",
    show_colorbar: bool = True,
) -> Axes:
    """Plot a single matrix as a heatmap."""

    _validate_matrix(matrix)

    if ax is None:
        _, ax = plt.subplots(figsize=(4, 3))

    image = ax.imshow(matrix, cmap=cmap, aspect="equal")
    ax.set_title(title)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_yticks(range(matrix.shape[0]))

    if show_colorbar:
        ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    return ax


def plot_matrix_grid(
    matrices: Mapping[str, np.ndarray],
    *,
    output_path: str | Path | None = None,
    cmap: str = "viridis",
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """Plot several matrices side by side as heatmaps.

    Args:
        matrices: Mapping of plot titles to 2D arrays.
        output_path: Optional path where the figure should be saved.
        cmap: Matplotlib colormap name.
        figsize: Optional figure size. A sensible default is chosen from the
            number of matrices.
    """

    if not matrices:
        raise ValueError("matrices must contain at least one matrix")

    titles = list(matrices.keys())
    arrays = list(matrices.values())
    for matrix in arrays:
        _validate_matrix(matrix)

    if figsize is None:
        figsize = (4.0 * len(arrays), 3.5)

    fig, axes = plt.subplots(1, len(arrays), figsize=figsize, squeeze=False)

    for title, matrix, ax in zip(titles, arrays, axes[0], strict=True):
        plot_matrix_heatmap(matrix, title=title, ax=ax, cmap=cmap)

    fig.tight_layout()

    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=160, bbox_inches="tight")

    return fig


def plot_singular_values(
    matrix: np.ndarray,
    *,
    title: str = "Singular Values",
    ax: Axes | None = None,
    marker: str = "o",
    log_scale: bool = False,
) -> Axes:
    """Plot the singular values of one matrix."""

    _validate_matrix(matrix)

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 3.5))

    values = singular_values(matrix)
    indices = np.arange(1, values.size + 1)
    ax.plot(indices, values, marker=marker)
    ax.set_title(title)
    ax.set_xlabel("Index")
    ax.set_ylabel("Singular value")
    ax.set_xticks(indices)

    if log_scale:
        ax.set_yscale("log")

    return ax


def plot_spectrum_comparison(
    matrices: Mapping[str, np.ndarray],
    *,
    ax: Axes | None = None,
    title: str = "Singular Value Spectra",
    output_path: str | Path | None = None,
    log_scale: bool = False,
    figsize: tuple[float, float] = (6.5, 4.0),
) -> Figure:
    """Plot singular-value spectra for multiple matrices on one axis."""

    if not matrices:
        raise ValueError("matrices must contain at least one matrix")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    for label, matrix in matrices.items():
        _validate_matrix(matrix)
        values = singular_values(matrix)
        indices = np.arange(1, values.size + 1)
        ax.plot(
            indices,
            values,
            marker="o",
            label=label,
            linestyle=_spectrum_linestyle(label),
            linewidth=_spectrum_linewidth(label),
            alpha=_spectrum_alpha(label),
        )

    ax.set_title(title)
    ax.set_xlabel("Index")
    ax.set_ylabel("Singular value")
    ax.legend()

    if log_scale:
        ax.set_yscale("log")

    if output_path is not None or ax is None:
        fig.tight_layout()

    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=160, bbox_inches="tight")

    return fig


def plot_quantization_summary(
    original: np.ndarray,
    result: QuantizationResult,
    *,
    output_path: str | Path | None = None,
    title: str = "Quantization Summary",
    cmap: str = "coolwarm",
    metrics: QuantizationMetrics | None = None,
    figsize: tuple[float, float] = (18.0, 8.0),
) -> Figure:
    """Plot matrices, residuals, spectra, and metric summary."""

    _validate_matrix(original)
    if result.dequantized.shape != original.shape:
        raise ValueError("dequantized matrix must have the same shape as original")
    if result.quantized.shape != original.shape:
        raise ValueError("quantized matrix must have the same shape as original")

    if metrics is None:
        metrics = compute_quantization_metrics(
            original,
            result.dequantized,
            quantized=result.quantized,
            qmin=result.qmin,
            qmax=result.qmax,
        )

    residual = result.dequantized - original
    fig, axes = plt.subplots(2, 4, figsize=figsize)

    fig.suptitle(title, fontsize=14)
    plot_matrix_heatmap(original, title="Original", ax=axes[0, 0], cmap=cmap)
    plot_matrix_heatmap(result.quantized, title="Quantized Codes", ax=axes[0, 1], cmap=cmap)
    plot_matrix_heatmap(result.dequantized, title="Dequantized", ax=axes[0, 2], cmap=cmap)
    plot_matrix_heatmap(residual, title="Residual", ax=axes[0, 3], cmap=cmap)

    plot_spectrum_comparison(
        {
            "Original": original,
            "Quantized codes": result.quantized.astype(np.float64),
            "Dequantized": result.dequantized,
        },
        ax=axes[1, 0],
        title="Singular Value Spectra",
    )

    axes[1, 1].axis("off")
    axes[1, 2].axis("off")
    axes[1, 3].axis("off")
    axes[1, 1].text(
        0.0,
        1.0,
        _format_quantization_summary(original, result, metrics),
        va="top",
        ha="left",
        family="monospace",
        fontsize=9,
    )

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))

    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=160, bbox_inches="tight")

    return fig


def plot_quantization_comparison(
    original: np.ndarray,
    results: Mapping[str, QuantizationResult],
    *,
    output_path: str | Path | None = None,
    title: str = "Quantization Comparison",
    cmap: str = "coolwarm",
    metrics: Mapping[str, QuantizationMetrics] | None = None,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """Compare residuals, spectra, and metrics for multiple quantizers."""

    _validate_matrix(original)
    if not results:
        raise ValueError("results must contain at least one quantization result")

    if metrics is None:
        metrics = {
            label: compute_quantization_metrics(
                original,
                result.dequantized,
                quantized=result.quantized,
                qmin=result.qmin,
                qmax=result.qmax,
            )
            for label, result in results.items()
        }

    _validate_comparison_inputs(original, results, metrics)

    labels = list(results.keys())
    column_count = len(labels) + 1
    if figsize is None:
        figsize = (5.0 * column_count, 11.0)

    fig, axes = plt.subplots(3, column_count, figsize=figsize, squeeze=False)
    fig.suptitle(title, fontsize=14)

    plot_matrix_heatmap(original, title="Original", ax=axes[0, 0], cmap=cmap)
    axes[1, 0].axis("off")
    axes[2, 0].axis("off")
    axes[2, 0].text(
        0.0,
        1.0,
        _format_original_summary(original),
        va="top",
        ha="left",
        family="monospace",
        fontsize=9,
    )

    for column, label in enumerate(labels, start=1):
        result = results[label]
        residual = result.dequantized - original
        display_label = label.upper()

        plot_matrix_heatmap(
            residual,
            title=f"{display_label} Residual",
            ax=axes[0, column],
            cmap=cmap,
        )
        plot_spectrum_comparison(
            {
                "Original": original,
                f"{display_label} dequantized": result.dequantized,
            },
            ax=axes[1, column],
            title=f"{display_label} Spectra",
        )
        axes[2, column].axis("off")
        axes[2, column].text(
            0.0,
            1.0,
            _format_quantization_method_summary(result, metrics[label]),
            va="top",
            ha="left",
            family="monospace",
            fontsize=9,
        )

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))

    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=160, bbox_inches="tight")

    return fig


def _validate_matrix(matrix: np.ndarray) -> None:
    if matrix.ndim != 2:
        raise ValueError("matrix must be a 2D array")


def _validate_comparison_inputs(
    original: np.ndarray,
    results: Mapping[str, QuantizationResult],
    metrics: Mapping[str, QuantizationMetrics],
) -> None:
    if set(results.keys()) != set(metrics.keys()):
        raise ValueError("results and metrics must have matching labels")

    for result in results.values():
        if result.dequantized.shape != original.shape:
            raise ValueError("dequantized matrices must have the same shape as original")
        if result.quantized.shape != original.shape:
            raise ValueError("quantized matrices must have the same shape as original")


def _format_original_summary(original: np.ndarray) -> str:
    spectrum = analyze_spectrum(original)
    return "\n".join(
        [
            "Original",
            f"  dtype: {original.dtype}",
            f"  bits: {original.dtype.itemsize * 8}",
            f"  shape: {original.shape[0]}x{original.shape[1]}",
            "",
            "Spectrum",
            f"  rank: {spectrum.rank}",
            f"  stable_rank: {_format_float(spectrum.stable_rank)}",
            f"  top_sv: {_format_top_singular_values(spectrum.singular_values)}",
        ]
    )


def _format_quantization_method_summary(
    result: QuantizationResult,
    metrics: QuantizationMetrics,
) -> str:
    metric_values = asdict(metrics)
    return "\n".join(
        [
            f"{result.bitwidth}-bit Quantizer",
            f"  range: [{result.qmin}, {result.qmax}]",
            f"  scale: {_format_float(result.scale)}",
            f"  storage: {result.quantized.dtype}",
            f"  storage_bits: {result.quantized.dtype.itemsize * 8}",
            "",
            "Error Metrics",
            f"  mse: {_format_float(metric_values['mse'])}",
            f"  mae: {_format_float(metric_values['mae'])}",
            "  rel_frob: "
            f"{_format_float(metric_values['relative_frobenius_error'])}",
            f"  cosine: {_format_float(metric_values['cosine_similarity'])}",
            f"  snr_db: {_format_float(metric_values['snr_db'])}",
            f"  max_abs: {_format_float(metric_values['max_abs_error'])}",
            f"  mean_err: {_format_float(metric_values['mean_error'])}",
            f"  err_std: {_format_float(metric_values['error_std'])}",
            "",
            "Diagnostics",
            f"  sat_frac: {_format_optional_float(metric_values['saturation_fraction'])}",
            f"  zero_frac: {_format_optional_float(metric_values['zero_fraction'])}",
            "",
            "Spectrum",
            "  rel_spec_err: "
            f"{_format_float(metric_values['relative_spectrum_l2_error'])}",
            "  stable_rank_delta: "
            f"{_format_float(metric_values['stable_rank_change'])}",
            f"  rank: {int(metric_values['reference_rank'])} -> "
            f"{int(metric_values['candidate_rank'])}",
        ]
    )


def _format_quantization_summary(
    original: np.ndarray,
    result: QuantizationResult,
    metrics: QuantizationMetrics,
) -> str:
    metric_values = asdict(metrics)
    original_spectrum = analyze_spectrum(original)
    quantized_spectrum = analyze_spectrum(result.quantized.astype(np.float64))
    dequantized_spectrum = analyze_spectrum(result.dequantized)
    lines = [
        "Data",
        f"  original_dtype: {original.dtype}",
        f"  original_bits: {original.dtype.itemsize * 8}",
        f"  quantized_dtype: {result.quantized.dtype}",
        f"  storage_bits: {result.quantized.dtype.itemsize * 8}",
        "",
        "Quantizer",
        f"  bitwidth: {result.bitwidth}",
        f"  range: [{result.qmin}, {result.qmax}]",
        f"  scale: {_format_float(result.scale)}",
        "",
        "Error Metrics",
        f"  mse: {_format_float(metric_values['mse'])}",
        f"  mae: {_format_float(metric_values['mae'])}",
        "  rel_frob: "
        f"{_format_float(metric_values['relative_frobenius_error'])}",
        f"  cosine: {_format_float(metric_values['cosine_similarity'])}",
        f"  snr_db: {_format_float(metric_values['snr_db'])}",
        f"  max_abs: {_format_float(metric_values['max_abs_error'])}",
        f"  mean_err: {_format_float(metric_values['mean_error'])}",
        f"  err_std: {_format_float(metric_values['error_std'])}",
        "",
        "Diagnostics",
        f"  sat_frac: {_format_optional_float(metric_values['saturation_fraction'])}",
        f"  zero_frac: {_format_optional_float(metric_values['zero_fraction'])}",
        "",
        "Spectrum",
        "  rel_spec_err: "
        f"{_format_float(metric_values['relative_spectrum_l2_error'])}",
        "  stable_rank_delta: "
        f"{_format_float(metric_values['stable_rank_change'])}",
        f"  rank: {int(metric_values['reference_rank'])} -> "
        f"{int(metric_values['candidate_rank'])}",
        "  stable_rank:",
        f"    orig  {_format_float(original_spectrum.stable_rank)}",
        f"    qcode {_format_float(quantized_spectrum.stable_rank)}",
        f"    deq   {_format_float(dequantized_spectrum.stable_rank)}",
        "  top_sv:",
        f"    orig  {_format_top_singular_values(original_spectrum.singular_values)}",
        f"    qcode {_format_top_singular_values(quantized_spectrum.singular_values)}",
        f"    deq   {_format_top_singular_values(dequantized_spectrum.singular_values)}",
    ]
    return "\n".join(lines)


def _format_optional_float(value: float | None) -> str:
    if value is None:
        return "n/a"
    return _format_float(value)


def _format_float(value: float) -> str:
    if not isfinite(value):
        return str(value)
    if abs(value) >= 1000 or 0 < abs(value) < 0.001:
        return f"{value:.4e}"
    return f"{value:.6f}"


def _format_top_singular_values(values: np.ndarray, *, count: int = 3) -> str:
    top_values = values[:count]
    return "[" + ", ".join(_format_float(float(value)) for value in top_values) + "]"


def _spectrum_linestyle(label: str) -> str:
    if label.lower() == "original":
        return ":"
    return "-"


def _spectrum_linewidth(label: str) -> float:
    if label.lower() == "original":
        return 2.4
    return 1.2


def _spectrum_alpha(label: str) -> float:
    if label.lower() == "original":
        return 0.95
    return 0.85
