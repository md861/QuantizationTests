"""Tests for plotting helpers."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from quant.visualize import (
    plot_matrix_grid,
    plot_matrix_heatmap,
    plot_quantization_comparison,
    plot_quantization_summary,
    plot_singular_values,
    plot_spectrum_comparison,
)
from quant.quantizer import quantize_int4, quantize_int8


def test_plot_matrix_heatmap_returns_axes() -> None:
    matrix = np.arange(9, dtype=np.float32).reshape(3, 3)

    ax = plot_matrix_heatmap(matrix, title="Example")

    assert ax.get_title() == "Example"
    plt.close(ax.figure)


def test_plot_matrix_grid_saves_png(tmp_path) -> None:
    matrices = {
        "A": np.zeros((3, 3), dtype=np.float32),
        "B": np.ones((3, 3), dtype=np.float32),
    }
    output_path = tmp_path / "grid.png"

    fig = plot_matrix_grid(matrices, output_path=output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0
    plt.close(fig)


def test_plot_matrix_grid_requires_at_least_one_matrix() -> None:
    with pytest.raises(ValueError, match="at least one"):
        plot_matrix_grid({})


def test_plot_matrix_heatmap_rejects_non_2d_input() -> None:
    with pytest.raises(ValueError, match="2D"):
        plot_matrix_heatmap(np.zeros((2, 2, 2), dtype=np.float32))


def test_plot_singular_values_returns_axes() -> None:
    matrix = np.eye(3, dtype=np.float32)

    ax = plot_singular_values(matrix, title="Spectrum")

    assert ax.get_title() == "Spectrum"
    plt.close(ax.figure)


def test_plot_spectrum_comparison_saves_png(tmp_path) -> None:
    matrices = {
        "Identity": np.eye(3, dtype=np.float32),
        "Scaled": np.eye(3, dtype=np.float32) * 2.0,
    }
    output_path = tmp_path / "spectra.png"

    fig = plot_spectrum_comparison(matrices, output_path=output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0
    plt.close(fig)


def test_plot_quantization_summary_saves_png(tmp_path) -> None:
    matrix = np.arange(9, dtype=np.float32).reshape(3, 3)
    result = quantize_int4(matrix)
    output_path = tmp_path / "quantization_summary.png"

    fig = plot_quantization_summary(matrix, result, output_path=output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0
    plt.close(fig)


def test_plot_quantization_comparison_saves_png(tmp_path) -> None:
    matrix = np.arange(16, dtype=np.float32).reshape(4, 4)
    results = {
        "int8": quantize_int8(matrix),
        "int4": quantize_int4(matrix),
    }
    output_path = tmp_path / "quantization_comparison.png"

    fig = plot_quantization_comparison(matrix, results, output_path=output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0
    assert len(fig.axes) >= 9
    plt.close(fig)


def test_plot_quantization_summary_requires_matching_shapes() -> None:
    matrix = np.zeros((2, 2), dtype=np.float32)
    result = quantize_int4(np.zeros((3, 3), dtype=np.float32))

    with pytest.raises(ValueError, match="same shape"):
        plot_quantization_summary(matrix, result)


def test_plot_quantization_comparison_requires_matching_shapes() -> None:
    matrix = np.zeros((2, 2), dtype=np.float32)
    results = {"int4": quantize_int4(np.zeros((3, 3), dtype=np.float32))}

    with pytest.raises(ValueError, match="same shape"):
        plot_quantization_comparison(matrix, results)
