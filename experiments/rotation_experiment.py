"""Compare INT4 quantization with rotation and scaling preprocessing."""

from __future__ import annotations

import csv
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from quant.matrix_factory import outlier_matrix
from quant.metrics import QuantizationMetrics, compute_quantization_metrics
from quant.quantizer import QuantizationResult, quantize_int4
from quant.rotations import apply_rotation, rotate_channel_pair
from quant.scaling import balance_channel_max_abs, column_max_abs, invert_channel_scaling
from quant.spectrum import singular_values
from quant.visualize import plot_matrix_heatmap


@dataclass(frozen=True)
class RotationExperimentConfig:
    """Configuration for the first Milestone 2 transformation experiment."""

    shape: tuple[int, int] = (64, 64)
    seed: int = 211
    base_std: float = 1.0
    outlier_fraction: float = 0.02
    outlier_scale: float = 12.0
    rotation_search_steps: int = 360
    results_dir: Path = Path("results")
    plots_dir: Path = Path("plots")
    save_plots: bool = True


@dataclass(frozen=True)
class RotationExperimentRecord:
    """One row of rotation/scaling experiment output."""

    method: str
    shape: str
    seed: int
    base_std: float
    outlier_fraction: float
    outlier_scale: float
    rotation_i: int | None
    rotation_j: int | None
    rotation_theta: float | None
    scaling_target_max_abs: float | None
    transform_column_max_abs_spread: float
    quantization_scale: float
    quantizer: str
    quantizer_bitwidth: int
    qmin: int
    qmax: int
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
    saturation_fraction: float | None
    zero_fraction: float | None
    plot_path: str


@dataclass(frozen=True)
class _MethodResult:
    method: str
    transformed: np.ndarray
    recovered: np.ndarray
    quantization: QuantizationResult
    metrics: QuantizationMetrics
    rotation_i: int | None
    rotation_j: int | None
    rotation_theta: float | None
    scaling_target_max_abs: float | None


def run_rotation_experiment(
    config: RotationExperimentConfig = RotationExperimentConfig(),
) -> list[RotationExperimentRecord]:
    """Run baseline, rotation-only, scaling-only, and combined INT4 paths."""

    _validate_config(config)
    config.results_dir.mkdir(parents=True, exist_ok=True)
    if config.save_plots:
        config.plots_dir.mkdir(parents=True, exist_ok=True)

    matrix = outlier_matrix(
        config.shape,
        base_std=config.base_std,
        outlier_fraction=config.outlier_fraction,
        outlier_scale=config.outlier_scale,
        seed=config.seed,
    )
    rotation_i, rotation_j = _rotation_pair(matrix)
    method_results = _run_methods(matrix, config, rotation_i, rotation_j)

    plot_path = ""
    if config.save_plots:
        plot_path = str(config.plots_dir / "rotation_scaling_comparison.png")
        fig = plot_rotation_experiment_dashboard(
            matrix,
            method_results,
            output_path=plot_path,
            title="Rotation and Scaling INT4 Comparison",
        )
        plt.close(fig)

    records = [
        _make_record(
            matrix=matrix,
            method_result=result,
            config=config,
            plot_path=plot_path,
        )
        for result in method_results
    ]
    _write_csv(config.results_dir / "rotation_metrics.csv", records)
    return records


def print_summary(records: list[RotationExperimentRecord]) -> None:
    """Print a compact experiment summary."""

    print("Rotation/scaling quantization experiment")
    print("method, mse, rel_frob, snr_db, zero_frac, sat_frac")
    for record in records:
        print(
            f"{record.method}, "
            f"{record.mse:.8f}, "
            f"{record.relative_frobenius_error:.8f}, "
            f"{record.snr_db:.4f}, "
            f"{_format_optional(record.zero_fraction)}, "
            f"{_format_optional(record.saturation_fraction)}"
        )


def plot_rotation_experiment_dashboard(
    original: np.ndarray,
    method_results: list[_MethodResult],
    *,
    output_path: str | Path | None = None,
    title: str = "Rotation and Scaling INT4 Comparison",
    cmap: str = "coolwarm",
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """Plot transformed matrices, residuals, spectra, and metrics by method."""

    if not method_results:
        raise ValueError("method_results must contain at least one method")
    for result in method_results:
        if result.transformed.shape != original.shape or result.recovered.shape != original.shape:
            raise ValueError("method matrices must have the same shape as original")

    if figsize is None:
        figsize = (5.0 * len(method_results), 18.0)

    fig, axes = plt.subplots(5, len(method_results), figsize=figsize, squeeze=False)
    fig.suptitle(title, fontsize=14)
    x_positions = np.arange(original.shape[1])

    for column, result in enumerate(method_results):
        display_label = result.method.replace("_", " ").title()
        residual = result.recovered - original

        plot_matrix_heatmap(
            result.transformed,
            title=f"{display_label} Transform",
            ax=axes[0, column],
            cmap=cmap,
        )
        _set_sparse_matrix_ticks(axes[0, column], original.shape)
        plot_matrix_heatmap(
            residual,
            title=f"{display_label} Residual",
            ax=axes[1, column],
            cmap=cmap,
        )
        _set_sparse_matrix_ticks(axes[1, column], original.shape)

        axes[2, column].bar(
            x_positions,
            column_max_abs(result.transformed),
            color="tab:blue",
            alpha=0.82,
        )
        axes[2, column].set_title(f"{display_label} Column Max-Abs")
        axes[2, column].set_xlabel("Column")
        axes[2, column].set_ylabel("Max abs")
        axes[2, column].grid(axis="y", alpha=0.25)

        axes[3, column].bar(
            x_positions,
            _per_column_mse(original, result.recovered),
            color="tab:orange",
            alpha=0.82,
        )
        axes[3, column].set_title(f"{display_label} Per-Column MSE")
        axes[3, column].set_xlabel("Column")
        axes[3, column].set_ylabel("MSE")
        axes[3, column].grid(axis="y", alpha=0.25)

        _plot_spectrum_pair(
            original,
            result.recovered,
            label=display_label,
            ax=axes[4, column],
        )

    fig.text(
        0.01,
        0.01,
        _format_dashboard_summary(method_results),
        va="bottom",
        ha="left",
        family="monospace",
        fontsize=9,
    )
    fig.tight_layout(rect=(0.0, 0.05, 1.0, 0.96))

    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=160, bbox_inches="tight")

    return fig


def main() -> None:
    """Run the default rotation/scaling experiment."""

    records = run_rotation_experiment()
    print_summary(records)
    print("Saved metrics to results/rotation_metrics.csv")
    print("Saved plot to plots/rotation_scaling_comparison.png")


def _run_methods(
    matrix: np.ndarray,
    config: RotationExperimentConfig,
    rotation_i: int,
    rotation_j: int,
) -> list[_MethodResult]:
    baseline = _quantize_path(
        method="baseline",
        original=matrix,
        transformed=matrix,
        recovered_transform=lambda dequantized: dequantized,
        rotation_i=None,
        rotation_j=None,
        rotation_theta=None,
        scaling_target_max_abs=None,
    )

    rotated, theta = rotate_channel_pair(
        matrix,
        rotation_i,
        rotation_j,
        n_search=config.rotation_search_steps,
    )
    rotation_only = _quantize_path(
        method="rotation_only",
        original=matrix,
        transformed=rotated,
        recovered_transform=lambda dequantized: apply_rotation(
            dequantized,
            rotation_i,
            rotation_j,
            -theta,
        ),
        rotation_i=rotation_i,
        rotation_j=rotation_j,
        rotation_theta=theta,
        scaling_target_max_abs=None,
    )

    scaled, scaling = balance_channel_max_abs(matrix)
    scaling_only = _quantize_path(
        method="scaling_only",
        original=matrix,
        transformed=scaled,
        recovered_transform=lambda dequantized: invert_channel_scaling(
            dequantized,
            scaling,
        ),
        rotation_i=None,
        rotation_j=None,
        rotation_theta=None,
        scaling_target_max_abs=scaling.target_max_abs,
    )

    rotated_scaled, rotated_scaling = balance_channel_max_abs(rotated)
    rotation_scaling = _quantize_path(
        method="rotation_scaling",
        original=matrix,
        transformed=rotated_scaled,
        recovered_transform=lambda dequantized: apply_rotation(
            invert_channel_scaling(dequantized, rotated_scaling),
            rotation_i,
            rotation_j,
            -theta,
        ),
        rotation_i=rotation_i,
        rotation_j=rotation_j,
        rotation_theta=theta,
        scaling_target_max_abs=rotated_scaling.target_max_abs,
    )

    return [baseline, rotation_only, scaling_only, rotation_scaling]


def _quantize_path(
    *,
    method: str,
    original: np.ndarray,
    transformed: np.ndarray,
    recovered_transform,
    rotation_i: int | None,
    rotation_j: int | None,
    rotation_theta: float | None,
    scaling_target_max_abs: float | None,
) -> _MethodResult:
    result = quantize_int4(transformed)
    recovered = recovered_transform(result.dequantized)
    metrics = compute_quantization_metrics(
        original,
        recovered,
        quantized=result.quantized,
        qmin=result.qmin,
        qmax=result.qmax,
    )
    return _MethodResult(
        method=method,
        transformed=transformed,
        recovered=recovered,
        quantization=result,
        metrics=metrics,
        rotation_i=rotation_i,
        rotation_j=rotation_j,
        rotation_theta=rotation_theta,
        scaling_target_max_abs=scaling_target_max_abs,
    )


def _rotation_pair(matrix: np.ndarray) -> tuple[int, int]:
    max_abs = column_max_abs(matrix)
    order = np.argsort(max_abs)
    return int(order[-1]), int(order[-2])


def _make_record(
    *,
    matrix: np.ndarray,
    method_result: _MethodResult,
    config: RotationExperimentConfig,
    plot_path: str,
) -> RotationExperimentRecord:
    metric_values = asdict(method_result.metrics)
    q = method_result.quantization
    return RotationExperimentRecord(
        method=method_result.method,
        shape=f"{matrix.shape[0]}x{matrix.shape[1]}",
        seed=config.seed,
        base_std=config.base_std,
        outlier_fraction=config.outlier_fraction,
        outlier_scale=config.outlier_scale,
        rotation_i=method_result.rotation_i,
        rotation_j=method_result.rotation_j,
        rotation_theta=method_result.rotation_theta,
        scaling_target_max_abs=method_result.scaling_target_max_abs,
        transform_column_max_abs_spread=float(np.ptp(column_max_abs(method_result.transformed))),
        quantization_scale=q.scale,
        quantizer="int4",
        quantizer_bitwidth=q.bitwidth,
        qmin=q.qmin,
        qmax=q.qmax,
        mse=metric_values["mse"],
        mae=metric_values["mae"],
        relative_frobenius_error=metric_values["relative_frobenius_error"],
        cosine_similarity=metric_values["cosine_similarity"],
        snr_db=metric_values["snr_db"],
        max_abs_error=metric_values["max_abs_error"],
        mean_error=metric_values["mean_error"],
        error_std=metric_values["error_std"],
        spectrum_l2_error=metric_values["spectrum_l2_error"],
        relative_spectrum_l2_error=metric_values["relative_spectrum_l2_error"],
        reference_rank=metric_values["reference_rank"],
        candidate_rank=metric_values["candidate_rank"],
        reference_stable_rank=metric_values["reference_stable_rank"],
        candidate_stable_rank=metric_values["candidate_stable_rank"],
        stable_rank_change=metric_values["stable_rank_change"],
        saturation_fraction=metric_values["saturation_fraction"],
        zero_fraction=metric_values["zero_fraction"],
        plot_path=plot_path,
    )


def _write_csv(path: Path, records: list[RotationExperimentRecord]) -> None:
    if not records:
        raise ValueError("records must contain at least one row")

    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(asdict(records[0]).keys()))
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def _plot_spectrum_pair(
    original: np.ndarray,
    recovered: np.ndarray,
    *,
    label: str,
    ax: plt.Axes,
) -> None:
    indices = np.arange(1, min(original.shape) + 1)
    ax.plot(indices, singular_values(original), marker="o", linestyle=":", label="Original")
    ax.plot(indices, singular_values(recovered), marker="o", label=label)
    ax.set_title(f"{label} Spectrum")
    ax.set_xlabel("Index")
    ax.set_ylabel("Singular value")
    ax.legend()


def _format_dashboard_summary(method_results: list[_MethodResult]) -> str:
    lines = [
        "Summary",
        "method              mse       rel_frob    snr_db    zero_frac  sat_frac",
    ]
    for result in method_results:
        values = asdict(result.metrics)
        lines.append(
            f"{result.method:<17}"
            f"{values['mse']:>9.4f} "
            f"{values['relative_frobenius_error']:>10.4f} "
            f"{values['snr_db']:>9.3f} "
            f"{_format_optional(values['zero_fraction']):>10} "
            f"{_format_optional(values['saturation_fraction']):>9}"
        )
    return "\n".join(lines)


def _per_column_mse(original: np.ndarray, reconstructed: np.ndarray) -> np.ndarray:
    return np.mean((reconstructed - original) ** 2, axis=0)


def _set_sparse_matrix_ticks(ax: plt.Axes, shape: tuple[int, int]) -> None:
    row_ticks = _sparse_ticks(shape[0])
    col_ticks = _sparse_ticks(shape[1])
    ax.set_xticks(col_ticks)
    ax.set_yticks(row_ticks)
    ax.set_xticklabels([str(value) for value in col_ticks])
    ax.set_yticklabels([str(value) for value in row_ticks])


def _sparse_ticks(size: int) -> list[int]:
    if size <= 8:
        return list(range(size))
    ticks = np.linspace(0, size - 1, num=5, dtype=int)
    return sorted(set(int(value) for value in ticks))


def _format_optional(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.6f}"


def _validate_config(config: RotationExperimentConfig) -> None:
    if len(config.shape) != 2 or config.shape[0] <= 0 or config.shape[1] <= 1:
        raise ValueError("shape must be a 2D tuple with at least two columns")
    if config.base_std <= 0:
        raise ValueError("base_std must be positive")
    if not 0 <= config.outlier_fraction <= 1:
        raise ValueError("outlier_fraction must be between 0 and 1")
    if config.outlier_scale <= 0:
        raise ValueError("outlier_scale must be positive")
    if config.rotation_search_steps < 1:
        raise ValueError("rotation_search_steps must be at least 1")


if __name__ == "__main__":
    main()
