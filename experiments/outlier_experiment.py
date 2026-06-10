"""Sweep outlier severity for symmetric INT8 and INT4 quantization."""

from __future__ import annotations

import csv
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from quant.matrix_factory import outlier_matrix
from quant.metrics import QuantizationMetrics, compute_quantization_metrics
from quant.quantizer import QuantizationResult, quantize_int4, quantize_int8
from quant.visualize import plot_quantization_comparison


Quantizer = Callable[[np.ndarray], QuantizationResult]


@dataclass(frozen=True)
class OutlierExperimentConfig:
    """Configuration for the outlier-severity quantization sweep."""

    shape: tuple[int, int] = (64, 64)
    seed: int = 123
    base_std: float = 1.0
    outlier_fractions: tuple[float, ...] = (0.001, 0.005, 0.01, 0.02)
    outlier_scales: tuple[float, ...] = (4.0, 10.0, 20.0)
    results_dir: Path = Path("results")
    plots_dir: Path = Path("plots")
    save_plots: bool = True


@dataclass(frozen=True)
class OutlierRecord:
    """One row of outlier experiment output."""

    matrix_kind: str
    quantizer: str
    shape: str
    seed: int
    base_std: float
    outlier_fraction: float
    outlier_scale: float
    outlier_count: int
    quantization_scale: float
    quantizer_bitwidth: int
    qmin: int
    qmax: int
    original_dtype: str
    original_bits: int
    quantized_dtype: str
    quantized_storage_bits: int
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


def run_outlier_experiment(
    config: OutlierExperimentConfig = OutlierExperimentConfig(),
) -> list[OutlierRecord]:
    """Run the outlier-severity sweep and write CSV and optional plots."""

    _validate_config(config)
    config.results_dir.mkdir(parents=True, exist_ok=True)
    if config.save_plots:
        config.plots_dir.mkdir(parents=True, exist_ok=True)

    records: list[OutlierRecord] = []
    for outlier_fraction in config.outlier_fractions:
        for outlier_scale in config.outlier_scales:
            matrix = outlier_matrix(
                config.shape,
                base_std=config.base_std,
                outlier_fraction=outlier_fraction,
                outlier_scale=outlier_scale,
                seed=config.seed,
            )
            outlier_count = int(round(matrix.size * outlier_fraction))

            quantization_results: dict[str, QuantizationResult] = {}
            metric_results: dict[str, QuantizationMetrics] = {}
            for quantizer_name, quantizer in _quantizers().items():
                result = quantizer(matrix)
                metrics = compute_quantization_metrics(
                    matrix,
                    result.dequantized,
                    quantized=result.quantized,
                    qmin=result.qmin,
                    qmax=result.qmax,
                )
                quantization_results[quantizer_name] = result
                metric_results[quantizer_name] = metrics

            plot_path = ""
            if config.save_plots:
                plot_path = str(config.plots_dir / _plot_filename(outlier_fraction, outlier_scale))
                fig = plot_quantization_comparison(
                    matrix,
                    quantization_results,
                    output_path=plot_path,
                    title=(
                        "Outlier sweep quantization comparison "
                        f"fraction={outlier_fraction:g} "
                        f"scale={outlier_scale:g}"
                    ),
                    metrics=metric_results,
                )
                plt.close(fig)

            for quantizer_name, result in quantization_results.items():
                records.append(
                    _make_record(
                        matrix=matrix,
                        result=result,
                        metrics=metric_results[quantizer_name],
                        config=config,
                        quantizer_name=quantizer_name,
                        outlier_fraction=outlier_fraction,
                        outlier_scale=outlier_scale,
                        outlier_count=outlier_count,
                        plot_path=plot_path,
                    )
                )

    _write_csv(config.results_dir / "outlier_metrics.csv", records)
    return records


def print_summary(records: list[OutlierRecord]) -> None:
    """Print a compact experiment summary."""

    print("Outlier quantization experiment")
    print("fraction, scale, quantizer, mse, rel_frob, cosine, snr_db, zero_frac, sat_frac")
    for record in records:
        print(
            f"{record.outlier_fraction:g}, {record.outlier_scale:g}, "
            f"{record.quantizer}, "
            f"{record.mse:.8f}, "
            f"{record.relative_frobenius_error:.8f}, "
            f"{record.cosine_similarity:.8f}, "
            f"{record.snr_db:.4f}, "
            f"{_format_optional(record.zero_fraction)}, "
            f"{_format_optional(record.saturation_fraction)}"
        )


def main() -> None:
    """Run the default outlier-severity experiment."""

    records = run_outlier_experiment()
    print_summary(records)
    print("Saved metrics to results/outlier_metrics.csv")
    print("Saved plots to plots/outlier_fraction_<fraction>_scale_<scale>_comparison.png")


def _quantizers() -> dict[str, Quantizer]:
    return {
        "int8": quantize_int8,
        "int4": quantize_int4,
    }


def _make_record(
    *,
    matrix: np.ndarray,
    result: QuantizationResult,
    metrics: QuantizationMetrics,
    config: OutlierExperimentConfig,
    quantizer_name: str,
    outlier_fraction: float,
    outlier_scale: float,
    outlier_count: int,
    plot_path: str,
) -> OutlierRecord:
    metric_values = asdict(metrics)
    return OutlierRecord(
        matrix_kind="outlier",
        quantizer=quantizer_name,
        shape=f"{matrix.shape[0]}x{matrix.shape[1]}",
        seed=config.seed,
        base_std=config.base_std,
        outlier_fraction=outlier_fraction,
        outlier_scale=outlier_scale,
        outlier_count=outlier_count,
        quantization_scale=result.scale,
        quantizer_bitwidth=result.bitwidth,
        qmin=result.qmin,
        qmax=result.qmax,
        original_dtype=str(matrix.dtype),
        original_bits=matrix.dtype.itemsize * 8,
        quantized_dtype=str(result.quantized.dtype),
        quantized_storage_bits=result.quantized.dtype.itemsize * 8,
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


def _write_csv(path: Path, records: list[OutlierRecord]) -> None:
    if not records:
        raise ValueError("records must contain at least one row")

    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(asdict(records[0]).keys()))
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def _plot_filename(outlier_fraction: float, outlier_scale: float) -> str:
    fraction_label = _number_label(outlier_fraction)
    scale_label = _number_label(outlier_scale)
    return f"outlier_fraction_{fraction_label}_scale_{scale_label}_comparison.png"


def _number_label(value: float) -> str:
    return f"{value:g}".replace("-", "neg").replace(".", "p")


def _format_optional(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.6f}"


def _validate_config(config: OutlierExperimentConfig) -> None:
    if len(config.shape) != 2 or config.shape[0] <= 0 or config.shape[1] <= 0:
        raise ValueError("shape must be a 2D tuple with positive dimensions")
    if config.base_std <= 0:
        raise ValueError("base_std must be positive")
    if not config.outlier_fractions:
        raise ValueError("outlier_fractions must contain at least one value")
    if not config.outlier_scales:
        raise ValueError("outlier_scales must contain at least one value")
    if any(not 0 <= fraction <= 1 for fraction in config.outlier_fractions):
        raise ValueError("outlier_fractions must be between 0 and 1")
    if any(scale <= 0 for scale in config.outlier_scales):
        raise ValueError("outlier_scales must be positive")


if __name__ == "__main__":
    main()
